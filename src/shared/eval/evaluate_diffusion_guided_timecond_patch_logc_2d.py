from typing import Dict, List, Optional

import numpy as np
import torch

from src.obj2.conference.models.flow_history_encoder import FlowHistoryEncoder
from src.shared.eval.centroid_error import compute_plume_centroid_error
from src.shared.eval.flow_conditioning import compute_plume_flow_channels
from src.shared.eval.plume_ssim import compute_global_ssim, compute_plume_ssim


def normalize_k(k_field: np.ndarray, stats: Dict, use_logK: bool, eps_k: float) -> np.ndarray:
    k_field = k_field.astype(np.float64)
    if use_logK:
        k_field = np.log(np.clip(k_field, eps_k, None))
    k_field = (k_field - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return k_field.astype(np.float32)


def log10_phys(c_phys: np.ndarray, eps_c: float) -> np.ndarray:
    return np.log10(np.clip(c_phys, 0.0, None) + eps_c).astype(np.float32)


def log10_to_phys(c_log10: np.ndarray, eps_c: float) -> np.ndarray:
    c_phys = (10.0 ** c_log10) - eps_c
    return np.clip(c_phys, 0.0, None).astype(np.float32)


def build_time_input_channels(
    k_patch: np.ndarray,
    t_norm: float,
    time_encoding: str,
    context_patch: Optional[np.ndarray] = None,
) -> np.ndarray:
    patch = k_patch.shape[0]
    if time_encoding == "linear":
        t_chan = np.full((patch, patch), t_norm, dtype=np.float32)
        chans = [k_patch, t_chan]
    elif time_encoding == "fourier":
        twopi = 2.0 * np.pi
        sin_t = np.full((patch, patch), np.sin(twopi * t_norm), dtype=np.float32)
        cos_t = np.full((patch, patch), np.cos(twopi * t_norm), dtype=np.float32)
        chans = [k_patch, sin_t, cos_t]
    else:
        raise ValueError(f"Unknown time_encoding={time_encoding}")
    if context_patch is not None:
        chans.append(context_patch.astype(np.float32))
    return np.stack(chans, axis=0)


def _stitch_coords(height: int, width: int, patch: int, stride: int) -> tuple:
    ys = list(range(0, max(height - patch + 1, 1), stride))
    xs = list(range(0, max(width - patch + 1, 1), stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != height - patch:
        ys.append(height - patch)
    if xs[-1] != width - patch:
        xs.append(width - patch)
    return ys, xs


@torch.no_grad()
def predict_fullfield_one_t_guide(
    guide_model: torch.nn.Module,
    k_norm: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str = "linear",
) -> np.ndarray:
    height, width = k_norm.shape
    ys, xs = _stitch_coords(height, width, patch, stride)
    coords = [(y0, x0) for y0 in ys for x0 in xs]

    pred_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    cond_batch = []
    for y0, x0 in coords:
        k_patch = k_norm[y0:y0 + patch, x0:x0 + patch]
        cond_batch.append(build_time_input_channels(k_patch, t_norm, time_encoding))

    cond_batch = np.stack(cond_batch, axis=0)
    cond_batch_t = torch.from_numpy(cond_batch).to(device)
    pred_batch = guide_model(cond_batch_t).squeeze(1).cpu().numpy().astype(np.float32)

    for idx, (y0, x0) in enumerate(coords):
        pred_sum[y0:y0 + patch, x0:x0 + patch] += pred_batch[idx]
        weight_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

    return (pred_sum / np.maximum(weight_sum, 1e-6)).astype(np.float32)


@torch.no_grad()
def predict_fullfield_one_t_guided_diffusion(
    model: torch.nn.Module,
    guide_model: torch.nn.Module,
    k_norm: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    diff_steps: int,
    alpha_bars: torch.Tensor,
    sample_clip_min: float,
    sample_clip_max: float,
    method_style: str,
    time_encoding: str = "linear",
    seed: int = 0,
    context_full: Optional[np.ndarray] = None,
    flow_full: Optional[np.ndarray] = None,
) -> np.ndarray:
    height, width = k_norm.shape
    ys, xs = _stitch_coords(height, width, patch, stride)
    coords = [(y0, x0) for y0 in ys for x0 in xs]

    guide_full = predict_fullfield_one_t_guide(
        guide_model=guide_model,
        k_norm=k_norm,
        t_norm=t_norm,
        patch=patch,
        stride=stride,
        device=device,
        time_encoding=time_encoding,
    )

    rng = np.random.default_rng(seed)
    if method_style in {"dyffusion_style", "dydiff_style"}:
        state_full = guide_full + rng.standard_normal((height, width), dtype=np.float32)
    else:
        state_full = rng.standard_normal((height, width), dtype=np.float32)
    self_cond_full = np.zeros((height, width), dtype=np.float32)

    for step in reversed(range(diff_steps)):
        pred_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        next_self_cond_sum = np.zeros((height, width), dtype=np.float32)

        step_norm = float(step) / float(max(diff_steps - 1, 1))
        alpha_bar_t = float(alpha_bars[step].item())
        sqrt_ab_t = np.sqrt(max(alpha_bar_t, 1e-6))
        sqrt_one_minus_ab_t = np.sqrt(max(1.0 - alpha_bar_t, 1e-6))

        if step > 0:
            alpha_bar_prev = float(alpha_bars[step - 1].item())
            sqrt_ab_prev = np.sqrt(max(alpha_bar_prev, 1e-6))
            sqrt_one_minus_ab_prev = np.sqrt(max(1.0 - alpha_bar_prev, 1e-6))

        cond_batch = []
        base_batch = []
        state_batch = []
        self_cond_batch = []
        for y0, x0 in coords:
            k_patch = k_norm[y0:y0 + patch, x0:x0 + patch]
            context_patch = None if context_full is None else context_full[y0:y0 + patch, x0:x0 + patch]
            cond_patch = build_time_input_channels(k_patch, t_norm, time_encoding, context_patch=context_patch)
            if flow_full is not None:
                flow_patch = flow_full[:, y0:y0 + patch, x0:x0 + patch]
                cond_patch = np.concatenate([cond_patch, flow_patch.astype(np.float32)], axis=0)
            cond_batch.append(cond_patch)
            base_batch.append(guide_full[y0:y0 + patch, x0:x0 + patch])
            state_batch.append(state_full[y0:y0 + patch, x0:x0 + patch])
            self_cond_batch.append(self_cond_full[y0:y0 + patch, x0:x0 + patch])

        cond_batch = np.stack(cond_batch, axis=0)
        base_batch = np.stack(base_batch, axis=0)[:, None, ...]
        state_batch = np.stack(state_batch, axis=0)[:, None, ...]
        self_cond_batch = np.stack(self_cond_batch, axis=0)[:, None, ...]
        step_batch = np.full((len(coords), 1, patch, patch), step_norm, dtype=np.float32)
        if method_style == "dydiff_style":
            model_in = np.concatenate([cond_batch, base_batch, state_batch, step_batch, self_cond_batch], axis=1)
        else:
            model_in = np.concatenate([cond_batch, base_batch, state_batch, step_batch], axis=1)
        model_in_t = torch.from_numpy(model_in).to(device)
        pred_noise_batch = model(model_in_t).squeeze(1).cpu().numpy().astype(np.float32)

        for idx, (y0, x0) in enumerate(coords):
            base_patch = base_batch[idx, 0]
            state_patch = state_batch[idx, 0]
            pred_noise = pred_noise_batch[idx]

            if method_style in {"dyffusion_style", "dydiff_style"}:
                x0_pred = (state_patch - (1.0 - sqrt_ab_t) * base_patch - sqrt_one_minus_ab_t * pred_noise) / sqrt_ab_t
                x0_pred = np.clip(x0_pred, sample_clip_min, sample_clip_max)
                if step == 0:
                    x_prev = x0_pred
                else:
                    x_prev = (1.0 - sqrt_ab_prev) * base_patch + sqrt_ab_prev * x0_pred + sqrt_one_minus_ab_prev * pred_noise
                next_self_cond = x0_pred - base_patch
            else:
                residual_pred = (state_patch - sqrt_one_minus_ab_t * pred_noise) / sqrt_ab_t
                x0_pred = np.clip(base_patch + residual_pred, sample_clip_min, sample_clip_max)
                if step == 0:
                    x_prev = x0_pred
                else:
                    residual_prev = sqrt_ab_prev * (x0_pred - base_patch) + sqrt_one_minus_ab_prev * pred_noise
                    x_prev = np.clip(base_patch + residual_prev, sample_clip_min, sample_clip_max)
                next_self_cond = x0_pred - base_patch

            pred_sum[y0:y0 + patch, x0:x0 + patch] += x_prev.astype(np.float32)
            next_self_cond_sum[y0:y0 + patch, x0:x0 + patch] += next_self_cond.astype(np.float32)
            weight_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

        state_full = pred_sum / np.maximum(weight_sum, 1e-6)
        self_cond_full = next_self_cond_sum / np.maximum(weight_sum, 1e-6)

    return state_full.astype(np.float32)


@torch.no_grad()
def evaluate_timecond_guided_diffusion(
    model: torch.nn.Module,
    guide_model: torch.nn.Module,
    test_files: List[str],
    stats: Dict,
    patch: int,
    stride: int,
    device: str,
    use_logK: bool,
    eps_k: float,
    eps_c: float,
    diff_steps: int,
    alpha_bars: torch.Tensor,
    sample_clip_min: float,
    sample_clip_max: float,
    method_style: str,
    timestep_indices: Optional[List[int]] = None,
    time_encoding: str = "linear",
    base_seed: int = 0,
    use_context_cond: bool = False,
    use_flow_cond: bool = False,
    flow_recent_window: int = 15,
    flow_recency_power: float = 0.0,
    flow_encoder: Optional[FlowHistoryEncoder] = None,
) -> Dict:
    all_ssim = []
    all_plume_ssim = []
    all_centroid_err = []
    all_l1 = []
    all_mse = []
    all_mass = []
    ssim_per_t = None
    plume_ssim_per_t = None
    plume_valid_count_per_t = None
    centroid_err_per_t = None
    centroid_valid_count_per_t = None
    l1_per_t = None
    mse_per_t = None
    mass_per_t = None
    count_per_t = None

    for file_idx, fp in enumerate(test_files):
        with np.load(fp) as data:
            k_field = data["K"].astype(np.float32)
            c_field = data["C"].astype(np.float32)
            times = data["times"].astype(np.float32)

        total_t, _, _ = c_field.shape
        k_norm = normalize_k(k_field, stats, use_logK, eps_k)
        context_full = None
        seen_sequence_full = None
        if use_context_cond:
            if total_t <= 14:
                raise ValueError(f"Context timestep index 14 is unavailable for T={total_t} in {fp}")
            context_full = log10_phys(c_field[14], eps_c)
        if use_flow_cond:
            if total_t <= 14:
                raise ValueError(f"Flow conditioning requires seen timesteps 0..14, but T={total_t} in {fp}")
            seen_sequence_full = log10_phys(c_field[:15], eps_c)
            if flow_encoder is not None:
                flow_encoder.eval()

        if ssim_per_t is None:
            ssim_per_t = np.zeros(total_t, dtype=np.float64)
            plume_ssim_per_t = np.zeros(total_t, dtype=np.float64)
            plume_valid_count_per_t = np.zeros(total_t, dtype=np.int64)
            centroid_err_per_t = np.zeros(total_t, dtype=np.float64)
            centroid_valid_count_per_t = np.zeros(total_t, dtype=np.int64)
            l1_per_t = np.zeros(total_t, dtype=np.float64)
            mse_per_t = np.zeros(total_t, dtype=np.float64)
            mass_per_t = np.zeros(total_t, dtype=np.float64)
            count_per_t = np.zeros(total_t, dtype=np.int64)

        t0, t1 = float(times.min()), float(times.max())
        active_indices = timestep_indices if timestep_indices is not None else list(range(total_t))

        for t_idx in active_indices:
            t_norm = (float(times[t_idx]) - t0) / (t1 - t0 + 1e-12)
            sample_seed = int(base_seed + 1009 * file_idx + 53 * int(t_idx))
            flow_full = None
            if use_flow_cond:
                base_flow_full = compute_plume_flow_channels(
                    seen_sequence=seen_sequence_full,
                    target_timestep=int(t_idx),
                    H=k_field.shape[0],
                    W=k_field.shape[1],
                    recent_window=flow_recent_window,
                    recency_power=flow_recency_power,
                )
                if flow_encoder is not None:
                    seen_seq_t = torch.from_numpy(seen_sequence_full[None, ...]).to(device=device, dtype=torch.float32)
                    base_flow_t = torch.from_numpy(base_flow_full[None, ...]).to(device=device, dtype=torch.float32)
                    flow_full = flow_encoder(seen_seq_t, base_flow_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
                else:
                    flow_full = base_flow_full
            pred_log = predict_fullfield_one_t_guided_diffusion(
                model=model,
                guide_model=guide_model,
                k_norm=k_norm,
                t_norm=t_norm,
                patch=patch,
                stride=stride,
                device=device,
                diff_steps=diff_steps,
                alpha_bars=alpha_bars,
                sample_clip_min=sample_clip_min,
                sample_clip_max=sample_clip_max,
                method_style=method_style,
                time_encoding=time_encoding,
                seed=sample_seed,
                context_full=context_full,
                flow_full=flow_full,
            )
            gt_log = log10_phys(c_field[t_idx], eps_c)
            score = compute_global_ssim(gt_log, pred_log)
            all_ssim.append(score)
            ssim_per_t[t_idx] += score
            count_per_t[t_idx] += 1

            plume_score, plume_valid_count = compute_plume_ssim(
                gt_phys=np.clip(c_field[t_idx], 0.0, None),
                pred_log=pred_log,
                eps_c=eps_c,
            )
            if not np.isnan(plume_score):
                all_plume_ssim.append(float(plume_score))
                plume_ssim_per_t[t_idx] += float(plume_score)
                plume_valid_count_per_t[t_idx] += 1

            pred_phys = log10_to_phys(pred_log, eps_c)
            gt_phys = np.clip(c_field[t_idx], 0.0, None)

            centroid_err, _ = compute_plume_centroid_error(gt_phys=gt_phys, pred_phys=pred_phys)
            if not np.isnan(centroid_err):
                all_centroid_err.append(float(centroid_err))
                centroid_err_per_t[t_idx] += float(centroid_err)
                centroid_valid_count_per_t[t_idx] += 1

            diff = pred_phys - gt_phys
            l1_value = float(np.mean(np.abs(diff)))
            mse_value = float(np.mean(diff * diff))
            mass_value = float(np.abs(pred_phys.sum() - gt_phys.sum()))
            all_l1.append(l1_value)
            all_mse.append(mse_value)
            all_mass.append(mass_value)
            l1_per_t[t_idx] += l1_value
            mse_per_t[t_idx] += mse_value
            mass_per_t[t_idx] += mass_value

    ssim_per_t = ssim_per_t / np.maximum(count_per_t, 1)
    plume_ssim_per_t = plume_ssim_per_t / np.maximum(plume_valid_count_per_t, 1)
    centroid_err_per_t = centroid_err_per_t / np.maximum(centroid_valid_count_per_t, 1)
    l1_per_t = l1_per_t / np.maximum(count_per_t, 1)
    mse_per_t = mse_per_t / np.maximum(count_per_t, 1)
    mass_per_t = mass_per_t / np.maximum(count_per_t, 1)

    return {
        "mean_ssim_log": float(np.mean(all_ssim)),
        "mean_plume_ssim_log": float(np.mean(all_plume_ssim)) if all_plume_ssim else float("nan"),
        "mean_centroid_err_px": float(np.mean(all_centroid_err)) if all_centroid_err else float("nan"),
        "ssim_per_timestep_log": ssim_per_t.tolist(),
        "plume_ssim_per_timestep_log": plume_ssim_per_t.tolist(),
        "plume_valid_count_per_timestep": plume_valid_count_per_t.tolist(),
        "centroid_err_per_timestep_px": centroid_err_per_t.tolist(),
        "centroid_valid_count_per_timestep": centroid_valid_count_per_t.tolist(),
        "mean_l1_phys": float(np.mean(all_l1)),
        "l1_per_timestep_phys": l1_per_t.tolist(),
        "mean_mse_phys": float(np.mean(all_mse)),
        "mse_per_timestep_phys": mse_per_t.tolist(),
        "mean_mass_err_phys": float(np.mean(all_mass)),
        "mass_err_per_timestep_phys": mass_per_t.tolist(),
        "n_test_files": len(test_files),
        "patch": int(patch),
        "stride": int(stride),
        "timestep_indices": timestep_indices,
        "time_encoding": time_encoding,
        "use_context_cond": bool(use_context_cond),
        "use_flow_cond": bool(use_flow_cond),
        "flow_recent_window": int(flow_recent_window),
        "flow_recency_power": float(flow_recency_power),
        "diff_steps": int(diff_steps),
        "method_style": method_style,
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
    }
