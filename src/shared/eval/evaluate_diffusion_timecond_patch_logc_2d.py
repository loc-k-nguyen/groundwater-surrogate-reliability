from typing import Dict, List, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


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


def build_time_input_channels(k_patch: np.ndarray, t_norm: float, time_encoding: str) -> np.ndarray:
    patch = k_patch.shape[0]
    if time_encoding == "linear":
        t_chan = np.full((patch, patch), t_norm, dtype=np.float32)
        return np.stack([k_patch, t_chan], axis=0)
    if time_encoding == "fourier":
        twopi = 2.0 * np.pi
        sin_t = np.full((patch, patch), np.sin(twopi * t_norm), dtype=np.float32)
        cos_t = np.full((patch, patch), np.cos(twopi * t_norm), dtype=np.float32)
        return np.stack([k_patch, sin_t, cos_t], axis=0)
    raise ValueError(f"Unknown time_encoding={time_encoding}")


def ssim_log(gt_log: np.ndarray, pr_log: np.ndarray) -> float:
    data_range = float(np.max(gt_log) - np.min(gt_log))
    if data_range < 1e-8:
        data_range = 1e-8
    return float(ssim(gt_log, pr_log, data_range=data_range))


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
def predict_fullfield_one_t_diffusion(
    model: torch.nn.Module,
    k_norm: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    diff_steps: int,
    alpha_bars: torch.Tensor,
    sample_clip_min: float,
    sample_clip_max: float,
    time_encoding: str = "linear",
    seed: int = 0,
) -> np.ndarray:
    height, width = k_norm.shape
    ys, xs = _stitch_coords(height, width, patch, stride)
    coords = [(y0, x0) for y0 in ys for x0 in xs]

    rng = np.random.default_rng(seed)
    x_t_full = rng.standard_normal((height, width), dtype=np.float32)

    for step in reversed(range(diff_steps)):
        pred_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        step_norm = float(step) / float(max(diff_steps - 1, 1))
        alpha_bar_t = float(alpha_bars[step].item())
        sqrt_ab_t = np.sqrt(max(alpha_bar_t, 1e-6))
        sqrt_one_minus_ab_t = np.sqrt(max(1.0 - alpha_bar_t, 1e-6))

        if step > 0:
            alpha_bar_prev = float(alpha_bars[step - 1].item())
            sqrt_ab_prev = np.sqrt(max(alpha_bar_prev, 1e-6))
            sqrt_one_minus_ab_prev = np.sqrt(max(1.0 - alpha_bar_prev, 1e-6))

        cond_batch = []
        x_batch = []
        for y0, x0 in coords:
            k_patch = k_norm[y0:y0 + patch, x0:x0 + patch]
            cond_batch.append(build_time_input_channels(k_patch, t_norm, time_encoding))
            x_batch.append(x_t_full[y0:y0 + patch, x0:x0 + patch])

        cond_batch = np.stack(cond_batch, axis=0)
        x_batch = np.stack(x_batch, axis=0)[:, None, ...]
        step_batch = np.full((len(coords), 1, patch, patch), step_norm, dtype=np.float32)
        model_in = np.concatenate([cond_batch, x_batch, step_batch], axis=1)
        model_in_t = torch.from_numpy(model_in).to(device)
        pred_noise_batch = model(model_in_t).squeeze(1).cpu().numpy().astype(np.float32)

        for idx, (y0, x0) in enumerate(coords):
            x0_pred = (x_batch[idx, 0] - sqrt_one_minus_ab_t * pred_noise_batch[idx]) / sqrt_ab_t
            x0_pred = np.clip(x0_pred, sample_clip_min, sample_clip_max)

            if step == 0:
                x_prev = x0_pred
            else:
                x_prev = sqrt_ab_prev * x0_pred + sqrt_one_minus_ab_prev * pred_noise_batch[idx]

            pred_sum[y0:y0 + patch, x0:x0 + patch] += x_prev.astype(np.float32)
            weight_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

        x_t_full = pred_sum / np.maximum(weight_sum, 1e-6)

    return x_t_full.astype(np.float32)


@torch.no_grad()
def evaluate_timecond_diffusion(
    model: torch.nn.Module,
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
    timestep_indices: Optional[List[int]] = None,
    time_encoding: str = "linear",
    base_seed: int = 0,
) -> Dict:
    all_ssim = []
    all_l1 = []
    all_mse = []
    all_mass = []
    ssim_per_t = None
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

        if ssim_per_t is None:
            ssim_per_t = np.zeros(total_t, dtype=np.float64)
            l1_per_t = np.zeros(total_t, dtype=np.float64)
            mse_per_t = np.zeros(total_t, dtype=np.float64)
            mass_per_t = np.zeros(total_t, dtype=np.float64)
            count_per_t = np.zeros(total_t, dtype=np.int64)

        t0, t1 = float(times.min()), float(times.max())
        active_indices = timestep_indices if timestep_indices is not None else list(range(total_t))

        for t_idx in active_indices:
            t_norm = (float(times[t_idx]) - t0) / (t1 - t0 + 1e-12)
            sample_seed = int(base_seed + 1009 * file_idx + 53 * int(t_idx))
            pred_log = predict_fullfield_one_t_diffusion(
                model=model,
                k_norm=k_norm,
                t_norm=t_norm,
                patch=patch,
                stride=stride,
                device=device,
                diff_steps=diff_steps,
                alpha_bars=alpha_bars,
                sample_clip_min=sample_clip_min,
                sample_clip_max=sample_clip_max,
                time_encoding=time_encoding,
                seed=sample_seed,
            )
            gt_log = log10_phys(c_field[t_idx], eps_c)
            score = ssim_log(gt_log, pred_log)
            all_ssim.append(score)
            ssim_per_t[t_idx] += score
            count_per_t[t_idx] += 1

            pred_phys = log10_to_phys(pred_log, eps_c)
            gt_phys = np.clip(c_field[t_idx], 0.0, None)
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
    l1_per_t = l1_per_t / np.maximum(count_per_t, 1)
    mse_per_t = mse_per_t / np.maximum(count_per_t, 1)
    mass_per_t = mass_per_t / np.maximum(count_per_t, 1)

    return {
        "mean_ssim_log": float(np.mean(all_ssim)),
        "ssim_per_timestep_log": ssim_per_t.tolist(),
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
        "diff_steps": int(diff_steps),
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
    }
