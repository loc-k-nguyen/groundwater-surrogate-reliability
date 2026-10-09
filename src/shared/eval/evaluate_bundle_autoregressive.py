from typing import Dict, List, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.shared.data.dataset_timecond_bundle import build_bundle_time_channels
from src.shared.eval.evaluate_timecond_patch_logc_2d import log10_phys, normalize_K


def ssim_log(gt_log: np.ndarray, pr_log: np.ndarray) -> float:
    dr = float(np.max(gt_log) - np.min(gt_log))
    if dr < 1e-8:
        dr = 1e-8
    return float(ssim(gt_log, pr_log, data_range=dr))


def log10_to_phys(C_log10: np.ndarray, eps_c: float) -> np.ndarray:
    C = (10.0 ** C_log10) - eps_c
    return np.clip(C, 0.0, None).astype(np.float32)


@torch.no_grad()
def predict_fullfield_bundle_one_t(
    model: torch.nn.Module,
    K_norm: np.ndarray,
    context_phys: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str,
    eps_c: float,
) -> np.ndarray:
    H, W = K_norm.shape
    pred_sum = None
    w_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)

    for y0 in ys:
        for x0 in xs:
            X = build_bundle_time_channels(
                K_norm=K_norm[y0:y0 + patch, x0:x0 + patch],
                context_raw=context_phys[:, y0:y0 + patch, x0:x0 + patch],
                t_norm=t_norm,
                eps_c=eps_c,
                time_encoding=time_encoding,
            )[None, ...]
            pred = model(torch.from_numpy(X).to(device)).squeeze(0).cpu().numpy().astype(np.float32)
            if pred_sum is None:
                pred_sum = np.zeros((pred.shape[0], H, W), dtype=np.float32)
            pred_sum[:, y0:y0 + patch, x0:x0 + patch] += pred
            w_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

    return pred_sum / np.maximum(w_sum[None, ...], 1e-6)


@torch.no_grad()
def rollout_bundle_sequence(
    model: torch.nn.Module,
    K_norm: np.ndarray,
    C0_phys: np.ndarray,
    times: np.ndarray,
    context_steps: int,
    pred_steps: int,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str,
    eps_c: float,
) -> np.ndarray:
    T = int(times.shape[0])
    preds = np.zeros((T, K_norm.shape[0], K_norm.shape[1]), dtype=np.float32)
    preds[0] = np.clip(C0_phys, 0.0, None)

    t_min = float(times.min())
    t_max = float(times.max())
    anchor = 0
    while anchor < T - 1:
        context = []
        for offset in range(context_steps):
            ctx_idx = max(0, anchor - context_steps + 1 + offset)
            context.append(preds[ctx_idx])
        context = np.stack(context, axis=0).astype(np.float32)
        t_norm = (float(times[anchor]) - t_min) / (t_max - t_min + 1e-12)
        pred_bundle = predict_fullfield_bundle_one_t(
            model=model,
            K_norm=K_norm,
            context_phys=context,
            t_norm=t_norm,
            patch=patch,
            stride=stride,
            device=device,
            time_encoding=time_encoding,
            eps_c=eps_c,
        )
        for step_offset in range(pred_steps):
            t_idx = anchor + 1 + step_offset
            if t_idx >= T:
                break
            preds[t_idx] = log10_to_phys(pred_bundle[step_offset], eps_c)
        anchor += pred_steps
    return preds


@torch.no_grad()
def evaluate_bundle_autoregressive(
    model: torch.nn.Module,
    test_files: List[str],
    stats: Dict,
    patch: int,
    stride: int,
    device: str,
    use_logK: bool,
    eps_k: float,
    eps_c: float,
    context_steps: int,
    pred_steps: int,
    timestep_indices: Optional[List[int]] = None,
    time_encoding: str = "linear",
) -> Dict:
    all_ssim = []
    all_l1 = []
    all_mse = []
    all_mass = []
    ssim_per_t = None
    count_per_t = None

    for fp in test_files:
        with np.load(fp) as d:
            K = d["K"].astype(np.float32)
            C = d["C"].astype(np.float32)
            times = d["times"].astype(np.float32)

        K_norm = normalize_K(K, stats, use_logK, eps_k)
        pred_seq = rollout_bundle_sequence(
            model=model,
            K_norm=K_norm,
            C0_phys=C[0],
            times=times,
            context_steps=context_steps,
            pred_steps=pred_steps,
            patch=patch,
            stride=stride,
            device=device,
            time_encoding=time_encoding,
            eps_c=eps_c,
        )

        T = int(C.shape[0])
        if ssim_per_t is None:
            ssim_per_t = np.zeros(T, dtype=np.float64)
            count_per_t = np.zeros(T, dtype=np.int64)

        active_indices = timestep_indices if timestep_indices is not None else list(range(T))
        for t_idx in active_indices:
            gt_log = log10_phys(C[t_idx], eps_c)
            pr_log = log10_phys(pred_seq[t_idx], eps_c)
            score = ssim_log(gt_log, pr_log)
            all_ssim.append(score)
            ssim_per_t[t_idx] += score
            count_per_t[t_idx] += 1

            diff = pred_seq[t_idx] - np.clip(C[t_idx], 0.0, None)
            all_l1.append(float(np.mean(np.abs(diff))))
            all_mse.append(float(np.mean(diff * diff)))
            all_mass.append(float(np.abs(pred_seq[t_idx].sum() - C[t_idx].sum())))

    ssim_curve = (ssim_per_t / np.maximum(count_per_t, 1)).tolist()
    return {
        "mean_ssim_log": float(np.mean(all_ssim)),
        "ssim_per_timestep_log": ssim_curve,
        "mean_l1_phys": float(np.mean(all_l1)),
        "mean_mse_phys": float(np.mean(all_mse)),
        "mean_mass_err_phys": float(np.mean(all_mass)),
        "n_test_files": len(test_files),
        "patch": int(patch),
        "stride": int(stride),
        "timestep_indices": timestep_indices,
        "time_encoding": time_encoding,
        "context_steps": int(context_steps),
        "pred_steps": int(pred_steps),
        "rollout": "autoregressive_bundle",
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
    }
