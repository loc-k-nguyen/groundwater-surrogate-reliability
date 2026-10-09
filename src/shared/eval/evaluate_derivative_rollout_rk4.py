from typing import Dict, List, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.shared.data.dataset_timecond_derivative import build_state_time_channels
from src.shared.eval.evaluate_timecond_patch_logc_2d import log10_phys, normalize_K


def ssim_log(gt_log: np.ndarray, pr_log: np.ndarray) -> float:
    dr = float(np.max(gt_log) - np.min(gt_log))
    if dr < 1e-8:
        dr = 1e-8
    return float(ssim(gt_log, pr_log, data_range=dr))


@torch.no_grad()
def predict_fullfield_derivative_one_t(
    model: torch.nn.Module,
    K_norm: np.ndarray,
    C_curr_phys: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str,
    eps_c: float,
) -> np.ndarray:
    H, W = K_norm.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
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
            X = build_state_time_channels(
                K_norm=K_norm[y0:y0 + patch, x0:x0 + patch],
                C_curr_raw=C_curr_phys[y0:y0 + patch, x0:x0 + patch],
                t_norm=t_norm,
                eps_c=eps_c,
                time_encoding=time_encoding,
            )[None, ...]
            pred = model(torch.from_numpy(X).to(device)).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            pred_sum[y0:y0 + patch, x0:x0 + patch] += pred
            w_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

    return pred_sum / np.maximum(w_sum, 1e-6)


def _norm_time(t_value: float, t_min: float, t_max: float) -> float:
    return (t_value - t_min) / (t_max - t_min + 1e-12)


@torch.no_grad()
def rk4_step_fullfield(
    model: torch.nn.Module,
    K_norm: np.ndarray,
    C_curr_phys: np.ndarray,
    t_value: float,
    dt: float,
    t_min: float,
    t_max: float,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str,
    eps_c: float,
) -> np.ndarray:
    t_norm = _norm_time(t_value, t_min, t_max)
    k1 = predict_fullfield_derivative_one_t(model, K_norm, C_curr_phys, t_norm, patch, stride, device, time_encoding, eps_c)

    s2 = np.clip(C_curr_phys + 0.5 * dt * k1, 0.0, None).astype(np.float32)
    t2_norm = _norm_time(t_value + 0.5 * dt, t_min, t_max)
    k2 = predict_fullfield_derivative_one_t(model, K_norm, s2, t2_norm, patch, stride, device, time_encoding, eps_c)

    s3 = np.clip(C_curr_phys + 0.5 * dt * k2, 0.0, None).astype(np.float32)
    k3 = predict_fullfield_derivative_one_t(model, K_norm, s3, t2_norm, patch, stride, device, time_encoding, eps_c)

    s4 = np.clip(C_curr_phys + dt * k3, 0.0, None).astype(np.float32)
    t4_norm = _norm_time(t_value + dt, t_min, t_max)
    k4 = predict_fullfield_derivative_one_t(model, K_norm, s4, t4_norm, patch, stride, device, time_encoding, eps_c)

    next_state = C_curr_phys + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(next_state, 0.0, None).astype(np.float32)


@torch.no_grad()
def rollout_derivative_sequence(
    model: torch.nn.Module,
    K_norm: np.ndarray,
    C0_phys: np.ndarray,
    times: np.ndarray,
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
    for idx in range(T - 1):
        dt = float(times[idx + 1] - times[idx])
        preds[idx + 1] = rk4_step_fullfield(
            model=model,
            K_norm=K_norm,
            C_curr_phys=preds[idx],
            t_value=float(times[idx]),
            dt=dt,
            t_min=t_min,
            t_max=t_max,
            patch=patch,
            stride=stride,
            device=device,
            time_encoding=time_encoding,
            eps_c=eps_c,
        )
    return preds


@torch.no_grad()
def evaluate_derivative_rollout(
    model: torch.nn.Module,
    test_files: List[str],
    stats: Dict,
    patch: int,
    stride: int,
    device: str,
    use_logK: bool,
    eps_k: float,
    eps_c: float,
    timestep_indices: Optional[List[int]] = None,
    time_encoding: str = "fourier",
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
        pred_seq = rollout_derivative_sequence(
            model=model,
            K_norm=K_norm,
            C0_phys=C[0],
            times=times,
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
        "rollout": "rk4",
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
    }
