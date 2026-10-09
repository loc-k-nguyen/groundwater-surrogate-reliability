import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.obj2.conference.models.unet2d import UNet2D
from src.shared.data.io import load_param_split, flatten_param_folders
from src.shared.eval.centroid_error import compute_plume_centroid_error
from src.shared.eval.plume_ssim import compute_global_ssim, compute_plume_ssim


# ----------------------------
# IO / transforms
# ----------------------------
def load_stats(stats_json_path: str) -> Dict:
    with open(stats_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["stats"] if isinstance(j, dict) and "stats" in j else j


def normalize_K(K: np.ndarray, stats: Dict, use_logK: bool, eps_k: float) -> np.ndarray:
    K = K.astype(np.float64)
    if use_logK:
        K = np.log(np.clip(K, eps_k, None))
    K = (K - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return K.astype(np.float32)


def log10_phys(C_phys: np.ndarray, eps_c: float) -> np.ndarray:
    return np.log10(np.clip(C_phys, 0.0, None) + eps_c).astype(np.float32)


def log10_to_phys(C_log10: np.ndarray, eps_c: float) -> np.ndarray:
    C = (10.0 ** C_log10) - eps_c
    return np.clip(C, 0.0, None).astype(np.float32)


def build_time_input_channels(K_patch: np.ndarray, t_norm: float, time_encoding: str) -> np.ndarray:
    patch = K_patch.shape[0]
    if time_encoding == "linear":
        t_chan = np.full((patch, patch), t_norm, dtype=np.float32)
        return np.stack([K_patch, t_chan], axis=0)
    if time_encoding == "fourier":
        twopi = 2.0 * np.pi
        sin_t = np.full((patch, patch), np.sin(twopi * t_norm), dtype=np.float32)
        cos_t = np.full((patch, patch), np.cos(twopi * t_norm), dtype=np.float32)
        return np.stack([K_patch, sin_t, cos_t], axis=0)
    raise ValueError(f"Unknown time_encoding={time_encoding}")


# ----------------------------
# Stitching
# ----------------------------
@torch.no_grad()
def predict_fullfield_one_t(
    model: torch.nn.Module,
    K_norm: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str = "linear",
) -> np.ndarray:
    """
    Predict one timestep full field log10(C+eps) by patch stitching.
    Returns (H,W) float32.
    """
    H, W = K_norm.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    # ensure we cover last patch boundary (in case stride doesn't land exactly)
    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    if ys[-1] != H - patch: ys.append(H - patch)
    if xs[-1] != W - patch: xs.append(W - patch)

    for y0 in ys:
        for x0 in xs:
            Kp = K_norm[y0:y0 + patch, x0:x0 + patch]
            X = build_time_input_channels(Kp, t_norm, time_encoding)[None, ...]
            X = torch.from_numpy(X).to(device)

            pred = model(X).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)  # (P,P)

            pred_sum[y0:y0 + patch, x0:x0 + patch] += pred
            w_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

    w_sum = np.maximum(w_sum, 1e-6)
    return pred_sum / w_sum


# ----------------------------
# Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate_timecond(
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
    time_encoding: str = "linear",
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
    count_per_t = None

    for fp in test_files:
        d = np.load(fp)
        K = d["K"].astype(np.float32)             # (H,W)
        C = d["C"].astype(np.float32)             # (T,H,W)
        times = d["times"].astype(np.float32)     # (T,)
        d.close()

        T, H, W = C.shape
        K_norm = normalize_K(K, stats, use_logK, eps_k)

        if ssim_per_t is None:
            ssim_per_t = np.zeros(T, dtype=np.float64)
            plume_ssim_per_t = np.zeros(T, dtype=np.float64)
            plume_valid_count_per_t = np.zeros(T, dtype=np.int64)
            centroid_err_per_t = np.zeros(T, dtype=np.float64)
            centroid_valid_count_per_t = np.zeros(T, dtype=np.int64)
            count_per_t = np.zeros(T, dtype=np.int64)

        t0, t1 = float(times.min()), float(times.max())

        active_indices = timestep_indices if timestep_indices is not None else list(range(T))

        for t_idx in active_indices:
            t = float(times[t_idx])
            t_norm = (t - t0) / (t1 - t0 + 1e-12)

            pr_log = predict_fullfield_one_t(model, K_norm, t_norm, patch, stride, device, time_encoding=time_encoding)
            gt_log = log10_phys(C[t_idx], eps_c)

            s = compute_global_ssim(gt_log, pr_log)
            all_ssim.append(s)
            ssim_per_t[t_idx] += s
            count_per_t[t_idx] += 1

            plume_score, plume_valid_count = compute_plume_ssim(
                gt_phys=np.clip(C[t_idx], 0.0, None),
                pred_log=pr_log,
                eps_c=eps_c,
            )
            if not np.isnan(plume_score):
                all_plume_ssim.append(float(plume_score))
                plume_ssim_per_t[t_idx] += float(plume_score)
                plume_valid_count_per_t[t_idx] += 1

            pr_phys = log10_to_phys(pr_log, eps_c)
            gt_phys = np.clip(C[t_idx], 0.0, None)

            centroid_err, _ = compute_plume_centroid_error(gt_phys=gt_phys, pred_phys=pr_phys)
            if not np.isnan(centroid_err):
                all_centroid_err.append(float(centroid_err))
                centroid_err_per_t[t_idx] += float(centroid_err)
                centroid_valid_count_per_t[t_idx] += 1

            diff = pr_phys - gt_phys
            all_l1.append(float(np.mean(np.abs(diff))))
            all_mse.append(float(np.mean(diff * diff)))
            all_mass.append(float(np.abs(pr_phys.sum() - gt_phys.sum())))

    ssim_per_t = ssim_per_t / np.maximum(count_per_t, 1)
    plume_ssim_per_t = plume_ssim_per_t / np.maximum(plume_valid_count_per_t, 1)
    centroid_err_per_t = centroid_err_per_t / np.maximum(centroid_valid_count_per_t, 1)

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
        "mean_mse_phys": float(np.mean(all_mse)),
        "mean_mass_err_phys": float(np.mean(all_mass)),
        "n_test_files": len(test_files),
        "patch": int(patch),
        "stride": int(stride),
        "timestep_indices": timestep_indices,
        "time_encoding": time_encoding,
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--stats_json", required=True)
    ap.add_argument("--ckpt", required=True)

    ap.add_argument("--patch", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)

    ap.add_argument("--use_logK", action="store_true")
    ap.add_argument("--eps_k", type=float, default=1e-6)
    ap.add_argument("--eps_c", type=float, default=1e-12)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_json", type=str, default="test_report_timecond_logc_obj1.json")
    ap.add_argument("--time_encoding", type=str, default="linear", choices=["linear", "fourier"])
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    # test split
    _, _, test_params = load_param_split(args.split_json)
    test_files = flatten_param_folders(args.data_root, test_params)

    # load stats
    stats = load_stats(args.stats_json)

    # load ckpt + model
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    base = int(cfg.get("base_channels", 64))

    in_ch = 2 if args.time_encoding == "linear" else 3
    model = UNet2D(in_ch=in_ch, out_ch=1, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = evaluate_timecond(
        model=model,
        test_files=test_files,
        stats=stats,
        patch=args.patch,
        stride=args.stride,
        device=device,
        use_logK=args.use_logK,
        eps_k=args.eps_k,
        eps_c=args.eps_c,
        time_encoding=args.time_encoding,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[Saved]", args.out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

# python -m src.eval.evaluate_timecond_patch_logc_2d ^
#   --data_root ".\data\T25_TSTEP_OVERRIDE_FINAL" ^
#   --split_json "splits\param_split_fixed.json" ^
#   --stats_json "stats\train_stats.json" ^
#   --ckpt "runs\obj1_timecond_logc_p320_seed2_mid_bias_loglin_a20_yb-12_yc-2_mass0.001_lr0.0001\best.pt" ^
#   --patch 320 ^
#   --stride 160 ^
#   --use_logK ^
#   --device cuda ^
#   --out_json "runs\obj1_timecond_logc_p320_seed2_mid_bias_loglin_a20_yb-12_yc-2_mass0.001_lr0.0001\test_report_logc_obj1.json"
