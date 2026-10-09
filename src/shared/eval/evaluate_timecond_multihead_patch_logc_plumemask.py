import os
import json
import argparse
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.shared.data.io import load_param_split, flatten_param_folders
from src.obj2.conference.models.unet2d_timecond_multihead import UNet2D_TimeCond_MultiHead


def load_stats(stats_json_path: str) -> dict:
    with open(stats_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["stats"] if isinstance(j, dict) and "stats" in j else j


def normalize_K(K: np.ndarray, stats: dict, use_logK: bool, eps_k: float) -> np.ndarray:
    K = K.astype(np.float64)
    if use_logK:
        K = np.log(np.clip(K, eps_k, None))
    K = (K - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return K.astype(np.float32)


def gt_log10(C_phys: np.ndarray, eps_c: float) -> np.ndarray:
    return np.log10(np.clip(C_phys, 0.0, None) + eps_c).astype(np.float32)


def log10_to_phys(C_log10: np.ndarray, eps_c: float) -> np.ndarray:
    C = (10.0 ** C_log10) - eps_c
    return np.clip(C, 0.0, None).astype(np.float32)


def build_time_input_channels(K_patch: np.ndarray, t_norm: float, time_encoding: str) -> np.ndarray:
    patch = K_patch.shape[0]
    if time_encoding == "linear":
        t_chan = np.full((patch, patch), float(t_norm), dtype=np.float32)
        return np.stack([K_patch, t_chan], axis=0)
    if time_encoding == "fourier":
        twopi = 2.0 * np.pi
        sin_t = np.full((patch, patch), np.sin(twopi * t_norm), dtype=np.float32)
        cos_t = np.full((patch, patch), np.cos(twopi * t_norm), dtype=np.float32)
        return np.stack([K_patch, sin_t, cos_t], axis=0)
    raise ValueError(f"Unknown time_encoding={time_encoding}")


def hann2d(patch: int) -> np.ndarray:
    w1 = np.hanning(patch).astype(np.float32)
    w2 = np.outer(w1, w1).astype(np.float32)
    return np.clip(w2, 1e-3, 1.0)


def ssim_log(gt: np.ndarray, pr: np.ndarray) -> float:
    dr = float(np.max(gt) - np.min(gt))
    if dr < 1e-8:
        dr = 1e-8
    return float(ssim(gt, pr, data_range=dr))


def bbox_from_mask(mask: np.ndarray, pad: int = 0):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, mask.shape[0] - 1)
    x1 = min(x1 + pad, mask.shape[1] - 1)
    return y0, y1 + 1, x0, x1 + 1


def plume_ssim_bbox(gtL: np.ndarray, prL: np.ndarray, mask: np.ndarray, pad: int, min_pixels: int) -> float:
    if int(mask.sum()) < int(min_pixels):
        return float("nan")
    bb = bbox_from_mask(mask, pad=pad)
    if bb is None:
        return float("nan")
    y0, y1, x0, x1 = bb
    gt_roi = gtL[y0:y1, x0:x1]
    pr_roi = prL[y0:y1, x0:x1]
    if gt_roi.shape[0] < 7 or gt_roi.shape[1] < 7:
        return float("nan")
    dr = float(np.max(gt_roi) - np.min(gt_roi))
    if dr < 1e-8:
        dr = 1e-8
    return float(ssim(gt_roi, pr_roi, data_range=dr))


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


@torch.no_grad()
def stitch_predict_one_t(
    model,
    K_norm: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
    time_encoding: str = "linear",
) -> np.ndarray:
    """
    Predict one timestep full field for multihead model.
    Returns pred_log10: (H,W)
    """
    H, W = K_norm.shape
    pred_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    if ys[-1] != H - patch: ys.append(H - patch)
    if xs[-1] != W - patch: xs.append(W - patch)

    w_patch = hann2d(patch)

    for y0 in ys:
        for x0 in xs:
            Kp = K_norm[y0:y0 + patch, x0:x0 + patch]
            X = build_time_input_channels(Kp, t_norm, time_encoding)[None, ...]
            X = torch.from_numpy(X).to(device)

            out = model(X)
            pred_map = out[0] if isinstance(out, (tuple, list)) else out  # (1,1,P,P)
            pred = pred_map.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (P,P)

            pred_sum[y0:y0 + patch, x0:x0 + patch] += pred * w_patch
            w_sum[y0:y0 + patch, x0:x0 + patch] += w_patch

    w_sum = np.maximum(w_sum, 1e-6)
    return pred_sum / w_sum


@torch.no_grad()
def evaluate(model, test_files, stats, patch, stride, device, use_logK, eps_k, eps_c,
             plume_mode: str, plume_thresh: float, plume_pad: int, plume_min_pixels: int,
             timestep_indices=None, time_encoding: str = "linear"):
    all_ssim = []
    all_l1 = []
    all_mse = []
    all_mass = []
    plume_ssim_all = []
    plume_area_frac_all = []
    centroid_error_all = []

    ssim_per_t = None
    plume_ssim_per_t = None
    plume_area_frac_per_t = None
    centroid_error_per_t = None
    centroid_count_per_t = None
    count_per_t = None

    for fp in test_files:
        d = np.load(fp)
        K = d["K"].astype(np.float32)
        C = d["C"].astype(np.float32)
        times = d["times"].astype(np.float32)
        d.close()

        T, H, W = C.shape
        K_norm = normalize_K(K, stats, use_logK, eps_k)

        tmin, tmax = float(times.min()), float(times.max())

        if ssim_per_t is None:
            ssim_per_t = np.zeros(T, dtype=np.float64)
            plume_ssim_per_t = np.zeros(T, dtype=np.float64)
            plume_area_frac_per_t = np.zeros(T, dtype=np.float64)
            centroid_error_per_t = np.zeros(T, dtype=np.float64)
            centroid_count_per_t = np.zeros(T, dtype=np.int64)
            count_per_t = np.zeros(T, dtype=np.int64)

        active_indices = timestep_indices if timestep_indices is not None else list(range(T))

        for t in active_indices:
            t_norm = (float(times[t]) - tmin) / (tmax - tmin + 1e-12)
            prL = stitch_predict_one_t(model, K_norm, t_norm, patch, stride, device, time_encoding=time_encoding)

            gt_phys = np.clip(C[t], 0.0, None)
            gtL = gt_log10(gt_phys, eps_c)

            # global SSIM
            s = ssim_log(gtL, prL)
            all_ssim.append(s)
            ssim_per_t[t] += s
            count_per_t[t] += 1

            # physical errors
            pr_phys = log10_to_phys(prL, eps_c)
            diff = pr_phys - gt_phys
            all_l1.append(float(np.mean(np.abs(diff))))
            all_mse.append(float(np.mean(diff * diff)))
            all_mass.append(float(np.abs(pr_phys.sum() - gt_phys.sum())))

            # plume mask from GT
            if plume_mode == "phys":
                mask = gt_phys > plume_thresh
                pr_mask = pr_phys > plume_thresh
            else:
                mask = gtL > plume_thresh
                pr_mask = prL > plume_thresh

            plume_area_frac = float(mask.mean())
            plume_area_frac_all.append(plume_area_frac)
            plume_area_frac_per_t[t] += plume_area_frac

            ps = plume_ssim_bbox(gtL, prL, mask, pad=plume_pad, min_pixels=plume_min_pixels)
            if np.isfinite(ps):
                plume_ssim_all.append(ps)
                plume_ssim_per_t[t] += ps

            gt_centroid = mask_centroid(mask)
            pr_centroid = mask_centroid(pr_mask)
            if gt_centroid is not None and pr_centroid is not None:
                cy = gt_centroid[0] - pr_centroid[0]
                cx = gt_centroid[1] - pr_centroid[1]
                centroid_err = float(np.sqrt(cy * cy + cx * cx))
                centroid_error_all.append(centroid_err)
                centroid_error_per_t[t] += centroid_err
                centroid_count_per_t[t] += 1

    ssim_per_t = ssim_per_t / np.maximum(count_per_t, 1)
    plume_area_frac_per_t = plume_area_frac_per_t / np.maximum(count_per_t, 1)
    plume_ssim_per_t = plume_ssim_per_t / np.maximum(count_per_t, 1)
    centroid_error_per_t = centroid_error_per_t / np.maximum(centroid_count_per_t, 1)

    return {
        "mean_ssim_log": float(np.mean(all_ssim)),
        "ssim_per_timestep_log": ssim_per_t.tolist(),

        "mean_plume_ssim_log_bbox": float(np.mean(plume_ssim_all)) if len(plume_ssim_all) else float("nan"),
        "plume_ssim_per_timestep_log_bbox": plume_ssim_per_t.tolist(),
        "mean_plume_area_fraction": float(np.mean(plume_area_frac_all)),
        "plume_area_fraction_per_timestep": plume_area_frac_per_t.tolist(),
        "mean_centroid_error_pixels": float(np.mean(centroid_error_all)) if len(centroid_error_all) else float("nan"),
        "centroid_error_per_timestep_pixels": centroid_error_per_t.tolist(),

        "mean_l1_phys": float(np.mean(all_l1)),
        "mean_mse_phys": float(np.mean(all_mse)),
        "mean_mass_err_phys": float(np.mean(all_mass)),

        "n_test_files": int(len(test_files)),
        "patch": int(patch),
        "stride": int(stride),
        "timestep_indices": timestep_indices,
        "time_encoding": time_encoding,
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),

        "model_class": "UNet2D_TimeCond_MultiHead",
        "plume_mask": {
            "mode": plume_mode,
            "thresh": float(plume_thresh),
            "pad": int(plume_pad),
            "min_pixels": int(plume_min_pixels),
            "definition": "mask built from GT; plume SSIM computed on tight bounding box around mask"
        }
    }


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

    ap.add_argument("--plume_mode", type=str, default="phys", choices=["phys", "log"])
    ap.add_argument("--plume_thresh", type=float, default=1e-8)
    ap.add_argument("--plume_pad", type=int, default=8)
    ap.add_argument("--plume_min_pixels", type=int, default=64)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--mass_mlp_hidden", type=int, default=256)
    ap.add_argument("--time_encoding", type=str, default="linear", choices=["linear", "fourier"])

    ap.add_argument("--out_json", type=str, default="test_report_logc_obj1_plumemask.json")
    args = ap.parse_args()

    stats = load_stats(args.stats_json)

    _, _, test_params = load_param_split(args.split_json)
    test_files = flatten_param_folders(args.data_root, test_params)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg = ckpt.get("cfg", {})

    base = int(cfg.get("base_channels", args.base_channels))
    hidden = int(cfg.get("mass_mlp_hidden", args.mass_mlp_hidden))
    time_encoding = str(cfg.get("time_encoding", args.time_encoding))

    in_ch = 2 if time_encoding == "linear" else 3
    model = UNet2D_TimeCond_MultiHead(in_ch=in_ch, base=base, mass_mlp_hidden=hidden).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = evaluate(
        model=model,
        test_files=test_files,
        stats=stats,
        patch=args.patch,
        stride=args.stride,
        device=args.device,
        use_logK=(args.use_logK or bool(cfg.get("use_logK", True))),
        eps_k=float(cfg.get("eps_k", args.eps_k)),
        eps_c=float(cfg.get("eps_c", args.eps_c)),
        plume_mode=args.plume_mode,
        plume_thresh=args.plume_thresh,
        plume_pad=args.plume_pad,
        plume_min_pixels=args.plume_min_pixels,
        time_encoding=time_encoding,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[Saved]", args.out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

# python -m src.eval.evaluate_timecond_multihead_patch_logc_plumemask ^
#   --data_root "./data/T25_TSTEP_OVERRIDE_FINAL" ^
#   --split_json "splits/param_split_fixed.json" ^
#   --stats_json "stats/train_stats.json" ^
#   --ckpt "runs/obj1_timecond_multihead_p320_seed0_midbias_sig0.18_loglin_a20_yb-12_yc-2_lmh0.05_lr0.0001/best.pt" ^
#   --patch 320 --stride 160 --use_logK ^
#   --plume_mode phys --plume_thresh 1e-8 --plume_pad 8 --plume_min_pixels 64 ^
#   --out_json "runs/obj1_timecond_multihead_p320_seed0_midbias_sig0.18_loglin_a20_yb-12_yc-2_lmh0.05_lr0.0001/test_report_logc_obj1_plumemask.json"
