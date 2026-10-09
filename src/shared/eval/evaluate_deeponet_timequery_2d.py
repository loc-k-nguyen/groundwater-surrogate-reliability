import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from src.shared.data.io import flatten_param_folders, load_param_split
from src.shared.eval.centroid_error import compute_plume_centroid_error
from src.shared.eval.plume_ssim import compute_global_ssim, compute_plume_ssim


def load_stats(stats_json_path: str) -> Dict:
    with open(stats_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["stats"] if isinstance(payload, dict) and "stats" in payload else payload


def normalize_K(k_field: np.ndarray, stats: Dict, use_logK: bool, eps_k: float) -> np.ndarray:
    k_field = k_field.astype(np.float64)
    if use_logK:
        k_field = np.log(np.clip(k_field, eps_k, None))
    k_field = (k_field - float(stats["k_mean"])) / (float(stats["k_std"]) + 1e-8)
    return k_field.astype(np.float32)


def log10_phys(c_field: np.ndarray, eps_c: float) -> np.ndarray:
    return np.log10(np.clip(c_field.astype(np.float32), 0.0, None) + eps_c).astype(np.float32)


def log10_to_phys(c_log10: np.ndarray, eps_c: float) -> np.ndarray:
    c_field = (10.0 ** c_log10) - eps_c
    return np.clip(c_field, 0.0, None).astype(np.float32)


@torch.no_grad()
def predict_fullfield_one_t(
    model: torch.nn.Module,
    k_norm: np.ndarray,
    t_norm: float,
    patch: int,
    stride: int,
    device: str,
) -> np.ndarray:
    hgt, wid = k_norm.shape
    pred_sum = np.zeros((hgt, wid), dtype=np.float32)
    weight_sum = np.zeros((hgt, wid), dtype=np.float32)

    ys = list(range(0, max(hgt - patch + 1, 1), stride))
    xs = list(range(0, max(wid - patch + 1, 1), stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != hgt - patch:
        ys.append(hgt - patch)
    if xs[-1] != wid - patch:
        xs.append(wid - patch)

    t_tensor = torch.tensor([[t_norm]], dtype=torch.float32, device=device)
    for y0 in ys:
        for x0 in xs:
            k_patch = k_norm[y0:y0 + patch, x0:x0 + patch]
            k_tensor = torch.from_numpy(k_patch[None, None, ...]).to(device)
            pred = model(k_tensor, t_tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            pred_sum[y0:y0 + patch, x0:x0 + patch] += pred
            weight_sum[y0:y0 + patch, x0:x0 + patch] += 1.0

    weight_sum = np.maximum(weight_sum, 1e-6)
    return pred_sum / weight_sum


@torch.no_grad()
def evaluate_deeponet_timequery(
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
        data = np.load(fp)
        k_field = data["K"].astype(np.float32)
        c_field = data["C"].astype(np.float32)
        times = data["times"].astype(np.float32)
        data.close()

        time_count = c_field.shape[0]
        if ssim_per_t is None:
            ssim_per_t = np.zeros(time_count, dtype=np.float64)
            plume_ssim_per_t = np.zeros(time_count, dtype=np.float64)
            plume_valid_count_per_t = np.zeros(time_count, dtype=np.int64)
            centroid_err_per_t = np.zeros(time_count, dtype=np.float64)
            centroid_valid_count_per_t = np.zeros(time_count, dtype=np.int64)
            count_per_t = np.zeros(time_count, dtype=np.int64)

        k_norm = normalize_K(k_field, stats, use_logK, eps_k)
        t0, t1 = float(times.min()), float(times.max())
        active = timestep_indices if timestep_indices is not None else list(range(time_count))

        for t_idx in active:
            t_value = float(times[t_idx])
            t_norm = (t_value - t0) / (t1 - t0 + 1e-12)
            pr_log = predict_fullfield_one_t(model, k_norm, t_norm, patch, stride, device)
            gt_log = log10_phys(c_field[t_idx], eps_c)

            ssim_value = compute_global_ssim(gt_log, pr_log)
            all_ssim.append(ssim_value)
            ssim_per_t[t_idx] += ssim_value
            count_per_t[t_idx] += 1

            plume_score, plume_valid_count = compute_plume_ssim(
                gt_phys=np.clip(c_field[t_idx], 0.0, None),
                pred_log=pr_log,
                eps_c=eps_c,
            )
            if not np.isnan(plume_score):
                all_plume_ssim.append(float(plume_score))
                plume_ssim_per_t[t_idx] += float(plume_score)
                plume_valid_count_per_t[t_idx] += 1

            pr_phys = log10_to_phys(pr_log, eps_c)
            gt_phys = np.clip(c_field[t_idx], 0.0, None)

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
    return {
        "mean_ssim_log": float(np.mean(all_ssim)),
        "mean_plume_ssim_log": float(np.mean(all_plume_ssim)) if all_plume_ssim else float("nan"),
        "mean_centroid_err_px": float(np.mean(all_centroid_err)) if all_centroid_err else float("nan"),
        "ssim_per_timestep_log": ssim_per_t.tolist(),
        "plume_ssim_per_timestep_log": (plume_ssim_per_t / np.maximum(plume_valid_count_per_t, 1)).tolist(),
        "plume_valid_count_per_timestep": plume_valid_count_per_t.tolist(),
        "centroid_err_per_timestep_px": (centroid_err_per_t / np.maximum(centroid_valid_count_per_t, 1)).tolist(),
        "centroid_valid_count_per_timestep": centroid_valid_count_per_t.tolist(),
        "mean_l1_phys": float(np.mean(all_l1)),
        "mean_mse_phys": float(np.mean(all_mse)),
        "mean_mass_err_phys": float(np.mean(all_mass)),
        "n_test_files": len(test_files),
        "patch": int(patch),
        "stride": int(stride),
        "timestep_indices": timestep_indices,
        "use_logK": bool(use_logK),
        "eps_k": float(eps_k),
        "eps_c": float(eps_c),
    }


def main() -> None:
    import argparse

    from src.obj2.conference.models.deeponet_timequery_2d import DeepONetTimeQuery2D

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
    ap.add_argument("--out_json", type=str, default="test_report_deeponet_timequery_obj2.json")
    args = ap.parse_args()

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    _, _, test_params = load_param_split(args.split_json)
    test_files = flatten_param_folders(args.data_root, test_params)
    stats = load_stats(args.stats_json)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = DeepONetTimeQuery2D(
        in_ch=1,
        out_ch=1,
        branch_width=int(cfg.get("branch_width", 128)),
        branch_latent=int(cfg.get("branch_latent", 384)),
        trunk_width=int(cfg.get("trunk_width", 384)),
        basis_rank=int(cfg.get("basis_rank", 96)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = evaluate_deeponet_timequery(
        model=model,
        test_files=test_files,
        stats=stats,
        patch=args.patch,
        stride=args.stride,
        device=device,
        use_logK=args.use_logK,
        eps_k=args.eps_k,
        eps_c=args.eps_c,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("[Saved]", args.out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
