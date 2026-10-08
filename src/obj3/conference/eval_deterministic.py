"""
Deterministic evaluation of MS-TMO-UNet on Obj3 splits.

Runs sliding-window inference on full-field data for each eval split
(iid_test, ood_test) and computes plume SSIM, global SSIM, and mass error.

Usage:
    python -m src.obj3.conference.eval_deterministic \\
        --checkpoint experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed0/best.pt \\
        --output_dir experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed0/eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.data_obj3_ood import (
    Obj3FullFieldDataset,
    collect_npz_files,
    load_obj3_split,
)
from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Evaluation protocol (fixed)
DATA_RANGE = 14.0
PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
EPS_C = 1e-12


def patch_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError(
            f"patch_size and stride must be positive, got patch_size={patch_size}, stride={stride}"
        )
    if patch_size >= length:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    end_position = length - patch_size
    if positions[-1] != end_position:
        positions.append(end_position)
    return positions


def sliding_window_inference(
    model: torch.nn.Module,
    K_full: torch.Tensor,
    patch_size: int = 320,
    stride: int = 160,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run sliding-window inference with Hann window blending.

    Args:
        model: trained model
        K_full: (1, 1, H, W) input tensor
        patch_size: patch size
        stride: stride (overlap = patch_size - stride)
        device: compute device

    Returns:
        pred: (1, 25, H, W) predictions in log10 space
    """
    _, _, H, W = K_full.shape
    out_channels = 25

    # Create Hann window for blending
    hann_1d_h = torch.hann_window(patch_size, periodic=False).to(device)
    hann_1d_w = torch.hann_window(patch_size, periodic=False).to(device)
    hann_2d = hann_1d_h.unsqueeze(1) * hann_1d_w.unsqueeze(0)  # (P, P)
    hann_2d = hann_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, P, P)
    hann_2d = torch.clamp(hann_2d, min=1e-3)

    output = torch.zeros(1, out_channels, H, W, device=device)
    weight = torch.zeros(1, 1, H, W, device=device)

    # Pad if needed
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    K_padded = F.pad(K_full, [0, pad_w, 0, pad_h], mode="reflect")
    H_pad, W_pad = K_padded.shape[2], K_padded.shape[3]

    output_padded = torch.zeros(1, out_channels, H_pad, W_pad, device=device)
    weight_padded = torch.zeros(1, 1, H_pad, W_pad, device=device)

    row_positions = patch_positions(H_pad, patch_size, stride)
    col_positions = patch_positions(W_pad, patch_size, stride)
    for r in row_positions:
        for c in col_positions:
            patch = K_padded[:, :, r : r + patch_size, c : c + patch_size]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                pred_patch = model(patch)  # (1, 25, P, P)
            pred_patch = pred_patch.float()

            output_padded[:, :, r : r + patch_size, c : c + patch_size] += pred_patch * hann_2d
            weight_padded[:, :, r : r + patch_size, c : c + patch_size] += hann_2d

    # Unpad
    output = output_padded[:, :, :H, :W]
    weight = weight_padded[:, :, :H, :W]
    if torch.any(weight <= 0):
        raise RuntimeError(
            f"Sliding-window coverage failure for field={(H, W)}, patch={patch_size}, "
            f"stride={stride}, rows={row_positions}, cols={col_positions}"
        )

    return output / weight


def evaluate_split(
    model: torch.nn.Module,
    dataset: Obj3FullFieldDataset,
    device: str = "cuda",
    patch_size: int = 320,
    stride: int = 160,
    pred_clamp_min: float = -12.0,
    pred_clamp_max: float = 2.0,
    limit: int = 0,
) -> Dict[str, object]:
    """
    Evaluate a model on a full-field dataset split.

    Returns:
        metrics: dict with per-sample and aggregate metrics
    """
    model.eval()
    results: List[Dict] = []

    n_samples = len(dataset) if limit <= 0 else min(limit, len(dataset))

    for i in range(n_samples):
        sample = dataset[i]
        K = sample["K"].unsqueeze(0).to(device)       # (1, 1, H, W)
        C_log_gt = sample["C_log"].numpy()              # (25, H, W)
        C_phys_gt = sample["C_phys"].numpy()             # (25, H, W)
        param_id = sample["param_id"]
        real_id = sample["real_id"]

        pred_log = sliding_window_inference(model, K, patch_size, stride, device)
        pred_log = torch.clamp(pred_log, pred_clamp_min, pred_clamp_max)
        pred_log_np = pred_log.squeeze(0).cpu().numpy()  # (25, H, W)

        # Per-timestep metrics
        timestep_metrics = []
        for t in range(25):
            gt_t = C_log_gt[t]
            pred_t = pred_log_np[t]
            gt_phys_t = np.clip(C_phys_gt[t], 0.0, None)

            # Global SSIM
            global_ssim = float(ssim(gt_t, pred_t, data_range=DATA_RANGE))

            # Plume SSIM
            plume_ssim_val, plume_pixels = compute_plume_ssim(
                gt_phys_t, pred_t, EPS_C, PLUME_THRESH, PLUME_PAD, PLUME_MIN_PIXELS, DATA_RANGE
            )

            # Mass error
            pred_phys_t = np.clip(10.0 ** pred_t - EPS_C, 0.0, None)
            gt_mass = float(gt_phys_t.sum())
            pred_mass = float(pred_phys_t.sum())
            mass_err = abs(pred_mass - gt_mass) / max(gt_mass, 1e-12)

            timestep_metrics.append({
                "t": t,
                "global_ssim": global_ssim,
                "plume_ssim": plume_ssim_val,
                "mass_error": mass_err,
                "plume_pixels": plume_pixels,
            })

        # Aggregate over timesteps
        valid_plume = [m for m in timestep_metrics if np.isfinite(m["plume_ssim"])]
        mean_plume_ssim = float(np.mean([m["plume_ssim"] for m in valid_plume])) if valid_plume else float("nan")
        mean_global_ssim = float(np.mean([m["global_ssim"] for m in timestep_metrics]))
        mean_mass_error = float(np.mean([m["mass_error"] for m in timestep_metrics]))

        results.append({
            "param_id": param_id,
            "real_id": real_id,
            "mean_plume_ssim": mean_plume_ssim,
            "mean_global_ssim": mean_global_ssim,
            "mean_mass_error": mean_mass_error,
            "timestep_metrics": timestep_metrics,
            "pred_log": pred_log_np,  # stored for ensemble/conformal use
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] %s/%s  plume_ssim=%.4f  global_ssim=%.4f",
                         i + 1, n_samples, param_id, real_id, mean_plume_ssim, mean_global_ssim)

    # Aggregate over samples
    all_plume = [r["mean_plume_ssim"] for r in results if np.isfinite(r["mean_plume_ssim"])]
    all_global = [r["mean_global_ssim"] for r in results]
    all_mass = [r["mean_mass_error"] for r in results]

    return {
        "per_sample": results,
        "aggregate": {
            "plume_ssim_mean": float(np.mean(all_plume)) if all_plume else float("nan"),
            "plume_ssim_std": float(np.std(all_plume)) if all_plume else float("nan"),
            "global_ssim_mean": float(np.mean(all_global)),
            "global_ssim_std": float(np.std(all_global)),
            "mass_error_mean": float(np.mean(all_mass)),
            "mass_error_std": float(np.std(all_mass)),
            "n_samples": len(results),
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Obj3 deterministic evaluation")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, default=str(REPO_ROOT / "splits" / "param_split_obj3_ood.json"))
    ap.add_argument("--main_data_root", type=str,
                    default=str(REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL"))
    ap.add_argument("--extra_data_root", type=str,
                    default=str(REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"))
    ap.add_argument("--stats_json", type=str,
                    default=str(REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"))
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--eval_splits", nargs="+", default=["iid_test", "ood_test"])
    ap.add_argument("--limit", type=int, default=0, help="Limit samples per split (0=all)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt.get("cfg", {})
    stats = ckpt.get("stats", None)
    if stats is None:
        with open(args.stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

    model = UNet_ASPP_Attn(
        in_ch=1, out_ch=25,
        base=cfg.get("base_channels", 64),
        attn_heads=cfg.get("attn_heads", 4),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint: %s (epoch %d)", args.checkpoint, ckpt.get("epoch", -1))

    # Load splits
    splits = load_obj3_split(args.split_json)

    for split_name in args.eval_splits:
        if split_name not in splits:
            logger.warning("Split '%s' not found, skipping", split_name)
            continue

        files = collect_npz_files(splits[split_name], args.main_data_root, args.extra_data_root)
        dataset = Obj3FullFieldDataset(files, stats)
        logger.info("Evaluating split '%s': %d files", split_name, len(files))

        metrics = evaluate_split(
            model, dataset, device=args.device,
            patch_size=args.patch_size, stride=args.stride,
            limit=args.limit,
        )

        # Save results (without large prediction arrays)
        report = {
            "split": split_name,
            "checkpoint": args.checkpoint,
            "aggregate": metrics["aggregate"],
            "per_sample": [
                {k: v for k, v in s.items() if k != "pred_log" and k != "timestep_metrics"}
                for s in metrics["per_sample"]
            ],
        }
        report_path = output_dir / f"metrics_{split_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Split '%s': plume_ssim=%.4f±%.4f  global_ssim=%.4f±%.4f  mass_err=%.4f±%.4f",
                     split_name,
                     metrics["aggregate"]["plume_ssim_mean"], metrics["aggregate"]["plume_ssim_std"],
                     metrics["aggregate"]["global_ssim_mean"], metrics["aggregate"]["global_ssim_std"],
                     metrics["aggregate"]["mass_error_mean"], metrics["aggregate"]["mass_error_std"])


if __name__ == "__main__":
    main()
