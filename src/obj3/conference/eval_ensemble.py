"""
Deep Ensemble evaluation for Obj3.

Aggregates predictions from M trained seeds, computes ensemble mean + variance,
then evaluates accuracy and uncertainty metrics on IID/OOD test splits.

Usage:
    python -m src.obj3.conference.eval_ensemble \\
        --checkpoints \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed0/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed1/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed2/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed3/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed4/best.pt \\
        --output_dir experiments/obj3/conference/runs/obj3_conf_ms_tmo_ensemble5_full
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.data_obj3_ood import (
    Obj3FullFieldDataset,
    collect_npz_files,
    load_obj3_split,
)
from src.obj3.conference.eval_deterministic import sliding_window_inference
from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RANGE = 14.0
PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
EPS_C = 1e-12


def load_ensemble_models(
    checkpoint_paths: List[str],
    device: str = "cuda",
) -> List[torch.nn.Module]:
    """Load M trained models from checkpoints."""
    models = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt.get("cfg", {})
        model = UNet_ASPP_Attn(
            in_ch=1, out_ch=25,
            base=cfg.get("base_channels", 64),
            attn_heads=cfg.get("attn_heads", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
        logger.info("Loaded: %s (epoch %d)", path, ckpt.get("epoch", -1))
    return models


def ensemble_inference(
    models: List[torch.nn.Module],
    K: torch.Tensor,
    patch_size: int = 320,
    stride: int = 160,
    device: str = "cuda",
    pred_clamp_min: float = -12.0,
    pred_clamp_max: float = 2.0,
) -> tuple:
    """
    Run inference for all ensemble members and compute mean + variance.

    Returns:
        mean_pred: (25, H, W) log10 predictions
        var_pred: (25, H, W) pixelwise ensemble variance
        all_preds: (M, 25, H, W) individual member predictions
    """
    all_preds = []
    for model in models:
        pred = sliding_window_inference(model, K, patch_size, stride, device)
        pred = torch.clamp(pred, pred_clamp_min, pred_clamp_max)
        all_preds.append(pred.squeeze(0).cpu().numpy())

    all_preds_np = np.stack(all_preds, axis=0)  # (M, 25, H, W)
    mean_pred = all_preds_np.mean(axis=0)         # (25, H, W)
    var_pred = all_preds_np.var(axis=0)           # (25, H, W)

    return mean_pred, var_pred, all_preds_np


def evaluate_ensemble_on_split(
    models: List[torch.nn.Module],
    dataset: Obj3FullFieldDataset,
    device: str = "cuda",
    patch_size: int = 320,
    stride: int = 160,
    limit: int = 0,
) -> dict:
    """Evaluate ensemble on a full-field dataset split."""
    results = []
    n_samples = len(dataset) if limit <= 0 else min(limit, len(dataset))

    for i in range(n_samples):
        sample = dataset[i]
        K = sample["K"].unsqueeze(0).to(device)
        C_log_gt = sample["C_log"].numpy()
        C_phys_gt = sample["C_phys"].numpy()
        param_id = sample["param_id"]
        real_id = sample["real_id"]

        mean_pred, var_pred, _ = ensemble_inference(models, K, patch_size, stride, device)

        # Per-timestep metrics on ensemble mean
        plume_ssims = []
        global_ssims = []
        mass_errors = []

        for t in range(25):
            gs = float(ssim(C_log_gt[t], mean_pred[t], data_range=DATA_RANGE))
            gt_phys_t = np.clip(C_phys_gt[t], 0.0, None)
            ps, _ = compute_plume_ssim(gt_phys_t, mean_pred[t], EPS_C, PLUME_THRESH, PLUME_PAD, PLUME_MIN_PIXELS, DATA_RANGE)

            pred_phys_t = np.clip(10.0 ** mean_pred[t] - EPS_C, 0.0, None)
            gt_mass = float(gt_phys_t.sum())
            me = abs(float(pred_phys_t.sum()) - gt_mass) / max(gt_mass, 1e-12)

            global_ssims.append(gs)
            if np.isfinite(ps):
                plume_ssims.append(ps)
            mass_errors.append(me)

        # Sample-level uncertainty = mean pixelwise variance
        sample_uncertainty = float(var_pred.mean())

        results.append({
            "param_id": param_id,
            "real_id": real_id,
            "mean_plume_ssim": float(np.mean(plume_ssims)) if plume_ssims else float("nan"),
            "mean_global_ssim": float(np.mean(global_ssims)),
            "mean_mass_error": float(np.mean(mass_errors)),
            "mean_ensemble_var": sample_uncertainty,
            "plume_ensemble_var": float(var_pred[:, :, :].mean()),  # will refine with plume mask
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] %s/%s  plume=%.4f  ens_var=%.6f",
                         i + 1, n_samples, param_id, real_id,
                         results[-1]["mean_plume_ssim"], sample_uncertainty)

    all_plume = [r["mean_plume_ssim"] for r in results if np.isfinite(r["mean_plume_ssim"])]
    all_global = [r["mean_global_ssim"] for r in results]
    all_mass = [r["mean_mass_error"] for r in results]
    all_var = [r["mean_ensemble_var"] for r in results]

    return {
        "per_sample": results,
        "aggregate": {
            "plume_ssim_mean": float(np.mean(all_plume)) if all_plume else float("nan"),
            "plume_ssim_std": float(np.std(all_plume)) if all_plume else float("nan"),
            "global_ssim_mean": float(np.mean(all_global)),
            "global_ssim_std": float(np.std(all_global)),
            "mass_error_mean": float(np.mean(all_mass)),
            "mass_error_std": float(np.std(all_mass)),
            "ensemble_var_mean": float(np.mean(all_var)),
            "ensemble_var_std": float(np.std(all_var)),
            "n_samples": len(results),
            "n_members": len(models),
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Obj3 Deep Ensemble evaluation")
    ap.add_argument("--checkpoints", nargs="+", required=True)
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_ensemble_models(args.checkpoints, args.device)
    logger.info("Loaded ensemble: %d members", len(models))

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    splits = load_obj3_split(args.split_json)

    for split_name in args.eval_splits:
        if split_name not in splits:
            continue

        files = collect_npz_files(splits[split_name], args.main_data_root, args.extra_data_root)
        dataset = Obj3FullFieldDataset(files, stats)
        logger.info("Evaluating ensemble on '%s': %d files", split_name, len(files))

        metrics = evaluate_ensemble_on_split(
            models, dataset, device=args.device,
            patch_size=args.patch_size, stride=args.stride, limit=args.limit,
        )

        report = {
            "split": split_name,
            "checkpoints": args.checkpoints,
            "aggregate": metrics["aggregate"],
            "per_sample": metrics["per_sample"],
        }
        report_path = output_dir / f"ensemble_metrics_{split_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Ensemble '%s': plume=%.4f±%.4f  var=%.6f",
                     split_name,
                     metrics["aggregate"]["plume_ssim_mean"],
                     metrics["aggregate"]["plume_ssim_std"],
                     metrics["aggregate"]["ensemble_var_mean"])


if __name__ == "__main__":
    main()
