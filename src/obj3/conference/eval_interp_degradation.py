"""
Evaluate the trained Obj3 ensemble on the interpolation OOD split.

This produces a monotonic degradation summary across sigma^2_Y levels:
  0.3, 0.7, 1.3, 1.7, 2.0

The conformal quantile is loaded from an existing conformal IID report and is
applied directly without recalibration.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.data_obj3_ood import build_interp_eval_datasets
from src.obj3.conference.eval_ensemble import load_ensemble_models, ensemble_inference
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RANGE = 14.0
PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
EPS_C = 1e-12


def _load_scp_quantile(conformal_json: str | Path, alpha: float = 0.10) -> float:
    with open(conformal_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    key = f"alpha_{alpha:.2f}"
    scp = report.get("scp", {})
    if key not in scp:
        raise KeyError(f"Missing {key} in SCP report: {conformal_json}")
    return float(scp[key]["quantile"])


def _load_ood_summary(ood_metrics: str | Path) -> dict:
    with open(ood_metrics, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate_interp_split(
    models: List[torch.nn.Module],
    dataset,
    quantile: float,
    device: str,
    patch_size: int,
    stride: int,
) -> Dict[str, float]:
    plume_means: List[float] = []
    global_means: List[float] = []
    mass_means: List[float] = []
    coverage_values: List[float] = []
    width_values: List[float] = []
    plume_coverage_values: List[float] = []
    plume_width_values: List[float] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        K = sample["K"].unsqueeze(0).to(device)
        C_log_gt = sample["C_log"].numpy()
        C_phys_gt = sample["C_phys"].numpy()

        mean_pred, _, _ = ensemble_inference(
            models,
            K,
            patch_size=patch_size,
            stride=stride,
            device=device,
        )

        lower = mean_pred - quantile
        upper = mean_pred + quantile
        covered = (C_log_gt >= lower) & (C_log_gt <= upper)
        width = upper - lower

        plume_mask = C_phys_gt > PLUME_THRESH
        coverage_values.append(float(np.mean(covered)))
        width_values.append(float(np.mean(width)))
        if plume_mask.any():
            plume_coverage_values.append(float(np.mean(covered[plume_mask])))
            plume_width_values.append(float(np.mean(width[plume_mask])))

        plume_ssims = []
        global_ssims = []
        mass_errors = []
        for t in range(25):
            global_ssims.append(float(ssim(C_log_gt[t], mean_pred[t], data_range=DATA_RANGE)))
            gt_phys_t = np.clip(C_phys_gt[t], 0.0, None)
            plume_ssim_val, _ = compute_plume_ssim(
                gt_phys_t,
                mean_pred[t],
                EPS_C,
                PLUME_THRESH,
                PLUME_PAD,
                PLUME_MIN_PIXELS,
                DATA_RANGE,
            )
            if np.isfinite(plume_ssim_val):
                plume_ssims.append(float(plume_ssim_val))

            pred_phys = np.clip(10.0 ** mean_pred[t] - EPS_C, 0.0, None)
            gt_mass = float(gt_phys_t.sum())
            mass_errors.append(abs(float(pred_phys.sum()) - gt_mass) / max(gt_mass, 1e-12))

        if plume_ssims:
            plume_means.append(float(np.mean(plume_ssims)))
        global_means.append(float(np.mean(global_ssims)))
        mass_means.append(float(np.mean(mass_errors)))

    return {
        "plume_ssim_mean": float(np.mean(plume_means)) if plume_means else float("nan"),
        "plume_ssim_std": float(np.std(plume_means)) if plume_means else float("nan"),
        "global_ssim_mean": float(np.mean(global_means)),
        "global_ssim_std": float(np.std(global_means)),
        "mass_error_mean": float(np.mean(mass_means)),
        "mass_error_std": float(np.std(mass_means)),
        "coverage_at_90": float(np.mean(coverage_values)),
        "mean_interval_width": float(np.mean(width_values)),
        "plume_coverage_at_90": float(np.mean(plume_coverage_values)) if plume_coverage_values else float("nan"),
        "plume_mean_interval_width": float(np.mean(plume_width_values)) if plume_width_values else float("nan"),
        "n_samples": int(len(dataset)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Evaluate interpolation degradation for Obj3 ensemble")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--conformal_json", type=str, required=True)
    ap.add_argument("--ood_metrics", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--interp_split_path", type=str,
                    default=str(REPO_ROOT / "splits" / "param_split_obj3_interp.json"))
    ap.add_argument("--interp_data_root", type=str,
                    default=str(REPO_ROOT / "simulation" / "datasets" / "obj3_interpolation"))
    ap.add_argument("--stats_json", type=str,
                    default=str(REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"))
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    quantile = _load_scp_quantile(args.conformal_json, args.alpha)
    models = load_ensemble_models(args.checkpoints, args.device)
    interp_datasets, _ = build_interp_eval_datasets(
        interp_split_path=args.interp_split_path,
        interp_data_root=args.interp_data_root,
        stats_path=args.stats_json,
    )

    sigma_levels: List[float] = []
    n_samples: List[int] = []
    plume_ssim_mean: List[float] = []
    plume_ssim_std: List[float] = []
    coverage_at_90: List[float] = []
    mean_interval_width: List[float] = []

    per_sigma: Dict[str, dict] = {}
    for split_name in ["interp_03", "interp_07", "interp_13", "interp_17"]:
        if split_name not in interp_datasets:
            continue
        sigma = float(split_name.split("_")[1]) / 10.0
        logger.info("Evaluating interpolation split %s (sigma^2_Y=%.1f)", split_name, sigma)
        summary = _evaluate_interp_split(
            models,
            interp_datasets[split_name]["dataset"],
            quantile,
            args.device,
            args.patch_size,
            args.stride,
        )
        sigma_levels.append(sigma)
        n_samples.append(summary["n_samples"])
        plume_ssim_mean.append(summary["plume_ssim_mean"])
        plume_ssim_std.append(summary["plume_ssim_std"])
        coverage_at_90.append(summary["coverage_at_90"])
        mean_interval_width.append(summary["mean_interval_width"])
        per_sigma[split_name] = summary

    # Append sigma^2_Y = 2.0 from existing OOD ensemble metrics.
    ood_report = _load_ood_summary(args.ood_metrics)
    ood_aggregate = ood_report.get("aggregate", {})
    ood_per_sample = ood_report.get("per_sample", [])
    sigma_levels.append(2.0)
    n_samples.append(int(ood_aggregate.get("n_samples", len(ood_per_sample))))
    plume_ssim_mean.append(float(ood_aggregate.get("plume_ssim_mean", float("nan"))))
    plume_ssim_std.append(float(ood_aggregate.get("plume_ssim_std", float("nan"))))

    # Coverage/width for sigma=2.0 come from conformal OOD report if present.
    with open(args.conformal_json.replace("iid_test", "ood_test"), "r", encoding="utf-8") as f:
        conformal_ood = json.load(f)
    alpha_key = f"alpha_{args.alpha:.2f}"
    ood_scp = conformal_ood["scp"][alpha_key]
    coverage_at_90.append(float(ood_scp["global_coverage"]))
    mean_interval_width.append(float(ood_scp["global_mean_width"]))

    report = {
        "alpha": float(args.alpha),
        "scp_quantile": float(quantile),
        "sigma_levels": sigma_levels,
        "n_samples": n_samples,
        "plume_ssim_mean": plume_ssim_mean,
        "plume_ssim_std": plume_ssim_std,
        "coverage_at_90": coverage_at_90,
        "mean_interval_width": mean_interval_width,
        "per_sigma": per_sigma,
        "ood_sigma_20": {
            "aggregate": ood_aggregate,
            "scp_alpha": ood_scp,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "interp_degradation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved interpolation degradation report -> %s", out_path)


if __name__ == "__main__":
    main()
