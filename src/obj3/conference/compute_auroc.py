"""
Compute Obj3 AUROC summaries from existing ensemble metrics JSON files.

This script computes:
  1. OOD AUROC using ensemble uncertainty as the score
  2. Failure AUROC using ensemble uncertainty as the score

Failure is defined using the 10th percentile of IID plume SSIM.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.obj3.conference.metrics_uq import compute_auroc_failure, compute_auroc_ood

logger = logging.getLogger(__name__)


def _load_metrics(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_per_sample_arrays(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    per_sample = metrics.get("per_sample", [])
    uncertainty = np.asarray(
        [row.get("mean_ensemble_var", np.nan) for row in per_sample],
        dtype=np.float32,
    )
    plume_ssim = np.asarray(
        [row.get("mean_plume_ssim", np.nan) for row in per_sample],
        dtype=np.float32,
    )
    return uncertainty, plume_ssim


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Compute Obj3 OOD and failure AUROC")
    ap.add_argument("--iid_metrics", type=str, required=True)
    ap.add_argument("--ood_metrics", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--failure_quantile", type=float, default=0.10)
    args = ap.parse_args()

    iid_metrics = _load_metrics(args.iid_metrics)
    ood_metrics = _load_metrics(args.ood_metrics)

    iid_uncertainty, iid_plume_ssim = _extract_per_sample_arrays(iid_metrics)
    ood_uncertainty, ood_plume_ssim = _extract_per_sample_arrays(ood_metrics)

    finite_iid_plume = iid_plume_ssim[np.isfinite(iid_plume_ssim)]
    if finite_iid_plume.size == 0:
        raise ValueError("IID plume SSIM array is empty or non-finite; cannot define failure threshold.")

    failure_threshold = float(np.quantile(finite_iid_plume, args.failure_quantile))
    ood_auroc = compute_auroc_ood(iid_uncertainty, ood_uncertainty)

    combined_uncertainty = np.concatenate([iid_uncertainty, ood_uncertainty], axis=0)
    combined_plume_ssim = np.concatenate([iid_plume_ssim, ood_plume_ssim], axis=0)
    failure_auroc = compute_auroc_failure(
        combined_uncertainty,
        combined_plume_ssim,
        failure_threshold,
    )

    iid_failures = int(np.sum(np.isfinite(iid_plume_ssim) & (iid_plume_ssim < failure_threshold)))
    ood_failures = int(np.sum(np.isfinite(ood_plume_ssim) & (ood_plume_ssim < failure_threshold)))

    report = {
        "iid_metrics_path": str(args.iid_metrics),
        "ood_metrics_path": str(args.ood_metrics),
        "failure_quantile": float(args.failure_quantile),
        "failure_threshold_plume_ssim": failure_threshold,
        "auroc_ood": float(ood_auroc),
        "auroc_failure": float(failure_auroc),
        "n_iid_samples": int(iid_uncertainty.size),
        "n_ood_samples": int(ood_uncertainty.size),
        "n_iid_failures": iid_failures,
        "n_ood_failures": ood_failures,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "auroc_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved AUROC summary -> %s", out_path)
    logger.info("OOD AUROC: %.4f | Failure AUROC: %.4f | Failure threshold: %.4f",
                report["auroc_ood"], report["auroc_failure"], report["failure_threshold_plume_ssim"])


if __name__ == "__main__":
    main()
