"""
Uncertainty-error localization analysis for Obj3.

For each sample, measures spatial overlap between:
  - High-uncertainty region: top-Q% of ensemble std pixels (within plume)
  - High-error region: top-Q% of |y_true - y_pred| pixels (within plume)

Reports mean IoU (Intersection over Union) on IID vs OOD test sets.
A high IoU means uncertainty correctly localizes where errors occur.

Reads existing ensemble NPZ files (no GPU required).

Usage (Raapoi, CPU job):
    python -m src.obj3.conference.compute_uncertainty_localization \\
        --iid_npz .../iid_test_preds.npz \\
        --ood_npz .../ood_test_preds.npz \\
        --output_dir .../paper_assets/data
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUME_THRESH = 1e-8
TOP_QUANTILE = 0.75   # top 25% = "high" region
MIN_PLUME_PIXELS = 64
CHUNK = 20


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Binary IoU between two boolean masks."""
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    if union == 0:
        return math.nan
    return inter / union


def _top_quantile_mask(
    field: np.ndarray,  # (H, W)
    plume_mask: np.ndarray,  # (H, W) bool
    q: float = TOP_QUANTILE,
) -> np.ndarray:
    """Boolean mask of pixels in the top (1-q) fraction within the plume."""
    if plume_mask.sum() < MIN_PLUME_PIXELS:
        return np.zeros_like(plume_mask)
    vals = field[plume_mask]
    threshold = np.quantile(vals, q)
    result = np.zeros_like(plume_mask)
    result[plume_mask] = field[plume_mask] >= threshold
    return result


# ---------------------------------------------------------------------------
# Sample-level computation
# ---------------------------------------------------------------------------


def _localization_sample(
    y_true: np.ndarray,       # (T, H, W) log-concentration
    y_pred: np.ndarray,       # (T, H, W) log-concentration
    y_pred_std: np.ndarray,   # (T, H, W) ensemble std
    c_phys: np.ndarray,       # (T, H, W) physical concentration
) -> dict:
    """Compute per-timestep IoU and return per-sample aggregates."""
    T = y_true.shape[0]
    iou_per_t = []

    for t in range(T):
        plume_mask = c_phys[t] > PLUME_THRESH
        if plume_mask.sum() < MIN_PLUME_PIXELS:
            continue

        abs_err = np.abs(y_true[t] - y_pred[t])
        unc = y_pred_std[t]

        mask_err = _top_quantile_mask(abs_err, plume_mask)
        mask_unc = _top_quantile_mask(unc, plume_mask)

        iou = _iou(mask_unc, mask_err)
        if not math.isnan(iou):
            iou_per_t.append(iou)

    if not iou_per_t:
        return {"mean_iou": math.nan, "n_valid_timesteps": 0}

    return {
        "mean_iou": float(np.mean(iou_per_t)),
        "n_valid_timesteps": len(iou_per_t),
    }


# ---------------------------------------------------------------------------
# Split-level processing
# ---------------------------------------------------------------------------


def process_split(npz_path: Path, split_name: str) -> dict:
    """Process one split from NPZ and return aggregate localization metrics."""
    logger.info("Loading %s ...", npz_path)
    data = np.load(npz_path, mmap_mode="r")

    y_true = data["y_true"]         # (N, T, H, W)
    y_pred = data["y_pred"]         # (N, T, H, W)
    y_pred_std = data["y_pred_std"] # (N, T, H, W)
    c_phys = data["c_phys"]         # (N, T, H, W)
    N = y_true.shape[0]
    logger.info("%s: N=%d samples", split_name, N)

    all_iou = []

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        if start % 100 == 0:
            logger.info("  %s: sample %d / %d", split_name, start, N)

        for i in range(end - start):
            m = _localization_sample(
                y_true[start + i].astype(np.float32),
                y_pred[start + i].astype(np.float32),
                y_pred_std[start + i].astype(np.float32),
                c_phys[start + i].astype(np.float32),
            )
            if not math.isnan(m["mean_iou"]):
                all_iou.append(m["mean_iou"])

    result = {
        "split": split_name,
        "n_samples": N,
        "top_quantile": TOP_QUANTILE,
        "iou_uncertainty_vs_error": {
            "mean": float(np.mean(all_iou)),
            "std": float(np.std(all_iou)),
            "note": (
                f"Mean IoU between top-{int((1-TOP_QUANTILE)*100)}% uncertainty "
                f"and top-{int((1-TOP_QUANTILE)*100)}% error regions within the plume"
            ),
        },
    }

    logger.info(
        "%s  IoU(unc, err) = %.4f ± %.4f",
        split_name,
        result["iou_uncertainty_vs_error"]["mean"],
        result["iou_uncertainty_vs_error"]["std"],
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="Uncertainty-error localization IoU from ensemble NPZ"
    )
    ap.add_argument(
        "--iid_npz",
        type=str,
        default=str(
            REPO_ROOT
            / "experiments/obj3/conference/runs"
            / "obj3_conf_ms_tmo_ensemble5_full/preds/iid_test_preds.npz"
        ),
    )
    ap.add_argument(
        "--ood_npz",
        type=str,
        default=str(
            REPO_ROOT
            / "experiments/obj3/conference/runs"
            / "obj3_conf_ms_tmo_ensemble5_full/preds/ood_test_preds.npz"
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(
            REPO_ROOT / "experiments/obj3/conference/paper_assets/data"
        ),
    )
    args = ap.parse_args()

    iid_result = process_split(Path(args.iid_npz), "iid_test")
    ood_result = process_split(Path(args.ood_npz), "ood_test")

    output = {
        "iid_test": iid_result,
        "ood_test": ood_result,
        "delta_iou": (
            ood_result["iou_uncertainty_vs_error"]["mean"]
            - iid_result["iou_uncertainty_vs_error"]["mean"]
        ),
        "note": (
            "Spatial IoU between high-uncertainty and high-error plume regions. "
            "Higher IoU indicates uncertainty correctly localizes prediction errors."
        ),
    }

    out_path = Path(args.output_dir) / "uncertainty_localization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    logger.info("Saved -> %s", out_path)

    print("\n=== Uncertainty localization (IoU) ===")
    for split, res in [("IID", iid_result), ("OOD", ood_result)]:
        iou = res["iou_uncertainty_vs_error"]
        print(f"  {split}: IoU = {iou['mean']:.4f} ± {iou['std']:.4f}")
    print(f"  Delta (OOD-IID): {output['delta_iou']:+.4f}")


if __name__ == "__main__":
    main()
