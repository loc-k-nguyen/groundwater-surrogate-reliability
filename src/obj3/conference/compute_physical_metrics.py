"""
Physical metrics from ensemble NPZ predictions for Obj3.

Computes for each sample and timestep:
  - Plume centroid error (L2 distance between predicted and true centre of mass)
  - Plume area error (|predicted active pixels - true active pixels|)
  - Total contaminant mass error (|pred_mass - true_mass| / true_mass)

All metrics use physical concentration (c_phys) with threshold 1e-8.
Reads existing ensemble NPZ files — no GPU required.

Usage (CPU analysis):
    python -m src.obj3.conference.compute_physical_metrics \\
        --iid_npz .../preds/iid_test_preds.npz \\
        --ood_npz .../preds/ood_test_preds.npz \\
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
CHUNK = 20  # samples per chunk to bound memory


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def _centroid(concentration: np.ndarray) -> tuple[float, float]:
    """Compute centre of mass (row, col) weighted by concentration.

    Args:
        concentration: (H, W) non-negative concentration field.

    Returns:
        (row_centroid, col_centroid) or (nan, nan) if empty.
    """
    total = concentration.sum()
    if total < 1e-30:
        return math.nan, math.nan
    h, w = concentration.shape
    rows = np.arange(h, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(w, dtype=np.float64).reshape(1, -1)
    row_c = float((rows * concentration).sum() / total)
    col_c = float((cols * concentration).sum() / total)
    return row_c, col_c


def _physical_metrics_sample(
    c_true: np.ndarray,  # (T, H, W)
    c_pred: np.ndarray,  # (T, H, W)  exp(log-pred)
) -> dict:
    """Compute per-sample (averaged over T) physical metrics.

    Returns dict with centroid_error_mean, area_error_mean,
    rel_mass_error_mean, centroid_errors (per-T), area_errors (per-T),
    rel_mass_errors (per-T).
    """
    T = c_true.shape[0]
    centroid_errors = []
    area_errors = []
    rel_mass_errors = []

    for t in range(T):
        true_t = c_true[t]
        pred_t = c_pred[t]

        mask_true = true_t > PLUME_THRESH
        mask_pred = pred_t > PLUME_THRESH

        # Centroid error
        r_t, c_t = _centroid(np.where(mask_true, true_t, 0.0))
        r_p, c_p = _centroid(np.where(mask_pred, pred_t, 0.0))
        if math.isnan(r_t) or math.isnan(r_p):
            centroid_errors.append(math.nan)
        else:
            centroid_errors.append(math.hypot(r_p - r_t, c_p - c_t))

        # Area error (pixels)
        area_errors.append(abs(int(mask_pred.sum()) - int(mask_true.sum())))

        # Relative mass error
        true_mass = float(true_t[mask_true].sum())
        pred_mass = float(pred_t[mask_pred].sum())
        if true_mass > 1e-30:
            rel_mass_errors.append(abs(pred_mass - true_mass) / true_mass)
        else:
            rel_mass_errors.append(math.nan)

    # Aggregate (ignore nan)
    valid_ce = [v for v in centroid_errors if not math.isnan(v)]
    valid_ae = area_errors
    valid_rm = [v for v in rel_mass_errors if not math.isnan(v)]

    return {
        "centroid_error_mean": float(np.mean(valid_ce)) if valid_ce else math.nan,
        "area_error_mean": float(np.mean(valid_ae)),
        "rel_mass_error_mean": float(np.mean(valid_rm)) if valid_rm else math.nan,
        "centroid_errors": centroid_errors,
    }


# ---------------------------------------------------------------------------
# Split-level processing
# ---------------------------------------------------------------------------


def process_split(npz_path: Path, split_name: str) -> dict:
    """Load NPZ and compute physical metrics for all samples.

    NPZ keys: y_true (N,T,H,W) log-concentration, y_pred (N,T,H,W),
              y_pred_std (N,T,H,W), c_phys (N,T,H,W) physical concentration.
    """
    logger.info("Loading %s ...", npz_path)
    data = np.load(npz_path, mmap_mode="r")

    # c_phys is true physical concentration; derive pred physical concentration
    # from y_pred (log space).
    y_pred_log = data["y_pred"]   # (N, T, H, W)
    c_phys_true = data["c_phys"]  # (N, T, H, W)
    N = y_pred_log.shape[0]
    logger.info("%s: N=%d samples", split_name, N)

    all_centroid_err = []
    all_area_err = []
    all_rel_mass_err = []

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        if start % 100 == 0:
            logger.info("  %s: sample %d / %d", split_name, start, N)

        y_pred_chunk = y_pred_log[start:end].astype(np.float64)
        c_true_chunk = c_phys_true[start:end].astype(np.float64)

        # Physical predicted concentration: inverse of log10(C + eps_c).
        # The Obj3 target uses log10 concentration, not the natural logarithm.
        c_pred_chunk = np.clip(
            np.power(10.0, np.clip(y_pred_chunk, -12.0, 2.0)) - 1e-12,
            0.0,
            None,
        )

        for i in range(end - start):
            m = _physical_metrics_sample(c_true_chunk[i], c_pred_chunk[i])
            if not math.isnan(m["centroid_error_mean"]):
                all_centroid_err.append(m["centroid_error_mean"])
            all_area_err.append(m["area_error_mean"])
            if not math.isnan(m["rel_mass_error_mean"]):
                all_rel_mass_err.append(m["rel_mass_error_mean"])

    result = {
        "split": split_name,
        "n_samples": N,
        "centroid_error": {
            "mean": float(np.mean(all_centroid_err)),
            "std": float(np.std(all_centroid_err)),
            "unit": "pixels",
        },
        "plume_area_error": {
            "mean": float(np.mean(all_area_err)),
            "std": float(np.std(all_area_err)),
            "unit": "pixels",
        },
        "relative_mass_error": {
            "mean": float(np.mean(all_rel_mass_err)),
            "std": float(np.std(all_rel_mass_err)),
            "unit": "fraction",
        },
    }

    logger.info(
        "%s  centroid_err=%.2f±%.2f px  area_err=%.1f±%.1f px"
        "  rel_mass_err=%.4f±%.4f",
        split_name,
        result["centroid_error"]["mean"],
        result["centroid_error"]["std"],
        result["plume_area_error"]["mean"],
        result["plume_area_error"]["std"],
        result["relative_mass_error"]["mean"],
        result["relative_mass_error"]["std"],
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    ap = argparse.ArgumentParser(description="Physical metrics from ensemble NPZ")
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
        "delta": {
            "centroid_error_px": (
                ood_result["centroid_error"]["mean"]
                - iid_result["centroid_error"]["mean"]
            ),
            "relative_mass_error": (
                ood_result["relative_mass_error"]["mean"]
                - iid_result["relative_mass_error"]["mean"]
            ),
        },
        "note": (
            "Physical metrics from ensemble mean predictions. "
            "Centroid error is L2 pixel distance between predicted and true "
            "plume centre of mass, averaged over 25 timesteps."
        ),
    }

    out_path = Path(args.output_dir) / "physical_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    logger.info("Saved -> %s", out_path)

    # Print summary for paper
    print("\n=== Physical metrics summary ===")
    for split, res in [("IID", iid_result), ("OOD", ood_result)]:
        ce = res["centroid_error"]
        rm = res["relative_mass_error"]
        print(
            f"  {split}: centroid err = {ce['mean']:.1f} ± {ce['std']:.1f} px  |  "
            f"rel mass err = {rm['mean']:.4f} ± {rm['std']:.4f}"
        )
    d = output["delta"]
    print(
        f"  Delta (OOD-IID): centroid err Δ = {d['centroid_error_px']:+.1f} px  |  "
        f"rel mass err Δ = {d['relative_mass_error']:+.4f}"
    )


if __name__ == "__main__":
    main()
