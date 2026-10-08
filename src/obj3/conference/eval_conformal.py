"""
Conformal Prediction evaluation for Obj3.

Implements:
  1. Vanilla Split Conformal (SCP) - absolute residual score
  2. Normalized Ensemble Conformal (NEC) - residual normalized by ensemble std

Calibrates on val_calib + extra_calib, evaluates on iid_test and ood_test.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

logger = logging.getLogger(__name__)

PLUME_THRESH = 1e-8
EPS_NEC = 1e-4
DEFAULT_SCORE_MASK = "plume"
DEFAULT_SIGMA_FLOOR_QUANTILE = 0.10


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile for coverage level 1-alpha."""
    n = len(scores)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, level, interpolation="higher"))


def _iter_selected_arrays(
    values: np.ndarray,
    c_phys: np.ndarray | None,
    score_mask: str,
) -> list[np.ndarray]:
    """Collect flattened values one sample at a time under the requested score mask."""
    chunks: list[np.ndarray] = []
    for idx in range(values.shape[0]):
        sample_values = values[idx]
        if score_mask == "plume":
            if c_phys is None:
                raise ValueError("Plume score masking requires c_phys arrays.")
            mask = c_phys[idx] > PLUME_THRESH
            if not np.any(mask):
                continue
            selected = sample_values[mask]
        else:
            selected = sample_values.reshape(-1)

        if selected.size > 0:
            chunks.append(np.asarray(selected, dtype=np.float32))
    return chunks


def _concat_chunks(chunks: list[np.ndarray], label: str) -> np.ndarray:
    if not chunks:
        raise ValueError(f"No calibration values found for '{label}'.")
    return np.concatenate(chunks, axis=0)


def evaluate_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cal_quantile: float,
    plume_mask: np.ndarray | None = None,
    chunk_size: int = 4,
) -> Dict[str, float]:
    """Evaluate SCP coverage and interval width in chunks."""
    n_samples = y_true.shape[0]
    total_count = 0
    total_covered = 0
    total_width = 0.0
    plume_count = 0
    plume_covered = 0
    plume_width = 0.0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        y_true_chunk = y_true[start:end]
        y_pred_chunk = y_pred[start:end]
        lower = y_pred_chunk - cal_quantile
        upper = y_pred_chunk + cal_quantile
        covered = (y_true_chunk >= lower) & (y_true_chunk <= upper)
        width = upper - lower

        total_count += covered.size
        total_covered += int(np.count_nonzero(covered))
        total_width += float(width.sum(dtype=np.float64))

        if plume_mask is not None:
            plume_chunk = plume_mask[start:end]
            plume_pixels = int(np.count_nonzero(plume_chunk))
            if plume_pixels > 0:
                plume_count += plume_pixels
                plume_covered += int(np.count_nonzero(covered[plume_chunk]))
                plume_width += float(width[plume_chunk].sum(dtype=np.float64))

    result = {
        "global_coverage": float(total_covered / max(total_count, 1)),
        "global_mean_width": float(total_width / max(total_count, 1)),
    }
    if plume_count > 0:
        result["plume_coverage"] = float(plume_covered / plume_count)
        result["plume_mean_width"] = float(plume_width / plume_count)
    else:
        result["plume_coverage"] = float("nan")
        result["plume_mean_width"] = float("nan")
    return result


def evaluate_nec_coverage(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    cal_quantile_nec: float,
    sigma_floor: float,
    plume_mask: np.ndarray | None = None,
    chunk_size: int = 4,
) -> Dict[str, float]:
    """Evaluate NEC coverage and interval width in chunks."""
    n_samples = y_true.shape[0]
    total_count = 0
    total_covered = 0
    total_width = 0.0
    plume_count = 0
    plume_covered = 0
    plume_width = 0.0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        y_true_chunk = y_true[start:end]
        y_pred_mean_chunk = y_pred_mean[start:end]
        y_pred_std_chunk = y_pred_std[start:end]

        half_width = cal_quantile_nec * np.maximum(y_pred_std_chunk, sigma_floor)
        lower = y_pred_mean_chunk - half_width
        upper = y_pred_mean_chunk + half_width
        covered = (y_true_chunk >= lower) & (y_true_chunk <= upper)
        width = upper - lower

        total_count += covered.size
        total_covered += int(np.count_nonzero(covered))
        total_width += float(width.sum(dtype=np.float64))

        if plume_mask is not None:
            plume_chunk = plume_mask[start:end]
            plume_pixels = int(np.count_nonzero(plume_chunk))
            if plume_pixels > 0:
                plume_count += plume_pixels
                plume_covered += int(np.count_nonzero(covered[plume_chunk]))
                plume_width += float(width[plume_chunk].sum(dtype=np.float64))

    result = {
        "global_coverage": float(total_covered / max(total_count, 1)),
        "global_mean_width": float(total_width / max(total_count, 1)),
    }
    if plume_count > 0:
        result["plume_coverage"] = float(plume_covered / plume_count)
        result["plume_mean_width"] = float(plume_width / plume_count)
    else:
        result["plume_coverage"] = float("nan")
        result["plume_mean_width"] = float("nan")
    return result


def compute_calibration_quantiles(
    cal_y_true: np.ndarray,
    cal_y_pred: np.ndarray,
    cal_y_pred_std: np.ndarray | None,
    cal_c_phys: np.ndarray | None,
    alpha_list: Sequence[float],
    score_mask: str = DEFAULT_SCORE_MASK,
    nec_sigma_floor_quantile: float = DEFAULT_SIGMA_FLOOR_QUANTILE,
    nec_sigma_floor_min: float = EPS_NEC,
) -> tuple[Dict[str, dict], Dict[str, dict], Dict[str, float | int | str]]:
    """Compute calibration quantiles once and reuse them for every test split."""
    residual = np.abs(cal_y_true - cal_y_pred).astype(np.float32, copy=False)
    residual_chunks = _iter_selected_arrays(residual, cal_c_phys, score_mask)
    scp_scores = _concat_chunks(residual_chunks, label=f"{score_mask}_scp_scores")

    calibration_info: Dict[str, float | int | str] = {
        "score_mask": score_mask,
        "n_calibration_scores": int(scp_scores.size),
    }

    scp_quantiles: Dict[str, dict] = {}
    for alpha in alpha_list:
        scp_quantiles[f"alpha_{alpha:.2f}"] = {
            "alpha": float(alpha),
            "quantile": conformal_quantile(scp_scores, alpha),
        }
    del scp_scores
    gc.collect()

    nec_quantiles: Dict[str, dict] = {}
    if cal_y_pred_std is not None:
        std_chunks = _iter_selected_arrays(cal_y_pred_std, cal_c_phys, score_mask)
        std_scores = _concat_chunks(std_chunks, label=f"{score_mask}_sigma_values")
        positive_std = std_scores[std_scores > 0.0]
        if positive_std.size > 0:
            sigma_floor = float(np.quantile(positive_std, nec_sigma_floor_quantile))
        else:
            sigma_floor = float(nec_sigma_floor_min)
        sigma_floor = max(float(nec_sigma_floor_min), sigma_floor)

        nec_score_chunks: list[np.ndarray] = []
        for residual_chunk, std_chunk in zip(residual_chunks, std_chunks):
            nec_score_chunks.append(
                np.divide(
                    residual_chunk,
                    np.maximum(std_chunk, sigma_floor),
                    dtype=np.float32,
                )
            )
        nec_scores = _concat_chunks(nec_score_chunks, label=f"{score_mask}_nec_scores")

        for alpha in alpha_list:
            nec_quantiles[f"alpha_{alpha:.2f}"] = {
                "alpha": float(alpha),
                "quantile_nec": conformal_quantile(nec_scores, alpha),
                "sigma_floor": sigma_floor,
            }

        calibration_info["nec_sigma_floor"] = sigma_floor
        calibration_info["nec_sigma_floor_quantile"] = float(nec_sigma_floor_quantile)
        calibration_info["n_nec_scores"] = int(nec_scores.size)
        del std_scores, positive_std, nec_scores
        gc.collect()

    del residual, residual_chunks
    gc.collect()
    return scp_quantiles, nec_quantiles, calibration_info


def run_conformal_pipeline(
    test_y_true: np.ndarray,
    test_y_pred: np.ndarray,
    test_y_pred_std: np.ndarray | None,
    scp_quantiles: Dict[str, dict],
    nec_quantiles: Dict[str, dict],
    test_c_phys: np.ndarray | None = None,
    split_name: str = "test",
    chunk_size: int = 4,
    score_mask: str = DEFAULT_SCORE_MASK,
) -> Dict[str, object]:
    """Evaluate precomputed conformal quantiles on one test split."""
    test_plume_mask = None
    if test_c_phys is not None:
        test_plume_mask = test_c_phys > PLUME_THRESH

    results = {"split": split_name, "score_mask": score_mask, "scp": {}, "nec": {}}

    for key, val in scp_quantiles.items():
        cov = evaluate_coverage(
            test_y_true,
            test_y_pred,
            float(val["quantile"]),
            plume_mask=test_plume_mask,
            chunk_size=chunk_size,
        )
        results["scp"][key] = {**val, **cov}

    if nec_quantiles and test_y_pred_std is not None:
        for key, val in nec_quantiles.items():
            cov_nec = evaluate_nec_coverage(
                test_y_true,
                test_y_pred,
                test_y_pred_std,
                float(val["quantile_nec"]),
                sigma_floor=float(val.get("sigma_floor", EPS_NEC)),
                plume_mask=test_plume_mask,
                chunk_size=chunk_size,
            )
            results["nec"][key] = {**val, **cov_nec}

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Obj3 Conformal Prediction evaluation")
    ap.add_argument("--calibration_preds", type=str, required=True,
                    help="NPZ with cal_y_true, cal_y_pred, cal_y_pred_std, c_phys")
    ap.add_argument("--iid_preds", type=str, required=True,
                    help="NPZ with test_y_true, test_y_pred, test_y_pred_std, test_c_phys")
    ap.add_argument("--ood_preds", type=str, required=True,
                    help="NPZ with test_y_true, test_y_pred, test_y_pred_std, test_c_phys")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--alpha_list", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--chunk_size", type=int, default=4)
    ap.add_argument("--score_mask", choices=["plume", "all"], default=DEFAULT_SCORE_MASK)
    ap.add_argument("--nec_sigma_floor_quantile", type=float, default=DEFAULT_SIGMA_FLOOR_QUANTILE)
    ap.add_argument("--nec_sigma_floor_min", type=float, default=EPS_NEC)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_data = np.load(args.calibration_preds)
    cal_y_true = cal_data["y_true"]
    cal_y_pred = cal_data["y_pred"]
    cal_y_pred_std = cal_data.get("y_pred_std", None)
    cal_c_phys = cal_data.get("c_phys", None)
    logger.info("Calibration: %d samples", cal_y_true.shape[0])
    scp_quantiles, nec_quantiles, calibration_info = compute_calibration_quantiles(
        cal_y_true,
        cal_y_pred,
        cal_y_pred_std,
        cal_c_phys,
        args.alpha_list,
        score_mask=args.score_mask,
        nec_sigma_floor_quantile=args.nec_sigma_floor_quantile,
        nec_sigma_floor_min=args.nec_sigma_floor_min,
    )
    logger.info(
        "Calibration score mask=%s  n_scores=%d%s",
        calibration_info["score_mask"],
        calibration_info["n_calibration_scores"],
        (
            f"  sigma_floor={calibration_info['nec_sigma_floor']:.6f}"
            if "nec_sigma_floor" in calibration_info
            else ""
        ),
    )
    del cal_data, cal_y_true, cal_y_pred, cal_y_pred_std, cal_c_phys
    gc.collect()

    for split_tag, pred_path in [("iid_test", args.iid_preds), ("ood_test", args.ood_preds)]:
        test_data = np.load(pred_path)
        test_y_true = test_data["y_true"]
        test_y_pred = test_data["y_pred"]
        test_y_pred_std = test_data.get("y_pred_std", None)
        test_c_phys = test_data.get("c_phys", None)

        logger.info("Evaluating '%s': %d samples", split_tag, test_y_true.shape[0])
        results = run_conformal_pipeline(
            test_y_true,
            test_y_pred,
            test_y_pred_std,
            scp_quantiles,
            nec_quantiles,
            test_c_phys=test_c_phys,
            split_name=split_tag,
            chunk_size=args.chunk_size,
            score_mask=args.score_mask,
        )
        results["calibration"] = calibration_info

        report_path = output_dir / f"conformal_{split_tag}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        for method in ["scp", "nec"]:
            if results[method]:
                for key, val in results[method].items():
                    logger.info(
                        "  %s %s %s: cov=%.4f  width=%.4f  plume_cov=%.4f",
                        split_tag,
                        method,
                        key,
                        val["global_coverage"],
                        val["global_mean_width"],
                        val.get("plume_coverage", float("nan")),
                    )

        del test_data, test_y_true, test_y_pred, test_y_pred_std, test_c_phys
        gc.collect()


if __name__ == "__main__":
    main()
