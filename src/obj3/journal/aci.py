from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PLUME_THRESH = 1e-8
EPS_ALPHA = 1e-6


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Match the conference SCP quantile rule for comparability."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores_arr.size == 0:
        raise ValueError("Calibration scores must be non-empty.")
    level = np.ceil((scores_arr.size + 1) * (1.0 - alpha)) / scores_arr.size
    level = float(np.clip(level, 0.0, 1.0))
    try:
        return float(np.quantile(scores_arr, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores_arr, level, interpolation="higher"))


def plume_residual_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_phys: np.ndarray,
    plume_thresh: float = PLUME_THRESH,
) -> np.ndarray:
    """Flatten absolute plume residuals using the conference SCP score mask."""
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    c_phys_arr = np.asarray(c_phys, dtype=np.float32)
    if y_true_arr.shape != y_pred_arr.shape or y_true_arr.shape != c_phys_arr.shape:
        raise ValueError("y_true, y_pred, and c_phys must share the same shape.")

    residual = np.abs(y_true_arr - y_pred_arr)
    mask = c_phys_arr > float(plume_thresh)
    if not np.any(mask):
        raise ValueError("No plume pixels found in calibration arrays.")
    return residual[mask].astype(np.float32, copy=False)


def sample_keys_from_metrics(metrics_json: str | Path) -> list[tuple[int, int]]:
    """Read sample ordering from existing metrics JSON files."""
    with open(metrics_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    keys: list[tuple[int, int]] = []
    for sample in payload.get("per_sample", []):
        param_id = int(str(sample["param_id"]).replace("param_", ""))
        real_id = int(str(sample["real_id"]).replace("real_", ""))
        keys.append((param_id, real_id))
    return keys


def reorder_indices_for_keys(sample_keys: Iterable[tuple[int, int]]) -> np.ndarray:
    keys = list(sample_keys)
    if not keys:
        raise ValueError("sample_keys must be non-empty.")
    order = sorted(range(len(keys)), key=lambda idx: (keys[idx][0], keys[idx][1]))
    return np.asarray(order, dtype=np.int64)


@dataclass(frozen=True)
class ACIStep:
    index: int
    param_id: int
    real_id: int
    alpha_before: float
    quantile: float
    plume_pixel_coverage: float
    global_pixel_coverage: float
    plume_mean_width: float
    sample_plume_mae: float
    miss_indicator: float


def run_aci_sequence(
    calibration_scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_phys: np.ndarray,
    sample_keys: Iterable[tuple[int, int]],
    alpha: float = 0.10,
    gamma: float = 0.01,
    plume_thresh: float = PLUME_THRESH,
) -> dict[str, object]:
    """Run adaptive conformal inference over a reproducible sample sequence.

    The scalar ACI miss indicator is adapted to the plume-focused Obj3 setting:
    a sample counts as a miss when its plume-pixel coverage falls below the
    target coverage (1 - alpha). This keeps the online update aligned with the
    journal comparison metric while preserving the original binary update rule.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}.")

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    c_phys_arr = np.asarray(c_phys, dtype=np.float32)
    if y_true_arr.shape != y_pred_arr.shape or y_true_arr.shape != c_phys_arr.shape:
        raise ValueError("y_true, y_pred, and c_phys must share the same shape.")

    keys = list(sample_keys)
    if len(keys) != y_true_arr.shape[0]:
        raise ValueError(
            f"sample_keys length {len(keys)} does not match sample count {y_true_arr.shape[0]}."
        )

    target_coverage = 1.0 - float(alpha)
    current_alpha = float(alpha)
    steps: list[ACIStep] = []

    for idx, (param_id, real_id) in enumerate(keys):
        quantile = conformal_quantile(calibration_scores, current_alpha)
        lower = y_pred_arr[idx] - quantile
        upper = y_pred_arr[idx] + quantile
        covered = (y_true_arr[idx] >= lower) & (y_true_arr[idx] <= upper)
        plume_mask = c_phys_arr[idx] > float(plume_thresh)
        if not np.any(plume_mask):
            plume_mask = np.ones_like(covered, dtype=bool)

        plume_pixel_coverage = float(np.mean(covered[plume_mask]))
        global_pixel_coverage = float(np.mean(covered))
        plume_mean_width = float(np.mean((upper - lower)[plume_mask]))
        sample_plume_mae = float(np.mean(np.abs(y_true_arr[idx] - y_pred_arr[idx])[plume_mask]))
        miss_indicator = float(plume_pixel_coverage < target_coverage)

        steps.append(
            ACIStep(
                index=idx,
                param_id=param_id,
                real_id=real_id,
                alpha_before=current_alpha,
                quantile=quantile,
                plume_pixel_coverage=plume_pixel_coverage,
                global_pixel_coverage=global_pixel_coverage,
                plume_mean_width=plume_mean_width,
                sample_plume_mae=sample_plume_mae,
                miss_indicator=miss_indicator,
            )
        )

        current_alpha = float(
            np.clip(current_alpha + gamma * (float(alpha) - miss_indicator), EPS_ALPHA, 1.0 - EPS_ALPHA)
        )

    final_quantile = conformal_quantile(calibration_scores, current_alpha)
    return {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "target_coverage": target_coverage,
        "final_alpha": current_alpha,
        "final_quantile": final_quantile,
        "aggregate": {
            "plume_pixel_coverage": float(np.mean([step.plume_pixel_coverage for step in steps])),
            "global_pixel_coverage": float(np.mean([step.global_pixel_coverage for step in steps])),
            "plume_mean_width": float(np.mean([step.plume_mean_width for step in steps])),
            "sample_plume_mae": float(np.mean([step.sample_plume_mae for step in steps])),
            "sample_miss_rate": float(np.mean([step.miss_indicator for step in steps])),
            "n_samples": len(steps),
        },
        "per_sample": [
            {
                "index": step.index,
                "param_id": step.param_id,
                "real_id": step.real_id,
                "alpha_before": step.alpha_before,
                "quantile": step.quantile,
                "plume_pixel_coverage": step.plume_pixel_coverage,
                "global_pixel_coverage": step.global_pixel_coverage,
                "plume_mean_width": step.plume_mean_width,
                "sample_plume_mae": step.sample_plume_mae,
                "miss_indicator": step.miss_indicator,
            }
            for step in steps
        ],
    }


def select_best_gamma(results_by_gamma: dict[str, dict[str, object]], target_coverage: float) -> tuple[str, dict[str, object]]:
    """Choose the gamma with the smallest OOD coverage gap, then the tighter width."""
    if not results_by_gamma:
        raise ValueError("results_by_gamma must be non-empty.")

    best_key = min(
        results_by_gamma,
        key=lambda key: (
            abs(results_by_gamma[key]["ood"]["aggregate"]["plume_pixel_coverage"] - target_coverage),
            results_by_gamma[key]["ood"]["aggregate"]["plume_mean_width"],
            float(key),
        ),
    )
    return best_key, results_by_gamma[best_key]
