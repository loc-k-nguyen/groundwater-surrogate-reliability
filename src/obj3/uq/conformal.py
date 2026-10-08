from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def absolute_residual_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must share the same shape, got {y_true_arr.shape} vs {y_pred_arr.shape}."
        )
    return np.abs(y_pred_arr - y_true_arr).astype(np.float32)


@dataclass
class CoverageSummary:
    alpha: float
    quantile: float
    global_coverage: float
    global_mean_width: float
    plume_coverage: float
    plume_mean_width: float


class SplitConformalPredictor:
    def __init__(self, one_sided_upper: bool = False, plume_thresh: float = 1e-8) -> None:
        self.one_sided_upper = bool(one_sided_upper)
        self.plume_thresh = float(plume_thresh)
        self.calibration_scores: np.ndarray | None = None

    def fit(self, scores: np.ndarray) -> None:
        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.size == 0:
            raise ValueError("Calibration scores must be non-empty.")
        if np.any(~np.isfinite(scores_arr)):
            raise ValueError("Calibration scores contain NaN or Inf.")
        self.calibration_scores = scores_arr.reshape(-1)

    def _require_fitted(self) -> np.ndarray:
        if self.calibration_scores is None:
            raise RuntimeError("SplitConformalPredictor.fit must be called before prediction.")
        return self.calibration_scores

    def _quantile(self, alpha: float) -> float:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}.")
        scores = self._require_fitted()
        level = np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size
        level = float(np.clip(level, 0.0, 1.0))
        try:
            return float(np.quantile(scores, level, method="higher"))
        except TypeError:
            return float(np.quantile(scores, level, interpolation="higher"))

    def predict_interval(self, y_pred: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        quantile = self._quantile(alpha)
        if self.one_sided_upper:
            lower = np.full_like(y_pred_arr, -np.inf, dtype=np.float32)
            upper = y_pred_arr + quantile
        else:
            lower = y_pred_arr - quantile
            upper = y_pred_arr + quantile
        return lower.astype(np.float32), upper.astype(np.float32)

    def coverage_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha_list: Iterable[float] = (0.05, 0.10, 0.20),
    ) -> dict[str, dict[str, float]]:
        y_true_arr = np.asarray(y_true, dtype=np.float32)
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"y_true and y_pred must share the same shape, got {y_true_arr.shape} vs {y_pred_arr.shape}."
            )

        plume_mask = y_true_arr > self.plume_thresh
        report: dict[str, dict[str, float]] = {}

        for alpha in alpha_list:
            quantile = self._quantile(float(alpha))
            lower, upper = self.predict_interval(y_pred_arr, float(alpha))
            if self.one_sided_upper:
                covered = y_true_arr <= upper
            else:
                covered = (y_true_arr >= lower) & (y_true_arr <= upper)

            width = upper - lower
            if self.one_sided_upper:
                width = upper - y_pred_arr

            if np.any(plume_mask):
                plume_coverage = float(np.mean(covered[plume_mask]))
                plume_mean_width = float(np.mean(width[plume_mask]))
            else:
                plume_coverage = float("nan")
                plume_mean_width = float("nan")

            summary = CoverageSummary(
                alpha=float(alpha),
                quantile=quantile,
                global_coverage=float(np.mean(covered)),
                global_mean_width=float(np.mean(width)),
                plume_coverage=plume_coverage,
                plume_mean_width=plume_mean_width,
            )
            report[f"alpha_{float(alpha):.2f}"] = {
                "alpha": summary.alpha,
                "quantile": summary.quantile,
                "global_coverage": summary.global_coverage,
                "global_mean_width": summary.global_mean_width,
                "plume_coverage": summary.plume_coverage,
                "plume_mean_width": summary.plume_mean_width,
                "mode": "one_sided_upper" if self.one_sided_upper else "two_sided",
            }

        return report
