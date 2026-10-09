from typing import Tuple

import numpy as np


CENTROID_THRESH = 1e-8
CENTROID_MIN_PIXELS = 64


def _weighted_centroid(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    rows, cols = np.where(mask)
    weights = values[mask].astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return float(values.shape[0] // 2), float(values.shape[1] // 2)
    row_c = float(np.sum(rows.astype(np.float64) * weights) / weight_sum)
    col_c = float(np.sum(cols.astype(np.float64) * weights) / weight_sum)
    return row_c, col_c


def compute_plume_centroid_error(
    gt_phys: np.ndarray,
    pred_phys: np.ndarray,
    thresh: float = CENTROID_THRESH,
    min_pixels: int = CENTROID_MIN_PIXELS,
) -> Tuple[float, int]:
    gt_mask = np.asarray(gt_phys) > float(thresh)
    gt_valid_count = int(gt_mask.sum())
    if gt_valid_count < int(min_pixels):
        return float("nan"), gt_valid_count

    gt_row, gt_col = _weighted_centroid(np.asarray(gt_phys, dtype=np.float64), gt_mask)

    pred_mask = np.asarray(pred_phys) > float(thresh)
    if int(pred_mask.sum()) < int(min_pixels):
        pred_row = float(gt_phys.shape[0] // 2)
        pred_col = float(gt_phys.shape[1] // 2)
    else:
        pred_row, pred_col = _weighted_centroid(np.asarray(pred_phys, dtype=np.float64), pred_mask)

    err = float(np.sqrt((gt_row - pred_row) ** 2 + (gt_col - pred_col) ** 2))
    return err, gt_valid_count
