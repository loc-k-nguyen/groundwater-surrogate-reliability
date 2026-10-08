from typing import Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim


PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
LOG10_SSIM_DATA_RANGE = 14.0
PLUME_SSIM_DATA_RANGE = LOG10_SSIM_DATA_RANGE


def compute_global_ssim(
    gt_log: np.ndarray,
    pred_log: np.ndarray,
    data_range: float = LOG10_SSIM_DATA_RANGE,
) -> float:
    return float(ssim(np.asarray(gt_log), np.asarray(pred_log), data_range=float(data_range)))


def compute_plume_ssim(
    gt_phys: np.ndarray,
    pred_log: np.ndarray,
    eps_c: float,
    thresh: float = PLUME_THRESH,
    pad: int = PLUME_PAD,
    min_pixels: int = PLUME_MIN_PIXELS,
    data_range: float = PLUME_SSIM_DATA_RANGE,
) -> Tuple[float, int]:
    plume_mask = np.asarray(gt_phys) > float(thresh)
    valid_count = int(plume_mask.sum())
    if valid_count < int(min_pixels):
        return float("nan"), valid_count

    rows, cols = np.where(plume_mask)
    if rows.size == 0 or cols.size == 0:
        return float("nan"), valid_count

    height, width = gt_phys.shape
    r0 = max(0, int(rows.min()) - int(pad))
    r1 = min(height, int(rows.max()) + int(pad) + 1)
    c0 = max(0, int(cols.min()) - int(pad))
    c1 = min(width, int(cols.max()) + int(pad) + 1)

    gt_crop = np.log10(np.clip(gt_phys[r0:r1, c0:c1], 0.0, None) + float(eps_c)).astype(np.float32)
    pred_crop = pred_log[r0:r1, c0:c1].astype(np.float32)
    return float(ssim(gt_crop, pred_crop, data_range=float(data_range))), valid_count
