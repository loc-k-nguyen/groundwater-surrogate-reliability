from typing import Tuple

import numpy as np


def _weighted_centroid(frame_log: np.ndarray, thresh: float, min_pixels: int) -> Tuple[float, float]:
    height, width = frame_log.shape
    mask = frame_log > thresh
    if int(mask.sum()) < int(min_pixels):
        return float(height // 2), float(width // 2)

    rows, cols = np.nonzero(mask)
    weights = np.exp(frame_log[mask]).astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float(height // 2), float(width // 2)

    cy = float(np.sum(rows.astype(np.float64) * weights) / weight_sum)
    cx = float(np.sum(cols.astype(np.float64) * weights) / weight_sum)
    return cy, cx


def compute_plume_flow_channels(
    seen_sequence: np.ndarray,
    target_timestep: int,
    H: int,
    W: int,
    thresh: float = -8.0,
    centroid_sigma_frac: float = 0.05,
    recent_window: int = 15,
    recency_power: float = 0.0,
) -> np.ndarray:
    """Return [flow_x, flow_y, centroid_blob] conditioning maps for ACDiff."""
    if seen_sequence.shape != (15, H, W):
        raise ValueError(f"Expected seen_sequence shape (15, {H}, {W}), got {seen_sequence.shape}")

    min_pixels = 64
    centroids_y = np.zeros(15, dtype=np.float64)
    centroids_x = np.zeros(15, dtype=np.float64)
    for t_idx in range(15):
        cy, cx = _weighted_centroid(seen_sequence[t_idx], thresh=thresh, min_pixels=min_pixels)
        centroids_y[t_idx] = cy
        centroids_x[t_idx] = cx

    recent_window = int(np.clip(recent_window, 2, 15))
    start_idx = 15 - recent_window
    t_axis = np.arange(start_idx, 15, dtype=np.float64)
    fit_y = centroids_y[start_idx:]
    fit_x = centroids_x[start_idx:]

    if float(recency_power) > 0.0:
        weights = np.arange(1, recent_window + 1, dtype=np.float64) ** float(recency_power)
        vy, cy0 = np.polyfit(t_axis, fit_y, deg=1, w=weights)
        vx, cx0 = np.polyfit(t_axis, fit_x, deg=1, w=weights)
    else:
        vy, cy0 = np.polyfit(t_axis, fit_y, deg=1)
        vx, cx0 = np.polyfit(t_axis, fit_x, deg=1)

    delta_t = float(target_timestep - 14)
    expected_cx = np.clip((cx0 + vx * 14.0) + vx * delta_t, 0.0, float(W - 1))
    expected_cy = np.clip((cy0 + vy * 14.0) + vy * delta_t, 0.0, float(H - 1))

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    sigma = max(float(centroid_sigma_frac) * float(np.sqrt(H * H + W * W)), 1e-6)
    centroid_map = np.exp(-((yy - expected_cy) ** 2 + (xx - expected_cx) ** 2) / (2.0 * sigma * sigma))
    centroid_max = float(centroid_map.max())
    if centroid_max > 0.0:
        centroid_map = centroid_map / centroid_max

    flow_map_x = np.full((H, W), float(vx) / max(float(W) / 10.0, 1e-6), dtype=np.float32)
    flow_map_y = np.full((H, W), float(vy) / max(float(H) / 10.0, 1e-6), dtype=np.float32)
    return np.stack([flow_map_x, flow_map_y, centroid_map.astype(np.float32)], axis=0).astype(np.float32)
