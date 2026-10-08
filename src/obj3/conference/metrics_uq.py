"""
UQ evaluation metrics for Obj3: ECE, CRPS, AUROC, reliability diagrams.

All functions operate on numpy arrays at the sample level.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# ============================================================================
# Expected Calibration Error (ECE)
# ============================================================================


def compute_ece(
    predicted_confidence: np.ndarray,
    observed_coverage: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error for regression prediction intervals.

    Args:
        predicted_confidence: array of nominal coverage levels (1-alpha) for each sample
        observed_coverage: binary array, 1 if true value is within interval
        n_bins: number of bins for calibration

    Returns:
        ECE value (lower is better)
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(predicted_confidence)

    for i in range(n_bins):
        mask = (predicted_confidence >= bin_edges[i]) & (predicted_confidence < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = predicted_confidence[mask].mean()
        avg_coverage = observed_coverage[mask].mean()
        ece += mask.sum() / total * abs(avg_coverage - avg_confidence)

    return float(ece)


def compute_ece_from_coverage_table(
    alpha_list: Sequence[float],
    empirical_coverages: Sequence[float],
) -> float:
    """
    Simplified ECE from a discrete set of (alpha, coverage) pairs.

    Args:
        alpha_list: nominal error rates [0.05, 0.10, 0.20, ...]
        empirical_coverages: observed coverage at each alpha

    Returns:
        Mean absolute calibration error
    """
    errors = [abs(cov - (1.0 - alpha)) for alpha, cov in zip(alpha_list, empirical_coverages)]
    return float(np.mean(errors))


# ============================================================================
# Continuous Ranked Probability Score (CRPS)
# ============================================================================


def crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Closed-form CRPS for Gaussian predictive distribution.

    CRPS(F, y) = σ [z Φ(z) + φ(z) - 1/√π]
    where z = (y - μ) / σ, Φ = standard normal CDF, φ = standard normal PDF.
    """
    from scipy.stats import norm

    sigma_safe = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma_safe
    crps = sigma_safe * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return crps.astype(np.float32)


def mean_crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean CRPS assuming Gaussian predictive distribution."""
    return float(np.mean(crps_gaussian(y_true, mu, sigma)))


# ============================================================================
# AUROC for OOD / Failure Detection
# ============================================================================


def _binary_roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary AUROC in pure NumPy.

    This avoids adding a scikit-learn dependency on cluster environments where
    only lightweight numerical utilities are available.
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    valid = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct, labels_sorted.size - 1]

    tps = np.cumsum(labels_sorted == 1)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)

    tpr = np.r_[0.0, tps / pos]
    fpr = np.r_[0.0, fps / neg]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def compute_auroc_ood(
    uncertainty_iid: np.ndarray,
    uncertainty_ood: np.ndarray,
) -> float:
    """
    AUROC for detecting OOD samples using uncertainty as the score.

    Higher uncertainty on OOD → higher AUROC → better OOD detection.

    Args:
        uncertainty_iid: per-sample uncertainty scores for IID test set
        uncertainty_ood: per-sample uncertainty scores for OOD test set

    Returns:
        AUROC (0.5 = random, 1.0 = perfect)
    """
    labels = np.concatenate([
        np.zeros(len(uncertainty_iid)),   # IID = 0
        np.ones(len(uncertainty_ood)),     # OOD = 1
    ])
    scores = np.concatenate([uncertainty_iid, uncertainty_ood])

    if len(np.unique(labels)) < 2:
        return float("nan")

    return _binary_roc_auc_score(labels, scores)


def compute_auroc_failure(
    uncertainty: np.ndarray,
    plume_ssim: np.ndarray,
    failure_threshold: float,
) -> float:
    """
    AUROC for detecting failure cases (low SSIM) using uncertainty as the score.

    Args:
        uncertainty: per-sample uncertainty scores
        plume_ssim: per-sample plume-region SSIM
        failure_threshold: SSIM below this = failure (e.g. 10th percentile of IID SSIM)

    Returns:
        AUROC for failure detection
    """
    valid = np.isfinite(plume_ssim) & np.isfinite(uncertainty)
    if valid.sum() < 10:
        return float("nan")

    labels = (plume_ssim[valid] < failure_threshold).astype(np.int32)
    scores = uncertainty[valid]

    if len(np.unique(labels)) < 2:
        return float("nan")

    return _binary_roc_auc_score(labels, scores)


# ============================================================================
# Reliability diagram data
# ============================================================================


def reliability_diagram_data(
    alpha_list: Sequence[float],
    coverages_iid: Sequence[float],
    coverages_ood: Sequence[float],
) -> dict:
    """
    Prepare data for reliability diagram plotting.

    Returns dict with:
        nominal: list of 1-alpha values
        iid: list of IID empirical coverage
        ood: list of OOD empirical coverage
    """
    nominal = [1.0 - alpha for alpha in alpha_list]
    return {
        "nominal": nominal,
        "iid": list(coverages_iid),
        "ood": list(coverages_ood),
    }


# ============================================================================
# Coverage-width tradeoff
# ============================================================================


def coverage_width_sweep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibration_scores: np.ndarray,
    alpha_list: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50),
) -> list:
    """
    Compute coverage and mean width at multiple alpha levels.

    Returns list of dicts: [{alpha, coverage, mean_width}, ...]
    """
    n_cal = len(calibration_scores)
    results = []

    for alpha in alpha_list:
        level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
        level = float(np.clip(level, 0.0, 1.0))
        try:
            q = float(np.quantile(calibration_scores, level, method="higher"))
        except TypeError:
            q = float(np.quantile(calibration_scores, level, interpolation="higher"))

        lower = y_pred - q
        upper = y_pred + q
        covered = (y_true >= lower) & (y_true <= upper)

        results.append({
            "alpha": float(alpha),
            "nominal_coverage": 1.0 - float(alpha),
            "empirical_coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(upper - lower)),
            "quantile": float(q),
        })

    return results


# ============================================================================
# Per-sample uncertainty aggregation
# ============================================================================


def sample_uncertainty_from_ensemble(
    predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample ensemble mean and mean variance.

    Args:
        predictions: (M, N_samples, T, H, W) — M ensemble members

    Returns:
        mean_pred: (N_samples, T, H, W)
        mean_var: (N_samples,) — mean pixelwise variance per sample
    """
    mean_pred = predictions.mean(axis=0)
    var_pred = predictions.var(axis=0)
    mean_var = var_pred.mean(axis=(-3, -2, -1))  # average over T, H, W
    return mean_pred, mean_var
