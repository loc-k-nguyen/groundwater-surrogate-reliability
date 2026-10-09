from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PLUME_THRESH = 1e-8


def sample_plume_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    c_phys: np.ndarray,
    plume_thresh: float = PLUME_THRESH,
) -> np.ndarray:
    """Compute one plume-region MAE score per sample."""
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    c_phys_arr = np.asarray(c_phys, dtype=np.float32)
    if y_true_arr.shape != y_pred_arr.shape or y_true_arr.shape != c_phys_arr.shape:
        raise ValueError("y_true, y_pred, and c_phys must share the same shape.")

    risks = []
    for idx in range(y_true_arr.shape[0]):
        plume_mask = c_phys_arr[idx] > float(plume_thresh)
        if not np.any(plume_mask):
            plume_mask = np.ones_like(c_phys_arr[idx], dtype=bool)
        risks.append(float(np.mean(np.abs(y_true_arr[idx] - y_pred_arr[idx])[plume_mask])))
    return np.asarray(risks, dtype=np.float32)


def hoeffding_radius(n: int, delta: float) -> float:
    """Historical radius formula; validity requires assumptions not checked here."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    return float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


@dataclass(frozen=True)
class CRCSolution:
    risk_budget: float
    lambda_value: float
    empirical_excess_risk: float
    ucb_excess_risk: float
    attained: bool


def solve_crc_lambda(
    calibration_risks: np.ndarray,
    risk_budget: float,
    delta: float = 0.10,
) -> CRCSolution:
    """Select a tolerance by the historical descriptive calibration criterion.

    The criterion adds a Hoeffding-shaped radius to mean excess plume MAE.
    Plume MAE is unbounded and cases share geological draws, so this is not
    a valid upper confidence bound or a prospective risk-control certificate.
    ``attained`` records calibration-criterion attainment only, not test risk.
    Historical field names are retained for numerical artifact compatibility.
    """
    cal_risks = np.asarray(calibration_risks, dtype=np.float32).reshape(-1)
    if cal_risks.size == 0:
        raise ValueError("calibration_risks must be non-empty.")
    if risk_budget < 0.0:
        raise ValueError(f"risk_budget must be non-negative, got {risk_budget}.")

    radius = hoeffding_radius(cal_risks.size, delta)
    candidates = np.unique(np.sort(cal_risks))

    for lam in candidates:
        empirical_excess = float(np.mean(np.maximum(cal_risks - lam, 0.0)))
        ucb_excess = empirical_excess + radius
        if ucb_excess <= risk_budget:
            return CRCSolution(
                risk_budget=float(risk_budget),
                lambda_value=float(lam),
                empirical_excess_risk=empirical_excess,
                ucb_excess_risk=float(ucb_excess),
                attained=True,
            )

    lam = float(candidates[-1])
    empirical_excess = float(np.mean(np.maximum(cal_risks - lam, 0.0)))
    return CRCSolution(
        risk_budget=float(risk_budget),
        lambda_value=lam,
        empirical_excess_risk=empirical_excess,
        ucb_excess_risk=float(empirical_excess + radius),
        attained=False,
    )


def evaluate_crc_solution(test_risks: np.ndarray, solution: CRCSolution) -> dict[str, float]:
    test_arr = np.asarray(test_risks, dtype=np.float32).reshape(-1)
    excess = np.maximum(test_arr - solution.lambda_value, 0.0)
    return {
        "mean_plume_mae": float(np.mean(test_arr)),
        "std_plume_mae": float(np.std(test_arr)),
        "excess_risk_mean": float(np.mean(excess)),
        "fraction_within_lambda": float(np.mean(test_arr <= solution.lambda_value)),
        "n_samples": int(test_arr.size),
    }
