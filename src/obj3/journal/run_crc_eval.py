from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.obj3.journal.crc import evaluate_crc_solution, sample_plume_mae, solve_crc_lambda


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RISK_LEVELS = (0.05, 0.10, 0.20)
DEFAULT_CALIB_MAE = REPO_ROOT / "experiments" / "obj3" / "journal" / "calibration" / "calibration_plume_mae.npy"
DEFAULT_CALIB_PREDS = REPO_ROOT / "experiments" / "obj3" / "journal" / "calibration" / "calibration_preds.npz"
DEFAULT_IID_PREDS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "preds" / "iid_test_preds.npz"
DEFAULT_OOD_PREDS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "preds" / "ood_test_preds.npz"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "experiments" / "obj3" / "journal" / "reports" / "analysis" / "crc_results.json"


def _load_calibration_mae(calibration_mae: Path, calibration_preds: Path) -> np.ndarray:
    if calibration_mae.exists():
        return np.load(calibration_mae, allow_pickle=False)
    if calibration_preds.exists():
        payload = np.load(calibration_preds, allow_pickle=False)
        return sample_plume_mae(payload["y_true"], payload["y_pred"], payload["c_phys"])
    raise FileNotFoundError("Neither calibration_plume_mae.npy nor calibration_preds.npz is available.")


def _load_test_mae(pred_path: Path) -> np.ndarray:
    payload = np.load(pred_path, allow_pickle=False)
    return sample_plume_mae(payload["y_true"], payload["y_pred"], payload["c_phys"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Obj3 journal CRC evaluation.")
    ap.add_argument("--calibration_plume_mae", type=str, default=str(DEFAULT_CALIB_MAE))
    ap.add_argument("--calibration_preds", type=str, default=str(DEFAULT_CALIB_PREDS))
    ap.add_argument("--iid_preds", type=str, default=str(DEFAULT_IID_PREDS))
    ap.add_argument("--ood_preds", type=str, default=str(DEFAULT_OOD_PREDS))
    ap.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    args = ap.parse_args()

    calibration_risks = _load_calibration_mae(Path(args.calibration_plume_mae), Path(args.calibration_preds))
    iid_risks = _load_test_mae(Path(args.iid_preds))
    ood_risks = _load_test_mae(Path(args.ood_preds))

    results: dict[str, dict[str, float | bool]] = {}
    for risk_level in DEFAULT_RISK_LEVELS:
        solution = solve_crc_lambda(calibration_risks, risk_budget=risk_level, delta=0.10)
        iid_eval = evaluate_crc_solution(iid_risks, solution)
        ood_eval = evaluate_crc_solution(ood_risks, solution)
        key = f"{risk_level:.2f}"
        results[key] = {
            "delta_calibrated": float(solution.lambda_value),
            "iid_empirical_risk": float(iid_eval["mean_plume_mae"]),
            "ood_empirical_risk": float(ood_eval["mean_plume_mae"]),
            "excess_risk_iid": float(iid_eval["mean_plume_mae"] - solution.lambda_value),
            "excess_risk_ood": float(ood_eval["mean_plume_mae"] - solution.lambda_value),
            "iid_fraction_within_delta": float(iid_eval["fraction_within_lambda"]),
            "ood_fraction_within_delta": float(ood_eval["fraction_within_lambda"]),
            "attained": bool(solution.attained),
        }

    payload = {
        "risk_levels": results,
        "notes": [
            "delta_calibrated is the CRC-calibrated plume-MAE tolerance lambda.",
            "Excess risk is reported as empirical mean plume MAE minus delta_calibrated.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
