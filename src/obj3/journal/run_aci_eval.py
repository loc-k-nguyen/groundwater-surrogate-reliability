from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.obj3.journal.aci import (
    EPS_ALPHA,
    conformal_quantile,
    plume_residual_scores,
    reorder_indices_for_keys,
    sample_keys_from_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ALPHA = 0.10
DEFAULT_GAMMAS = (0.01, 0.02, 0.05, 0.10)
DEFAULT_CALIB_SCORES = REPO_ROOT / "experiments" / "obj3" / "journal" / "calibration" / "calibration_scores.npy"
DEFAULT_CALIB_PREDS = REPO_ROOT / "experiments" / "obj3" / "journal" / "calibration" / "calibration_preds.npz"
DEFAULT_IID_PREDS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "preds" / "iid_test_preds.npz"
DEFAULT_OOD_PREDS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "preds" / "ood_test_preds.npz"
DEFAULT_IID_METRICS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "eval" / "ensemble_metrics_iid_test.json"
DEFAULT_OOD_METRICS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "eval" / "ensemble_metrics_ood_test.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "experiments" / "obj3" / "journal" / "reports" / "analysis" / "aci_results.json"


def _load_calibration_scores(calibration_scores: Path, calibration_preds: Path) -> np.ndarray:
    if calibration_scores.exists():
        return np.load(calibration_scores, allow_pickle=False)
    if calibration_preds.exists():
        payload = np.load(calibration_preds, allow_pickle=False)
        return plume_residual_scores(payload["y_true"], payload["y_pred"], payload["c_phys"])
    raise FileNotFoundError("Neither calibration_scores.npy nor calibration_preds.npz is available.")


def _gamma_key(gamma: float) -> str:
    return f"{gamma:.2f}"


def _run_aci_sequence_streaming(
    calibration_scores: np.ndarray,
    payload: dict[str, np.ndarray],
    sample_keys: list[tuple[int, int]],
    order: np.ndarray,
    alpha: float,
    gamma: float,
) -> dict[str, object]:
    target_coverage = 1.0 - float(alpha)
    current_alpha = float(alpha)

    plume_coverages = []
    global_coverages = []
    plume_widths = []
    plume_maes = []
    misses = []

    for original_idx in order.tolist():
        y_true = payload["y_true"][original_idx]
        y_pred = payload["y_pred"][original_idx]
        c_phys = payload["c_phys"][original_idx]

        quantile = conformal_quantile(calibration_scores, current_alpha)
        lower = y_pred - quantile
        upper = y_pred + quantile
        covered = (y_true >= lower) & (y_true <= upper)
        plume_mask = c_phys > 1e-8
        if not np.any(plume_mask):
            plume_mask = np.ones_like(covered, dtype=bool)

        plume_pixel_coverage = float(np.mean(covered[plume_mask]))
        global_pixel_coverage = float(np.mean(covered))
        plume_mean_width = float(np.mean((upper - lower)[plume_mask]))
        sample_plume_mae = float(np.mean(np.abs(y_true - y_pred)[plume_mask]))
        miss_indicator = float(plume_pixel_coverage < target_coverage)

        plume_coverages.append(plume_pixel_coverage)
        global_coverages.append(global_pixel_coverage)
        plume_widths.append(plume_mean_width)
        plume_maes.append(sample_plume_mae)
        misses.append(miss_indicator)

        current_alpha = float(
            np.clip(current_alpha + gamma * (float(alpha) - miss_indicator), EPS_ALPHA, 1.0 - EPS_ALPHA)
        )

    return {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "target_coverage": target_coverage,
        "final_alpha": current_alpha,
        "final_quantile": conformal_quantile(calibration_scores, current_alpha),
        "aggregate": {
            "plume_pixel_coverage": float(np.mean(plume_coverages)),
            "global_pixel_coverage": float(np.mean(global_coverages)),
            "plume_mean_width": float(np.mean(plume_widths)),
            "sample_plume_mae": float(np.mean(plume_maes)),
            "sample_miss_rate": float(np.mean(misses)),
            "n_samples": int(len(order)),
        },
    }


def _select_best_gamma(results_by_gamma: dict[str, dict[str, object]], target_coverage: float) -> tuple[str, dict[str, object]]:
    best_key = min(
        results_by_gamma,
        key=lambda key: (
            abs(results_by_gamma[key]["ood"]["aggregate"]["plume_pixel_coverage"] - target_coverage),
            results_by_gamma[key]["ood"]["aggregate"]["plume_mean_width"],
            float(key),
        ),
    )
    return best_key, results_by_gamma[best_key]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Obj3 journal ACI gamma sweep.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--calibration_scores", type=str, default=str(DEFAULT_CALIB_SCORES))
    ap.add_argument("--calibration_preds", type=str, default=str(DEFAULT_CALIB_PREDS))
    ap.add_argument("--iid_preds", type=str, default=str(DEFAULT_IID_PREDS))
    ap.add_argument("--ood_preds", type=str, default=str(DEFAULT_OOD_PREDS))
    ap.add_argument("--iid_metrics", type=str, default=str(DEFAULT_IID_METRICS))
    ap.add_argument("--ood_metrics", type=str, default=str(DEFAULT_OOD_METRICS))
    ap.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    args = ap.parse_args()

    calibration_scores = _load_calibration_scores(Path(args.calibration_scores), Path(args.calibration_preds))
    # mmap_mode does not make compressed NPZ members memory mapped. Materialize
    # each member once so the online ACI loop does not repeatedly decompress the
    # same multi-gigabyte array for every sample and gamma.
    with np.load(args.iid_preds, allow_pickle=False) as iid_payload:
        iid = {key: iid_payload[key] for key in ("y_true", "y_pred", "c_phys")}
    with np.load(args.ood_preds, allow_pickle=False) as ood_payload:
        ood = {key: ood_payload[key] for key in ("y_true", "y_pred", "c_phys")}

    iid_keys = sample_keys_from_metrics(args.iid_metrics)
    ood_keys = sample_keys_from_metrics(args.ood_metrics)
    iid_order = reorder_indices_for_keys(iid_keys)
    ood_order = reorder_indices_for_keys(ood_keys)
    iid_sorted_keys = [iid_keys[idx] for idx in iid_order]
    ood_sorted_keys = [ood_keys[idx] for idx in ood_order]

    results_by_gamma: dict[str, dict[str, object]] = {}
    for gamma in DEFAULT_GAMMAS:
        gamma_key = _gamma_key(gamma)
        results_by_gamma[gamma_key] = {
            "iid": _run_aci_sequence_streaming(
                calibration_scores=calibration_scores,
                payload=iid,
                sample_keys=iid_sorted_keys,
                order=iid_order,
                alpha=args.alpha,
                gamma=gamma,
            ),
            "ood": _run_aci_sequence_streaming(
                calibration_scores=calibration_scores,
                payload=ood,
                sample_keys=ood_sorted_keys,
                order=ood_order,
                alpha=args.alpha,
                gamma=gamma,
            ),
        }

    best_gamma_key, best_full_result = _select_best_gamma(results_by_gamma, 1.0 - args.alpha)
    gamma_sweep = {
        gamma_key: {
            "iid": results_by_gamma[gamma_key]["iid"]["aggregate"],
            "ood": results_by_gamma[gamma_key]["ood"]["aggregate"],
        }
        for gamma_key in results_by_gamma
    }

    payload = {
        "alpha": float(args.alpha),
        "gamma_sweep": gamma_sweep,
        "best_gamma": best_gamma_key,
        "best_result": {
            "iid": best_full_result["iid"]["aggregate"],
            "ood": best_full_result["ood"]["aggregate"],
            "iid_final_alpha": best_full_result["iid"]["final_alpha"],
            "ood_final_alpha": best_full_result["ood"]["final_alpha"],
            "iid_final_quantile": best_full_result["iid"]["final_quantile"],
            "ood_final_quantile": best_full_result["ood"]["final_quantile"],
        },
        "notes": [
            "Calibration scores use the same plume-masked residual construction as the conference conformal setup.",
            "IID and OOD sequences are ordered by (param_id, real_id) before online updates.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
