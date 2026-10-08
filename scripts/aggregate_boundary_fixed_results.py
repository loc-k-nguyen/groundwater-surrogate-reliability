"""Aggregate isolated Obj3 boundary-fixed outputs without touching old results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import mannwhitneyu
from scipy.special import ndtr
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.data_obj3_ood import collect_npz_files, load_obj3_split
from src.shared.eval.plume_ssim import compute_plume_ssim


DATA_RANGE = 14.0
PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
EPS_C = 1e-12
FAILURE_QUANTILE = 0.10
CRPS_SIGMA_MIN = 1e-6
LOCALIZATION_QUANTILE = 0.75


class NpySampleStream:
    """Sequentially read one NPY member from a compressed NPZ archive."""

    def __init__(self, npz_path: Path, member: str) -> None:
        self._archive = zipfile.ZipFile(npz_path)
        self._handle = self._archive.open(f"{member}.npy")
        version = np.lib.format.read_magic(self._handle)
        self.shape, fortran, self.dtype = np.lib.format._read_array_header(self._handle, version)
        if fortran:
            raise ValueError(f"Fortran-order member is unsupported: {npz_path}/{member}")
        self.n = int(self.shape[0])
        self.sample_shape = tuple(int(value) for value in self.shape[1:])
        self._sample_bytes = int(np.prod(self.sample_shape)) * self.dtype.itemsize

    def read(self) -> np.ndarray:
        buffer = self._handle.read(self._sample_bytes)
        if len(buffer) != self._sample_bytes:
            raise EOFError("Short read from compressed NPZ member")
        return np.frombuffer(buffer, dtype=self.dtype).reshape(self.sample_shape)

    def close(self) -> None:
        self._handle.close()
        self._archive.close()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def auc_with_ties(positive: Sequence[float], negative: Sequence[float]) -> float:
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    wins = np.count_nonzero(pos[:, None] > neg[None, :])
    ties = np.count_nonzero(pos[:, None] == neg[None, :])
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def binary_roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct, labels.size - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    thresholds = sorted_scores[threshold_indices]
    true_positives = np.r_[0, true_positives]
    false_positives = np.r_[0, false_positives]
    # Keep the leading no-positive-prediction point while preserving strict JSON.
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), thresholds]
    return false_positives / negatives, true_positives / positives, thresholds


def ordered_sample_keys(split_name: str, main_data_root: Path, extra_data_root: Path) -> list[tuple[str, str]]:
    split = load_obj3_split()
    files = collect_npz_files(split[split_name], main_data_root, extra_data_root)
    return [(path.parent.name, path.stem) for path in files]


def gaussian_crps_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    plume_mask: np.ndarray,
) -> tuple[int, float, float]:
    """Return plume-pixel count, CRPS sum, and squared-CRPS sum."""
    if not np.any(plume_mask):
        return 0, 0.0, 0.0
    truth = np.asarray(y_true[plume_mask], dtype=np.float64)
    pred = np.asarray(y_pred[plume_mask], dtype=np.float64)
    raw_std = np.asarray(y_std[plume_mask], dtype=np.float64)
    zero_std = raw_std <= 0.0
    sigma = np.maximum(raw_std, CRPS_SIGMA_MIN)
    z = (truth - pred) / sigma
    crps = sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
        - 1.0 / math.sqrt(math.pi)
    )
    if np.any(zero_std):
        crps[zero_std] = np.abs(truth[zero_std] - pred[zero_std])
    return (
        int(crps.size),
        float(crps.sum(dtype=np.float64)),
        float(np.square(crps, dtype=np.float64).sum(dtype=np.float64)),
    )


def uncertainty_error_localization_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    c_phys: np.ndarray,
) -> float:
    """Mean timestep IoU of top-quartile uncertainty and absolute error."""
    values: list[float] = []
    for timestep in range(y_true.shape[0]):
        plume = c_phys[timestep] > PLUME_THRESH
        if int(plume.sum()) < PLUME_MIN_PIXELS:
            continue
        error = np.abs(y_true[timestep] - y_pred[timestep])
        error_threshold = float(np.quantile(error[plume], LOCALIZATION_QUANTILE))
        uncertainty_threshold = float(
            np.quantile(y_std[timestep][plume], LOCALIZATION_QUANTILE)
        )
        high_error = plume & (error >= error_threshold)
        high_uncertainty = plume & (y_std[timestep] >= uncertainty_threshold)
        union = int(np.count_nonzero(high_error | high_uncertainty))
        if union:
            values.append(
                int(np.count_nonzero(high_error & high_uncertainty)) / union
            )
    return float(np.mean(values)) if values else float("nan")


def evaluate_ensemble_cache(
    cache_path: Path,
    split_name: str,
    sample_keys: list[tuple[str, str]],
) -> dict[str, object]:
    streams = {
        key: NpySampleStream(cache_path, key)
        for key in ("y_true", "y_pred", "y_pred_std", "c_phys")
    }
    counts = {stream.n for stream in streams.values()}
    if counts != {len(sample_keys)}:
        raise ValueError(f"{split_name}: cache/sample-key count mismatch: {counts} vs {len(sample_keys)}")

    rows: list[dict[str, object]] = []
    crps_count = 0
    crps_sum = 0.0
    crps_sumsq = 0.0
    try:
        for sample_index, (param_id, real_id) in enumerate(sample_keys):
            y_true = streams["y_true"].read()
            y_pred = streams["y_pred"].read()
            y_std = streams["y_pred_std"].read()
            c_phys = streams["c_phys"].read()

            plume_ssims: list[float] = []
            global_ssims: list[float] = []
            mass_errors: list[float] = []
            for timestep in range(25):
                global_ssims.append(float(ssim(y_true[timestep], y_pred[timestep], data_range=DATA_RANGE)))
                gt_phys = np.clip(c_phys[timestep], 0.0, None)
                plume_ssim, _ = compute_plume_ssim(
                    gt_phys,
                    y_pred[timestep],
                    EPS_C,
                    PLUME_THRESH,
                    PLUME_PAD,
                    PLUME_MIN_PIXELS,
                    DATA_RANGE,
                )
                if np.isfinite(plume_ssim):
                    plume_ssims.append(float(plume_ssim))
                pred_phys = np.clip(10.0 ** y_pred[timestep] - EPS_C, 0.0, None)
                gt_mass = float(gt_phys.sum())
                mass_errors.append(abs(float(pred_phys.sum()) - gt_mass) / max(gt_mass, EPS_C))

            plume_mask = c_phys > PLUME_THRESH
            plume_std = float(np.mean(y_std[plume_mask])) if np.any(plume_mask) else float("nan")
            count, value_sum, value_sumsq = gaussian_crps_summary(
                y_true, y_pred, y_std, plume_mask
            )
            crps_count += count
            crps_sum += value_sum
            crps_sumsq += value_sumsq
            localization_iou = uncertainty_error_localization_iou(
                y_true, y_pred, y_std, c_phys
            )
            rows.append(
                {
                    "param_id": param_id,
                    "real_id": real_id,
                    "mean_plume_ssim": float(np.mean(plume_ssims)),
                    "mean_global_ssim": float(np.mean(global_ssims)),
                    "mean_mass_error": float(np.mean(mass_errors)),
                    "mean_ensemble_var": float(np.mean(np.square(y_std, dtype=np.float64))),
                    "plume_ensemble_std": plume_std,
                    "uncertainty_error_iou": localization_iou,
                }
            )
            if (sample_index + 1) % 20 == 0:
                print(f"{split_name}: evaluated {sample_index + 1}/{len(sample_keys)}", flush=True)
    finally:
        for stream in streams.values():
            stream.close()

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    if crps_count <= 0:
        raise ValueError(f"{split_name}: no plume pixels available for CRPS")
    crps_mean = crps_sum / crps_count
    crps_std = math.sqrt(max(crps_sumsq / crps_count - crps_mean * crps_mean, 0.0))

    aggregate = {
        "plume_ssim_mean": float(values("mean_plume_ssim").mean()),
        "plume_ssim_std": float(values("mean_plume_ssim").std()),
        "global_ssim_mean": float(values("mean_global_ssim").mean()),
        "global_ssim_std": float(values("mean_global_ssim").std()),
        "mass_error_mean": float(values("mean_mass_error").mean()),
        "mass_error_std": float(values("mean_mass_error").std()),
        "ensemble_var_mean": float(values("mean_ensemble_var").mean()),
        "ensemble_var_std": float(values("mean_ensemble_var").std()),
        "plume_ensemble_std_mean": float(values("plume_ensemble_std").mean()),
        "gaussian_crps_mean": crps_mean,
        "gaussian_crps_std": crps_std,
        "gaussian_crps_n_plume_pixels": crps_count,
        "uncertainty_error_iou_mean": float(values("uncertainty_error_iou").mean()),
        "uncertainty_error_iou_std": float(values("uncertainty_error_iou").std()),
        "uncertainty_error_iou_top_quantile": LOCALIZATION_QUANTILE,
        "n_samples": len(rows),
        "n_members": 5,
    }
    return {"split": split_name, "aggregate": aggregate, "per_sample": rows}


def aggregate_seed_family(run_root: Path, family: str) -> dict[str, object]:
    prefix = "metrics" if family == "deterministic" else "mcdrop_metrics"
    per_seed: list[dict[str, object]] = []
    for seed in range(10):
        for split in ("iid_test", "ood_test"):
            path = run_root / family / f"seed{seed}" / f"{prefix}_{split}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            aggregate = payload["aggregate"]
            if int(aggregate["n_samples"]) != (110 if split == "iid_test" else 270):
                raise ValueError(f"Unexpected sample count: {path}")
            per_seed.append(
                {
                    "seed": seed,
                    "split": split,
                    **{key: value for key, value in aggregate.items() if isinstance(value, (int, float))},
                }
            )

    summaries: list[dict[str, object]] = []
    metric_names = ["plume_ssim_mean", "global_ssim_mean", "mass_error_mean"]
    if family == "mcdrop":
        metric_names.append("mc_var_mean")
    for seed_set, seeds in (("seeds_0_4", range(5)), ("seeds_0_9", range(10))):
        for split in ("iid_test", "ood_test"):
            selected = [row for row in per_seed if row["seed"] in seeds and row["split"] == split]
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in selected], dtype=float)
                summaries.append(
                    {
                        "seed_set": seed_set,
                        "split": split,
                        "metric": metric,
                        "n_seeds": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
    return {"family": family, "per_seed": per_seed, "summaries": summaries}


def uncertainty_diagnostics(iid: dict[str, object], ood: dict[str, object]) -> dict[str, object]:
    iid_rows = iid["per_sample"]
    ood_rows = ood["per_sample"]
    iid_scores = np.asarray([row["mean_ensemble_var"] for row in iid_rows], dtype=float)
    ood_scores = np.asarray([row["mean_ensemble_var"] for row in ood_rows], dtype=float)
    labels = np.concatenate([np.zeros(len(iid_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([iid_scores, ood_scores])
    ood_auc = auc_with_ties(ood_scores, iid_scores)
    fpr, tpr, thresholds = binary_roc_curve(labels, scores)
    mwu = mannwhitneyu(ood_scores, iid_scores, alternative="greater", method="auto")

    all_rows = list(iid_rows) + list(ood_rows)
    iid_plume_ssim = np.asarray(
        [float(row["mean_plume_ssim"]) for row in iid_rows], dtype=float
    )
    failure_threshold = float(np.quantile(iid_plume_ssim, FAILURE_QUANTILE))
    failure_labels = np.asarray(
        [float(row["mean_plume_ssim"]) < failure_threshold for row in all_rows], dtype=int
    )
    if np.unique(failure_labels).size == 2:
        failure_auc: float | None = auc_with_ties(
            scores[failure_labels == 1], scores[failure_labels == 0]
        )
        fail_fpr, fail_tpr, fail_thresholds = binary_roc_curve(failure_labels, scores)
        failure_status = "defined"
    else:
        failure_auc = None
        fail_fpr = fail_tpr = fail_thresholds = np.asarray([], dtype=float)
        failure_status = "undefined_single_class"
    return {
        "score": "mean ensemble variance over the full field",
        "ood_detection": {
            "auc": ood_auc,
            "auc_rank_recomputed": auc_with_ties(ood_scores, iid_scores),
            "n_iid": len(iid_scores),
            "n_ood": len(ood_scores),
            "mann_whitney_u": float(mwu.statistic),
            "mann_whitney_p_one_sided": float(mwu.pvalue),
            "rank_biserial": 2.0 * ood_auc - 1.0,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "failure_detection": {
            "failure_quantile_iid": FAILURE_QUANTILE,
            "failure_threshold_plume_ssim": failure_threshold,
            "n_iid_failures": int(failure_labels[: len(iid_rows)].sum()),
            "n_ood_failures": int(failure_labels[len(iid_rows) :].sum()),
            "n_failures": int(failure_labels.sum()),
            "n_nonfailures": int((1 - failure_labels).sum()),
            "status": failure_status,
            "auc": failure_auc,
            "fpr": fail_fpr.tolist(),
            "tpr": fail_tpr.tolist(),
            "thresholds": fail_thresholds.tolist(),
        },
        "variance_distributions": {
            "iid": iid_scores.tolist(),
            "ood": ood_scores.tolist(),
        },
    }


def mcdrop_diagnostics(
    run_root: Path,
    seed_count: int,
    failure_threshold: float,
) -> dict[str, object]:
    by_split: dict[str, dict[tuple[str, str], list[dict[str, float]]]] = {
        "iid_test": {},
        "ood_test": {},
    }
    for seed in range(seed_count):
        for split in by_split:
            path = run_root / "mcdrop" / f"seed{seed}" / f"mcdrop_metrics_{split}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["per_sample"]:
                key = (row["param_id"], row["real_id"])
                by_split[split].setdefault(key, []).append(row)

    averaged: dict[str, list[dict[str, float]]] = {}
    for split, keyed in by_split.items():
        averaged[split] = [
            {
                "mc_var": float(np.mean([float(row["mean_mc_var"]) for row in rows])),
                "plume_ssim": float(np.mean([float(row["mean_plume_ssim"]) for row in rows])),
            }
            for rows in keyed.values()
        ]
    iid = averaged["iid_test"]
    ood = averaged["ood_test"]
    scores = np.asarray([row["mc_var"] for row in iid + ood], dtype=float)
    labels = np.concatenate([np.zeros(len(iid)), np.ones(len(ood))])
    failure = np.asarray(
        [row["plume_ssim"] < failure_threshold for row in iid + ood], dtype=int
    )
    ood_auc = auc_with_ties(
        scores[labels == 1],
        scores[labels == 0],
    )
    failure_auc = (
        auc_with_ties(scores[failure == 1], scores[failure == 0])
        if np.unique(failure).size == 2
        else None
    )
    return {
        "n_seeds": seed_count,
        "n_iid": len(iid),
        "n_ood": len(ood),
        "ood_auc": ood_auc,
        "failure_auc": failure_auc,
        "failure_status": "defined" if failure_auc is not None else "undefined_single_class",
        "failure_threshold_source": "corrected ensemble IID 10th percentile",
        "failure_threshold_plume_ssim": failure_threshold,
        "iid_mc_var_mean": float(np.mean([row["mc_var"] for row in iid])),
        "ood_mc_var_mean": float(np.mean([row["mc_var"] for row in ood])),
    }


def write_summary_csv(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--main-data-root",
        type=Path,
        default=Path("Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL"),
    )
    parser.add_argument(
        "--extra-data-root",
        type=Path,
        default=Path("simulation/datasets/obj3_calibration"),
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite postprocess output: {args.output_dir}")
    cache_cert = args.run_root / "certification/prediction_cache_certification.json"
    certification = json.loads(cache_cert.read_text(encoding="utf-8"))
    if certification.get("status") != "BOUNDARY_FIXED_CACHE_CERTIFIED":
        raise ValueError("Prediction cache certification gate did not pass")

    ensemble_root = args.output_dir / "ensemble/eval"
    ensemble_results: dict[str, dict[str, object]] = {}
    for split in ("iid_test", "ood_test"):
        sample_keys = ordered_sample_keys(split, args.main_data_root, args.extra_data_root)
        result = evaluate_ensemble_cache(
            args.run_root / f"predictions/{split}_preds.npz",
            split,
            sample_keys,
        )
        ensemble_results[split] = result
        json_dump(ensemble_root / f"ensemble_metrics_{split}.json", result)

    deterministic = aggregate_seed_family(args.run_root, "deterministic")
    mcdrop = aggregate_seed_family(args.run_root, "mcdrop")
    diagnostics = uncertainty_diagnostics(
        ensemble_results["iid_test"], ensemble_results["ood_test"]
    )
    failure_threshold = float(
        diagnostics["failure_detection"]["failure_threshold_plume_ssim"]
    )
    mcdrop_five = mcdrop_diagnostics(args.run_root, 5, failure_threshold)
    mcdrop_ten = mcdrop_diagnostics(args.run_root, 10, failure_threshold)

    json_dump(args.output_dir / "summaries/deterministic.json", deterministic)
    json_dump(args.output_dir / "summaries/mcdrop.json", mcdrop)
    json_dump(args.output_dir / "summaries/ensemble_uncertainty_diagnostics.json", diagnostics)
    json_dump(args.output_dir / "summaries/mcdrop_5seed_diagnostics.json", mcdrop_five)
    json_dump(args.output_dir / "summaries/mcdrop_10seed_diagnostics.json", mcdrop_ten)
    write_summary_csv(args.output_dir / "summaries/deterministic_summary.csv", deterministic["summaries"])
    write_summary_csv(args.output_dir / "summaries/mcdrop_summary.csv", mcdrop["summaries"])

    json_dump(
        args.output_dir / "aggregate_certification.json",
        {
            "status": "BOUNDARY_FIXED_AGGREGATES_CERTIFIED",
            "cache_certification": str(cache_cert),
            "ensemble_iid_n": ensemble_results["iid_test"]["aggregate"]["n_samples"],
            "ensemble_ood_n": ensemble_results["ood_test"]["aggregate"]["n_samples"],
            "deterministic_seed_count": 10,
            "mcdrop_seed_count": 10,
        },
    )
    print("BOUNDARY_FIXED_AGGREGATES_CERTIFIED")


if __name__ == "__main__":
    main()
