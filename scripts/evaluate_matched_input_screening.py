"""Audit simple K-only screening on the completed Axis-D matched pool.

The script is CPU-only and evaluation-only. It does not change datasets, split
files, checkpoints, or certified manuscript results.
"""

from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path, configure_script_paths

if __name__ == "__main__":
    DEFAULT_ROOT = configure_script_paths()

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = DEFAULT_ROOT
DEFAULT_MAIN = asset_path("data", ROOT / (
    "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL/"
    "T25_TSTEP_OVERRIDE_FINAL_FLIPPED"
))
DEFAULT_SPLIT = ROOT / "splits/param_split_obj3_ood.json"
DEFAULT_MATCHED = ROOT / (
    "experiments/obj3/journal/upgrade_2026_07_01/tier1_multi_axis_ood/"
    "axisD_transport_shift/sim/dataset_axisD"
)
DEFAULT_AUDIT = ROOT / (
    "experiments/obj3/journal/upgrade_2026_07_01/tier1_multi_axis_ood/"
    "axisD_transport_shift/AXISD_MATCHED_DATASET_AUDIT.json"
)
DEFAULT_OUT = ROOT / (
    "experiments/obj3/journal/analysis/"
    "obj3_matched_pool_input_screening_20260818"
)

FEATURES = (
    "k_mean",
    "k_std",
    "k_var",
    "k_q05",
    "k_q25",
    "k_q50",
    "k_q75",
    "k_q95",
    "logk_mean",
    "logk_std",
    "logk_var",
    "logk_q05",
    "logk_q25",
    "logk_q50",
    "logk_q75",
    "logk_q95",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_k(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        k_raw = np.asarray(data["K"], dtype=np.float64)
    target_shape = (600, 400)
    if k_raw.size != int(np.prod(target_shape)):
        raise ValueError(f"Cannot reshape K from {path}: {k_raw.shape} to {target_shape}")
    return k_raw.reshape(target_shape)


def k_hash(k_field: np.ndarray) -> str:
    canonical = np.asarray(k_field, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def feature_values(k_field: np.ndarray) -> dict[str, float]:
    logk = np.log10(k_field + 1e-6)
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    kq = np.quantile(k_field, quantiles)
    lq = np.quantile(logk, quantiles)
    return {
        "k_mean": float(np.mean(k_field)),
        "k_std": float(np.std(k_field)),
        "k_var": float(np.var(k_field)),
        "k_q05": float(kq[0]),
        "k_q25": float(kq[1]),
        "k_q50": float(kq[2]),
        "k_q75": float(kq[3]),
        "k_q95": float(kq[4]),
        "logk_mean": float(np.mean(logk)),
        "logk_std": float(np.std(logk)),
        "logk_var": float(np.var(logk)),
        "logk_q05": float(lq[0]),
        "logk_q25": float(lq[1]),
        "logk_q50": float(lq[2]),
        "logk_q75": float(lq[3]),
        "logk_q95": float(lq[4]),
    }


def rank_auc(negative: Iterable[float], positive: Iterable[float]) -> float:
    neg = np.asarray(list(negative), dtype=np.float64)
    pos = np.asarray(list(positive), dtype=np.float64)
    if neg.size == 0 or pos.size == 0:
        raise ValueError("AUROC requires both IID and matched samples")
    comparisons = pos[:, None] - neg[None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def bootstrap_auc_ci(
    negative: np.ndarray,
    positive: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        neg = negative[rng.integers(0, negative.size, negative.size)]
        pos = positive[rng.integers(0, positive.size, positive.size)]
        draws[index] = rank_auc(neg, pos)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def summarize_feature(
    rows: list[dict[str, object]],
    feature: str,
    analysis_unit: str,
    bootstrap_repeats: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    iid = np.asarray(
        [float(row[feature]) for row in rows if row["group"] == "iid"],
        dtype=np.float64,
    )
    matched = np.asarray(
        [float(row[feature]) for row in rows if row["group"] == "matched"],
        dtype=np.float64,
    )
    auc = rank_auc(iid, matched)
    ci_low, ci_high = bootstrap_auc_ci(
        iid, matched, repeats=bootstrap_repeats, rng=rng
    )
    return {
        "analysis_unit": analysis_unit,
        "score": feature,
        "auroc_high_score_is_matched": auc,
        "auroc_ci95_low": ci_low,
        "auroc_ci95_high": ci_high,
        "best_direction_auc": max(auc, 1.0 - auc),
        "best_direction": "higher_is_matched" if auc >= 0.5 else "lower_is_matched",
        "iid_mean": float(np.mean(iid)),
        "iid_std": float(np.std(iid)),
        "matched_mean": float(np.mean(matched)),
        "matched_std": float(np.std(matched)),
        "n_iid": int(iid.size),
        "n_matched": int(matched.size),
    }


def collect_rows(iid_files: list[Path], matched_files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, files in (("iid", iid_files), ("matched", matched_files)):
        for path in files:
            k_field = canonical_k(path)
            row: dict[str, object] = {
                "group": group,
                "param_id": path.parent.name,
                "real_id": path.stem,
                "path": str(path.relative_to(ROOT)),
                "k_sha256": k_hash(k_field),
            }
            row.update(feature_values(k_field))
            rows.append(row)
    return rows


def deduplicate_fields(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["group"]), str(row["k_sha256"]))
        unique.setdefault(key, row)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-data-root", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--matched-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    with args.matched_audit.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    expected_status = "EXPANDED_ALL_READY pass=12 fail=0 params=6/6"
    if audit.get("status") != expected_status:
        raise RuntimeError(f"Matched audit is not ready: {audit.get('status')}")

    with args.split_json.open(encoding="utf-8") as handle:
        split = json.load(handle)
    iid_files = [
        path
        for param_id in split["iid_test"]
        for path in sorted((args.main_data_root / f"param_{param_id:03d}").glob("real_*.npz"))
    ]
    matched_files = [
        path
        for param_id in audit["params"]
        for path in sorted((args.matched_root / f"param_{param_id}").glob("real_*.npz"))
    ]
    if len(iid_files) != 110:
        raise RuntimeError(f"Expected 110 IID files, found {len(iid_files)}")
    if len(matched_files) != 12:
        raise RuntimeError(f"Expected 12 matched files, found {len(matched_files)}")

    rows = collect_rows(iid_files, matched_files)
    unique_rows = deduplicate_fields(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    sample_columns = (
        "group",
        "param_id",
        "real_id",
        "path",
        "k_sha256",
        *FEATURES,
    )
    with (args.output_dir / "input_features_per_case.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_columns)
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(20260818)
    summaries: list[dict[str, object]] = []
    for analysis_unit, selected in (("case_weighted", rows), ("unique_k_field", unique_rows)):
        for feature in FEATURES:
            summaries.append(
                summarize_feature(
                    selected,
                    feature,
                    analysis_unit,
                    args.bootstrap_repeats,
                    rng,
                )
            )

    summary_columns = tuple(summaries[0].keys())
    with (args.output_dir / "input_screening_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summaries)

    prespecified = next(
        row
        for row in summaries
        if row["analysis_unit"] == "unique_k_field" and row["score"] == "logk_std"
    )
    exploratory_best = max(
        (row for row in summaries if row["analysis_unit"] == "unique_k_field"),
        key=lambda row: float(row["best_direction_auc"]),
    )
    iid_logk_std = [float(row["logk_std"]) for row in unique_rows if row["group"] == "iid"]
    matched_logk_std = [
        float(row["logk_std"]) for row in unique_rows if row["group"] == "matched"
    ]
    within_iid_range = np.mean(
        (np.asarray(matched_logk_std) >= min(iid_logk_std))
        & (np.asarray(matched_logk_std) <= max(iid_logk_std))
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OBJ3_MATCHED_INPUT_SCREENING_COMPLETE",
        "protocol": {
            "positive_class": "Axis-D matched transport-stress case",
            "negative_class": "certified IID test case",
            "prespecified_primary_score": "field standard deviation of log10(K)",
            "primary_analysis_unit": "unique K field",
            "secondary_analysis_unit": "simulation case, including repeated K fields",
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": 20260818,
            "caution": (
                "This audit tests simple scalar K statistics on a finite constructed pool. "
                "It does not establish generic non-identifiability from K."
            ),
        },
        "counts": {
            "iid_cases": sum(row["group"] == "iid" for row in rows),
            "matched_cases": sum(row["group"] == "matched" for row in rows),
            "iid_unique_k_fields": sum(row["group"] == "iid" for row in unique_rows),
            "matched_unique_k_fields": sum(
                row["group"] == "matched" for row in unique_rows
            ),
        },
        "prespecified_logk_std": prespecified,
        "matched_unique_fields_within_iid_logk_std_range_fraction": float(within_iid_range),
        "exploratory_best_simple_feature": exploratory_best,
        "interpretation": (
            "Use the prespecified logk_std result for Figure 1 wording. Treat the maximum "
            "over all simple features as exploratory because feature selection on this small "
            "pool is optimistic."
        ),
        "sources": {
            "split_json": {
                "path": str(args.split_json.relative_to(ROOT)),
                "sha256": file_sha256(args.split_json),
            },
            "matched_audit": {
                "path": str(args.matched_audit.relative_to(ROOT)),
                "sha256": file_sha256(args.matched_audit),
                "certified_status": audit["status"],
            },
            "main_data_root": str(args.main_data_root.relative_to(ROOT)),
            "matched_data_root": str(args.matched_root.relative_to(ROOT)),
        },
    }
    (args.output_dir / "input_screening_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
