"""Build the Obj3 EMS deployment-taxonomy table from available UQ eval outputs.

Merges plume-region predictive-uncertainty OOD-detection AUROC across monitors:
input-K screening, deep ensemble, heteroscedastic head, and FNO ensemble. The
table is CPU-only and can be rebuilt whenever new evaluation CSVs arrive.
"""
from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path

import csv
import json
from itertools import product
from pathlib import Path

REPO = DEFAULT_ROOT
RUNS = REPO / "experiments/obj3/journal/runs"
OUT = REPO / "experiments/obj3/journal/paper"

SOURCES = [
    (
        "deep_ensemble",
        [
            RUNS / "obj3_jnl_det_ensemble_uq_eval_axisDexpanded_20260706/det_ensemble_expanded_per_sample.csv",
            RUNS / "obj3_jnl_det_ensemble_uq_eval/det_ensemble_uq_per_sample.csv",
        ],
        "plume_total_std_log",
    ),
    (
        "hetero_head",
        [
            RUNS / "obj3_jnl_ms_tmo_hetero_eval_axisDexpanded_20260706/hetero_expanded_per_sample.csv",
            RUNS / "obj3_jnl_ms_tmo_hetero_eval_20260704/hetero_uq_per_sample.csv",
        ],
        "plume_total_std_log",
    ),
    (
        "fno_ensemble",
        [
            RUNS / "obj3_jnl_fno_uq_eval_axisDexpanded_20260706/fno_expanded_per_sample.csv",
            RUNS / "obj3_jnl_fno_uq_eval/fno_uq_per_sample.csv",
        ],
        "plume_total_std_log",
    ),
]
AXISD_PREFIXES = ("axisd",)


def rank_auc(negative: list[float | None], positive: list[float | None]) -> float | None:
    neg = [x for x in negative if x is not None]
    pos = [x for x in positive if x is not None]
    if not neg or not pos:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p, n in product(pos, neg)
    )
    return round(wins / (len(pos) * len(neg)), 4)


def load_rows(csv_path: Path, uncertainty_col: str) -> dict[str, list[float | None]] | None:
    if not csv_path.exists():
        return None
    groups: dict[str, list[float | None]] = {
        "iid": [],
        "ood": [],
        "axisD": [],
        "iid_ssim": [],
        "ood_ssim": [],
        "axisD_ssim": [],
    }
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = row.get("target", "")
            uncertainty = parse_float(row.get(uncertainty_col))
            plume_ssim = parse_float(row.get("mean_plume_ssim"))
            if target == "iid_test":
                groups["iid"].append(uncertainty)
                groups["iid_ssim"].append(plume_ssim)
            elif target == "ood_test":
                groups["ood"].append(uncertainty)
                groups["ood_ssim"].append(plume_ssim)
            elif target.lower().startswith(AXISD_PREFIXES):
                groups["axisD"].append(uncertainty)
                groups["axisD_ssim"].append(plume_ssim)
    return groups


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def parse_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def mean(values: list[float | None]) -> float | None:
    vals = [x for x in values if x is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def cell(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[str, object]] = {}

    for label, paths, col in SOURCES:
        path = first_existing(paths)
        if path is None:
            table[label] = {"status": "PENDING", "path": str(paths[0])}
            continue
        groups = load_rows(path, col)
        if groups is None:
            table[label] = {"status": "PENDING", "path": str(path)}
            continue
        table[label] = {
            "status": "READY",
            "path": str(path),
            "n": {
                "iid": len(groups["iid"]),
                "ood": len(groups["ood"]),
                "axisD": len(groups["axisD"]),
            },
            "plume_ssim": {
                "iid": mean(groups["iid_ssim"]),
                "ood": mean(groups["ood_ssim"]),
                "axisD": mean(groups["axisD_ssim"]),
            },
            "plume_unc_mean": {
                "iid": mean(groups["iid"]),
                "ood": mean(groups["ood"]),
                "axisD": mean(groups["axisD"]),
            },
            "auroc_sigma2Y_vs_iid": rank_auc(groups["iid"], groups["ood"]),
            "auroc_axisD_vs_iid": rank_auc(groups["iid"], groups["axisD"]),
        }

    table["input_K_screening"] = {
        "status": "READY(prior)",
        "auroc_sigma2Y_vs_iid": 1.0,
        "auroc_axisD_vs_iid": None,
        "note": "alphaL does not change K, so input screening is blind to transport shift",
    }

    (OUT / "OBJ3_EMS_AXISD_TAXONOMY_TABLE.json").write_text(
        json.dumps(table, indent=2),
        encoding="utf-8",
    )

    order = ["input_K_screening", "deep_ensemble", "hetero_head", "fno_ensemble"]
    lines = [
        "# Obj3 EMS Deployment Taxonomy Table (auto-built)",
        "Plume-region predictive-uncertainty OOD-detection AUROC (uncertainty > IID). "
        "Re-run `build_axisD_taxonomy_table.py` after new evaluation CSVs arrive.",
        "",
        "| Monitor | sigma2Y variance shift | Axis D alphaL transport shift | status |",
        "|---|--:|--:|---|",
    ]
    for key in order:
        row = table.get(key, {})
        lines.append(
            f"| {key} | {cell(row.get('auroc_sigma2Y_vs_iid'))} | "
            f"{cell(row.get('auroc_axisD_vs_iid'))} | {row.get('status', '?')} |"
        )

    lines += [
        "",
        "**Reading:** input-K screening catches the variance shift but is blind to the "
        "transport shift. The MS-TMO deep ensemble and heteroscedastic head show high "
        "Axis D uncertainty on the current small pool, whereas the FNO ensemble does "
        "not. This makes architecture dependence explicit rather than hidden. Axis D "
        "AUROC remains preliminary until the host pool expands.",
        "",
        "## Accuracy + uncertainty magnitude per method",
    ]
    for key in order:
        row = table.get(key, {})
        if str(row.get("status", "")).startswith("READY") and "plume_ssim" in row:
            plume_ssim = row["plume_ssim"]  # type: ignore[index]
            plume_unc = row["plume_unc_mean"]  # type: ignore[index]
            lines.append(
                f"- **{key}** (n={row.get('n')}): plume SSIM iid/ood/axisD = "
                f"{cell(plume_ssim['iid'])}/{cell(plume_ssim['ood'])}/{cell(plume_ssim['axisD'])}; "
                f"plume unc = {cell(plume_unc['iid'])}/{cell(plume_unc['ood'])}/{cell(plume_unc['axisD'])}"
            )
        elif row.get("status") == "PENDING":
            lines.append(f"- **{key}**: PENDING (eval job not yet complete)")

    output_path = OUT / "OBJ3_EMS_AXISD_TAXONOMY_TABLE.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print(output_path.resolve())


if __name__ == "__main__":
    main()
