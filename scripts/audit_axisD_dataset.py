"""Audit an Axis D dataset folder and write machine-readable readiness status.

The host simulator writes datasets as:

    dataset_axisD/param_9200/real_001.npz
    dataset_axisD/param_9200/real_002.npz
    ...

This script checks every requested param directory with the same schema checker
used by the original Axis D smoke gate, then writes a compact status file for
the follow-up GPU evaluation gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from axisD_npz_schema_check import check

DEFAULT_PARAMS = tuple(str(p) for p in range(9200, 9206))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit expanded Axis D NPZ dataset")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--params", nargs="+", default=list(DEFAULT_PARAMS))
    parser.add_argument("--min-pass-per-param", type=int, default=2)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    return parser.parse_args()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "param_id",
        "real_id",
        "path",
        "exists",
        "verdict",
        "failed_checks",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    rows: list[dict[str, Any]] = []
    by_param: dict[str, dict[str, Any]] = {}

    for raw_param in args.params:
        param = str(raw_param)
        param_dir = dataset_root / f"param_{param}"
        files = sorted(param_dir.glob("real_*.npz")) if param_dir.exists() else []
        param_rows: list[dict[str, Any]] = []
        for npz_path in files:
            result = check(npz_path)
            row = {
                "param_id": param,
                "real_id": npz_path.stem.replace("real_", ""),
                "path": str(npz_path),
                "exists": npz_path.exists(),
                "verdict": result.get("verdict", "FAIL"),
                "failed_checks": ";".join(result.get("failed_checks", [])),
            }
            param_rows.append(row)
            rows.append(row)

        pass_count = sum(1 for row in param_rows if row["verdict"] == "PASS")
        fail_count = sum(1 for row in param_rows if row["verdict"] != "PASS")
        by_param[param] = {
            "param_dir": str(param_dir),
            "n_files": len(files),
            "pass": pass_count,
            "fail": fail_count,
            "ready": pass_count >= args.min_pass_per_param and fail_count == 0,
        }
        if not files:
            rows.append(
                {
                    "param_id": param,
                    "real_id": "",
                    "path": str(param_dir),
                    "exists": False,
                    "verdict": "MISSING",
                    "failed_checks": "no_real_npz_files",
                }
            )

    total_pass = sum(1 for row in rows if row.get("verdict") == "PASS")
    total_fail = sum(1 for row in rows if row.get("verdict") != "PASS")
    ready_params = sum(1 for summary in by_param.values() if summary["ready"])
    all_ready = ready_params == len(args.params) and total_fail == 0
    status_line = (
        f"EXPANDED_ALL_READY pass={total_pass} fail={total_fail} params={ready_params}/{len(args.params)}"
        if all_ready
        else f"EXPANDED_PARTIAL pass={total_pass} fail={total_fail} params={ready_params}/{len(args.params)}"
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "params": list(args.params),
        "min_pass_per_param": args.min_pass_per_param,
        "status": status_line,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "ready_params": ready_params,
        "by_param": by_param,
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(rows, args.out_csv)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(
        status_line
        + "\n"
        + f"audit_json={args.out_json}\n"
        + f"audit_csv={args.out_csv}\n"
        + "GPU inference NOT launched by host batch. Submit the gated Raapoi eval next.\n",
        encoding="utf-8",
    )
    print(status_line, flush=True)


if __name__ == "__main__":
    main()
