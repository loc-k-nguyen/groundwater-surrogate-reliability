"""Follow-up to the Obj3 conductivity-twin audit: which pool holds each evaluation field's twin.

The first pass compared evaluation fields against training fields only. Two questions remain.
First, the 60 OOD fields without a training twin: are they twinned to the calibration or IID pools
instead, which the model also saw during checkpoint selection or which serve as the reference set?
Second, the per-cell structure: a twin is expected exactly when the same (anisotropy, correlation
length, realization) appears at a lower log-variance somewhere outside the OOD pool.

Read-only. Writes one JSON report.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--calibration-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if args.output.exists():
        parser.error('Output already exists; use a new isolated path')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def asset_path(kind, fallback):
        value = args.data_root if kind == 'data' else args.calibration_root
        path = (value or fallback).resolve()
        if not path.is_dir():
            parser.error(f'Restricted {kind} directory is missing: {path}. Supply the corresponding root option.')
        return path

    def metadata_path(name):
        path = args.repo_root / 'metadata' / name
        return path if path.exists() else args.repo_root / 'simulation' / name

    import csv, json
    from collections import defaultdict
    import numpy as np

    REPO = args.repo_root
    DATA = asset_path("data", REPO / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL/T25_TSTEP_OVERRIDE_FINAL_FLIPPED")
    OUT = args.output
    TWIN_R = 0.999

    reg = {int(r["param_id"]): r for r in csv.DictReader(open(metadata_path("param_registry_master.csv")))}
    split = json.load(open(REPO / "splits/param_split_obj3_ood.json"))
    POOLS = ("train", "val_calib", "extra_calib", "iid_test")


    def fields(ids):
        out = []
        for pid in sorted(ids):
            for r in range(1, 6):
                f = DATA / f"param_{pid:03d}" / f"real_{r:03d}.npz"
                if f.exists():
                    out.append((pid, r, np.log10(np.maximum(np.load(f)["K"], 1e-30)).ravel().astype(np.float64)))
        return out


    def z(rows):
        X = np.stack([v for _, _, v in rows])
        return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


    ref = []
    for pool in POOLS:
        ref += [(pool, pid, r, v) for pid, r, v in fields(split[pool])]
    ood = fields(split["ood_test"])
    Zr = np.stack([(v - v.mean()) / v.std() for _, _, _, v in ref])
    Zo = z(ood)
    C = (Zo @ Zr.T) / Zr.shape[1]
    best = C.argmax(1)

    by_cell = defaultdict(lambda: defaultdict(int))
    pool_counts = defaultdict(int)
    rows = []
    for i, (pid, r, _) in enumerate(ood):
        j = int(best[i])
        pool, tpid, treal = ref[j][0], ref[j][1], ref[j][2]
        rr = float(C[i, j])
        twin = rr > TWIN_R
        cell = f"aniso {reg[pid]['anisotropy']}, corr {reg[pid]['correlation_length']}"
        by_cell[cell]["n"] += 1
        by_cell[cell]["twins"] += int(twin)
        by_cell[cell][f"twin_in_{pool}" if twin else "no_twin"] += 1
        if twin:
            pool_counts[pool] += 1
        rows.append({"param_id": pid, "real": r, "cell": cell, "r": rr, "twin": twin,
                     "best_pool": pool, "best_param": tpid, "best_real": treal})

    untw = [x for x in rows if not x["twin"]]
    report = {
        "n_ood_fields": len(ood), "twin_threshold": TWIN_R,
        "n_twinned_anywhere": int(sum(x["twin"] for x in rows)),
        "twin_pool_counts": dict(pool_counts),
        "untwinned": {"n": len(untw), "r_median": float(np.median([x["r"] for x in untw])) if untw else None,
                      "r_max": float(max([x["r"] for x in untw])) if untw else None,
                      "cells": sorted({x["cell"] for x in untw})},
        "by_cell": {k: dict(v) for k, v in sorted(by_cell.items())},
        "fields": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"twinned anywhere: {report['n_twinned_anywhere']}/{len(ood)}  by pool: {dict(pool_counts)}")
    print("untwinned:", report["untwinned"]["n"], "median r", report["untwinned"]["r_median"],
          "max r", report["untwinned"]["r_max"], "cells", report["untwinned"]["cells"])
    for k, v in report["by_cell"].items():
        print(" ", k, dict(v))
    print("REPORT ->", OUT)


if __name__ == '__main__':
    main()
