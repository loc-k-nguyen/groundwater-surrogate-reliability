"""Paired same-pattern comparison of the variance shift, using the conductivity twins (CPU only).

Sixty of the 270 shifted fields are exact amplitude-rescaled twins of reference-set fields: same
anisotropy, correlation length and realization, differing only in log-variance. For those pairs the
spatial pattern is held fixed and only the shift parameter moves, which is the paired analogue of
the dispersivity ladder used for the omitted-control axis.

This computes, per monitor, the paired change in plume SSIM and in both uncertainty scores across
each twin pair, with a sign test over pairs. The independent unit is coarser than the pair: the 60
pairs come from two (anisotropy, correlation length) cells, so the sign test is reported as
descriptive, with that resolution limit stated in the output.

Reads the existing matched-pool per-sample files and the twin breakdown. No inference.
"""
from __future__ import annotations
import csv, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results/matched_pool"
BREAK = REPO / "results/twin_audit/obj3_conductivity_twin_breakdown.json"
OUT = REPO / "results/twin_audit/obj3_twin_paired_shift_effect.json"
ARCH = [("Deep ensemble (U-Net)", "det_ensemble"), ("Heteroscedastic (U-Net)", "hetero"),
        ("Fourier operator", "fno"), ("DeepONet", "deeponet")]
COLS = {"plume_ssim": "mean_plume_ssim", "oracle": "plume_total_std_log",
        "deployable": "mean_total_std_log"}

pairs = [(r["param_id"], r["real"], r["best_param"], r["best_real"], r["cell"])
         for r in json.load(open(BREAK))["fields"] if r["best_pool"] == "iid_test" and r["twin"]]

out = {"n_twin_pairs": len(pairs),
       "cells": sorted({p[4] for p in pairs}),
       "note": ("Pairs share anisotropy, correlation length and realization; only sigma2Y differs "
                "(2.0 versus the reference level). The independent unit is the cell, not the pair."),
       "monitors": {}}

for label, key in ARCH:
    f = glob.glob(str(RUNS / f"*{key}*axisDmatched_boundaryfix*/*per_sample.csv"))[0]
    idx = {}
    for r in csv.DictReader(open(f)):
        pid = int(str(r["param_id"]).split("_")[-1])
        rid = int(str(r["real_id"]).split("_")[-1])
        idx[(r["target"], pid, rid)] = r
    rec = {}
    for name, col in COLS.items():
        d, by_cell = [], defaultdict(list)
        for opid, oreal, rpid, rreal, cell in pairs:
            a = idx.get(("ood_test", opid, oreal))
            b = idx.get(("iid_test", rpid, rreal))
            if not a or not b:
                continue
            try:
                delta = float(a[col]) - float(b[col])
            except (TypeError, ValueError):
                continue
            d.append(delta); by_cell[cell].append(delta)
        d = np.array(d)
        rec[name] = {"n_pairs": int(d.size), "mean_change": float(d.mean()),
                     "median_change": float(np.median(d)),
                     "n_increase": int((d > 0).sum()),
                     "cell_means": {c: float(np.mean(v)) for c, v in by_cell.items()}}
    out["monitors"][label] = rec

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(f"{out['n_twin_pairs']} twin pairs from cells {out['cells']}")
for label, rec in out["monitors"].items():
    print(f"\n{label}")
    for name, r in rec.items():
        print(f"  {name:11s} mean {r['mean_change']:+.4f}  median {r['median_change']:+.4f}  "
              f"rises in {r['n_increase']}/{r['n_pairs']}  per cell "
              + ", ".join(f"{c.split(',')[0]}: {v:+.4f}" for c, v in r["cell_means"].items()))
print("REPORT ->", OUT)
