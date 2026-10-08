"""Paired same-pattern variance analysis over all 270 twin pairs (CPU only).

The first paired analysis covered only the 60 shifted cases whose twin sits in the reference pool,
because predictions existed for that pool alone. Once the twin-control job has produced predictions
for the 380 training fields, the remaining 210 pairs become available: 180 with a training twin and
30 with a calibration twin.

This joins the shifted per-sample rows to their twin's row, wherever the twin lives, and reports the
paired change in plume SSIM and in both uncertainty scores. Pairs are grouped by the pool holding
the twin, because only reference twins are unseen like the shifted field; training and
validation-calibration twins also contrast seen with unseen fields. Within a group, results are
also broken down by conductivity setting, which is the independent unit.

Run after the twin-control evaluation completes. Read-only; no inference.
"""
from __future__ import annotations
import csv, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results/matched_pool"
TRAIN_RUN = RUNS / "obj3_jnl_twin_control_train_20260924"
BREAK = REPO / "results/twin_audit/obj3_conductivity_twin_breakdown.json"
OUT = REPO / "results/twin_audit/obj3_twin_paired_shift_effect_full.json"
REG = {int(r["param_id"]): r for r in
       csv.DictReader(open(REPO / "metadata/param_registry_master.csv"))}
ARCH = [("Deep ensemble (U-Net)", "det_ensemble"), ("Heteroscedastic (U-Net)", "hetero"),
        ("Fourier operator", "fno"), ("DeepONet", "deeponet")]
COLS = {"plume_ssim": "mean_plume_ssim", "oracle": "plume_total_std_log",
        "deployable": "mean_total_std_log"}
# Twin pools that the matched-pool evaluation already covers, keyed by their CSV target name.
MATCHED_TARGET = {"iid_test": "iid_test", "val_calib": None, "train": None}


def rows_by_key(path: str):
    out = {}
    for r in csv.DictReader(open(path)):
        pid = int(str(r["param_id"]).split("_")[-1])
        rid = int(str(r["real_id"]).split("_")[-1])
        out[(r["target"], pid, rid)] = r
    return out


def setting(pid: int):
    r = REG[pid]
    return (r["anisotropy"], r["correlation_length"])


pairs = [p for p in json.load(open(BREAK))["fields"] if p["twin"]]
missing_train = not TRAIN_RUN.exists()
report = {"n_pairs_available": 0, "train_predictions_present": not missing_train, "monitors": {}}

for label, key in ARCH:
    matched = rows_by_key(glob.glob(str(RUNS / f"*{key}*axisDmatched_boundaryfix*/*per_sample.csv"))[0])
    # Training-pool and validation-calibration twins come from separate per-sample files.
    train_rows = {}
    for f in (sorted(TRAIN_RUN.glob(f"{key}_*per_sample.csv")) if not missing_train else []):
        train_rows.update(rows_by_key(str(f)))

    groups = defaultdict(lambda: defaultdict(list))
    n_used = 0
    for p in pairs:
        shifted = matched.get(("ood_test", p["param_id"], p["real"]))
        pool = p["best_pool"]
        if pool == "iid_test":
            twin = matched.get(("iid_test", p["best_param"], p["best_real"]))
        elif pool in ("train", "val_calib"):
            twin = train_rows.get(("train", p["best_param"], p["best_real"])) \
                or train_rows.get(("val_calib", p["best_param"], p["best_real"]))
        else:
            twin = None
        if not shifted or not twin:
            continue
        n_used += 1
        # Group by where the twin lives as well as by variance contrast. Reference twins are unseen
        # by the model, like the shifted field, so those pairs are the clean comparison. Training
        # twins were fitted and validation-calibration twins steered checkpoint selection, so those
        # pairs also contrast seen with unseen fields and are reported separately.
        contrast = f"{pool}: sigma2Y {float(REG[p['best_param']]['sigma2Y'])} to 2.0"
        for name, col in COLS.items():
            try:
                d = float(shifted[col]) - float(twin[col])
            except (TypeError, ValueError, KeyError):
                continue
            groups[contrast][name].append((d, setting(p["param_id"])))

    rec = {}
    for contrast, per_metric in groups.items():
        rec[contrast] = {}
        for name, vals in per_metric.items():
            d = np.array([v for v, _ in vals])
            by_set = defaultdict(list)
            for v, s in vals:
                by_set[str(s)].append(v)
            rec[contrast][name] = {
                "n_pairs": int(d.size), "mean_change": float(d.mean()),
                "median_change": float(np.median(d)), "n_increase": int((d > 0).sum()),
                "n_settings": len(by_set),
                "setting_means": {k: float(np.mean(v)) for k, v in sorted(by_set.items())},
                "settings_all_same_sign": bool(
                    len({np.mean(v) > 0 for v in by_set.values()}) == 1),
            }
    report["monitors"][label] = rec
    report["n_pairs_available"] = max(report["n_pairs_available"], n_used)

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(report, open(OUT, "w"), indent=1)
if missing_train:
    print("NOTE: twin-control predictions not found; only reference-pool pairs were used.")
print(f"pairs used per monitor: {report['n_pairs_available']}")
for label, rec in report["monitors"].items():
    print(f"\n{label}")
    for contrast, per_metric in sorted(rec.items()):
        for name, r in per_metric.items():
            print(f"  {contrast:22s} {name:11s} mean {r['mean_change']:+.4f} "
                  f"rises {r['n_increase']}/{r['n_pairs']} over {r['n_settings']} settings"
                  f"{' (all same sign)' if r['settings_all_same_sign'] else ''}")
print("REPORT ->", OUT)
