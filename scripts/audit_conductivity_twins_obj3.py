"""Audit: are the Obj3 OOD conductivity fields amplitude-rescaled twins of training fields?

Background. The field generator reads a fixed random seed from RandF.in, and the setup script
rewrites only the anisotropy/correlation-length and log-mean/log-variance lines. Two parameter
cells that share anisotropy and correlation length but differ in log-variance therefore draw the
same underlying normal-score field, rescaled. This was confirmed for Obj1. Obj3 withholds
sigma2Y = 2.0 as its out-of-distribution axis, so if the same mechanism applies, every OOD field
is a rescaled copy of a field seen in training or calibration.

This script measures it directly on the released fields. For every field it standardizes
log10 K to zero mean and unit variance, then correlates each evaluation field against every
training field. A correlation of 1.000 means identical spatial pattern. The regression slope of
the raw log10 K values against the matched training field is reported as well, because under the
rescaling hypothesis the slope should equal sqrt(sigma2Y_eval / sigma2Y_train).

Read-only. Writes one JSON report. No data is modified, regenerated or reordered.
"""
from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path
import csv, json
from pathlib import Path
import numpy as np

REPO = DEFAULT_ROOT
DATA = asset_path("data", REPO / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL/T25_TSTEP_OVERRIDE_FINAL_FLIPPED")
SPLIT = REPO / "splits/param_split_obj3_ood.json"
REG = metadata_path("param_registry_master.csv")
OUT = REPO / "experiments/obj3/journal/reports/analysis/obj3_conductivity_twin_audit.json"
TWIN_R = 0.999

reg = {int(r["param_id"]): r for r in csv.DictReader(open(REG))}
split = json.load(open(SPLIT))


def load(pid: int, real: int) -> np.ndarray | None:
    f = DATA / f"param_{pid:03d}" / f"real_{real:03d}.npz"
    if not f.exists():
        return None
    return np.log10(np.maximum(np.load(f)["K"], 1e-30)).ravel().astype(np.float64)


def collect(ids):
    out = []
    for pid in sorted(ids):
        for r in range(1, 6):
            v = load(pid, r)
            if v is not None:
                out.append((pid, r, v))
    return out


def standardize(rows):
    X = np.stack([v for _, _, v in rows])
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0), X, mu.ravel(), sd.ravel()


train = collect(split["train"])
calib = collect(split["val_calib"] + split["extra_calib"])
report = {"n_train_fields": len(train), "twin_threshold": TWIN_R, "pools": {}}
Zt, Xt, _, SDt = standardize(train)
n_pix = Zt.shape[1]

for pool in ("ood_test", "iid_test"):
    rows = collect(split[pool])
    Ze, Xe, _, SDe = standardize(rows)
    C = (Ze @ Zt.T) / n_pix                      # Pearson r against every training field
    best = C.argmax(1)
    rmax = C[np.arange(len(rows)), best]
    recs, same_real, slope_ok = [], 0, 0
    for i, (pid, r, _) in enumerate(rows):
        j = int(best[i])
        tpid, treal = train[j][0], train[j][1]
        slope = float(SDe[i] / SDt[j])
        exp = float(np.sqrt(float(reg[pid]["sigma2Y"]) / float(reg[tpid]["sigma2Y"])))
        same_real += int(treal == r)
        slope_ok += int(abs(slope - exp) < 0.02 * exp)
        recs.append({"param_id": pid, "real": r, "sigma2Y": float(reg[pid]["sigma2Y"]),
                     "best_train_param": tpid, "best_train_real": treal,
                     "best_train_sigma2Y": float(reg[tpid]["sigma2Y"]),
                     "r": float(rmax[i]), "slope": slope, "slope_expected": exp})
    report["pools"][pool] = {
        "n_fields": len(rows),
        "n_twins_r_gt_threshold": int((rmax > TWIN_R).sum()),
        "fraction_twinned": float((rmax > TWIN_R).mean()),
        "r_median": float(np.median(rmax)), "r_min": float(rmax.min()), "r_max": float(rmax.max()),
        "twin_shares_realization_id": int(same_real),
        "slope_matches_sqrt_variance_ratio": int(slope_ok),
        "median_offdiagonal_r": float(np.median(np.abs(C))),
        "fields": recs,
    }

# Same-pattern check restricted to calibration, which also saw the fields during model selection.
if calib:
    Zc, _, _, SDc = standardize(calib)
    Ccal = (Ze @ Zc.T) / n_pix
    report["ood_vs_calibration_r_median"] = float(np.median(Ccal.max(1)))

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(report, open(OUT, "w"), indent=1)
for pool, d in report["pools"].items():
    print(f"{pool}: {d['n_twins_r_gt_threshold']}/{d['n_fields']} twins at r>{TWIN_R}; "
          f"r median {d['r_median']:.4f} min {d['r_min']:.4f}; "
          f"same realization id {d['twin_shares_realization_id']}; "
          f"slope matches sqrt ratio {d['slope_matches_sqrt_variance_ratio']}; "
          f"median |r| off-target {d['median_offdiagonal_r']:.4f}")
print("ood vs calibration max-r median:", report.get("ood_vs_calibration_r_median"))
print("REPORT ->", OUT)
