"""Sensitivity of the split-conformal quantile to non-physical calibration realizations (CPU only).

A scan of the 225 calibration realizations finds isolated snapshots in which the transport solver
returned concentrations far outside the physical range. This script (1) re-derives that list from
the raw fields, (2) recomputes the plume-masked split-conformal quantile exactly as
src/obj3/conference/eval_conformal.py does, from the certified calibration prediction cache, and
(3) recomputes it with the affected realizations removed. The certified quantile is checked first so
that any mismatch in reconstruction is caught.

Runs on the cluster where the cache lives. Writes one JSON file. No training, no inference.
"""
from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path
import glob, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_ROOT
RUN = REPO / "experiments/obj3/journal/runs/obj3_boundaryfix_20260818"
CACHE = RUN / "predictions/calibration_preds.npz"
OUT = RUN / "postfix_final/conformal/calibration_nonphysical_sensitivity.json"
MAIN = asset_path("data", REPO / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL")
EXTRA = asset_path("calibration", REPO / "simulation/datasets/obj3_calibration")
PLUME_THRESH, ALPHAS = 1e-8, (0.05, 0.10, 0.20)
CERTIFIED_Q10 = 3.8334767818450928
SEVERE, MILD = 1e3, 10.0     # |C| thresholds on relative concentration (physical bound ~1)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
    return float(np.quantile(scores, level, method="higher"))


split = json.load(open(REPO / "splits/param_split_obj3_ood.json"))
files = []
for c in sorted(split["val_calib"] + split["extra_calib"]):
    base = MAIN if (MAIN / f"param_{c:03d}").is_dir() else EXTRA
    files += [(c, Path(f).name) for f in sorted(glob.glob(str(base / f"param_{c:03d}/real_*.npz")))]

flag = {}
for i, (c, name) in enumerate(files):
    base = MAIN if (MAIN / f"param_{c:03d}").is_dir() else EXTRA
    C = np.load(base / f"param_{c:03d}" / name)["C"]
    mx = np.abs(C).reshape(C.shape[0], -1).max(1)
    ts = np.where(mx > MILD)[0]
    if ts.size:
        flag[i] = {"config": c, "file": name, "timesteps": ts.tolist(), "max_abs": float(mx.max()),
                   "min": float(C.min()), "max": float(C.max()),
                   "class": "severe" if mx.max() > SEVERE else "mild"}

z = np.load(CACHE)
y_true, y_pred, c_phys = z["y_true"], z["y_pred"], z["c_phys"]
assert y_true.shape[0] == len(files), (y_true.shape, len(files))

chunks, owner, bad_ts_pixels = [], [], 0
for i in range(y_true.shape[0]):
    m = c_phys[i] > PLUME_THRESH
    s = np.abs(y_true[i] - y_pred[i])[m].astype(np.float32)
    chunks.append(s); owner.append(np.full(s.size, i, dtype=np.int16))
    if i in flag:
        for t in flag[i]["timesteps"]:
            bad_ts_pixels += int(m[t].sum())
scores, owner = np.concatenate(chunks), np.concatenate(owner)
N = scores.size

res = {"n_realizations": len(files), "n_scores": int(N), "flagged": list(flag.values()),
       "affected_timestep_plume_pixels": bad_ts_pixels,
       "affected_timestep_pixel_fraction": bad_ts_pixels / N, "quantiles": {}}
severe = [i for i, f in flag.items() if f["class"] == "severe"]
for a in ALPHAS:
    full = conformal_quantile(scores, a)
    keep_sev = ~np.isin(owner, severe)
    keep_all = ~np.isin(owner, list(flag))
    res["quantiles"][f"alpha_{a:.2f}"] = {
        "all": full,
        "without_severe": conformal_quantile(scores[keep_sev], a),
        "without_all_flagged": conformal_quantile(scores[keep_all], a)}
res["certified_q10_reproduced"] = abs(res["quantiles"]["alpha_0.10"]["all"] - CERTIFIED_Q10) < 1e-6
res["severe_realization_pixel_fraction"] = float(np.isin(owner, severe).mean())
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
