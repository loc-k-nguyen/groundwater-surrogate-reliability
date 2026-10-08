"""Tier 0 severity-conditioned conformal repair for Obj3 EMS upgrade (CPU-only).

Compares, under a STATED calibration-support gap (calib sigma2Y in {0.1,0.5,1.0,1.5},
OOD test sigma2Y=2.0 is OUTSIDE support):

  1. SCP                        - split conformal, global calibration
  2. NEC                        - ensemble-std-normalized conformal (reproduces baseline)
  3. NEC-recal                  - global-scalar variance recalibration (shown to be a
                                  coverage no-op; documented clarifying result)
  4. Mondrian (regime)          - per-regime quantile; OOD -> nearest calib regime (1.5)
  5. severity-conditioned       - DEPLOYABLE: fit q(severity) over calib regimes using an
                                  input-K severity score, extrapolate to test severity
  6. conservative-fallback SCP  - quantile from the widest calib regime (sigma2Y=1.5)

Every method is evaluated in two modes:
  * DEPLOYABLE  : region = predicted plume mask (10**y_pred > plume_thresh); severity from inputs
  * ORACLE      : region = GT plume mask (c_phys > plume_thresh); retrospective reference only

NO distribution-free OOD guarantee is claimed. See OBJ3_EMS_UPGRADE_FRAMING_LOCK_2026-07-01.md.

Memory-safe: NPZ members are streamed one sample-slice at a time (never fully materialized).
"""
from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path

import argparse
import csv
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logging.disable(logging.WARNING)  # silence dataset "config dir missing" chatter
import sys

REPO_ROOT = DEFAULT_ROOT
sys.path.insert(0, str(REPO_ROOT))
from src.obj3.conference.data_obj3_ood import collect_npz_files, load_obj3_split  # noqa: E402

PLUME_THRESH = 1e-8
# Deployable predicted-plume mask convention matches obj3_deployable_conformal.py:
# mask = (y_pred > PLUME_THRESH) in the log-prediction domain (dense predicted core).
PRED_THRESH = PLUME_THRESH
EPS_NEC = 1e-4
SIGMA_FLOOR_Q = 0.10
EPS_K = 1e-6
ALPHAS = (0.05, 0.10, 0.20)
MAIN_ROOT = asset_path("data", REPO_ROOT / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL")
EXTRA_ROOT = asset_path("calibration", REPO_ROOT / "simulation/datasets/obj3_calibration")
REGISTRY = metadata_path("param_registry_master.csv")


# ----------------------------- streaming reader -----------------------------
class NpySampleStream:
    """Sequential per-sample reader for one .npy member inside an (compressed) .npz."""

    def __init__(self, npz_path: Path, member: str) -> None:
        self._zf = zipfile.ZipFile(npz_path)
        self._fh = self._zf.open(member)
        ver = np.lib.format.read_magic(self._fh)
        self.shape, fortran, self.dtype = np.lib.format._read_array_header(self._fh, ver)
        if fortran:
            raise ValueError(f"{member}: fortran-order not supported")
        self.n = int(self.shape[0])
        self.sample_shape = tuple(self.shape[1:])
        self._count = int(np.prod(self.sample_shape))
        self._nbytes = self._count * self.dtype.itemsize

    def read_sample(self) -> np.ndarray:
        buf = self._fh.read(self._nbytes)
        if len(buf) != self._nbytes:
            raise EOFError("short read from npz member stream")
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.sample_shape)

    def close(self) -> None:
        self._fh.close()
        self._zf.close()


def open_streams(npz_path: Path, keys: Sequence[str]) -> Dict[str, NpySampleStream]:
    return {k: NpySampleStream(npz_path, f"{k}.npy") for k in keys}


# ----------------------------- regime / severity -----------------------------
def load_registry_sigma(registry: Path = REGISTRY) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with open(registry, newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["param_id"])] = float(r["sigma2Y"])
    return out


def ordered_files(
    config_ids: List[int],
    main_root: Path = MAIN_ROOT,
    extra_root: Path = EXTRA_ROOT,
) -> List[Path]:
    return collect_npz_files(config_ids, main_root, extra_root)


def _pid(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def sample_severity(path: Path) -> float:
    """Deployable per-sample severity = std of log10(K) of that realisation's K field."""
    with np.load(path, allow_pickle=False) as d:
        K = np.asarray(d["K"], dtype=np.float64)
    K = np.squeeze(K)
    if K.ndim > 2:
        K = K.reshape(K.shape[-2], K.shape[-1])
    return float(np.std(np.log10(K + EPS_K)))


# ----------------------------- conformal quantile -----------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    if n == 0:
        return float("nan")
    level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, level, interpolation="higher"))


# ----------------------------- calibration pass -----------------------------
def calibrate(calib_npz: Path, files: List[Path], sigma: Dict[int, float], limit: int) -> dict:
    """Stream calibration; build per-regime score pools for oracle & deployable masks."""
    keys = ("y_true", "y_pred", "y_pred_std", "c_phys")
    st = open_streams(calib_npz, keys)
    n = st["y_true"].n
    if len(files) != n:
        raise RuntimeError(f"CALIB ROW MISMATCH: files={len(files)} npz={n}")
    if limit > 0:
        n = min(n, limit)

    # pools[mask_mode][regime] -> list of arrays; nec pools + std pools similarly
    modes = ("deployable", "oracle")
    scp: Dict[str, Dict[float, list]] = {m: {} for m in modes}
    nec: Dict[str, Dict[float, list]] = {m: {} for m in modes}
    sevs: Dict[float, list] = {}  # regime -> per-sample severities
    # global std floor computed from deployable+oracle plume std (use oracle union)
    std_pool: list = []

    for i in range(n):
        yt = st["y_true"].read_sample()
        yp = st["y_pred"].read_sample()
        ys = st["y_pred_std"].read_sample()
        cp = st["c_phys"].read_sample()
        reg = sigma[_pid(files[i])]
        sevs.setdefault(reg, []).append(sample_severity(files[i]))
        resid = np.abs(yt - yp)
        masks = {"deployable": yp > PRED_THRESH, "oracle": cp > PLUME_THRESH}
        for m, mask in masks.items():
            if not np.any(mask):
                continue
            r = resid[mask].astype(np.float32)
            s = ys[mask].astype(np.float32)
            scp[m].setdefault(reg, []).append(r)
            nec[m].setdefault(reg, []).append((r, s))
            if m == "oracle":
                std_pool.append(s[s > 0])
    for s in st.values():
        s.close()

    # sigma floor from oracle plume std distribution
    all_std = np.concatenate(std_pool) if std_pool else np.array([EPS_NEC], np.float32)
    sigma_floor = max(EPS_NEC, float(np.quantile(all_std, SIGMA_FLOOR_Q)))

    regimes = sorted(scp["oracle"].keys())
    mean_sev = {r: float(np.mean(sevs[r])) for r in sevs}

    # per-regime & global quantiles
    out = {"sigma_floor": sigma_floor, "regimes": regimes, "mean_severity": mean_sev,
           "n_calib": n, "scp": {}, "nec": {}, "scp_regime": {}, "nec_regime": {}}
    for m in modes:
        # global
        allr = np.concatenate([np.concatenate(v) for v in scp[m].values()])
        out["scp"][m] = {a: conformal_quantile(allr, a) for a in ALPHAS}
        # NEC global
        rr = np.concatenate([np.concatenate([p[0] for p in v]) for v in nec[m].values()])
        ss = np.concatenate([np.concatenate([p[1] for p in v]) for v in nec[m].values()])
        nec_scores = rr / np.maximum(ss, sigma_floor)
        out["nec"][m] = {a: conformal_quantile(nec_scores, a) for a in ALPHAS}
        del allr, rr, ss, nec_scores
        # per regime
        out["scp_regime"][m] = {}
        out["nec_regime"][m] = {}
        for r in regimes:
            rscore = np.concatenate(scp[m][r])
            out["scp_regime"][m][r] = {a: conformal_quantile(rscore, a) for a in ALPHAS}
            nr = np.concatenate([p[0] for p in nec[m][r]])
            ns = np.concatenate([p[1] for p in nec[m][r]])
            out["nec_regime"][m][r] = {a: conformal_quantile(nr / np.maximum(ns, sigma_floor), a) for a in ALPHAS}
    return out


def fit_severity_curve(cal: dict, mode: str) -> dict:
    """Fit q_scp(severity) linearly per alpha over calib regimes; report monotonicity."""
    regimes = cal["regimes"]
    sev = np.array([cal["mean_severity"][r] for r in regimes])
    fit = {"severity_by_regime": {float(r): float(s) for r, s in zip(regimes, sev)}, "alpha": {}}
    for a in ALPHAS:
        q = np.array([cal["scp_regime"][mode][r][a] for r in regimes])
        order = np.argsort(sev)
        q_sorted = q[order]
        monotone = bool(np.all(np.diff(q_sorted) >= -1e-6))
        # linear fit q = m*sev + b
        A = np.vstack([sev, np.ones_like(sev)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, q, rcond=None)
        fit["alpha"][a] = {"slope": float(slope), "intercept": float(intercept),
                            "monotone": monotone, "q_by_regime": {float(r): float(qq) for r, qq in zip(regimes, q)}}
    return fit


def sev_predict(fit: dict, alpha: float, s: float) -> float:
    p = fit["alpha"][alpha]
    return max(0.0, p["slope"] * s + p["intercept"])


# ----------------------------- evaluation pass -----------------------------
def evaluate(test_npz: Path, files: List[Path], sigma: Dict[int, float], cal: dict,
             sevfit: Dict[str, dict], alpha_list: Sequence[float], limit: int) -> List[dict]:
    """Single streaming pass over the split; evaluate every method x mode x alpha."""
    keys = ("y_true", "y_pred", "y_pred_std", "c_phys")
    st = open_streams(test_npz, keys)
    n = st["y_true"].n
    if len(files) != n:
        raise RuntimeError(f"TEST ROW MISMATCH: files={len(files)} npz={n}")
    if limit > 0:
        n = min(n, limit)
    floor = cal["sigma_floor"]
    widest = max(cal["regimes"])  # sigma2Y=1.5 = nearest calibrated regime to OOD 2.0

    modes = ("deployable", "oracle")
    methods = ["scp", "nec", "nec_recal", "mondrian", "severity", "conservative"]
    # acc[method][mode][alpha] -> [covered, total, width_sum]
    acc = {mth: {m: {a: [0, 0, 0.0] for a in alpha_list} for m in modes} for mth in methods}

    for i in range(n):
        yt = st["y_true"].read_sample()
        yp = st["y_pred"].read_sample()
        ys = st["y_pred_std"].read_sample()
        cp = st["c_phys"].read_sample()
        reg = sigma[_pid(files[i])]
        s_in = sample_severity(files[i])
        masks = {"deployable": yp > PRED_THRESH, "oracle": cp > PLUME_THRESH}
        for m, mask in masks.items():
            if not np.any(mask):
                continue
            yt_m, yp_m, ys_m = yt[mask], yp[mask], ys[mask]
            err = np.abs(yt_m - yp_m)
            std_m = np.maximum(ys_m, floor)
            in_support = reg in cal["scp_regime"][m]
            use_reg = reg if in_support else widest  # OOD -> nearest calibrated regime
            for a in alpha_list:
                halves = {
                    "scp": cal["scp"][m][a],
                    "nec": cal["nec"][m][a] * std_m,
                    "nec_recal": cal["nec_regime"][m][use_reg][a] * std_m,   # per-regime NEC (nearest for OOD)
                    "mondrian": cal["scp_regime"][m][use_reg][a],
                    "severity": max(0.0, sev_predict(sevfit[m], a, s_in)),   # deployable severity extrapolation
                    "conservative": cal["scp_regime"][m][widest][a],
                }
                for mth, half in halves.items():
                    covered = int(np.count_nonzero(err <= half))
                    slot = acc[mth][m][a]
                    slot[0] += covered
                    slot[1] += int(yt_m.size)
                    slot[2] += float(np.sum(2.0 * half if np.isscalar(half) else 2.0 * half, dtype=np.float64)) \
                        if not np.isscalar(half) else float(2.0 * half * yt_m.size)
    for s in st.values():
        s.close()

    rows = []
    for mth in methods:
        for m in modes:
            for a in alpha_list:
                cov, tot, wsum = acc[mth][m][a]
                rows.append({"method": mth, "mode": m, "alpha": a,
                             "coverage": cov / max(tot, 1), "mean_width": wsum / max(tot, 1),
                             "n_pixels": tot})
    return rows


# ----------------------------- driver -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "experiments/obj3/journal/upgrade_2026_07_01/tier0_conformal_repair"))
    ap.add_argument("--calib_npz", default=str(REPO_ROOT / "experiments/obj3/journal/calibration/calibration_preds.npz"))
    ap.add_argument("--iid_npz", default=str(REPO_ROOT / "experiments/obj3/conference/runs/obj3_conf_ms_tmo_ensemble5_full/preds/iid_test_preds.npz"))
    ap.add_argument("--ood_npz", default=str(REPO_ROOT / "experiments/obj3/conference/runs/obj3_conf_ms_tmo_ensemble5_full/preds/ood_test_preds.npz"))
    ap.add_argument("--limit", type=int, default=0, help="cap samples per split (smoke); 0 = all")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--main_root", type=Path, default=MAIN_ROOT)
    ap.add_argument("--extra_root", type=Path, default=EXTRA_ROOT)
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--check_order_only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    t0 = time.time()
    sigma = load_registry_sigma(args.registry)
    split = load_obj3_split()

    calib_files = ordered_files(split["val_calib"] + split["extra_calib"], args.main_root, args.extra_root)
    iid_files = ordered_files(split["iid_test"], args.main_root, args.extra_root)
    ood_files = ordered_files(split["ood_test"], args.main_root, args.extra_root)
    counts = {"calib": len(calib_files), "iid": len(iid_files), "ood": len(ood_files)}
    print(f"[order] row counts: {counts}", flush=True)
    if args.check_order_only:
        expected = {"calib": 225, "iid": 110, "ood": 270}
        if counts != expected:
            raise RuntimeError(f"ORDER COUNT MISMATCH: expected={expected} actual={counts}")
        print("OBJ3_TIER0_INPUT_ORDER_CERTIFIED", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cal = calibrate(Path(args.calib_npz), calib_files, sigma, args.limit)
    sevfit = {m: fit_severity_curve(cal, m) for m in ("deployable", "oracle")}
    print(f"[calib] regimes={cal['regimes']} sigma_floor={cal['sigma_floor']:.4g} "
          f"mean_sev={ {k: round(v,4) for k,v in cal['mean_severity'].items()} }", flush=True)
    for m in ("deployable", "oracle"):
        for a in ALPHAS:
            print(f"[sevfit {m} a{a}] slope={sevfit[m]['alpha'][a]['slope']:.4f} "
                  f"monotone={sevfit[m]['alpha'][a]['monotone']} q_by_regime={ {k:round(v,3) for k,v in sevfit[m]['alpha'][a]['q_by_regime'].items()} }", flush=True)

    all_rows = []
    for split_name, npz, files in [("iid", args.iid_npz, iid_files), ("ood", args.ood_npz, ood_files)]:
        rows = evaluate(Path(npz), files, sigma, cal, sevfit, ALPHAS, args.limit)
        for r in rows:
            r["split"] = split_name
        all_rows.extend(rows)
        print(f"[eval {split_name}] {len(rows)} rows done", flush=True)

    # write outputs
    (out_dir / f"tier0_calibration_{args.tag}.json").write_text(json.dumps(
        {"counts": counts, "sigma_floor": cal["sigma_floor"], "regimes": cal["regimes"],
         "mean_severity": cal["mean_severity"], "scp": cal["scp"], "nec": cal["nec"],
         "scp_regime": {m: {str(r): v for r, v in cal["scp_regime"][m].items()} for m in cal["scp_regime"]},
         "severity_fit": sevfit, "n_calib": cal["n_calib"]}, indent=2, default=str), encoding="utf-8")
    with open(out_dir / f"tier0_results_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "method", "mode", "alpha", "coverage", "mean_width", "n_pixels"])
        w.writeheader()
        w.writerows(all_rows)
    runtime = time.time() - t0
    (out_dir / f"tier0_meta_{args.tag}.json").write_text(json.dumps(
        {"runtime_sec": runtime, "limit": args.limit, "counts": counts,
         "pred_thresh": PRED_THRESH, "alphas": list(ALPHAS)}, indent=2), encoding="utf-8")
    print(f"[done] runtime={runtime:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
