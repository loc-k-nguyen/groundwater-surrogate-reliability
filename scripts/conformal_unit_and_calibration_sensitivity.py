"""Two conformal sensitivity analyses on the certified prediction caches (CPU only).

1. Calibration independence. The 50 validation-calibration realizations were also used for
   checkpoint selection. Recompute the plume-masked split-conformal quantile from the 175
   additional-calibration realizations alone and re-evaluate coverage on both test sets.

2. Realization as the exchangeable unit. Pixelwise scores are strongly dependent within a
   field. Here each realization contributes one score, s_i = the 90th percentile of its absolute
   plume residuals, i.e. the half-width needed to cover 90% of that field's plume pixels. The
   conformal quantile of the s_i over calibration realizations gives a half-width Q such that,
   if realizations are exchangeable, a new realization has at least 90% of its plume pixels
   covered with probability at least 1 - delta. Reported on both test sets as the fraction of
   realizations meeting the 90% pixel-coverage target.

The certified pixelwise quantile and test coverages are reproduced first as a check on the
reconstruction. No training, no inference. Writes one JSON file.
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

    import glob, json, sys
    import numpy as np

    REPO = args.repo_root
    RUN = REPO / "experiments/obj3/journal/runs/obj3_boundaryfix_20260818"
    PRED = RUN / "predictions"
    OUT = args.output
    MAIN = asset_path("data", REPO / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL")
    EXTRA = asset_path("calibration", REPO / "simulation/datasets/obj3_calibration")
    PLUME_THRESH, ALPHA, PIX_LEVEL = 1e-8, 0.10, 0.90
    CERTIFIED = {"q": 3.8334767818450928, "iid": 0.932977394011011}


    def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
        return float(np.quantile(scores, level, method="higher"))


    def per_sample(path: Path):
        """Yield (abs plume residuals) per realization, loading one cache at a time."""
        z = np.load(path)
        yt, yp, cp = z["y_true"], z["y_pred"], z["c_phys"]
        for i in range(yt.shape[0]):
            m = cp[i] > PLUME_THRESH
            yield np.abs(yt[i] - yp[i])[m].astype(np.float32)
        del yt, yp, cp


    split = json.load(open(REPO / "splits/param_split_obj3_ood.json"))
    calib_cfg = []
    for c in sorted(split["val_calib"] + split["extra_calib"]):
        base = MAIN if (MAIN / f"param_{c:03d}").is_dir() else EXTRA
        calib_cfg += [c] * len(glob.glob(str(base / f"param_{c:03d}/real_*.npz")))
    is_val = np.array([c in set(split["val_calib"]) for c in calib_cfg])

    cal = list(per_sample(PRED / "calibration_preds.npz"))
    assert len(cal) == len(calib_cfg) == 225, (len(cal), len(calib_cfg))

    q_all = conformal_quantile(np.concatenate(cal), ALPHA)
    q_175 = conformal_quantile(np.concatenate([s for s, v in zip(cal, is_val) if not v]), ALPHA)
    s_real = np.array([np.quantile(s, PIX_LEVEL) for s in cal])
    Q_all = conformal_quantile(s_real, ALPHA)
    Q_175 = conformal_quantile(s_real[~is_val], ALPHA)
    del cal

    res = {"alpha": ALPHA, "pixel_level_within_realization": PIX_LEVEL,
           "n_calibration": {"all": 225, "additional_only": int((~is_val).sum())},
           "pixelwise_quantile": {"all": q_all, "additional_only": q_175},
           "realization_quantile": {"all": Q_all, "additional_only": Q_175},
           "certified_quantile_reproduced": abs(q_all - CERTIFIED["q"]) < 1e-6,
           "test": {}}

    for name in ("iid_test", "ood_test"):
        cov = {k: [0, 0] for k in ("pix_all", "pix_175")}
        frac_all, frac_175 = [], []
        for s in per_sample(PRED / f"{name}_preds.npz"):
            for k, q in (("pix_all", q_all), ("pix_175", q_175)):
                cov[k][0] += int((s <= q).sum()); cov[k][1] += s.size
            frac_all.append(float((s <= Q_all).mean()))
            frac_175.append(float((s <= Q_175).mean()))
        frac_all, frac_175 = np.array(frac_all), np.array(frac_175)
        res["test"][name] = {
            "n_realizations": int(frac_all.size),
            "pixel_coverage_all_calibration": cov["pix_all"][0] / cov["pix_all"][1],
            "pixel_coverage_additional_only": cov["pix_175"][0] / cov["pix_175"][1],
            "realization_level_all_calibration": {
                "fraction_meeting_90pct_pixels": float((frac_all >= PIX_LEVEL).mean()),
                "median_pixel_coverage": float(np.median(frac_all)), "width": 2 * Q_all},
            "realization_level_additional_only": {
                "fraction_meeting_90pct_pixels": float((frac_175 >= PIX_LEVEL).mean()),
                "median_pixel_coverage": float(np.median(frac_175)), "width": 2 * Q_175},
        }
    res["certified_iid_coverage_reproduced"] = abs(
        res["test"]["iid_test"]["pixel_coverage_all_calibration"] - CERTIFIED["iid"]) < 1e-6

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
