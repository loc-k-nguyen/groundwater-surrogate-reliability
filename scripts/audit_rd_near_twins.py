"""Near-twin check for the realization-disjoint control (CPU only).

The control trains and validates on realizations 1-3. For each evaluation field, this reports the
maximum standardized log10 K correlation against every field the control saw (training plus
validation-calibration), separately for the cells
whose exact twin was trained on (correlation length 20 and 50) and those whose exact twin was not
(correlation length 500), and for realizations 1-3 against 4-5.

It establishes that realization identity, not the cell, decides whether a pattern was seen: the
realization 1-3 fields of the correlation-length 500 cells are near-twins of training fields
(r around 0.98) even though no exact twin was trained on, while realizations 4-5 are unseen in
every cell. Reads the raw conductivity fields, so it is shipped for inspection. Read-only.
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
    import numpy as np

    REPO = args.repo_root
    DATA = asset_path("data", REPO / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL/T25_TSTEP_OVERRIDE_FINAL_FLIPPED")
    OUT = args.output
    REG = {int(r["param_id"]): r for r in csv.DictReader(open(metadata_path("param_registry_master.csv")))}
    SPLIT = json.load(open(REPO / "splits/param_split_obj3_ood.json"))


    def standardized(ids, reals, keep=lambda p: True):
        rows = []
        for p in sorted(ids):
            if not keep(p):
                continue
            for r in reals:
                f = DATA / f"param_{p:03d}" / f"real_{r:03d}.npz"
                if f.exists():
                    v = np.log10(np.maximum(np.load(f)["K"], 1e-30)).ravel().astype(np.float64)
                    rows.append((v - v.mean()) / v.std())
        return np.stack(rows)


    # Everything the control saw: training fields and the validation-calibration fields used for
    # checkpoint selection, realizations 1-3 of each.
    train = standardized(SPLIT["train"] + SPLIT["val_calib"], [1, 2, 3])
    res = {}
    for label, corr in (("corr 20/50 (exact twin in training)", {"20.0", "50.0"}),
                        ("corr 500 (exact or near twin in validation-calibration or other anisotropy)", {"500.0"})):
        for reals in ([1, 2, 3], [4, 5]):
            ev = standardized(SPLIT["ood_test"], reals, lambda p: REG[p]["correlation_length"] in corr)
            m = (ev @ train.T / train.shape[1]).max(1)
            key = f"{label}, realizations {reals}"
            res[key] = {"n": int(len(m)), "max_r_median": float(np.median(m)),
                        "max_r_min": float(m.min()), "max_r_max": float(m.max())}
            print(f"{key}: n={len(m)} median {np.median(m):.4f} (min {m.min():.4f}, max {m.max():.4f})")
    # Reference (iid_test) pool: the twin-free filter must hold there too.
    for reals in ([1, 2, 3], [4, 5]):
        ev = standardized(SPLIT["iid_test"], reals)
        m = (ev @ train.T / train.shape[1]).max(1)
        key = f"reference pool (iid_test), realizations {reals}"
        res[key] = {"n": int(len(m)), "max_r_median": float(np.median(m)),
                    "max_r_min": float(m.min()), "max_r_max": float(m.max())}
        print(f"{key}: n={len(m)} median {np.median(m):.4f} (min {m.min():.4f}, max {m.max():.4f})")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == '__main__':
    main()
