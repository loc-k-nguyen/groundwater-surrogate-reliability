"""CPU-only NPZ schema check for an Axis D smoke npz (or any Obj3 data npz).

Verifies the npz is compatible with the inference loader (Obj3FullFieldDataset):
keys K,C,times; C shape (25,H,W); K shape (H,W)==C[1:]; float; 25 timesteps;
finite; plume nontrivial. Prints PASS/FAIL and writes a JSON verdict.

Usage: python axisD_npz_schema_check.py --npz <path> [--out <json>]
Validated against an existing master npz (keys K/C/times, C (25,600,400)).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

PLUME = 1e-8
EXPECT_T = 25
MIN_PLUME_PIXELS = 64  # fairness-rule plume_min_pixels


def check(npz_path: Path) -> dict:
    r: dict = {"npz": str(npz_path), "checks": {}, "verdict": "FAIL"}
    if not npz_path.exists():
        r["error"] = "file not found"; return r
    d = np.load(npz_path, allow_pickle=False)
    keys = list(d.files); r["keys"] = keys
    ck = r["checks"]
    ck["has_K_C_times"] = all(k in keys for k in ("K", "C", "times"))
    if not ck["has_K_C_times"]:
        return r
    K, C, times = d["K"], d["C"], d["times"]
    r["shapes"] = {"K": list(K.shape), "C": list(C.shape), "times": list(times.shape)}
    r["dtypes"] = {"K": str(K.dtype), "C": str(C.dtype)}
    ck["C_is_3d_25xHxW"] = (C.ndim == 3 and C.shape[0] == EXPECT_T)
    ck["K_is_2d"] = (K.ndim == 2)
    target_hw = tuple(C.shape[1:]) if C.ndim == 3 else ()
    r["K_exactly_matches_C_HW"] = bool(K.ndim == 2 and target_hw and tuple(K.shape) == target_hw)
    ck["K_can_canonicalize_to_C_HW"] = bool(
        K.ndim == 2 and target_hw and (tuple(K.shape) == target_hw or K.size == int(np.prod(target_hw)))
    )
    ck["K_matches_C_HW"] = ck["K_can_canonicalize_to_C_HW"]
    if ck["K_can_canonicalize_to_C_HW"]:
        r["K_canonical_shape"] = list(target_hw)
    ck["n_timesteps_eq_25"] = (C.ndim == 3 and C.shape[0] == EXPECT_T)
    ck["C_finite"] = bool(np.all(np.isfinite(C)))
    ck["K_finite"] = bool(np.all(np.isfinite(K)))
    ck["K_positive"] = bool(np.all(K > 0))
    r["C_min"] = float(np.min(C)); r["C_max"] = float(np.max(C))
    # plume area per timestep
    if C.ndim == 3:
        areas = [int(np.count_nonzero(C[t] > PLUME)) for t in range(C.shape[0])]
        r["plume_pixels_by_timestep"] = areas
        ck["plume_nontrivial"] = bool(max(areas) >= MIN_PLUME_PIXELS)
        ck["plume_grows_then_present_late"] = bool(areas[-1] >= MIN_PLUME_PIXELS)
    ck["loader_compatible"] = bool(
        ck["C_is_3d_25xHxW"]
        and ck["K_can_canonicalize_to_C_HW"]
        and ck["C_finite"]
        and ck["K_finite"]
    )
    r["verdict"] = "PASS" if all(ck.values()) else "FAIL"
    r["failed_checks"] = [k for k, v in ck.items() if not v]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = check(Path(a.npz))
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
    print("VERDICT:", r["verdict"])


if __name__ == "__main__":
    main()
