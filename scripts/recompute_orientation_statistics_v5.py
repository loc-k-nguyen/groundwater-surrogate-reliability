"""CPU-only, one-case-at-a-time reporting from certified orientation caches."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

from package_paths import enable_package_imports

if __name__ == "__main__":
    enable_package_imports()

import numpy as np

from src.obj3.journal.aci import EPS_ALPHA, conformal_quantile
from src.obj3.journal.crc import solve_crc_lambda, evaluate_crc_solution

ALPHAS = (0.05, 0.10, 0.20)
THRESHOLD = 1e-8
CERT_SHA = "fd011bb76d00829df0c17a87847927d3c4c5271f5a43d7c15d22db88baf89637"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def predicted_mask(log_prediction):
    return log_prediction > np.log10(THRESHOLD)


def sorted_quantile(scores, alpha):
    """Exactly match NumPy's higher rule on an already sorted score pool."""
    n = len(scores)
    if n == 0 or not 0 < alpha < 1:
        raise ValueError("Empty scores or invalid alpha")
    level = float(np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0, 1))
    return float(scores[int(np.ceil(level * (n - 1)))])


def cases(path, expected, limit=0):
    keys = ("y_true", "y_pred", "y_pred_std", "c_phys")
    with zipfile.ZipFile(path) as z:
        handles = [z.open(k + ".npy") for k in keys]
        try:
            headers = []
            for f in handles:
                headers.append(np.lib.format._read_array_header(f, np.lib.format.read_magic(f)))
            shape, fortran, dtype = headers[0]
            if any(h != headers[0] for h in headers) or fortran or dtype != np.dtype("float32"):
                raise ValueError("Cache header mismatch")
            if shape != (expected, 25, 600, 400):
                raise ValueError(f"Unexpected cache shape {shape}")
            count = int(np.prod(shape[1:]))
            for i in range(min(expected, limit) if limit else expected):
                arrays = []
                for f in handles:
                    b = f.read(count * 4)
                    if len(b) != count * 4:
                        raise ValueError("Truncated cache")
                    a = np.frombuffer(b, dtype=dtype).reshape(shape[1:])
                    if not np.isfinite(a).all():
                        raise ValueError("Nonfinite cache values")
                    arrays.append(a)
                if arrays[2].min() < 0:
                    raise ValueError("Negative standard deviation")
                yield i, dict(zip(keys, arrays))
        finally:
            for f in handles:
                f.close()


def calibrate(entry, split, out, limit):
    records, flagged, risks = [], [], []
    for i, d in cases(entry["path"], len(entry["order"]), limit):
        p = Path(entry["order"][i])
        pid = int(p.parent.name.split("_")[-1])
        err = np.abs(d["y_true"] - d["y_pred"])
        oracle, predicted = d["c_phys"] > THRESHOLD, predicted_mask(d["y_pred"])
        maximum = float(np.abs(d["c_phys"]).max())
        r = {"pid": pid, "extra": pid not in split["val_calib"], "bad": maximum > 10,
             "severe": maximum > 1000, "real": p.stem,
             "oracle": err[oracle], "predicted": err[predicted],
             "oracle_std": d["y_pred_std"][oracle], "predicted_std": d["y_pred_std"][predicted]}
        records.append(r)
        risks.append(float(np.mean(r["oracle"], dtype=np.float32)))
        if r["bad"]:
            flagged.append({"case": str(p), "max_abs_concentration": maximum, "severe": r["severe"]})
        if (i + 1) % 25 == 0:
            print(f"calibration cases {i + 1}", flush=True)
    result = {"n_cases": len(records), "flagged_cases": flagged, "variants": {}}
    predicates = {"all": lambda r: True, "extra_only": lambda r: r["extra"],
                  "without_severe": lambda r: not r["severe"],
                  "without_flagged": lambda r: not r["bad"]}
    sorted_oracle = None
    for variant, select in predicates.items():
        selected = [r for r in records if select(r)]
        if not selected:
            continue
        v = {"n_cases": len(selected), "regions": {}}
        for mode in ("oracle", "predicted"):
            residuals = np.concatenate([r[mode] for r in selected])
            std = np.concatenate([r[mode + "_std"] for r in selected])
            positive = std[std > 0]
            floor = max(1e-4, float(np.quantile(positive, .1))) if len(positive) else 1e-4
            del positive
            normalized = residuals / np.maximum(std, floor)
            field_scores = np.array([np.quantile(r[mode], .9) for r in selected if len(r[mode])])
            values = {"n_pixels": len(residuals), "sigma_floor": floor,
                      "scp": {str(a): conformal_quantile(residuals, a) for a in ALPHAS},
                      "nec": {str(a): conformal_quantile(normalized, a) for a in ALPHAS},
                      "field90": {str(a): conformal_quantile(field_scores, a) for a in ALPHAS}}
            v["regions"][mode] = values
            if variant == "all" and mode == "oracle":
                sorted_oracle = np.sort(residuals)
                np.save(out / "sorted_oracle_scores.npy", sorted_oracle, allow_pickle=False)
            del residuals, std, normalized
        result["variants"][variant] = v
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    return result, sorted_oracle, np.asarray(risks, dtype=np.float32)


def evaluate(entry, cal, sorted_scores, limit):
    totals, field_targets, physical, risks = {}, {}, [], []
    aci = {(a, g): {"current": a, "rows": []} for a in ALPHAS for g in (.01, .02, .05, .10)}
    selection = {"true": 0, "predicted": 0, "intersection": 0}
    for i, d in cases(entry["path"], len(entry["order"]), limit):
        yt, yp, std, cp = (d[k] for k in ("y_true", "y_pred", "y_pred_std", "c_phys"))
        oracle, predicted = cp > THRESHOLD, predicted_mask(yp)
        masks = {"oracle": oracle, "predicted": predicted, "union": oracle | predicted,
                 "global": np.ones_like(oracle)}
        err = np.abs(yt - yp)
        risks.append(float(err[oracle].mean()))
        for name in ("true", "predicted", "intersection"):
            m = oracle if name == "true" else predicted if name == "predicted" else oracle & predicted
            selection[name] += int(m.sum())
        for variant, v in cal["variants"].items():
            for region, params in v["regions"].items():
                for a in ALPHAS:
                    for method in ("scp", "nec", "field90"):
                        q = params[method][str(a)]
                        half = q * np.maximum(std, params["sigma_floor"]) if method == "nec" else q
                        covered = (yt >= yp - half) & (yt <= yp + half)
                        key = f"{variant}/{region}/{method}/{a}"
                        accumulator = totals.setdefault(key, {})
                        for mask_name, m in masks.items():
                            n = int(m.sum())
                            slot = accumulator.setdefault(mask_name, [0, 0, 0.0])
                            slot[0] += int(covered[m].sum())
                            slot[1] += n
                            slot[2] += float(2 * half * n) if np.isscalar(half) else float((2 * half[m]).sum(dtype=np.float64))
                        if method == "field90":
                            field_targets.setdefault(key, []).append(float(covered[masks[region]].mean()) >= .9)
        for (a, g), state in aci.items():
            q = sorted_quantile(sorted_scores, state["current"])
            covered = (yt >= yp - q) & (yt <= yp + q)
            coverage = float(covered[oracle].mean())
            miss = float(coverage < 1 - a)
            state["rows"].append([coverage, float(covered.mean()), 2 * q, miss])
            state["current"] = float(np.clip(state["current"] + g * (a - miss), EPS_ALPHA, 1 - EPS_ALPHA))
        for t in range(25):
            gt = np.maximum(cp[t], 0)
            pred = np.maximum(10 ** yp[t] - 1e-12, 0)
            delta = yp[t] - yt[t]
            signed = float((pred.sum(dtype=np.float64) - gt.sum(dtype=np.float64)) / max(gt.sum(dtype=np.float64), 1e-12))
            om, pm = oracle[t], predicted[t]
            row = {"case_index": i, "timestep": t, "log_mae": float(np.abs(delta).mean()),
                   "log_rmse": float(np.sqrt(np.mean(delta ** 2))), "log_bias": float(delta.mean()),
                   "signed_relative_concentration_sum_error": signed, "absolute_relative_concentration_sum_error": abs(signed),
                   "true_plume_pixels": int(om.sum()), "predicted_plume_pixels": int(pm.sum()),
                   "missed_true_plume_fraction": float((om & ~pm).sum() / max(om.sum(), 1)),
                   "false_positive_pixels": int((pm & ~om).sum()),
                   "relative_peak_error": float((pred.max() - gt.max()) / max(gt.max(), 1e-12))}
            physical.append(row)
        if (i + 1) % 25 == 0:
            print(f"{entry['name']} cases {i + 1}", flush=True)
    results = {k: {m: {"coverage": x[0] / x[1] if x[1] else None,
                       "width": x[2] / x[1] if x[1] else None, "n_pixels": x[1]}
                   for m, x in values.items()} for k, values in totals.items()}
    for k, values in field_targets.items():
        results[k]["fraction_fields_meeting_90pct"] = float(np.mean(values))
    selection["true_plume_recall"] = selection["intersection"] / selection["true"]
    selection["missed_true_plume_fraction"] = 1 - selection["true_plume_recall"]
    online = {f"{a}/{g}": {"final_alpha": state["current"],
              "mean_oracle_pixel_coverage": float(np.mean(state["rows"], axis=0)[0]),
              "mean_global_pixel_coverage": float(np.mean(state["rows"], axis=0)[1]),
              "mean_width": float(np.mean(state["rows"], axis=0)[2]),
              "field_miss_rate": float(np.mean(state["rows"], axis=0)[3])}
              for (a, g), state in aci.items()}
    return {"coverage": results, "selection": selection, "aci_posthoc_sweep": online,
            "physical_per_case_time": physical}, np.asarray(risks, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--certificate", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists; preserve previous run and use a new path")
    if sha(args.certificate) != CERT_SHA:
        raise ValueError("Pinned cache certificate hash differs")
    certificate = json.loads(args.certificate.read_text())
    if certificate["status"] != "ORIENTATION_CACHES_CERTIFIED":
        raise ValueError("Certificate did not pass")
    args.output.mkdir(parents=True)
    started = time.time()
    (args.output / "RUNNING.json").write_text(json.dumps({"status": "RUNNING", "limit": args.limit}))
    cal_entry = next(c for c in certificate["caches"] if c["name"] == "calibration")
    split = json.loads(args.split.read_text())
    cal, scores, cal_risks = calibrate(cal_entry, split, args.output, args.limit)
    crc = {str(b): solve_crc_lambda(cal_risks, b) for b in (.05, .10, .20)}
    for entry in certificate["caches"]:
        if entry["name"] == "calibration":
            continue
        result, risks = evaluate(entry, cal, scores, args.limit)
        result["crc_descriptive_only_unbounded_loss"] = {b: {"calibration": asdict(s), "test": evaluate_crc_solution(risks, s)} for b, s in crc.items()}
        (args.output / (entry["name"] + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False))
    files = {f.name: sha(f) for f in args.output.glob("*.json") if f.name != "RUNNING.json"}
    result = {"status": "SMOKE_DONE" if args.limit else "ORIENTATION_STATISTICS_DONE",
              "certificate_sha256": sha(args.certificate), "split_sha256": sha(args.split),
              "script_sha256": sha(__file__), "input_cache_certificates": certificate["caches"],
              "files": files, "elapsed_seconds": time.time() - started,
              "predicted_physical_threshold": THRESHOLD, "predicted_log_threshold": -8.0,
              "scope": "Existing U-Net ensemble caches only; not four-family accuracy or physical solver validation.",
              "limitations": ["Pixel and geological dependence: no distribution-free guarantee.",
                              "ACI sweep is post hoc, not an independently validated optimum.",
                              "QC exclusion is sensitivity only; no frozen data changed.",
                              "Conditional/severity analysis and selective reporting remain separate gates."]}
    (args.output / "COMPLETED.json").write_text(json.dumps(result, indent=2))
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
