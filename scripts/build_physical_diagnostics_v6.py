"""Reproduce finite-design physical/time diagnostics from certified FP32 scalars."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"unet_det": "Deep ensemble", "hetero": "Heteroscedastic", "fno": "Fourier operator", "deeponet": "DeepONet"}
COLORS = ["#1965a8", "#7b3294", "#d95f02", "#188977"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(a, b):
    return float(a / b) if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT or ROOT in out.parents or out.exists():
        raise FileExistsError("Use a new output directory outside the package")
    out.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    meta = {}
    sources = []
    for name in ("param_axisD_matched.json", "param_axisD_severity.json"):
        path = ROOT / "metadata" / name
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
        for row in json.loads(path.read_text()):
            meta[f"param_{row['param_id']}"] = row
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7), constrained_layout=True)
    ladder_fig, ladder_axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    table_rows = []
    for color, (family, label) in zip(COLORS, FAMILIES.items()):
        folder = ROOT / "results/orientation_v5/quality" / family
        completion = json.loads((folder / "COMPLETED.json").read_text())
        path = folder / "per_time.csv"
        assert sha(path) == completion["per_time_sha256"]
        assert sha(folder / "quality.json") == completion["quality_sha256"]
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
        sources.append({"path": (folder / "quality.json").relative_to(ROOT).as_posix(), "sha256": sha(folder / "quality.json")})
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 10700
        by_case = {}
        for row in rows:
            by_case.setdefault((row["split"], row["param_id"], row["real_id"]), []).append(int(row["timestep"]))
        assert len(by_case) == 428 and all(sorted(times) == list(range(25)) for times in by_case.values())
        summary[family] = {"regions": {}, "paired_ladder": {}}
        for split, n, display in (("iid_test", 110, "R"), ("ood_test", 270, "V"), ("transport_ladder", 12, "T")):
            selected = [r for r in rows if r["split"] == split and
                        (split != "transport_ladder" or meta[r["param_id"]]["long_dispersivity"] == 200.)]
            assert len(selected) == n * 25
            tp, fn, fp = [sum(int(r[key]) for r in selected) for key in ("true_positive_pixels", "false_negative_pixels", "false_positive_pixels")]
            peak = np.abs([float(r["signed_relative_peak_error"]) for r in selected])
            centroid = np.asarray([float(r["centroid_error_pixels"]) if r["centroid_error_pixels"] else np.nan for r in selected])
            finite = centroid[np.isfinite(centroid)]
            record = {"n_cases": n, "n_case_times": len(selected), "true_positive_pixels": tp, "false_negative_pixels": fn,
                      "false_positive_pixels": fp, "pooled_plume_recall": ratio(tp, tp + fn), "pooled_plume_precision": ratio(tp, tp + fp),
                      "absolute_relative_peak_error_q50": float(np.quantile(peak, .5)), "absolute_relative_peak_error_q95": float(np.quantile(peak, .95)),
                      "centroid_error_metres_q50": float(np.quantile(finite, .5) * 5) if len(finite) else None,
                      "centroid_finite_case_times": len(finite), "time_profiles": []}
            for time in range(25):
                pool = [r for r in selected if int(r["timestep"]) == time]
                p_tp, p_fn, p_fp = [sum(int(r[k]) for r in pool) for k in ("true_positive_pixels", "false_negative_pixels", "false_positive_pixels")]
                record["time_profiles"].append({"timestep": time, "n_cases": len(pool),
                                               "median_log_rmse": float(np.median([float(r["log_rmse"]) for r in pool])),
                                               "median_signed_relative_peak_error": float(np.median([float(r["signed_relative_peak_error"]) for r in pool])),
                                               "pooled_recall": ratio(p_tp, p_tp + p_fn), "pooled_precision": ratio(p_tp, p_tp + p_fp)})
            summary[family]["regions"][display] = record
            table_rows.append(f"{label} & {display} & {record['pooled_plume_recall']:.3f} & {record['pooled_plume_precision']:.3f} & {record['absolute_relative_peak_error_q95']:.2f} & {record['centroid_error_metres_q50']:.1f} " + r"\\")
            if display in {"R", "V"}:
                i = 0 if display == "R" else 1
                for j, key in enumerate(("median_log_rmse", "median_signed_relative_peak_error", "pooled_recall")):
                    axes[i, j].plot(range(25), [p[key] for p in record["time_profiles"]], color=color, label=label)
                axes[i, 2].plot(range(25), [p["pooled_precision"] for p in record["time_profiles"]], color=color, linestyle="--")
        data = json.loads((folder / "quality.json").read_text())[family]["splits"]["transport_ladder"]
        groups = {}
        for row in data:
            m = meta[row["param_id"]]
            key = (m["anisotropy"], m["log_variance"], m["trans_dispersivity_ratio"], row["real_id"])
            groups.setdefault(key, {})[float(m["long_dispersivity"])] = row
        assert len(groups) == 12
        for style, transverse, count in (("-", .1, 8), ("--", 1., 4)):
            subset = [values for key, values in groups.items() if key[2] == transverse]
            assert len(subset) == count and all(sorted(v) == [80., 100., 140., 200.] for v in subset)
            spreads = [float(np.ptp([v[d]["mean_total_std_log"] for d in (80., 100., 140., 200.)])) for v in subset]
            assert max(spreads) == 0.
            deltas = [float(np.mean([v[d]["mean_plume_ssim"] - v[80.]["mean_plume_ssim"] for v in subset])) for d in (80., 100., 140., 200.)]
            summary[family]["paired_ladder"][str(transverse)] = {"n_pairs": count, "max_full_field_score_spread": max(spreads), "ssim_change_from_80m": deltas}
            ladder_axes[0].plot([80, 100, 140, 200], deltas, color=color, linestyle=style, marker="o", markersize=3, label=label if transverse == .1 else None)
        ladder_axes[1].plot([80, 100, 140, 200], [1., 1., 1., 1.], color=color, linewidth=1)
    for i, regime in enumerate(("Low-support reference", "Variance shift")):
        for j, title in enumerate(("Median full-field log RMSE", "Median relative peak bias", "Pooled plume recall / precision")):
            axes[i, j].set_title(regime + "\n" + title, fontsize=9)
            axes[i, j].set_xlabel("Output timestep index")
            axes[i, j].grid(alpha=.2)
        axes[i, 1].axhline(0, color=".5", linewidth=.7)
        axes[i, 2].set_ylim(0, 1.03)
    axes[0, 0].legend(fontsize=7)
    ladder_axes[0].set(xlabel="Longitudinal dispersivity (m)", ylabel="Mean plume SSIM change from 80 m", title="Paired morphology response")
    ladder_axes[0].axhline(0, color=".5", linewidth=.7)
    ladder_axes[0].legend(fontsize=7)
    ladder_axes[1].set(xlabel="Longitudinal dispersivity (m)", ylabel="Full-field score / same-case score at 80 m", title="Prediction-only scores are unchanged", ylim=(.98, 1.02))
    ladder_axes[1].text(.5, .18, "8 pairs at transverse ratio 0.1\n4 pairs at transverse ratio 1.0\nExact zero spread in both groups", transform=ladder_axes[1].transAxes, ha="center", fontsize=9)
    for name, figure in (("fig_physical_time_v6", fig), ("fig_paired_transport_v6", ladder_fig)):
        figure.savefig(out / (name + ".pdf"))
        figure.savefig(out / (name + ".png"), dpi=180)
        plt.close(figure)
    (out / "physical_threshold_rows_v6.tex").write_text("\n".join(table_rows) + "\n", encoding="utf-8")
    (out / "physical_diagnostics_v6.json").write_text(json.dumps({"sources": sources, "families": summary,
        "scope": "Certified FP32 scalar aggregation, not new inference or independent geological replication",
        "grid_cell_metres": 5, "numerical_plume_cutoff": 1e-8, "case_times": 42800,
        "not_available": ["Absolute truth/prediction peaks", "Pixel maps", "Concentration-regulatory criterion"]}, indent=2) + "\n", encoding="utf-8")
    print("PHYSICAL_V6_SCALAR_GATES_PASS: 42800 records; both ratio groups; no new inference")


if __name__ == "__main__":
    main()
