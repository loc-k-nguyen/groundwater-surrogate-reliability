"""Reproduce corrected v5 finite-design summaries and figures from certified sources."""
from __future__ import annotations

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
FAMILIES = {
    "Deep ensemble (U-Net)": "det_ensemble",
    "Heteroscedastic (U-Net)": "hetero",
    "Fourier operator": "fno",
    "DeepONet": "deeponet",
}
COLORS = ["#1965a8", "#7b3294", "#d95f02", "#188977"]


def auroc(negative, positive):
    n, p = np.asarray(negative), np.asarray(positive)
    return float(np.mean((p[:, None] > n).astype(float) + .5 * (p[:, None] == n)))


def tie_curve(scores, failures):
    """Expected random review within exact ties, with attainable tie-order bounds."""
    scores, failures = np.asarray(scores), np.asarray(failures, dtype=bool)
    if scores.ndim != 1 or scores.shape != failures.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and failures must be finite aligned vectors")
    total = int(failures.sum())
    if not total:
        raise ValueError("At least one failure is required")
    expected, lower, upper = [0.], [0.], [0.]
    completed = 0
    blocks = []
    for score in np.unique(scores)[::-1]:
        block = failures[scores == score]
        size, count = len(block), int(block.sum())
        blocks.append({"score": float(score), "n_cases": size, "n_failures": count})
        for k in range(1, size + 1):
            expected.append(completed + k * count / size)
            lower.append(completed + max(0, k - (size - count)))
            upper.append(completed + min(k, count))
        completed += count
    return np.asarray(expected) / total, np.asarray(lower) / total, np.asarray(upper) / total, blocks


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Use an empty destination; existing outputs are never replaced")
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.code_root.resolve()
    args.quality_root = (args.quality_root or ROOT / "results/orientation_v5/quality").resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    sources = []
    taxonomy, distributions, ladders = {}, {}, {}
    for name, key in FAMILIES.items():
        arch = "unet_det" if key == "det_ensemble" else key
        folder = args.quality_root / arch
        quality_file = folder / "quality.json"
        completed = json.loads((folder / "COMPLETED.json").read_text())
        if completed["status"] != "QUALITY_EVALUATION_DONE" or hashlib.sha256(quality_file.read_bytes()).hexdigest() != completed["quality_sha256"]:
            raise ValueError(f"Quality certificate failed: {arch}")
        sources += [quality_file]
        doc = json.loads(quality_file.read_text())[arch]
        groups = {
            "reference": doc["splits"]["iid_test"],
            "variance": doc["splits"]["ood_test"],
            "transport": [r for r in doc["splits"]["transport_ladder"] if 9210 <= int(r["param_id"].split("_")[-1]) <= 9215],
        }
        assert [len(groups[k]) for k in groups] == [110, 270, 12]
        taxonomy[name] = {}
        for region, column in [("oracle", "plume_total_std_log"), ("full_field", "mean_total_std_log")]:
            values = {s: [float(r[column]) for r in g] for s, g in groups.items()}
            taxonomy[name][region] = {s: auroc(values["reference"], values[s])
                                       for s in ("variance", "transport")}
        taxonomy[name]["ssim"] = {s: float(np.mean([float(r["mean_plume_ssim"]) for r in g]))
                                  for s, g in groups.items()}
        distributions[name] = {s: [float(r["plume_total_std_log"]) for r in g]
                                for s, g in groups.items()}
        lr = doc["splits"]["transport_ladder"]
        ladders[name] = {}
        for region, column in [("oracle", "plume_total_std_log"), ("full_field", "mean_total_std_log")]:
            ref = [float(r[column]) for r in groups["reference"]]
            values = []
            for level, ids in [(80, range(9220, 9226)), (100, range(9226, 9232)),
                               (140, range(9232, 9238)), (200, range(9210, 9216))]:
                positive = [float(r[column]) for r in lr if int(r["param_id"].split("_")[-1]) in ids]
                assert len(positive) == 12
                values.append({"alpha_L_m": level, "auroc": auroc(ref, positive)})
            ladders[name][region] = values

    stats_path = ROOT / "results/reporting_statistics/reporting_statistics.json"
    sources.append(stats_path)
    threshold = json.loads(stats_path.read_text())["failure_threshold_plume_ssim"]
    pooled, accuracy = [], {}
    for split in ("iid", "ood"):
        path = ROOT / f"results/postfix_final/ensemble/eval/ensemble_metrics_{split}_test.json"
        sources.append(path)
        doc = json.loads(path.read_text())
        pooled += doc["per_sample"]
        accuracy[split] = {k: v for k, v in doc["aggregate"].items()
                           if k.startswith(("mass_error", "global_ssim", "plume_ssim", "gaussian_crps"))}
    scores = np.array([r["mean_ensemble_var"] for r in pooled], dtype=float)
    fail = np.array([r["mean_plume_ssim"] < threshold for r in pooled])
    expected, low, high, blocks = tie_curve(scores, fail)
    points = []
    for fraction in (.05, .1, .2, .3, .5):
        k = round(fraction * len(scores))
        points.append({"reviewed_fraction": k / len(scores), "n_reviewed": k,
                       "expected_failures_caught": float(expected[k] * fail.sum()),
                       "expected_recall": float(expected[k]),
                       "recall_min": float(low[k]), "recall_max": float(high[k]),
                       "expected_precision": float(expected[k] * fail.sum() / k),
                       "lift": float(expected[k] / (k / len(scores))),
                       "lift_min": float(low[k] / (k / len(scores))),
                       "lift_max": float(high[k] / (k / len(scores)))})
    triage = {"n_cases": len(scores), "n_failures": int(fail.sum()),
              "n_distinct_scores": len(blocks), "threshold": threshold,
              "failure_auroc": auroc(scores[~fail], scores[fail]),
              "tie_policy": "uniform random selection within exact score ties",
              "bound_interpretation": "attainable tie-order bounds, not confidence intervals",
              "operating_points": points, "tie_blocks": blocks}
    summary = {"interpretation": "finite deterministic design; no population inference",
               "taxonomy": taxonomy, "transport_ladder": ladders, "triage": triage,
               "ensemble_accuracy": accuracy,
               "sources": [{"path": p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT)
                            else "quality_v5/" + p.relative_to(args.quality_root).as_posix(),
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
    (out / "v5_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=True, layout="constrained")
    for ax, region, title in zip(axes, ("oracle", "full_field"),
                                ("Oracle\nplume-region score", "Prediction-time\nfull-field score")):
        vals = np.array([[taxonomy[n][region][s] for s in ("variance", "transport")] for n in FAMILIES])
        im = ax.imshow(vals, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1], ["Variance\nshift", "Transport\npool"])
        ax.set_yticks(range(4), ["Deep ensemble\n(U-Net)", "Heteroscedastic\n(U-Net)",
                                "Fourier operator", "DeepONet"])
        ax.set_title(title)
        for i in range(4):
            for j in range(2):
                ax.text(j, i, f"{vals[i,j]:.3f}", ha="center", va="center",
                        color="white" if vals[i,j] < .35 else "black", fontweight="semibold")
    cbar = fig.colorbar(im, ax=axes, fraction=.035, pad=.02, label="AUROC vs low-support reference")
    cbar.set_ticks([0, .5, 1], labels=["0", "0.5 (chance)", "1"])
    fig.savefig(out / "fig3_taxonomy_v5.pdf")
    fig.savefig(out / "fig3_taxonomy_v5.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.8), layout="constrained")
    for ax, (name, values), color in zip(axes.flat, distributions.items(), COLORS):
        groups = [values[s] for s in ("reference", "variance", "transport")]
        ax.boxplot(groups, tick_labels=["Reference", "Variance", "Transport"], showfliers=False)
        for j, values in enumerate(groups, 1):
            offsets = np.linspace(-.17, .17, len(values))
            ax.scatter(j + offsets, values, s=7, color=color, alpha=.4, zorder=3)
        ax.set_title(name.replace(" (U-Net)", "\n(U-Net)"))
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Oracle plume uncertainty\n(log units)")
    fig.savefig(out / "fig4_distributions_v5.pdf")
    fig.savefig(out / "fig4_distributions_v5.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=True, layout="constrained")
    for ax, region in zip(axes, ("oracle", "full_field")):
        for (name, data), color, marker in zip(ladders.items(), COLORS, ("o", "s", "^", "D")):
            ax.plot([r["alpha_L_m"] for r in data[region]], [r["auroc"] for r in data[region]],
                    marker=marker, color=color, label=name, linewidth=1.5)
        ax.axhline(.5, color=".5", linestyle="--", linewidth=1)
        ax.axvline(62, color=".5", linestyle=":", linewidth=1)
        ax.set_ylim(0, 1.035)
        ax.set_xlabel("Longitudinal dispersivity (m)")
        ax.set_title("Oracle plume-region score" if region == "oracle" else "Prediction-time full-field score", fontsize=8)
    axes[0].set_ylabel("AUROC vs low-support reference")
    axes[0].text(64, .04, "trained maximum", fontsize=8)
    axes[1].legend(fontsize=7.6, loc="upper right")
    fig.savefig(out / "fig2_transport_ladder_v5.pdf")
    fig.savefig(out / "fig2_transport_ladder_v5.png", dpi=200)
    plt.close(fig)

    x = np.arange(len(scores) + 1) / len(scores)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), layout="constrained")
    axes[0].fill_between(x, low, high, color="#1965a8", alpha=.18, label="Tie-order bounds (not sampling CI)")
    axes[0].plot(x, expected, color="#1965a8", label="Expected within-tie review")
    axes[0].plot([0, 1], [0, 1], "--", color=".5", label="Random review")
    axes[0].set(xlim=(0,1), ylim=(0,1), xlabel="Fraction of cases reviewed",
                ylabel="Fraction of failures recovered", title=f"Finite-pool ranking (AUROC {triage['failure_auroc']:.3f})")
    axes[0].legend(fontsize=7, loc="lower right")
    ys = np.array([p["lift"] for p in points])
    err = np.array([[p["lift"]-p["lift_min"] for p in points],
                    [p["lift_max"]-p["lift"] for p in points]])
    axes[1].bar(range(5), ys, yerr=err, capsize=4, width=.55, color="#1965a8")
    axes[1].axhline(1, linestyle=":", color=".5")
    axes[1].set_xticks(range(5), ["5%", "10%", "20%", "30%", "50%"])
    axes[1].set(xlabel="Fraction of cases reviewed", ylabel="Expected recall / random recall",
                title="Tie-aware illustrative triage")
    axes[1].set_ylim(0, max(p["lift_max"] for p in points) * 1.2)
    for i,y in enumerate(ys):
        axes[1].text(i, points[i]["lift_max"]+.04, f"{y:.2f}x", ha="center", fontsize=8)
    fig.savefig(out / "fig5_triage_v5.pdf")
    fig.savefig(out / "fig5_triage_v5.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"n_cases": triage["n_cases"], "n_failures": triage["n_failures"],
                      "operating_points": points, "ensemble_accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
