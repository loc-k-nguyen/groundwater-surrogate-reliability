"""Check current manuscript tables against certified corrected reporting artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def numbers(line):
    return [float(n) for n in re.findall(r"-?\d+\.\d+", line)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manuscript", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Preserve previous audits")
    root = a.artifact_root
    tex = a.manuscript.read_text(encoding="utf-8")
    stats = load(root / "certification/v5_reporting_certification.json")
    gates = load(root / "certification/v5_reporting_gates.json")
    analysis = load(root / "figures_v5_verified_repair2/v5_analysis.json")
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)

    for report in (stats, gates):
        for path, digest in report["source_files"].items():
            check("source hash: " + path, hashlib.sha256((root / path).read_bytes()).hexdigest() == digest)
    for name, values in analysis["taxonomy"].items():
        line = next(line for line in tex.splitlines() if line.startswith(name + " &") and len(numbers(line)) == 6)
        expected = [values[r][s] for r in ("oracle", "full_field") for s in ("variance", "transport")]
        expected += [values["ssim"][s] for s in ("reference", "transport")]
        check("taxonomy: " + name, numbers(line) == [round(x, 3) for x in expected])
    code = Path(__file__).resolve().parents[1]
    for label, metric in (("Full-field SSIM", "global_ssim_mean"), ("Oracle plume SSIM", "plume_ssim_mean"),
                          ("Relative concentration-sum error", "mass_error_mean"), ("Oracle Gaussian CRPS", "gaussian_crps_mean")):
        line = next(line for line in tex.splitlines() if line.startswith(label + " &"))
        check("native-cache accuracy: " + metric, numbers(line) == [round(analysis["ensemble_accuracy"][s][metric], 3) for s in ("iid", "ood")])
    controls = [load(code / ("results/twin_audit/obj3_realization_disjoint_control" + suffix + ".json"))
                for suffix in ("", "_hetero", "_fno", "_deeponet")]
    for label, region in (("Variance AUROC, oracle score", "oracle"), ("Variance AUROC, deployable score", "deployable")):
        line = next(line for line in tex.splitlines() if line.startswith(label + " &"))
        check("inherited two-draw control: " + region, numbers(line) == [round(d["twin_free_pooled"]["auroc_" + region]["auroc"], 3) for d in controls])
    reference = load(code / "figures/v5_corrected/v5_analysis_portable.json")
    check("native triage matches shipped corrected reference", analysis["triage"] == reference["triage"])
    triage = analysis["triage"]
    check("review counts", (triage["n_cases"], triage["n_failures"], triage["n_distinct_scores"]) == (380, 74, 65))
    for point in triage["operating_points"]:
        check("expected review lift: " + str(point["reviewed_fraction"]), f'{point["lift"]:.2f}' in tex)
    blocks = re.search(r"\\label\{tab:physical\}(.*?)\\end\{table\}", tex, re.S).group(1)
    lines = [line for line in blocks.splitlines() if " & " in line and len(numbers(line)) == 6]
    check("four-family physical rows", len(lines) == 12)
    metrics = ("log_mae", "log_rmse", "signed_log_bias", "signed_relative_integrated_concentration_error",
               "absolute_relative_integrated_concentration_error", "centroid_error_pixels")
    expected_rows = []
    for family in ("unet_det", "hetero", "fno", "deeponet"):
        for split in ("iid_test", "ood_test", "transport_primary_200m"):
            values = stats["quality"][family]["splits"][split]["case_mean"]
            expected_rows.append([round(values[m]["mean"], 2 if m == "centroid_error_pixels" else 3) for m in metrics])
    check("physical values including signed errors", [numbers(line) for line in lines] == expected_rows)
    methods = (("Realization-level (oracle)", "extra_only/oracle/field90/0.1", True),
               ("Pixel-pooled SCP", "extra_only/oracle/scp/0.1", False),
               ("Pixel-pooled SCP", "all/oracle/scp/0.1", False),
               ("Normalized ensemble (NEC)", "all/oracle/nec/0.1", False))
    table = re.search(r"\\label\{tab:conformal\}(.*?)\\end\{table\}", tex, re.S).group(1)
    rows = [line for line in table.splitlines() if " & " in line and len(numbers(line)) == 4]
    check("conformal row count", len(rows) == 7)
    for index, (label, key, field) in enumerate(methods):
        expected = []
        for split in ("iid_test", "ood_test"):
            result = stats["coverage"][split]["coverage"][key]
            expected += [round(result["fraction_fields_meeting_90pct"] if field else result["oracle"]["coverage"], 3),
                         round(result["oracle"]["width"], 2)]
        check("conformal: " + key, rows[index].startswith(label + " &") and numbers(rows[index]) == expected)
    mondrian = [r for r in gates["conditional_rows"] if r["method"] == "mondrian" and r["mode"] == "oracle" and r["alpha"] == "0.1"]
    check("oracle-regime row", numbers(rows[4]) == [round(float(r[k]), 3 if k == "coverage" else 2)
          for r in mondrian for k in ("coverage", "mean_width")])
    sweep = stats["coverage"]["ood_test"]["aci_posthoc_sweep"]
    selected = min((.01, .02, .05, .1), key=lambda g: (abs(sweep[f"0.1/{g}"]["mean_oracle_pixel_coverage"] - .9), sweep[f"0.1/{g}"]["mean_width"]))
    expected = []
    for split in ("iid_test", "ood_test"):
        result = stats["coverage"][split]["aci_posthoc_sweep"][f"0.1/{selected}"]
        expected += [round(result["mean_oracle_pixel_coverage"], 3), round(result["mean_width"], 2)]
    check("ACI posthoc selected row", numbers(rows[5]) == expected and selected == .1)
    expected = []
    for split in ("iid_test", "ood_test"):
        result = stats["coverage"][split]["coverage"]["all/predicted/scp/0.1"]["predicted"]
        expected += [round(result["coverage"], 3), round(result["width"], 2)]
    check("physical predicted-positive row", numbers(rows[6]) == expected)
    bib = a.manuscript.with_name("references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib)
    citations = [key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex) for key in group.split(",")]
    check("unique bibliography keys", len(keys) == len(set(keys)))
    check("cited keys exist", set(citations) <= set(keys))
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    check("unique labels", len(labels) == len(set(labels)))
    check("references resolve", set(re.findall(r"\\ref\{([^}]+)\}", tex)) <= set(labels))
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    check("current figures", len(figures) == 5 and all("_v5.pdf" in f for f in figures))
    for figure in figures:
        check("figure exists: " + figure, (a.manuscript.parent / "figures" / figure).is_file())
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    check("abstract word limit", len(abstract.split()) <= 150)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"checks": checks, "scope": "Current taxonomy, physical/conformal/native-cache accuracy tables, inherited control AUROC rows, exact unchanged triage, source checksums and linkage. Other prose/control rows, citation accuracy and visual QA require separate checks."}, indent=2), encoding="utf-8")
    print(f"{len(checks)} checks passed")


if __name__ == "__main__":
    main()
