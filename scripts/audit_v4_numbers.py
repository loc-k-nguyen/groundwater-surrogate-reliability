"""Check the numerical tables, current sources, references, and figure linkage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tex = args.manuscript.read_text(encoding="utf-8")
    summary = json.loads((ROOT / "figures/v4_final/v4_analysis.json").read_text())
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})
        if not condition:
            raise AssertionError(name)

    for source in summary["sources"]:
        path = ROOT / source["path"]
        check("source SHA-256: " + source["path"],
              hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"])
    for name, values in summary["taxonomy"].items():
        line = next(l for l in tex.splitlines() if l.startswith(name + " &")
                    and len(re.findall(r"\d+\.\d+", l)) == 6)
        reported = [float(n) for n in re.findall(r"\d+\.\d+", line)]
        expected = [values[r][s] for r in ("oracle","full_field") for s in ("variance","transport")]
        expected += [values["ssim"]["reference"], values["ssim"]["transport"]]
        check("taxonomy: " + name, reported == [round(v,3) for v in expected])
    accuracy = summary["ensemble_accuracy"]
    for label, key in [("Full-field SSIM","global_ssim_mean"), ("Oracle plume SSIM","plume_ssim_mean"),
                       ("Relative concentration-sum error","mass_error_mean"),
                       ("Oracle Gaussian CRPS","gaussian_crps_mean")]:
        line = next(l for l in tex.splitlines() if l.startswith(label + " &"))
        reported = [float(n) for n in re.findall(r"\d+\.\d+", line)]
        check("accuracy: " + key, reported == [round(accuracy[s][key],3) for s in ("iid","ood")])
    names = ["", "_hetero", "_fno", "_deeponet"]
    rd = [json.loads((ROOT / ("figures/v4_final/rd/obj3_realization_disjoint_control"+s+".json")).read_text())
          for s in names]
    for label, region in [("Variance AUROC, oracle score", "oracle"),
                          ("Variance AUROC, deployable score", "deployable")]:
        line = next(l for l in tex.splitlines() if l.startswith(label + " &"))
        reported = [float(n) for n in re.findall(r"\d+\.\d+", line)]
        check("two-draw control: " + region,
              reported == [round(d["twin_free_pooled"]["auroc_"+region]["auroc"],3) for d in rd])
    triage = summary["triage"]
    check("triage case/failure/tie counts", (triage["n_cases"],triage["n_failures"],triage["n_distinct_scores"]) == (380,74,65))
    for point in triage["operating_points"]:
        check("expected triage lift at " + str(point["reviewed_fraction"]),
              f'{point["lift"]:.2f}' in tex)
    check("triage AUROC", f'{triage["failure_auroc"]:.3f}' in tex)
    with (ROOT / "results/postfix_final/tier0/tier0_results_boundaryfix.csv").open() as handle:
        predicted = [r for r in csv.DictReader(handle) if r["method"]=="scp" and r["mode"]=="deployable" and r["alpha"]=="0.1"]
    line = next(l for l in tex.splitlines() if l.startswith("Pixel-pooled SCP (predicted region)"))
    expected = [float(r[k]) for r in predicted for k in ("coverage","mean_width")]
    check("predicted-region conformal row",
          [float(n) for n in re.findall(r"\d+\.\d+", line)] == [round(v,3) for v in expected])
    capacities = json.loads((ROOT / "figures/v4_final/model_capacity_v4.json").read_text())
    for name, count in capacities["counts"].items():
        check("capacity: " + name, f'{count["real_scalar_parameters_per_member"]/1e6:.3f}' in tex)
    for path, digest in capacities["sources"].items():
        check("capacity source hash: " + path, hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    bib = args.manuscript.with_name("references.bib").read_text()
    keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib)
    citations = [key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}",tex) for key in group.split(",")]
    check("bibliography keys are unique", len(keys)==len(set(keys)))
    check("all cited keys exist", set(citations) <= set(keys))
    labels = re.findall(r"\\label\{([^}]+)\}",tex)
    refs = re.findall(r"\\ref\{([^}]+)\}",tex)
    check("labels are unique", len(labels)==len(set(labels)))
    check("all references exist", set(refs) <= set(labels))
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",tex)
    check("five current figures", len(figures)==5 and all("_v4.pdf" in f for f in figures))
    for figure in figures:
        check("figure exists: "+figure,(args.manuscript.parent/"figures"/figure).is_file())
    check("unsupported significance removed", not re.search(r"\$p\s*[=<>]|95\\%.*interval|taxonomy-ci",tex))
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",tex,re.S).group(1)
    words = len(abstract.split())
    check("abstract no more than 150 words",words<=150)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checks":checks,"abstract_words":words,
                                     "bibliography_entries":len(keys),
                                     "scope":"current taxonomy, accuracy, control AUROCs, triage, predicted-region conformal, capacities, linkage; inherited prose and raw recomputation need separate checks"},
                                    indent=2),encoding="utf-8")
    print(f"{len(checks)} checks passed; abstract {words} words; {len(keys)} bibliography entries")


if __name__ == "__main__":
    main()
