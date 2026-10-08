"""Audit release contents and compare regenerated numerical summaries."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import re

from make_manifest import ROOT, rows

FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".exe", ".dll",
                      ".bat", ".btn", ".ucn", ".hds", ".pem", ".key", ".zip",
                      ".docx", ".tex", ".log"}
FORBIDDEN_PARTS = {".claude", ".codex", "checkpoints", "prediction_caches", "data"}
TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".txt", ".cff", ".yml", ".sha256", ".svg"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "access token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16})"),
    "local host path": re.compile(r"(?:[A-Za-z]:[\\/](?:Users|Khanh_Loc)|/nfs"
                                   r"/scratch/|/local" r"/scratch/)", re.I),
    "simulator invocation": re.compile(r"run\.bat\s+\d", re.I),
}


def audit_contents():
    problems, source_count = [], 0
    package_rows = rows()
    for relative, size, _ in package_rows:
        path = ROOT / relative
        if path.is_symlink():
            problems.append(f"Symlink not permitted: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_PARTS.intersection(path.relative_to(ROOT).parts):
            problems.append(f"Restricted file category: {relative}")
        if size > 5 * 1024 * 1024:
            problems.append(f"Unexpected large file: {relative}")
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
            content = path.read_text(encoding="utf-8-sig")
            for label, pattern in PATTERNS.items():
                if pattern.search(content):
                    problems.append(f"{label} detected in {relative}")
            if path.suffix == ".py":
                ast.parse(content, filename=relative)
                source_count += 1
        if path.parent.name == "metadata" and path.name.startswith("param_"):
            if path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    if "run_bat_code" in csv.DictReader(handle).fieldnames:
                        problems.append(f"Simulator command column: {relative}")
            elif path.suffix == ".json":
                records = json.loads(path.read_text())
                if isinstance(records, list) and any("run_bat_code" in r for r in records):
                    problems.append(f"Simulator command field: {relative}")
    for path in ROOT.glob("*.md"):
        for match in re.finditer(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = match.group(1)
            if "://" not in target and not (ROOT / target.split("#", 1)[0]).exists():
                problems.append(f"Broken local link in {path.name}: {target}")
    if problems:
        raise ValueError("\n".join(problems))
    print(f"Repository content checks passed: {len(package_rows)} files, {source_count} Python sources.")


def compare_reproduction(directory: Path, require_controls=False):
    reference = ROOT / "figures/v4_final"
    generated = json.loads((directory / "v4_analysis.json").read_text())
    expected = json.loads((reference / "v4_analysis.json").read_text())
    if generated != expected:
        raise ValueError("Regenerated summary differs from the reference numerical inputs/results")
    for name in ("fig1_workflow_v4.pdf", "fig1_workflow_v4.svg", "fig2_transport_ladder_v4.pdf",
                 "fig3_taxonomy_v4.pdf", "fig4_distributions_v4.pdf", "fig5_triage_v4.pdf"):
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise ValueError(f"Missing generated figure: {name}")
    count = 1
    if require_controls:
        for expected_path in sorted((reference / "rd").glob("*.json")):
            actual = directory / "rd" / expected_path.name
            if json.loads(actual.read_text()) != json.loads(expected_path.read_text()):
                raise ValueError(f"Regenerated control differs: {actual.name}")
            count += 1
        capacity = json.loads((directory / "model_capacity_v4.json").read_text())
        recorded = json.loads((reference / "model_capacity_v4.json").read_text())
        for key in ("counts", "sources", "method"):
            if capacity[key] != recorded[key]:
                raise ValueError(f"Parameter-count reproduction differs: {key}")
        count += 1
    print(f"Numerical reproduction passed: {count} JSON comparisons and six figure-presence checks.")
    print("PDF pixel equivalence is not certified by this cross-platform numerical check.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduced-dir", type=Path)
    parser.add_argument("--require-controls", action="store_true")
    args = parser.parse_args()
    if args.require_controls and args.reproduced_dir is None:
        parser.error("--require-controls requires --reproduced-dir")
    audit_contents()
    if args.reproduced_dir is not None:
        compare_reproduction(args.reproduced_dir, args.require_controls)


if __name__ == "__main__":
    main()
