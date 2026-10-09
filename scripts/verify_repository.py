"""Check code-only contents and reconstructed corrected finite-design outputs."""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from make_manifest import ROOT, rows

FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".exe", ".dll", ".bat",
                      ".btn", ".ucn", ".hds", ".pem", ".key", ".zip", ".docx", ".tex", ".log"}
FORBIDDEN_PARTS = {".claude", ".codex", ".git", "checkpoints", "prediction_caches", "data"}
TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".txt", ".cff", ".yml", ".yaml", ".svg", ".drawio"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential shape": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16})\b"),
    "machine path": re.compile(r"[A-Za-z]:[/\\](?:Users|Khanh_Loc)|/nfs" r"/scratch/|/local" r"/scratch/", re.I),
    "simulator command": re.compile(r"run\.bat\s+\d", re.I),
}
FIGURES = ("fig2_transport_ladder_v5", "fig3_taxonomy_v5", "fig4_distributions_v5", "fig5_triage_v5")
SCIENCE = ("taxonomy", "transport_ladder", "triage", "ensemble_accuracy")


def audit_embedded_media(text, depth=0):
    if depth > 8:
        raise ValueError("Embedded media exceeds reviewed nesting limit")
    count = 0
    for kind, payload in re.findall(r"data:image/(svg\+xml|png|jpeg)(?:;base64)?,([A-Za-z0-9+/=]+)", text):
        raw = base64.b64decode(payload, validate=True)
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError("Embedded media exceeds size cap")
        count += 1
        if kind == "svg+xml":
            decoded = raw.decode("utf-8-sig")
            ET.fromstring(decoded)
            for name, pattern in PATTERNS.items():
                if pattern.search(decoded):
                    raise ValueError(name + " in decoded vector")
            count += audit_embedded_media(decoded, depth + 1)
        else:
            with Image.open(io.BytesIO(raw)) as image:
                metadata = str(image.info)
                for name, pattern in PATTERNS.items():
                    if pattern.search(metadata):
                        raise ValueError(name + " in embedded raster metadata")
    return count


def audit_contents():
    problems, sources, embedded = [], 0, 0
    actual = rows()
    for relative, size, _ in actual:
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            problems.append("Nonlocal file: " + relative)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_PARTS.intersection(Path(relative).parts):
            problems.append("Restricted category: " + relative)
        if size > 5 * 1024 * 1024:
            problems.append("Exceeds size cap: " + relative)
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8-sig")
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    problems.append(name + " in " + relative)
            if path.suffix in {".svg", ".drawio"}:
                try:
                    embedded += audit_embedded_media(text)
                except ValueError as error:
                    problems.append(str(error) + " in " + relative)
            if path.suffix == ".py":
                ast.parse(text, filename=relative)
                sources += 1
        if path.parent.name == "metadata" and path.name.startswith("param_"):
            if path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    if "run_bat_code" in csv.DictReader(handle).fieldnames:
                        problems.append("Simulator field: " + relative)
            elif path.suffix == ".json":
                records = json.loads(path.read_text())
                if isinstance(records, list) and any("run_bat_code" in row for row in records):
                    problems.append("Simulator field: " + relative)
    for path in [*ROOT.glob("*.md"), ROOT / "results/README.md"]:
        for match in re.finditer(r"\]\(([^)]+)\)", path.read_text()):
            target = match.group(1).split("#", 1)[0]
            if target and "://" not in target and not (path.parent / target).exists():
                problems.append("Broken document link in " + path.relative_to(ROOT).as_posix())
    if problems:
        raise ValueError("\n".join(problems))
    print(f"Content screen passed: {len(actual)} files, {sources} Python sources, {embedded} decoded media payloads; not rights or complete secret certification")


def compare_reproduction(directory):
    expected = json.loads((ROOT / "figures/v5_corrected/v5_analysis_portable.json").read_text())
    generated = json.loads((directory / "v5_analysis.json").read_text())
    for section in SCIENCE:
        if expected[section] != generated[section]:
            raise ValueError("Scientific section differs: " + section)
    for record in generated["sources"]:
        relative = Path(record["path"])
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Nonlocal source provenance")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Source hash mismatch")
    for stem in FIGURES:
        for suffix in (".pdf", ".png"):
            path = directory / (stem + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Missing figure: " + stem + suffix)
        with Image.open(ROOT / "figures/v5_corrected" / (stem + ".png")) as reference, \
                Image.open(directory / (stem + ".png")) as actual:
            if not np.array_equal(np.asarray(reference.convert("RGBA")), np.asarray(actual.convert("RGBA"))):
                raise ValueError("Figure pixels differ: " + stem)
    print(f"Reproduction passed: {len(SCIENCE)} exact scientific sections, {len(generated['sources'])} local source hashes, 4 pixel-identical PNGs")


def compare_supplement(path):
    expected = json.loads((ROOT / "results/orientation_v5/selection_crc_baselines.json").read_text())
    actual = json.loads(path.read_text())
    if expected != actual:
        raise ValueError("Selection/excess-risk/baseline supplement differs")
    for source in actual["sources"]:
        relative = Path(source["path"])
        local = (ROOT / relative).resolve()
        if relative.is_absolute() or not local.is_relative_to(ROOT.resolve()):
            raise ValueError("Nonlocal supplement source")
        if hashlib.sha256(local.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Supplement source hash mismatch")
    print("Supplement passed: exact reconstruction and five local source hashes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduced-dir", type=Path)
    parser.add_argument("--supplement", type=Path)
    args = parser.parse_args()
    audit_contents()
    if args.reproduced_dir is not None:
        compare_reproduction(args.reproduced_dir.resolve())
    if args.supplement is not None:
        compare_supplement(args.supplement.resolve())


if __name__ == "__main__":
    main()
