"""Write or verify SHA-256 hashes for the files in this code package."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_CODE_REVIEW.csv"


def rows() -> list[tuple[str, int, str]]:
    result = []
    for path in sorted(ROOT.rglob("*"), key=lambda p: p.relative_to(ROOT).as_posix()):
        if (not path.is_file() or path == MANIFEST or ".git" in path.parts
                or "__pycache__" in path.parts or path.suffix == ".pyc"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest.hexdigest()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without modifying the manifest")
    args = parser.parse_args()
    actual = rows()
    if args.check:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            expected = [(r["path"], int(r["bytes"]), r["sha256"]) for r in csv.DictReader(handle)]
        if len({r[0] for r in expected}) != len(expected):
            raise SystemExit("release manifest contains duplicate paths")
        if sorted(actual) != sorted(expected):
            raise SystemExit("release manifest does not match package files")
        print(f"verified {len(actual)} files")
        return
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "bytes", "sha256"))
        writer.writerows(actual)
    print(f"recorded {len(actual)} files")


if __name__ == "__main__":
    main()
