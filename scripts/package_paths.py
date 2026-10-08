"""Configure raw-script roots without relying on the original repository depth."""
import argparse
import os
from pathlib import Path
import sys


def project_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    known, remaining = parser.parse_known_args()
    if any(value is not None for value in vars(known).values()):
        sys.argv[1:] = remaining
    for key in ("data_root", "calibration_root"):
        value = getattr(known, key)
        if value is not None:
            os.environ["SURROGATE_" + key.upper()] = str(value.resolve())
    root = (known.repo_root or Path(os.environ.get("SURROGATE_REPO_ROOT",
                                                Path(__file__).resolve().parents[1]))).resolve()
    if not (root / "src").is_dir() or not (root / "splits").is_dir():
        raise FileNotFoundError(f"Expected src/ and splits/ below repository root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


DEFAULT_ROOT = project_root()


def asset_path(kind, fallback):
    variable = {"data": "SURROGATE_DATA_ROOT", "calibration": "SURROGATE_CALIBRATION_ROOT"}[kind]
    return Path(os.environ.get(variable, fallback)).resolve()


def metadata_path(name):
    packaged = DEFAULT_ROOT / "metadata" / name
    return packaged if packaged.exists() else DEFAULT_ROOT / "simulation" / name
