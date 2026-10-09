"""Configure raw-script roots without relying on the original repository depth."""
import argparse
import os
from pathlib import Path
import sys


def project_root(root=None):
    """Resolve a root without parsing arguments or changing process state."""
    return Path(root or os.environ.get("SURROGATE_REPO_ROOT",
                                      Path(__file__).resolve().parents[1])).resolve()


def enable_package_imports():
    """Prioritize shipped code at explicit launch, without consuming CLI options."""
    code_root = Path(__file__).resolve().parents[1]
    sys.path[:] = [str(code_root)] + [path for path in sys.path if path != str(code_root)]
    return code_root


def configure_script_paths():
    """Consume common path options explicitly at script launch, never at import."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    known, remaining = parser.parse_known_args()
    root = project_root(known.repo_root)
    if not (root / "src").is_dir() or not (root / "splits").is_dir():
        parser.error(f"Expected src/ and splits/ below repository root: {root}")
    if any(value is not None for value in vars(known).values()):
        sys.argv[1:] = remaining
    for key in ("data_root", "calibration_root"):
        value = getattr(known, key)
        if value is not None:
            os.environ["SURROGATE_" + key.upper()] = str(value.resolve())
    # Code always comes from this package; an alternate root locates external assets.
    enable_package_imports()
    return root


DEFAULT_ROOT = project_root()


def asset_path(kind, fallback):
    variable = {"data": "SURROGATE_DATA_ROOT", "calibration": "SURROGATE_CALIBRATION_ROOT"}[kind]
    return Path(os.environ.get(variable, fallback)).resolve()


def metadata_path(name, root=None):
    root = project_root(root)
    packaged = root / "metadata" / name
    return packaged if packaged.exists() else root / "simulation" / name
