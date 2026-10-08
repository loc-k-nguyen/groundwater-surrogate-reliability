"""
Compute K-field normalization statistics for the Obj3 training split.

Usage:
    python -m src.obj3.conference.compute_obj3_train_stats

Output: experiments/obj3/conference/configs/obj3_train_stats.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.obj3.conference.data_obj3_ood import collect_npz_files, load_obj3_split

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_DATA = REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL"
EXTRA_DATA = REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"
OUTPUT_PATH = REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"

EPS_K = 1e-6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    splits = load_obj3_split()
    train_ids = splits["train"]
    files = collect_npz_files(train_ids, MAIN_DATA, EXTRA_DATA)
    logger.info("Obj3 training set: %d configs → %d files", len(train_ids), len(files))

    # Accumulate statistics in streaming fashion
    n_files = 0
    k_sum = 0.0
    k_sq_sum = 0.0
    n_pixels = 0
    c_max_global = 0.0

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            K = data["K"].astype(np.float64)
            C = data["C"].astype(np.float64)

        log_K = np.log10(K + EPS_K)
        k_sum += log_K.sum()
        k_sq_sum += (log_K ** 2).sum()
        n_pixels += log_K.size
        c_max_global = max(c_max_global, float(C.max()))
        n_files += 1

    k_mean = k_sum / n_pixels
    k_std = np.sqrt(k_sq_sum / n_pixels - k_mean ** 2)

    stats = {
        "n_train_params": len(train_ids),
        "n_train_files": n_files,
        "file_glob": "real_*.npz",
        "use_logK": True,
        "eps_k": EPS_K,
        "k_mean": float(k_mean),
        "k_std": float(k_std),
        "c_max": float(c_max_global),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote Obj3 train stats to %s", OUTPUT_PATH)
    logger.info("  k_mean=%.6f  k_std=%.6f  c_max=%.1f", k_mean, k_std, c_max_global)


if __name__ == "__main__":
    main()
