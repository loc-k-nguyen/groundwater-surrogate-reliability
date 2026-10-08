"""
Obj3 data loading with OOD split support.

Loads the Obj3 OOD split (param_split_obj3_ood.json) and resolves config IDs
to physical NPZ file paths across three data sources:
  - Main dataset (configs 0-161):   T25_TSTEP_OVERRIDE_FINAL/
  - Extra calibration (200-235):    simulation/datasets/obj3_calibration/
  - Interpolation OOD (276-355):    simulation/datasets/obj3_interpolation/

The interpolation split (param_split_obj3_interp.json) is a separate file used
ONLY for per-sigma degradation analysis. It is never mixed into training or
conformal calibration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 25
MAIN_DATASET_MAX_CONFIG = 161
EXTRA_CALIB_MIN_CONFIG = 200
INTERP_MIN_CONFIG = 276


def _canonicalize_K_shape(K_raw: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Normalize K storage layout to the spatial field shape implied by C.

    The main Obj1 dataset stores K as (H, W), but Obj3 calibration/interpolation
    archives store the same field flattened as (24000, 10). We do not rewrite
    the dataset on disk; instead we reshape on load using the spatial size from C.
    """
    target_hw = tuple(C.shape[-2:])
    if K_raw.shape == target_hw:
        return K_raw

    if K_raw.size == int(np.prod(target_hw)):
        logger.debug("Reshaping K from %s to %s", K_raw.shape, target_hw)
        return K_raw.reshape(target_hw)

    raise ValueError(
        f"Cannot canonicalize K with shape {K_raw.shape} to target spatial shape {target_hw}"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_obj3_split(split_path: Optional[str] = None) -> Dict[str, List[int]]:
    """Load the Obj3 OOD split JSON and return config ID lists per split."""
    if split_path is None:
        split_path = str(_repo_root() / "splits" / "param_split_obj3_ood.json")
    with open(split_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "train": raw["train"],
        "val_calib": raw["val_calib"],
        "extra_calib": raw["extra_calib"],
        "iid_test": raw["iid_test"],
        "ood_test": raw["ood_test"],
    }


def load_obj3_interp_split(split_path: Optional[str] = None) -> Dict[str, List[int]]:
    """Load the interpolation OOD split for per-sigma degradation analysis."""
    if split_path is None:
        split_path = str(_repo_root() / "splits" / "param_split_obj3_interp.json")
    with open(split_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k.startswith("interp_")}


def _resolve_config_dir(
    config_id: int,
    main_data_root: Path,
    extra_data_root: Path,
    interp_data_root: Optional[Path] = None,
) -> Path:
    """Map a config ID to its physical directory.

    Routing:
      0–161   → main_data_root  (T25_TSTEP_OVERRIDE_FINAL)
      200–235 → extra_data_root (obj3_calibration)
      276–355 → interp_data_root (obj3_interpolation)
    """
    if config_id <= MAIN_DATASET_MAX_CONFIG:
        return main_data_root / f"param_{config_id:03d}"
    if config_id >= INTERP_MIN_CONFIG:
        if interp_data_root is None:
            raise ValueError(
                f"config_id={config_id} requires interp_data_root but none was provided"
            )
        return interp_data_root / f"param_{config_id:03d}"
    return extra_data_root / f"param_{config_id:03d}"


def collect_npz_files(
    config_ids: Sequence[int],
    main_data_root: str | Path,
    extra_data_root: str | Path,
    interp_data_root: Optional[str | Path] = None,
) -> List[Path]:
    """Collect all NPZ files for the given config IDs, sorted by (config, realisation)."""
    main_root = Path(main_data_root)
    extra_root = Path(extra_data_root)
    interp_root = Path(interp_data_root) if interp_data_root is not None else None
    files: List[Tuple[int, int, Path]] = []

    for cid in config_ids:
        param_dir = _resolve_config_dir(cid, main_root, extra_root, interp_root)
        if not param_dir.exists():
            logger.warning("Config dir missing: %s (config_id=%d)", param_dir, cid)
            continue
        for npz in sorted(param_dir.glob("real_*.npz")):
            real_id = int(npz.stem.split("_")[-1])
            files.append((cid, real_id, npz))

    files.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in files]


class Obj3PatchDataset(Dataset):
    """
    Patch-based dataset for Obj3. Returns log10(C) targets for all 25 timesteps.

    Args:
        file_list: list of NPZ file paths
        stats: dict with k_mean, k_std from Obj3 training set
        patch_size: spatial crop size
        use_logK: whether to apply log10 to K before normalization
        eps_k: epsilon for log10(K)
        eps_c: epsilon for log10(C)
    """

    def __init__(
        self,
        file_list: List[Path],
        stats: Dict[str, float],
        patch_size: int = 320,
        use_logK: bool = True,
        eps_k: float = 1e-6,
        eps_c: float = 1e-12,
        legacy_crop_sampling: bool = False,
    ) -> None:
        self.files = file_list
        self.patch = patch_size
        self.use_logK = use_logK
        self.eps_k = eps_k
        self.eps_c = eps_c
        self.legacy_crop_sampling = legacy_crop_sampling
        self.k_mean = float(stats["k_mean"])
        self.k_std = float(stats["k_std"])

    def __len__(self) -> int:
        return len(self.files)

    def _normalize_K(self, K: np.ndarray) -> np.ndarray:
        K = K.astype(np.float64)
        if self.use_logK:
            K = np.log10(K + self.eps_k)
        return ((K - self.k_mean) / (self.k_std + 1e-12)).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as data:
            C = data["C"].astype(np.float32)
            K_raw = _canonicalize_K_shape(data["K"], C)

        K_norm = self._normalize_K(K_raw)
        H, W = K_norm.shape

        if self.patch > H or self.patch > W:
            raise ValueError("Patch exceeds the field dimensions")
        # Historical checkpoints excluded the final origin; retain that option explicitly.
        increment = 0 if self.legacy_crop_sampling else 1
        r = int(np.random.randint(0, H - self.patch + increment)) if H > self.patch else 0
        c = int(np.random.randint(0, W - self.patch + increment)) if W > self.patch else 0

        K_patch = K_norm[r : r + self.patch, c : c + self.patch]
        C_patch = C[:, r : r + self.patch, c : c + self.patch]

        # log10 transform for concentration
        C_log = np.log10(np.clip(C_patch, 0.0, None) + self.eps_c).astype(np.float32)

        return {
            "K": torch.from_numpy(K_patch[None, ...]),        # (1, P, P)
            "C_log": torch.from_numpy(C_log),                  # (25, P, P)
            "path": str(path),
        }


class Obj3FullFieldDataset(Dataset):
    """
    Full-field dataset for Obj3 evaluation (no cropping).

    Returns the full 600×400 field for sliding-window inference.
    """

    def __init__(
        self,
        file_list: List[Path],
        stats: Dict[str, float],
        use_logK: bool = True,
        eps_k: float = 1e-6,
        eps_c: float = 1e-12,
    ) -> None:
        self.files = file_list
        self.use_logK = use_logK
        self.eps_k = eps_k
        self.eps_c = eps_c
        self.k_mean = float(stats["k_mean"])
        self.k_std = float(stats["k_std"])

    def __len__(self) -> int:
        return len(self.files)

    def _normalize_K(self, K: np.ndarray) -> np.ndarray:
        K = K.astype(np.float64)
        if self.use_logK:
            K = np.log10(K + self.eps_k)
        return ((K - self.k_mean) / (self.k_std + 1e-12)).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as data:
            C = data["C"].astype(np.float32)
            K_raw = _canonicalize_K_shape(data["K"], C)

        K_norm = self._normalize_K(K_raw)
        C_log = np.log10(np.clip(C, 0.0, None) + self.eps_c).astype(np.float32)

        # Parse identifiers from path
        param_id = path.parent.name   # e.g. "param_012"
        real_id = path.stem           # e.g. "real_001"

        return {
            "K": torch.from_numpy(K_norm[None, ...]),         # (1, H, W)
            "C_log": torch.from_numpy(C_log),                  # (25, H, W)
            "C_phys": torch.from_numpy(C),                     # (25, H, W)
            "param_id": param_id,
            "real_id": real_id,
            "path": str(path),
        }


def build_obj3_dataloaders(
    split_path: Optional[str] = None,
    main_data_root: Optional[str] = None,
    extra_data_root: Optional[str] = None,
    stats_path: Optional[str] = None,
    patch_size: int = 320,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, object]:
    """Build all Obj3 dataloaders from the split file."""
    root = _repo_root()
    if main_data_root is None:
        main_data_root = str(root / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL")
    if extra_data_root is None:
        extra_data_root = str(root / "simulation" / "datasets" / "obj3_calibration")
    if stats_path is None:
        stats_path = str(root / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json")

    splits = load_obj3_split(split_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    result = {}
    for name, config_ids in splits.items():
        # interp configs not present in main split — no interp_data_root needed here
        files = collect_npz_files(config_ids, main_data_root, extra_data_root)
        logger.info("Split '%s': %d configs → %d files", name, len(config_ids), len(files))
        if name == "train":
            ds = Obj3PatchDataset(files, stats, patch_size=patch_size)
        else:
            ds = Obj3FullFieldDataset(files, stats)
        result[name] = {
            "dataset": ds,
            "files": files,
            "config_ids": config_ids,
        }

    return result, stats


def build_interp_eval_datasets(
    interp_split_path: Optional[str] = None,
    interp_data_root: Optional[str] = None,
    stats_path: Optional[str] = None,
) -> Dict[str, object]:
    """Build per-sigma evaluation datasets from the interpolation OOD split.

    Used ONLY for the per-σ²Y coverage degradation analysis (supplementary figure).
    These datasets are NEVER used for training or conformal calibration.

    Returns dict keyed by sigma level string: "interp_03", "interp_07", "interp_13", "interp_17".
    Each value has {"dataset": Obj3FullFieldDataset, "files": list, "config_ids": list}.
    """
    root = _repo_root()
    if interp_data_root is None:
        interp_data_root = str(root / "simulation" / "datasets" / "obj3_interpolation")
    if stats_path is None:
        stats_path = str(root / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json")

    splits = load_obj3_interp_split(interp_split_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Dummy roots — interp configs only route to interp_data_root
    dummy_main = root / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL"
    dummy_extra = root / "simulation" / "datasets" / "obj3_calibration"

    result = {}
    for name, config_ids in splits.items():
        files = collect_npz_files(config_ids, dummy_main, dummy_extra, interp_data_root)
        logger.info(
            "Interp split '%s': %d configs → %d files (σ²Y=%.1f)",
            name, len(config_ids), len(files), float(name.split("_")[1]) / 10,
        )
        ds = Obj3FullFieldDataset(files, stats)
        result[name] = {
            "dataset": ds,
            "files": files,
            "config_ids": config_ids,
        }

    return result, stats
