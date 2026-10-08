"""
MC Dropout CRPS computation for Obj3.

Runs MC Dropout inference (T forward passes) on IID and OOD test sets,
accumulates per-pixel Gaussian CRPS on plume pixels, and saves the result.

Does NOT store full prediction NPZ — CRPS is accumulated on-the-fly to
avoid writing 15+ GB of intermediate arrays.

Usage:
    python -m src.obj3.conference.eval_mcdrop_crps \\
        --checkpoint experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed0/best.pt \\
        --output_dir experiments/obj3/conference/runs/obj3_conf_ms_tmo_mcdrop_full_seed0 \\
        --n_passes 50 --dropout_rate 0.1 --eval_splits iid_test ood_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from src.obj3.conference.data_obj3_ood import (
    Obj3FullFieldDataset,
    collect_npz_files,
    load_obj3_split,
)
from src.obj3.conference.eval_mc_dropout import (
    UNet_ASPP_Attn_MCDrop,
    mc_dropout_inference,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RANGE = 14.0
PLUME_THRESH = 1e-8
SIGMA_MIN = 1e-6

_DEFAULT_SPLIT_JSON = str(
    REPO_ROOT / "splits" / "param_split_obj3_ood.json"
)
_DEFAULT_MAIN_DATA = str(
    REPO_ROOT
    / "Obj1"
    / "obj1_surrogate_conference"
    / "data"
    / "T25_TSTEP_OVERRIDE_FINAL"
)
_DEFAULT_EXTRA_DATA = str(
    REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"
)
_DEFAULT_STATS = str(
    REPO_ROOT
    / "experiments"
    / "obj3"
    / "conference"
    / "configs"
    / "obj3_train_stats.json"
)


# ---------------------------------------------------------------------------
# CRPS helpers
# ---------------------------------------------------------------------------


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a CPU NumPy view for dataset fields stored as arrays or tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _as_batched_float_tensor(
    value: np.ndarray | torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Convert a single sample field to a batched float tensor on device."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.from_numpy(value)
    return tensor.unsqueeze(0).float().to(device)


def _gaussian_crps_pixel(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Closed-form Gaussian CRPS per pixel.

    CRPS(N(mu, sigma), y) = sigma * [z * (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.
    """
    from scipy.stats import norm

    sig = np.maximum(sigma, SIGMA_MIN)
    z = (y_true - mu) / sig
    crps = sig * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )
    return crps.astype(np.float64)


def _accumulate_crps(
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    c_phys: np.ndarray,
) -> tuple[int, float, float]:
    """Compute plume-pixel CRPS for one sample.

    Args:
        y_true: (25, H, W) log-concentration ground truth.
        mean_pred: (25, H, W) MC mean prediction.
        var_pred:  (25, H, W) MC variance.
        c_phys:    (25, H, W) physical concentration (for plume mask).

    Returns:
        (n_plume_pixels, crps_sum, crps_sumsq)
    """
    plume_mask = c_phys > PLUME_THRESH
    std_pred = np.sqrt(np.maximum(var_pred, 0.0))

    crps_map = _gaussian_crps_pixel(y_true, mean_pred, std_pred)
    plume_crps = crps_map[plume_mask]

    n = int(plume_crps.size)
    s = float(plume_crps.sum(dtype=np.float64))
    sq = float(np.square(plume_crps, dtype=np.float64).sum(dtype=np.float64))
    return n, s, sq


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_split_crps(
    model: torch.nn.Module,
    dataset: Obj3FullFieldDataset,
    n_passes: int,
    patch_size: int,
    stride: int,
    device: str,
    limit: int = 0,
) -> dict:
    """Run MC Dropout and accumulate CRPS over the full split."""
    n_samples = len(dataset) if limit <= 0 else min(limit, len(dataset))
    total_n = 0
    total_sum = 0.0
    total_sumsq = 0.0

    for idx in range(n_samples):
        if idx % 50 == 0:
            logger.info("  sample %d / %d", idx, n_samples)

        sample = dataset[idx]
        K = _as_batched_float_tensor(sample["K"], device)  # (1, 1, H, W)
        y_true = _as_numpy(sample["C_log"])   # (25, H, W) log-concentration
        c_phys = _as_numpy(sample["C_phys"])  # (25, H, W) physical concentration

        mean_pred, var_pred = mc_dropout_inference(
            model, K, n_passes=n_passes,
            patch_size=patch_size, stride=stride, device=device,
        )

        n, s, sq = _accumulate_crps(y_true, mean_pred, var_pred, c_phys)
        total_n += n
        total_sum += s
        total_sumsq += sq

    if total_n == 0:
        raise ValueError("No plume pixels found across the split.")

    mean = total_sum / total_n
    var = max((total_sumsq / total_n) - mean * mean, 0.0)
    std = math.sqrt(var)
    return {"crps_mean": mean, "crps_std": std, "n_plume_pixels": total_n}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    ap = argparse.ArgumentParser(description="Obj3 MC Dropout CRPS evaluation")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--n_passes", type=int, default=50)
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument("--split_json", type=str, default=_DEFAULT_SPLIT_JSON)
    ap.add_argument("--main_data_root", type=str, default=_DEFAULT_MAIN_DATA)
    ap.add_argument("--extra_data_root", type=str, default=_DEFAULT_EXTRA_DATA)
    ap.add_argument("--stats_json", type=str, default=_DEFAULT_STATS)
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--eval_splits", nargs="+", default=["iid_test", "ood_test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt.get("cfg", {})
    model = UNet_ASPP_Attn_MCDrop(
        dropout_rate=args.dropout_rate,
        in_ch=1,
        out_ch=25,
        base=cfg.get("base_channels", 64),
        attn_heads=cfg.get("attn_heads", 4),
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info(
        "Loaded: %s  dropout_rate=%.2f  T=%d",
        args.checkpoint,
        args.dropout_rate,
        args.n_passes,
    )

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    splits = load_obj3_split(args.split_json)
    results: dict[str, dict] = {}

    for split_name in args.eval_splits:
        if split_name not in splits:
            logger.warning("Split '%s' not found, skipping.", split_name)
            continue

        files = collect_npz_files(
            splits[split_name], args.main_data_root, args.extra_data_root
        )
        dataset = Obj3FullFieldDataset(files, stats)
        logger.info(
            "MC Dropout CRPS on '%s': %d samples, T=%d",
            split_name,
            len(dataset),
            args.n_passes,
        )

        split_result = evaluate_split_crps(
            model,
            dataset,
            n_passes=args.n_passes,
            patch_size=args.patch_size,
            stride=args.stride,
            device=args.device,
            limit=args.limit,
        )
        results[split_name] = split_result
        logger.info(
            "  %s CRPS: %.4f ± %.4f  (plume pixels=%d)",
            split_name,
            split_result["crps_mean"],
            split_result["crps_std"],
            split_result["n_plume_pixels"],
        )

    results["note"] = (
        "Gaussian CRPS for MC Dropout on plume pixels (c_phys > 1e-8). "
        "Computed from per-pass mean and std accumulated on-the-fly."
    )
    results["checkpoint"] = args.checkpoint
    results["n_passes"] = args.n_passes
    results["dropout_rate"] = args.dropout_rate

    out_path = output_dir / "mcdrop_crps.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
