"""
Evaluate the frozen Obj3 conference ensemble on the extended OOD dataset.

Extended OOD: sigma^2_Y in {2.5, 3.0}, param_236-275.
  - sigma^2_Y = 2.5: param_236-255  (67/100 survived MT3D late-stage failures)
  - sigma^2_Y = 3.0: param_256-275  (74/100 survived)
  - Total: 141/200 accepted (70.5% retention)

IMPORTANT — selection bias:
  Failed simulations are not random. They are concentrated in parameter
  configurations with extreme heterogeneity. The reported plume SSIM values
  are conditioned on simulator success and represent an UPPER BOUND on true
  surrogate reliability in this regime. This script embeds that note in every
  output JSON it produces.

Usage:
    python -m src.obj3.journal.eval_extended_ood \
        --checkpoints path/to/seed{0..4}/best.pt [x5] \
        --extended_ood_root simulation/datasets/obj3_ood_extended \
        --conformal_json path/to/conformal_report_iid.json \
        --stats_json experiments/obj3/conference/configs/obj3_train_stats.json \
        --output_dir experiments/obj3/journal/reports/analysis \
        [--patch_size 320] [--stride 160] [--alpha 0.10] [--device cuda]

Output files:
    {output_dir}/extended_sigma_eval.json   <- consumed by scripts/obj3_jnl_sigma_curve.py
    {output_dir}/extended_ood_detail.json   <- full per-sample breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.eval_ensemble import load_ensemble_models, ensemble_inference
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Evaluation constants — must match conference fairness rules.
DATA_RANGE = 14.0
PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64
EPS_C = 1e-12
EPS_K = 1e-6

# Extended OOD param_id ranges (from OBJ3_OOD_EXTENDED_GENERATION_NOTE.md).
SIGMA_25_PARAM_RANGE = range(236, 256)  # param_236 .. param_255
SIGMA_30_PARAM_RANGE = range(256, 276)  # param_256 .. param_275

EXTENDED_SURVIVORS = 141
EXTENDED_TOTAL = 200

SELECTION_BIAS_NOTE = (
    f"Only {EXTENDED_SURVIVORS}/{EXTENDED_TOTAL} files survived late-stage MT3D "
    f"failures ({EXTENDED_SURVIVORS / EXTENDED_TOTAL:.1%} retention). "
    "Failed simulations are concentrated in parameter configurations with extreme "
    "heterogeneity. Reported plume SSIM is conditioned on simulator success and "
    "represents an UPPER BOUND on true surrogate reliability in this shift regime. "
    "Report in appendix only with explicit bias disclosure."
)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _collect_extended_files(
    extended_ood_root: Path,
    param_range: range,
) -> List[Tuple[int, int, Path]]:
    """Collect (param_id, real_id, path) tuples for a given param_id range.

    Scans the extended OOD directory directly; no split JSON required because
    all files in this directory are test-only by construction.
    """
    records: List[Tuple[int, int, Path]] = []
    for param_id in param_range:
        param_dir = extended_ood_root / f"param_{param_id:03d}"
        if not param_dir.exists():
            logger.warning("Extended OOD dir missing: %s", param_dir)
            continue
        for npz in sorted(param_dir.glob("real_*.npz")):
            real_id = int(npz.stem.split("_")[-1])
            records.append((param_id, real_id, npz))
    records.sort(key=lambda t: (t[0], t[1]))
    return records


def _load_stats(stats_json: str | Path) -> Dict[str, float]:
    with open(stats_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_K(K_raw: np.ndarray, k_mean: float, k_std: float) -> np.ndarray:
    K = K_raw.astype(np.float64)
    K = np.log10(K + EPS_K)
    return ((K - k_mean) / (k_std + 1e-12)).astype(np.float32)


def _canonicalize_K(K_raw: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Reshape K to match the spatial layout of C if stored as flat array."""
    target_hw = tuple(C.shape[-2:])
    if K_raw.shape == target_hw:
        return K_raw
    if K_raw.size == int(np.prod(target_hw)):
        return K_raw.reshape(target_hw)
    raise ValueError(
        f"Cannot canonicalize K shape {K_raw.shape} to spatial target {target_hw}"
    )


# ---------------------------------------------------------------------------
# Inference per sample
# ---------------------------------------------------------------------------

def _eval_single(
    models: List[torch.nn.Module],
    npz_path: Path,
    k_mean: float,
    k_std: float,
    scp_quantile: float,
    patch_size: int,
    stride: int,
    device: str,
) -> Dict[str, float]:
    """Run ensemble inference on one NPZ file and return per-sample metrics."""
    with np.load(npz_path, allow_pickle=False) as data:
        C = data["C"].astype(np.float32)          # (25, H, W)
        K_raw = _canonicalize_K(data["K"], C)

    K_norm = _normalize_K(K_raw, k_mean, k_std)   # (H, W)
    K_tensor = torch.from_numpy(K_norm[None, None, ...])  # (1, 1, H, W)
    K_tensor = K_tensor.to(device)

    mean_pred, _, _ = ensemble_inference(
        models,
        K_tensor,
        patch_size=patch_size,
        stride=stride,
        device=device,
    )
    # mean_pred: (25, H, W) numpy float32 in log10 space

    C_log_gt = np.log10(np.clip(C, 0.0, None) + EPS_C)

    # Conformal coverage
    lower = mean_pred - scp_quantile
    upper = mean_pred + scp_quantile
    covered = (C_log_gt >= lower) & (C_log_gt <= upper)

    plume_ssims, global_ssims, mass_errors = [], [], []
    plume_coverage_vals, plume_width_vals = [], []

    for t in range(25):
        global_ssims.append(float(ssim(C_log_gt[t], mean_pred[t], data_range=DATA_RANGE)))

        C_phys_t = np.clip(C[t], 0.0, None)
        plume_mask_t = C_phys_t > PLUME_THRESH

        ps, _ = compute_plume_ssim(
            C_phys_t, mean_pred[t], EPS_C, PLUME_THRESH,
            PLUME_PAD, PLUME_MIN_PIXELS, DATA_RANGE,
        )
        if np.isfinite(ps):
            plume_ssims.append(float(ps))

        pred_phys_t = np.clip(10.0 ** mean_pred[t] - EPS_C, 0.0, None)
        gt_mass = float(C_phys_t.sum())
        if gt_mass > 1e-12:
            mass_errors.append(abs(float(pred_phys_t.sum()) - gt_mass))

        if plume_mask_t.any():
            plume_coverage_vals.append(float(np.mean(covered[t][plume_mask_t])))
            plume_width_vals.append(float(np.mean((upper[t] - lower[t])[plume_mask_t])))

    return {
        "plume_ssim_mean": float(np.mean(plume_ssims)) if plume_ssims else float("nan"),
        "global_ssim_mean": float(np.mean(global_ssims)),
        "mass_error_mean": float(np.mean(mass_errors)) if mass_errors else float("nan"),
        "global_coverage": float(np.mean(covered)),
        "plume_coverage": float(np.mean(plume_coverage_vals)) if plume_coverage_vals else float("nan"),
        "plume_mean_width": float(np.mean(plume_width_vals)) if plume_width_vals else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-sigma evaluation loop
# ---------------------------------------------------------------------------

def _eval_sigma_bucket(
    models: List[torch.nn.Module],
    records: List[Tuple[int, int, Path]],
    k_mean: float,
    k_std: float,
    scp_quantile: float,
    patch_size: int,
    stride: int,
    device: str,
    sigma_label: str,
    survived: int,
    attempted: int,
) -> Tuple[Dict, List[Dict]]:
    """Run evaluation on all surviving files for one sigma bucket."""
    per_sample = []

    for param_id, real_id, npz_path in records:
        logger.info(
            "sigma=%s  param_%03d  real_%d  (%d/%d)",
            sigma_label, param_id, real_id, len(per_sample) + 1, len(records),
        )
        try:
            metrics = _eval_single(
                models, npz_path, k_mean, k_std, scp_quantile,
                patch_size, stride, device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed %s: %s", npz_path, exc)
            continue

        per_sample.append({
            "param_id": param_id,
            "real_id": real_id,
            "path": str(npz_path),
            **metrics,
        })

    # Aggregate over surviving samples only.
    plume_ssims = [s["plume_ssim_mean"] for s in per_sample if np.isfinite(s["plume_ssim_mean"])]
    global_ssims = [s["global_ssim_mean"] for s in per_sample]
    mass_errs = [s["mass_error_mean"] for s in per_sample if np.isfinite(s["mass_error_mean"])]
    plume_covs = [s["plume_coverage"] for s in per_sample if np.isfinite(s["plume_coverage"])]

    aggregate = {
        "sigma2Y": sigma_label,
        "n_survived": survived,
        "n_attempted": attempted,
        "retention_rate": survived / attempted if attempted > 0 else 0.0,
        "n_evaluated": len(per_sample),
        "mean": float(np.mean(plume_ssims)) if plume_ssims else None,
        "std": float(np.std(plume_ssims)) if plume_ssims else None,
        "n": len(plume_ssims),
        "global_ssim_mean": float(np.mean(global_ssims)) if global_ssims else None,
        "mass_error_mean": float(np.mean(mass_errs)) if mass_errs else None,
        "plume_coverage_mean": float(np.mean(plume_covs)) if plume_covs else None,
        "selection_bias_note": SELECTION_BIAS_NOTE,
    }
    return aggregate, per_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_scp_quantile(conformal_json: str | Path, alpha: float = 0.10) -> float:
    with open(conformal_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    key = f"alpha_{alpha:.2f}"
    scp = report.get("scp", {})
    if key not in scp:
        raise KeyError(f"Missing {key} in SCP report: {conformal_json}")
    return float(scp[key]["quantile"])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Evaluate extended OOD (sigma^2_Y=2.5/3.0) — journal only, appendix tier"
    )
    ap.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to the 5 ensemble best.pt checkpoints (seeds 0-4 from conference runs).",
    )
    ap.add_argument(
        "--extended_ood_root", type=str,
        default=str(REPO_ROOT / "simulation" / "datasets" / "obj3_ood_extended"),
        help="Root directory of the extended OOD NPZ files.",
    )
    ap.add_argument(
        "--conformal_json", type=str, required=True,
        help="Path to the conference conformal IID report JSON (for SCP quantile).",
    )
    ap.add_argument(
        "--stats_json", type=str,
        default=str(REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"),
    )
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    # --- Validate inputs -------------------------------------------------------
    extended_root = Path(args.extended_ood_root)
    if not extended_root.exists():
        raise FileNotFoundError(f"Extended OOD root not found: {extended_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load models and calibration ------------------------------------------
    logger.info("Loading ensemble models from %d checkpoints", len(args.checkpoints))
    models = load_ensemble_models(args.checkpoints, args.device)

    logger.info("Loading SCP quantile at alpha=%.2f from %s", args.alpha, args.conformal_json)
    scp_quantile = _load_scp_quantile(args.conformal_json, args.alpha)
    logger.info("SCP quantile: %.4f", scp_quantile)

    stats = _load_stats(args.stats_json)
    k_mean = float(stats["k_mean"])
    k_std = float(stats["k_std"])

    # --- Collect files per sigma bucket ----------------------------------------
    records_25 = _collect_extended_files(extended_root, SIGMA_25_PARAM_RANGE)
    records_30 = _collect_extended_files(extended_root, SIGMA_30_PARAM_RANGE)

    logger.info("sigma^2_Y=2.5: found %d files (expected ~67)", len(records_25))
    logger.info("sigma^2_Y=3.0: found %d files (expected ~74)", len(records_30))

    if not records_25 and not records_30:
        raise RuntimeError(
            f"No extended OOD files found under {extended_root}. "
            "Verify that simulation/datasets/obj3_ood_extended/ is present on this machine."
        )

    # --- Evaluate each sigma bucket --------------------------------------------
    logger.info("=== Evaluating sigma^2_Y = 2.5 ===")
    agg_25, per_sample_25 = _eval_sigma_bucket(
        models, records_25, k_mean, k_std, scp_quantile,
        args.patch_size, args.stride, args.device,
        sigma_label="2.5", survived=67, attempted=100,
    )

    logger.info("=== Evaluating sigma^2_Y = 3.0 ===")
    agg_30, per_sample_30 = _eval_sigma_bucket(
        models, records_30, k_mean, k_std, scp_quantile,
        args.patch_size, args.stride, args.device,
        sigma_label="3.0", survived=74, attempted=100,
    )

    # --- Write extended_sigma_eval.json (consumed by sigma_curve script) -------
    sigma_eval = {
        "status": "complete",
        "alpha": float(args.alpha),
        "scp_quantile": float(scp_quantile),
        "selection_bias_note": SELECTION_BIAS_NOTE,
        "levels": {
            "2.5": {
                "mean": agg_25["mean"],
                "std": agg_25["std"],
                "n": agg_25["n"],
                "n_survived": agg_25["n_survived"],
                "n_attempted": agg_25["n_attempted"],
                "retention_rate": agg_25["retention_rate"],
                "global_ssim_mean": agg_25["global_ssim_mean"],
                "mass_error_mean": agg_25["mass_error_mean"],
                "plume_coverage_mean": agg_25["plume_coverage_mean"],
            },
            "3.0": {
                "mean": agg_30["mean"],
                "std": agg_30["std"],
                "n": agg_30["n"],
                "n_survived": agg_30["n_survived"],
                "n_attempted": agg_30["n_attempted"],
                "retention_rate": agg_30["retention_rate"],
                "global_ssim_mean": agg_30["global_ssim_mean"],
                "mass_error_mean": agg_30["mass_error_mean"],
                "plume_coverage_mean": agg_30["plume_coverage_mean"],
            },
        },
    }
    sigma_eval_path = output_dir / "extended_sigma_eval.json"
    with open(sigma_eval_path, "w", encoding="utf-8") as f:
        json.dump(sigma_eval, f, indent=2)
    logger.info("Saved sigma eval summary -> %s", sigma_eval_path)

    # --- Write extended_ood_detail.json (full per-sample breakdown) -----------
    detail = {
        "status": "complete",
        "alpha": float(args.alpha),
        "scp_quantile": float(scp_quantile),
        "selection_bias_note": SELECTION_BIAS_NOTE,
        "sigma_2p5": {"aggregate": agg_25, "per_sample": per_sample_25},
        "sigma_3p0": {"aggregate": agg_30, "per_sample": per_sample_30},
    }
    detail_path = output_dir / "extended_ood_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)
    logger.info("Saved per-sample detail -> %s", detail_path)

    # --- Console summary -------------------------------------------------------
    logger.info("=== EXTENDED OOD SUMMARY ===")
    for sigma_key, agg in [("2.5", agg_25), ("3.0", agg_30)]:
        mean_str = f"{agg['mean']:.4f}" if agg["mean"] is not None else "N/A"
        std_str = f"{agg['std']:.4f}" if agg["std"] is not None else "N/A"
        logger.info(
            "sigma^2_Y=%s  plume_SSIM=%s ± %s  n=%d/%d  retention=%.1f%%",
            sigma_key, mean_str, std_str,
            agg["n_survived"], agg["n_attempted"],
            agg["retention_rate"] * 100,
        )
    logger.info("Selection bias applies — report in appendix only.")


if __name__ == "__main__":
    main()
