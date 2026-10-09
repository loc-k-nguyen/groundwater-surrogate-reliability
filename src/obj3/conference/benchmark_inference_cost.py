"""
Inference cost benchmark for Obj3 UQ methods vs MODFLOW baseline.

Measures wall-clock time per sample for:
  1. Deterministic MS-TMO-UNet (1 forward pass)
  2. Deep Ensemble (5 members × 1 pass each)
  3. MC Dropout (T=50 stochastic passes, single model)

Compares against MODFLOW per-realisation cost from config file.

Reports mean ± std over N_WARMUP + N_REPS timed runs (GPU synced).
Output: experiments/obj3/conference/paper_assets/data/inference_cost.json

Usage (Raapoi GPU job):
    python -m src.obj3.conference.benchmark_inference_cost \\
        --checkpoint_dir .../experiments/obj3/conference/runs \\
        --output_dir .../paper_assets/data
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Benchmark parameters
N_WARMUP = 5     # discarded warm-up repetitions
N_REPS = 20      # timed repetitions
MC_T = 50        # MC Dropout passes
N_ENSEMBLE = 5   # ensemble members

# MODFLOW per-realisation cost (seconds) — from project timing records.
# One full MODFLOW-NWT + MT3D-USGS run for a single (K, C) realisation
# at 25 timesteps takes approximately 90 seconds on a single CPU core.
MODFLOW_SECONDS_PER_SAMPLE = 90.0

# Input tensor shape: (1, T_in, H, W) — matches training shape
# Obj3 uses seen timesteps 0..14 as input (15 timesteps × 320 × 320)
BATCH_SIZE = 1
IN_CH = 1
H, W = 320, 320


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a single MS-TMO-UNet checkpoint."""
    from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = UNet_ASPP_Attn(
        in_ch=1,
        out_ch=25,
        base=cfg.get("base_channels", 64),
        attn_heads=cfg.get("attn_heads", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def _enable_dropout(model: torch.nn.Module) -> None:
    """Switch Dropout layers to training mode (MC Dropout)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_warmup: int,
    n_reps: int,
    device: torch.device,
) -> list[float]:
    """Run timed inference with GPU synchronisation."""
    times = []
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def _time_mcdrop(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_passes: int,
    n_warmup: int,
    n_reps: int,
    device: torch.device,
) -> list[float]:
    """Time T stochastic MC Dropout passes."""
    _enable_dropout(model)
    times = []

    def _run_passes():
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(model(x))
        return torch.stack(preds)

    # warm-up
    for _ in range(n_warmup):
        _run_passes()
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(n_reps):
        t0 = time.perf_counter()
        _run_passes()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return times


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(checkpoint_dir: Path, device: torch.device) -> dict:
    """Load models and run all timing experiments."""
    # Find checkpoints in the actual Obj3 conference layout.
    ckpt_paths = [
        checkpoint_dir / f"obj3_conf_ms_tmo_det_full_seed{s}" / "best.pt"
        for s in range(N_ENSEMBLE)
    ]
    ckpt_paths = [p for p in ckpt_paths if p.exists()]
    if not ckpt_paths:
        ckpt_paths = sorted(checkpoint_dir.glob("obj3_conf_ms_tmo_det_full_seed*/best.pt"))
    if not ckpt_paths:
        ckpt_paths = sorted(checkpoint_dir.glob("seed*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. "
            "Expected obj3_conf_ms_tmo_det_full_seed*/best.pt or seed*.pt layout."
        )
    logger.info("Found %d checkpoints: %s", len(ckpt_paths), [p.name for p in ckpt_paths])

    # Load all ensemble members
    models = [_load_model(p, device) for p in ckpt_paths[:N_ENSEMBLE]]
    logger.info("Loaded %d models onto %s", len(models), device)

    # Dummy input — same shape as real data
    x = torch.randn(BATCH_SIZE, IN_CH, H, W, device=device)

    results: dict = {}

    # 1. Deterministic (seed 0 only)
    logger.info("Benchmarking deterministic (1 pass, seed 0) ...")
    det_times = _time_forward(models[0], x, N_WARMUP, N_REPS, device)
    results["deterministic"] = {
        "mean_s": float(np.mean(det_times)),
        "std_s": float(np.std(det_times)),
        "n_reps": N_REPS,
        "description": "Single forward pass, seed 0",
    }
    logger.info(
        "  deterministic: %.4f ± %.4f s", results["deterministic"]["mean_s"],
        results["deterministic"]["std_s"],
    )

    # 2. Deep Ensemble (all 5 members, sequential)
    logger.info("Benchmarking deep ensemble (%d members) ...", len(models))
    ens_times = []
    for _ in range(N_WARMUP):
        for m in models:
            with torch.no_grad():
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        for m in models:
            with torch.no_grad():
                _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ens_times.append(time.perf_counter() - t0)

    results["deep_ensemble"] = {
        "mean_s": float(np.mean(ens_times)),
        "std_s": float(np.std(ens_times)),
        "n_members": len(models),
        "n_reps": N_REPS,
        "description": f"{len(models)} members, sequential",
    }
    logger.info(
        "  deep_ensemble: %.4f ± %.4f s", results["deep_ensemble"]["mean_s"],
        results["deep_ensemble"]["std_s"],
    )

    # 3. MC Dropout (T=50 passes, seed 0 model)
    logger.info("Benchmarking MC Dropout (T=%d, seed 0) ...", MC_T)
    mc_times = _time_mcdrop(models[0], x, MC_T, N_WARMUP, N_REPS, device)
    results["mc_dropout"] = {
        "mean_s": float(np.mean(mc_times)),
        "std_s": float(np.std(mc_times)),
        "n_passes": MC_T,
        "n_reps": N_REPS,
        "description": f"T={MC_T} stochastic passes, seed 0",
    }
    logger.info(
        "  mc_dropout: %.4f ± %.4f s", results["mc_dropout"]["mean_s"],
        results["mc_dropout"]["std_s"],
    )

    # Speedup vs MODFLOW
    for key in ("deterministic", "deep_ensemble", "mc_dropout"):
        t = results[key]["mean_s"]
        results[key]["speedup_vs_modflow"] = MODFLOW_SECONDS_PER_SAMPLE / t

    results["modflow_reference"] = {
        "mean_s": MODFLOW_SECONDS_PER_SAMPLE,
        "description": "MODFLOW-NWT + MT3D-USGS, single realisation, 25 timesteps, 1 CPU core",
    }
    results["device"] = str(device)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    ap = argparse.ArgumentParser(description="Inference cost benchmark for Obj3 UQ methods")
    ap.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(
            REPO_ROOT
            / "experiments/obj3/conference/runs"
            / "obj3_conf_ms_tmo_ensemble5_full/checkpoints"
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "experiments/obj3/conference/paper_assets/data"),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    results = run_benchmark(Path(args.checkpoint_dir), device)

    out_path = Path(args.output_dir) / "inference_cost.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    logger.info("Saved -> %s", out_path)

    print("\n=== Inference cost summary ===")
    modflow_s = MODFLOW_SECONDS_PER_SAMPLE
    for key in ("deterministic", "deep_ensemble", "mc_dropout"):
        r = results[key]
        print(
            f"  {key:20s}: {r['mean_s']*1000:7.1f} ± {r['std_s']*1000:5.1f} ms  "
            f"(×{r['speedup_vs_modflow']:,.0f} vs MODFLOW)"
        )
    print(f"  {'MODFLOW':20s}: {modflow_s:.0f} s  (reference)")


if __name__ == "__main__":
    main()
