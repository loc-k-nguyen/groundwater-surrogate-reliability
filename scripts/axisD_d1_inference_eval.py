"""
Axis D D1 explicit-dataset inference and evaluation for Obj3 journal upgrades.

This wrapper intentionally does not read or modify Obj3 split JSON files. It
evaluates a supplied Axis D dataset directory, such as param_9200, with one or
more existing Obj3 checkpoints and writes isolated prediction and metric files.
"""

from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.data_obj3_ood import Obj3FullFieldDataset
from src.obj3.conference.eval_deterministic import (
    DATA_RANGE,
    EPS_C,
    PLUME_MIN_PIXELS,
    PLUME_PAD,
    PLUME_THRESH,
    sliding_window_inference,
)
from src.obj3.conference.export_predictions import load_models
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = DEFAULT_ROOT
DEFAULT_DATA_ROOT = (
    REPO_ROOT
    / "experiments"
    / "obj3"
    / "journal"
    / "upgrade_2026_07_01"
    / "tier1_multi_axis_ood"
    / "axisD_transport_shift"
    / "sim"
    / "dataset_axisD"
    / "param_9200"
)
DEFAULT_STATS = REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def ensure_clean_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new isolated output directory or pass --force intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_axisd_files(data_root: Path, limit: int) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Axis D dataset directory does not exist: {data_root}")
    files = sorted(data_root.glob("real_*.npz"))
    if not files:
        raise FileNotFoundError(f"No real_*.npz files found under: {data_root}")
    if limit > 0:
        files = files[:limit]
    return files


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = []
    for value in values:
        if value is None:
            continue
        if np.isfinite(value):
            vals.append(float(value))
    if not vals:
        return None
    return float(np.mean(vals))


def std_or_none(values: Iterable[float]) -> float | None:
    vals = []
    for value in values:
        if value is None:
            continue
        if np.isfinite(value):
            vals.append(float(value))
    if not vals:
        return None
    return float(np.std(vals))


def evaluate_member_mean(
    mean_pred_log: np.ndarray,
    pred_std_log: np.ndarray,
    sample: Dict[str, object],
) -> Dict[str, Any]:
    c_log_gt = sample["C_log"].numpy()
    c_phys_gt = sample["C_phys"].numpy()
    timestep_metrics: List[Dict[str, Any]] = []

    for t in range(mean_pred_log.shape[0]):
        gt_log_t = c_log_gt[t]
        pred_log_t = mean_pred_log[t]
        gt_phys_t = np.clip(c_phys_gt[t], 0.0, None)

        global_ssim = float(ssim(gt_log_t, pred_log_t, data_range=DATA_RANGE))
        plume_ssim_val, plume_pixels = compute_plume_ssim(
            gt_phys_t,
            pred_log_t,
            EPS_C,
            PLUME_THRESH,
            PLUME_PAD,
            PLUME_MIN_PIXELS,
            DATA_RANGE,
        )

        pred_phys_t = np.clip(10.0 ** pred_log_t - EPS_C, 0.0, None)
        gt_mass = float(gt_phys_t.sum())
        pred_mass = float(pred_phys_t.sum())
        mass_error = abs(pred_mass - gt_mass) / max(gt_mass, 1e-12)

        plume_mask = gt_phys_t > PLUME_THRESH
        plume_pred_std = (
            float(np.mean(pred_std_log[t][plume_mask])) if np.any(plume_mask) else None
        )

        timestep_metrics.append(
            {
                "t": int(t),
                "global_ssim": global_ssim,
                "plume_ssim": finite_or_none(plume_ssim_val),
                "mass_error": mass_error,
                "plume_pixels": int(plume_pixels),
                "mean_pred_std_log": float(np.mean(pred_std_log[t])),
                "plume_mean_pred_std_log": plume_pred_std,
            }
        )

    return {
        "param_id": sample["param_id"],
        "real_id": sample["real_id"],
        "path": sample["path"],
        "mean_plume_ssim": mean_or_none(m["plume_ssim"] for m in timestep_metrics),
        "mean_global_ssim": mean_or_none(m["global_ssim"] for m in timestep_metrics),
        "mean_mass_error": mean_or_none(m["mass_error"] for m in timestep_metrics),
        "mean_pred_std_log": mean_or_none(m["mean_pred_std_log"] for m in timestep_metrics),
        "plume_mean_pred_std_log": mean_or_none(
            m["plume_mean_pred_std_log"] for m in timestep_metrics
        ),
        "timestep_metrics": timestep_metrics,
    }


def write_per_sample_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "param_id",
        "real_id",
        "path",
        "mean_plume_ssim",
        "mean_global_ssim",
        "mean_mass_error",
        "mean_pred_std_log",
        "plume_mean_pred_std_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: finite_or_none(row.get(name)) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explicit Axis D D1 inference/evaluation without touching Obj3 split files."
    )
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stats_json", type=str, default=str(DEFAULT_STATS))
    parser.add_argument("--patch_size", type=int, default=320)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pred_clamp_min", type=float, default=-12.0)
    parser.add_argument("--pred_clamp_max", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    ensure_clean_output_dir(output_dir, args.force)

    data_root = Path(args.data_root).resolve()
    files = load_axisd_files(data_root, args.limit)
    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    models = load_models(args.checkpoints, args.device)
    dataset = Obj3FullFieldDataset(files, stats)
    logger.info("Axis D D1 files: %d", len(dataset))
    logger.info("Loaded checkpoints: %d", len(models))

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []
    all_y_pred_std: List[np.ndarray] = []
    all_c_phys: List[np.ndarray] = []
    per_sample: List[Dict[str, Any]] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        k_full = sample["K"].unsqueeze(0).to(args.device)
        member_preds: List[np.ndarray] = []

        for model in models:
            with torch.no_grad():
                pred = sliding_window_inference(
                    model,
                    k_full,
                    patch_size=args.patch_size,
                    stride=args.stride,
                    device=args.device,
                )
                pred = torch.clamp(pred, args.pred_clamp_min, args.pred_clamp_max)
            member_preds.append(pred.squeeze(0).cpu().numpy())

        member_preds_np = np.stack(member_preds, axis=0)
        mean_pred_log = member_preds_np.mean(axis=0)
        pred_std_log = member_preds_np.std(axis=0)

        all_y_true.append(sample["C_log"].numpy())
        all_y_pred.append(mean_pred_log)
        all_y_pred_std.append(pred_std_log)
        all_c_phys.append(sample["C_phys"].numpy())
        per_sample.append(evaluate_member_mean(mean_pred_log, pred_std_log, sample))

        logger.info(
            "[%d/%d] %s/%s plume_ssim=%s global_ssim=%s",
            i + 1,
            len(dataset),
            sample["param_id"],
            sample["real_id"],
            per_sample[-1]["mean_plume_ssim"],
            per_sample[-1]["mean_global_ssim"],
        )

    predictions_path = output_dir / "predictions_axisD_d1.npz"
    np.savez_compressed(
        str(predictions_path),
        y_true=np.stack(all_y_true, axis=0),
        y_pred=np.stack(all_y_pred, axis=0),
        y_pred_std=np.stack(all_y_pred_std, axis=0),
        c_phys=np.stack(all_c_phys, axis=0),
        paths=np.array([str(p) for p in files]),
        checkpoints=np.array([str(p) for p in args.checkpoints]),
    )

    aggregate = {
        "plume_ssim_mean": mean_or_none(row["mean_plume_ssim"] for row in per_sample),
        "plume_ssim_std": std_or_none(row["mean_plume_ssim"] for row in per_sample),
        "global_ssim_mean": mean_or_none(row["mean_global_ssim"] for row in per_sample),
        "global_ssim_std": std_or_none(row["mean_global_ssim"] for row in per_sample),
        "mass_error_mean": mean_or_none(row["mean_mass_error"] for row in per_sample),
        "mass_error_std": std_or_none(row["mean_mass_error"] for row in per_sample),
        "mean_pred_std_log": mean_or_none(row["mean_pred_std_log"] for row in per_sample),
        "plume_mean_pred_std_log": mean_or_none(
            row["plume_mean_pred_std_log"] for row in per_sample
        ),
        "n_samples": len(per_sample),
        "n_checkpoints": len(models),
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "axisD_d1_explicit_dataset_inference_eval",
        "data_root": str(data_root),
        "files": [str(p) for p in files],
        "checkpoints": [str(p) for p in args.checkpoints],
        "output_dir": str(output_dir),
        "predictions_npz": str(predictions_path),
        "evaluation_protocol": {
            "data_range": DATA_RANGE,
            "plume_thresh": PLUME_THRESH,
            "plume_pad": PLUME_PAD,
            "plume_min_pixels": PLUME_MIN_PIXELS,
            "eps_c": EPS_C,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "pred_clamp_min": args.pred_clamp_min,
            "pred_clamp_max": args.pred_clamp_max,
        },
        "aggregate": aggregate,
        "per_sample": [
            {k: finite_or_none(v) for k, v in row.items() if k != "timestep_metrics"}
            for row in per_sample
        ],
    }

    metrics_path = output_dir / "metrics_axisD_d1.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    timestep_path = output_dir / "metrics_axisD_d1_timestep.json"
    with timestep_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": report["created_utc"],
                "per_sample_timestep_metrics": [
                    {
                        "param_id": row["param_id"],
                        "real_id": row["real_id"],
                        "timestep_metrics": row["timestep_metrics"],
                    }
                    for row in per_sample
                ],
            },
            f,
            indent=2,
        )

    write_per_sample_csv(per_sample, output_dir / "metrics_axisD_d1_per_sample.csv")

    manifest_path = output_dir / "manifest_axisD_d1.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": report["created_utc"],
                "script": str(Path(__file__).resolve()),
                "data_root": str(data_root),
                "files": [str(p) for p in files],
                "checkpoints": [str(p) for p in args.checkpoints],
                "outputs": {
                    "predictions_npz": str(predictions_path),
                    "metrics_json": str(metrics_path),
                    "timestep_json": str(timestep_path),
                    "per_sample_csv": str(output_dir / "metrics_axisD_d1_per_sample.csv"),
                },
            },
            f,
            indent=2,
        )

    logger.info("Saved predictions: %s", predictions_path)
    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Aggregate: %s", aggregate)


if __name__ == "__main__":
    main()
