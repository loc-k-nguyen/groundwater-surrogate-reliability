"""
Export model predictions to NPZ files for conformal calibration and evaluation.

Run this for each split (val_calib, extra_calib, iid_test, ood_test) after
ensemble training, to produce input files for eval_conformal.py.

For ensemble: exports mean predictions + std (for NEC).
For single-model: exports deterministic predictions only.

Usage (ensemble):
    python -m src.obj3.conference.export_predictions \\
        --checkpoints \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed0/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed1/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed2/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed3/best.pt \\
            experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed4/best.pt \\
        --splits val_calib extra_calib iid_test ood_test \\
        --output_dir experiments/obj3/conference/runs/obj3_conf_ms_tmo_ensemble5_full/preds
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.obj3.conference.data_obj3_ood import (
    Obj3FullFieldDataset,
    collect_npz_files,
    load_obj3_split,
)
from src.obj3.conference.eval_deterministic import sliding_window_inference
from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EPS_C = 1e-12


def load_models(checkpoint_paths: List[str], device: str) -> List[torch.nn.Module]:
    models = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt.get("cfg", {})
        model = UNet_ASPP_Attn(
            in_ch=1, out_ch=25,
            base=cfg.get("base_channels", 64),
            attn_heads=cfg.get("attn_heads", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
    return models


def export_split(
    models: List[torch.nn.Module],
    dataset: Obj3FullFieldDataset,
    output_path: Path,
    device: str,
    patch_size: int = 320,
    stride: int = 160,
    pred_clamp_min: float = -12.0,
    pred_clamp_max: float = 2.0,
    limit: int = 0,
) -> None:
    """Export predictions for one split into a single NPZ file."""
    n_samples = len(dataset) if limit <= 0 else min(limit, len(dataset))
    if n_samples == 0:
        raise ValueError(
            f"No samples resolved for export target '{output_path}'. "
            "This usually means the required dataset directories were not synced to the current machine."
        )
    all_y_true = []
    all_y_pred = []
    all_y_pred_std = []
    all_c_phys = []

    for i in range(n_samples):
        sample = dataset[i]
        K = sample["K"].unsqueeze(0).to(device)
        C_log_gt = sample["C_log"].numpy()           # (25, H, W)
        C_phys_gt = sample["C_phys"].numpy()          # (25, H, W)

        # Run all models
        member_preds = []
        for model in models:
            with torch.no_grad():
                pred = sliding_window_inference(model, K, patch_size, stride, device)
                pred = torch.clamp(pred, pred_clamp_min, pred_clamp_max)
                member_preds.append(pred.squeeze(0).cpu().numpy())

        member_preds_np = np.stack(member_preds, axis=0)   # (M, 25, H, W)
        mean_pred = member_preds_np.mean(axis=0)            # (25, H, W)
        std_pred = member_preds_np.std(axis=0)              # (25, H, W)

        all_y_true.append(C_log_gt)
        all_y_pred.append(mean_pred)
        all_y_pred_std.append(std_pred)
        all_c_phys.append(C_phys_gt)

        if (i + 1) % 20 == 0:
            logger.info("  Exported %d/%d", i + 1, n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        y_true=np.stack(all_y_true, axis=0),         # (N, 25, H, W)
        y_pred=np.stack(all_y_pred, axis=0),          # (N, 25, H, W)
        y_pred_std=np.stack(all_y_pred_std, axis=0),  # (N, 25, H, W)
        c_phys=np.stack(all_c_phys, axis=0),          # (N, 25, H, W)
    )
    logger.info("Saved %d samples → %s", n_samples, output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Export Obj3 predictions for conformal evaluation")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, default=str(REPO_ROOT / "splits" / "param_split_obj3_ood.json"))
    ap.add_argument("--splits", nargs="+", default=["val_calib", "extra_calib", "iid_test", "ood_test"])
    ap.add_argument("--main_data_root", type=str,
                    default=str(REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL"))
    ap.add_argument("--extra_data_root", type=str,
                    default=str(REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"))
    ap.add_argument("--stats_json", type=str,
                    default=str(REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"))
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    models = load_models(args.checkpoints, args.device)
    logger.info("Loaded %d model(s)", len(models))

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Combine val_calib + extra_calib for calibration export
    split_data = load_obj3_split(args.split_json)

    for split_name in args.splits:
        if split_name not in split_data:
            logger.warning("Split '%s' not found", split_name)
            continue
        out_path = output_dir / f"{split_name}_preds.npz"
        if out_path.exists():
            logger.info("Split '%s' already exported -> %s (skipping)", split_name, out_path)
            continue
        files = collect_npz_files(split_data[split_name], args.main_data_root, args.extra_data_root)
        dataset = Obj3FullFieldDataset(files, stats)
        logger.info("Split '%s': %d files", split_name, len(files))

        if split_name == "extra_calib" and len(files) == 0:
            raise FileNotFoundError(
                "Split 'extra_calib' resolved to 0 files. "
                f"Expected calibration configs under '{args.extra_data_root}'. "
                "Sync simulation/datasets/obj3_calibration to the execution machine before rerunning."
            )

        export_split(models, dataset, out_path, args.device,
                     args.patch_size, args.stride, limit=args.limit)

    # Also export combined calibration (val_calib + extra_calib merged)
    if "val_calib" in args.splits and "extra_calib" in args.splits:
        calib_out = output_dir / "calibration_preds.npz"
        if calib_out.exists():
            logger.info("Combined calibration already exported -> %s (skipping)", calib_out)
            return
        calib_ids = split_data["val_calib"] + split_data["extra_calib"]
        calib_files = collect_npz_files(calib_ids, args.main_data_root, args.extra_data_root)
        if len(calib_files) == 0:
            raise FileNotFoundError(
                "Combined calibration export resolved to 0 files. "
                "Sync both the main Obj3 dataset and extra calibration dataset to the execution machine."
            )
        calib_ds = Obj3FullFieldDataset(calib_files, stats)
        logger.info("Combined calibration: %d files", len(calib_files))
        export_split(models, calib_ds, calib_out,
                     args.device, args.patch_size, args.stride, limit=args.limit)


if __name__ == "__main__":
    main()
