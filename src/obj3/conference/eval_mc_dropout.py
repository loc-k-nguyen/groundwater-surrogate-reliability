"""
MC Dropout evaluation for Obj3.

Modifies inference to keep dropout active, runs T stochastic forward passes,
and computes MC mean + variance as uncertainty estimate.

Usage:
    python -m src.obj3.conference.eval_mc_dropout \\
        --checkpoint experiments/obj3/conference/runs/obj3_conf_ms_tmo_det_full_seed0/best.pt \\
        --output_dir experiments/obj3/conference/runs/obj3_conf_ms_tmo_mcdrop_full_seed0 \\
        --n_passes 50 --dropout_rate 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

from src.obj3.conference.data_obj3_ood import (
    Obj3FullFieldDataset,
    collect_npz_files,
    load_obj3_split,
)
from src.obj3.conference.eval_deterministic import sliding_window_inference
from src.obj3.conference.train_ms_tmo_obj3 import UNet_ASPP_Attn
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RANGE = 14.0
EPS_C = 1e-12
PLUME_THRESH = 1e-8
PLUME_PAD = 8
PLUME_MIN_PIXELS = 64


class UNet_ASPP_Attn_MCDrop(UNet_ASPP_Attn):
    """MS-TMO-UNet with dropout inserted at the bottleneck for MC Dropout inference."""

    def __init__(self, dropout_rate: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bottleneck_dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)

        m = self.mid(x4)
        m = self.aspp(m)
        m = self.bottleneck_dropout(m)  # Dropout at bottleneck
        m = self.attn(m)

        y = self.u3(m, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)
        return self.out(y)


def enable_mc_dropout(model: nn.Module) -> None:
    """Set only dropout layers to train mode (keep BN/GN in eval)."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_dropout_inference(
    model: nn.Module,
    K: torch.Tensor,
    n_passes: int = 50,
    patch_size: int = 320,
    stride: int = 160,
    device: str = "cuda",
    pred_clamp_min: float = -12.0,
    pred_clamp_max: float = 2.0,
) -> tuple:
    """Run T stochastic forward passes and compute MC mean + variance."""
    enable_mc_dropout(model)
    all_preds = []

    for _ in range(n_passes):
        pred = sliding_window_inference(model, K, patch_size, stride, device)
        pred = torch.clamp(pred, pred_clamp_min, pred_clamp_max)
        all_preds.append(pred.squeeze(0).cpu().numpy())

    all_preds_np = np.stack(all_preds, axis=0)  # (T, 25, H, W)
    mean_pred = all_preds_np.mean(axis=0)
    var_pred = all_preds_np.var(axis=0)

    return mean_pred, var_pred


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Obj3 MC Dropout evaluation")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--n_passes", type=int, default=50)
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument("--split_json", type=str, default=str(REPO_ROOT / "splits" / "param_split_obj3_ood.json"))
    ap.add_argument("--main_data_root", type=str,
                    default=str(REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL"))
    ap.add_argument("--extra_data_root", type=str,
                    default=str(REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"))
    ap.add_argument("--stats_json", type=str,
                    default=str(REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"))
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--eval_splits", nargs="+", default=["iid_test", "ood_test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weights into MC Dropout model
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt.get("cfg", {})
    model = UNet_ASPP_Attn_MCDrop(
        dropout_rate=args.dropout_rate,
        in_ch=1, out_ch=25,
        base=cfg.get("base_channels", 64),
        attn_heads=cfg.get("attn_heads", 4),
    ).to(args.device)

    # Load state dict (skip missing bottleneck_dropout — it's new)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info("Loaded checkpoint: %s (with MC dropout rate=%.2f, T=%d)",
                 args.checkpoint, args.dropout_rate, args.n_passes)

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    splits = load_obj3_split(args.split_json)

    for split_name in args.eval_splits:
        if split_name not in splits:
            continue

        files = collect_npz_files(splits[split_name], args.main_data_root, args.extra_data_root)
        dataset = Obj3FullFieldDataset(files, stats)
        n_samples = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
        logger.info("MC Dropout eval on '%s': %d samples, T=%d passes", split_name, n_samples, args.n_passes)

        results = []
        for i in range(n_samples):
            sample = dataset[i]
            K = sample["K"].unsqueeze(0).to(args.device)
            C_log_gt = sample["C_log"].numpy()
            C_phys_gt = sample["C_phys"].numpy()

            mean_pred, var_pred = mc_dropout_inference(
                model, K, args.n_passes, args.patch_size, args.stride, args.device,
            )

            plume_ssims = []
            global_ssims = []
            mass_errors = []
            for t in range(25):
                gs = float(ssim(C_log_gt[t], mean_pred[t], data_range=DATA_RANGE))
                gt_phys_t = np.clip(C_phys_gt[t], 0.0, None)
                ps, _ = compute_plume_ssim(gt_phys_t, mean_pred[t], EPS_C,
                                            PLUME_THRESH, PLUME_PAD, PLUME_MIN_PIXELS, DATA_RANGE)
                pred_phys = np.clip(10.0 ** mean_pred[t] - EPS_C, 0.0, None)
                gt_mass = float(gt_phys_t.sum())
                me = abs(float(pred_phys.sum()) - gt_mass) / max(gt_mass, 1e-12)

                global_ssims.append(gs)
                if np.isfinite(ps):
                    plume_ssims.append(ps)
                mass_errors.append(me)

            results.append({
                "param_id": sample["param_id"],
                "real_id": sample["real_id"],
                "mean_plume_ssim": float(np.mean(plume_ssims)) if plume_ssims else float("nan"),
                "mean_global_ssim": float(np.mean(global_ssims)),
                "mean_mass_error": float(np.mean(mass_errors)),
                "mean_mc_var": float(var_pred.mean()),
            })

            if (i + 1) % 5 == 0:
                logger.info("  [%d/%d] plume=%.4f mc_var=%.6f",
                             i + 1, n_samples, results[-1]["mean_plume_ssim"], results[-1]["mean_mc_var"])

        all_plume = [r["mean_plume_ssim"] for r in results if np.isfinite(r["mean_plume_ssim"])]
        all_global = [r["mean_global_ssim"] for r in results]
        all_mass = [r["mean_mass_error"] for r in results]
        all_var = [r["mean_mc_var"] for r in results]

        report = {
            "split": split_name, "checkpoint": args.checkpoint,
            "n_passes": args.n_passes, "dropout_rate": args.dropout_rate,
            "aggregate": {
                "plume_ssim_mean": float(np.mean(all_plume)) if all_plume else float("nan"),
                "plume_ssim_std": float(np.std(all_plume)) if all_plume else float("nan"),
                "global_ssim_mean": float(np.mean(all_global)) if all_global else float("nan"),
                "global_ssim_std": float(np.std(all_global)) if all_global else float("nan"),
                "mass_error_mean": float(np.mean(all_mass)) if all_mass else float("nan"),
                "mass_error_std": float(np.std(all_mass)) if all_mass else float("nan"),
                "mc_var_mean": float(np.mean(all_var)),
                "mc_var_std": float(np.std(all_var)),
                "n_samples": len(results),
            },
            "per_sample": results,
        }
        report_path = output_dir / f"mcdrop_metrics_{split_name}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("MC Dropout '%s': plume=%.4f±%.4f  mc_var=%.6f",
                     split_name, report["aggregate"]["plume_ssim_mean"],
                     report["aggregate"]["plume_ssim_std"], report["aggregate"]["mc_var_mean"])


if __name__ == "__main__":
    main()
