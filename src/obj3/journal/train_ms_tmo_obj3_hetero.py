"""Train a HETEROSCEDASTIC MS-TMO-UNet on the Obj3 OOD split (Tier 2 UQ head).

Journal-extension UQ method: the backbone predicts BOTH the mean and a per-pixel,
per-timestep log-variance; trained with a weighted Gaussian NLL. This gives an
aleatoric predictive-uncertainty head as a comparison to the deep ensemble /
MC dropout — evaluated later on the reliability taxonomy (incl. Axis D transport).

Reuses the exact MS-TMO-UNet blocks + data + fairness settings from
train_ms_tmo_obj3.py; only the output head (25 -> 50 channels) and the loss
(weighted Gaussian NLL) change. NOT a new predictor for the benchmark.

Smoke:
    python -m src.obj3.journal.train_ms_tmo_obj3_hetero --seed 0 --epochs 1 \
        --batch_size 4 --no_amp --max_train_files 6 --max_val_files 4 \
        --run_name obj3_jnl_ms_tmo_hetero_smoke_seed0
Full:
    python -m src.obj3.journal.train_ms_tmo_obj3_hetero --seed 0 --epochs 200 --batch_size 16
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.shared.utils.seed import set_seed
from src.obj3.conference.data_obj3_ood import Obj3PatchDataset, collect_npz_files, load_obj3_split
from src.obj3.conference.train_ms_tmo_obj3 import filter_realizations
from src.obj3.conference.train_ms_tmo_obj3 import (
    UNet_ASPP_Attn, make_weight_log_linear, timestep_weights, CFG as DET_CFG,
)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]

T_OUT = 25
LOGVAR_MIN, LOGVAR_MAX = -8.0, 6.0


class HeteroUNet(nn.Module):
    """MS-TMO-UNet backbone with a mean+log-variance head (out=2*T)."""

    def __init__(self, base: int = 64, attn_heads: int = 4) -> None:
        super().__init__()
        self.net = UNet_ASPP_Attn(in_ch=1, out_ch=2 * T_OUT, base=base, attn_heads=attn_heads)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        mean, logvar = out[:, :T_OUT], out[:, T_OUT:]
        logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
        return mean, logvar


def weighted_gaussian_nll(mean, logvar, target, weight):
    # 0.5 * ( exp(-logvar) * (mean-target)^2 + logvar ), weighted, mean-reduced
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (inv_var * (mean - target) ** 2 + logvar)
    return (weight * nll).mean()


def _loss(mean, logvar, Y_f, cfg):
    predL = torch.clamp(mean, cfg.pred_clamp_min, cfg.pred_clamp_max)
    w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
    T = Y_f.shape[1]
    w_t = timestep_weights(T, cfg.t_weight_mode, cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, T, 1, 1)
    w = w_y * w_t
    nll = weighted_gaussian_nll(predL, logvar, Y_f, w)
    d2 = (predL[:, 2:] - 2 * predL[:, 1:-1] + predL[:, :-2])
    loss_temp = d2.abs().mean()
    return nll + cfg.lambda_temp * loss_temp, nll, loss_temp


def parse_args():
    ap = argparse.ArgumentParser(description="Train heteroscedastic MS-TMO-UNet on Obj3 OOD split")
    for f, d in [("main_data_root", DET_CFG.main_data_root), ("extra_data_root", DET_CFG.extra_data_root),
                 ("split_json", DET_CFG.split_json), ("stats_json", DET_CFG.stats_json)]:
        ap.add_argument(f"--{f}", type=str, default=d)
    ap.add_argument("--out_root", type=str, default=str(REPO_ROOT / "experiments/obj3/journal/runs"))
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--legacy_crop_sampling", action="store_true")
    ap.add_argument("--patch_size", type=int, default=DET_CFG.patch_size)
    ap.add_argument("--batch_size", type=int, default=DET_CFG.batch_size)
    ap.add_argument("--epochs", type=int, default=DET_CFG.epochs)
    ap.add_argument("--lr", type=float, default=DET_CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=DET_CFG.weight_decay)
    ap.add_argument("--grad_clip_norm", type=float, default=DET_CFG.grad_clip_norm)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--base_channels", type=int, default=DET_CFG.base_channels)
    ap.add_argument("--attn_heads", type=int, default=DET_CFG.attn_heads)
    ap.add_argument("--max_train_files", type=int, default=0)
    ap.add_argument("--max_val_files", type=int, default=0)
    ap.add_argument("--realizations", type=int, nargs="*", default=None,
                    help="Restrict to these realization numbers, e.g. 1 2 3. Default: all. Used by the realization-disjoint twin control, since amplitude twins share a realization id.")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = DET_CFG()
    for field in cfg.__dataclass_fields__:
        if hasattr(a, field):
            setattr(cfg, field, getattr(a, field))
    cfg.amp = not a.no_amp
    cfg._max_train = a.max_train_files
    cfg._max_val = a.max_val_files
    cfg._resume = a.resume
    return cfg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = parse_args()
    cfg.runtime = set_seed(cfg.seed, deterministic=cfg.deterministic)
    if not cfg.run_name:
        cfg.run_name = f"obj3_jnl_ms_tmo_hetero_seed{cfg.seed}"
    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}, f, indent=2)

    splits = load_obj3_split(cfg.split_json)
    with open(cfg.stats_json) as f:
        stats = json.load(f)
    train_files = collect_npz_files(splits["train"], cfg.main_data_root, cfg.extra_data_root)
    val_files = collect_npz_files(splits["val_calib"], cfg.main_data_root, cfg.extra_data_root)
    if getattr(cfg, "realizations", None):
        train_files = filter_realizations(train_files, cfg.realizations)
        val_files = filter_realizations(val_files, cfg.realizations)
    if cfg._max_train > 0:
        train_files = train_files[: cfg._max_train]
    if cfg._max_val > 0:
        val_files = val_files[: cfg._max_val]
    logger.info("Train %d | Val %d files", len(train_files), len(val_files))

    train_ds = Obj3PatchDataset(train_files, stats, patch_size=cfg.patch_size,
                               legacy_crop_sampling=cfg.legacy_crop_sampling)
    val_ds = Obj3PatchDataset(val_files, stats, patch_size=cfg.patch_size,
                             legacy_crop_sampling=cfg.legacy_crop_sampling)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = HeteroUNet(base=cfg.base_channels, attn_heads=cfg.attn_heads).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt"); last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_nll", "val_nll", "lr"])

    for epoch in range(1, cfg.epochs + 1):
        model.train(); tl = tn = 0.0; n = 0
        for batch in train_loader:
            K = batch["K"].to(cfg.device, non_blocking=True)
            Y = batch["C_log"].to(cfg.device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                mean, logvar = model(K)
            mean = torch.nan_to_num(mean.float(), nan=0.0, posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
            logvar = logvar.float()
            loss, nll, _ = _loss(mean, logvar, Y, cfg)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
                if not torch.isfinite(g):
                    opt.zero_grad(set_to_none=True); scaler.update(); continue
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
                opt.step()
            tl += float(loss.item()); tn += float(nll.item()); n += 1
        sched.step()
        # val
        model.eval(); vl = vn = 0.0; m = 0
        with torch.no_grad():
            for batch in val_loader:
                K = batch["K"].to(cfg.device); Y = batch["C_log"].to(cfg.device).float()
                mean, logvar = model(K)
                mean = torch.nan_to_num(mean.float(), nan=0.0, posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
                loss, nll, _ = _loss(mean, logvar.float(), Y, cfg)
                if torch.isfinite(loss):
                    vl += float(loss.item()); vn += float(nll.item()); m += 1
        tr_loss, tr_nll = tl / max(n, 1), tn / max(n, 1)
        va_loss, va_nll = vl / max(m, 1), vn / max(m, 1)
        lr_now = sched.get_last_lr()[0]
        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__, "stats": stats,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict() if cfg.amp else None, "best_val": best_val}, last_path)
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__,
                        "stats": stats, "best_val": best_val}, best_path)
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, va_loss, tr_nll, va_nll, lr_now])
        logger.info("Epoch %03d | train %.5f val %.5f | nll(tr %.4f va %.4f) | lr %.2e | best %.5f",
                    epoch, tr_loss, va_loss, tr_nll, va_nll, lr_now, best_val)
    # smoke sentinel
    if cfg.epochs <= 2 and os.path.exists(best_path):
        print("OBJ3_HETERO_SMOKE_PASS", flush=True)
    logger.info("Done. best=%s", best_path)


if __name__ == "__main__":
    main()
