"""
Train MS-TMO-UNet (ASPP + self-attention) on the Obj3 OOD split.

Architecture reused from src/obj1/conference/train_unet_multiout_logc_B_aspp_attn.py.
Retrained from scratch on the Obj3 training set (σ²Y ∈ {0.1, 0.5, 1.0, 1.5}).

Usage:
    python -m src.obj3.conference.train_ms_tmo_obj3 \\
        --seed 0 --epochs 200 --batch_size 16 --amp

Smoke test:
    python -m src.obj3.conference.train_ms_tmo_obj3 \\
        --seed 0 --epochs 1 --batch_size 4 --no_amp \\
        --run_name obj3_conf_ms_tmo_det_full_smoke_seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.shared.utils.seed import set_seed
from src.obj3.conference.data_obj3_ood import (
    Obj3PatchDataset,
    collect_npz_files,
    load_obj3_split,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ============================================================================
# Architecture: UNet + ASPP + Self-Attention (MS-TMO-UNet)
# Identical to src/obj1/conference/train_unet_multiout_logc_B_aspp_attn.py
# ============================================================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class ASPP(nn.Module):
    def __init__(self, ch: int, rates: tuple = (1, 6, 12, 18)) -> None:
        super().__init__()
        groups = 8 if ch % 8 == 0 else 1
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, padding=r, dilation=r, bias=True),
                nn.GroupNorm(groups, ch),
                nn.SiLU(inplace=True),
            )
            for r in rates
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(ch * len(rates), ch, 1, 1, 0, bias=True),
            nn.GroupNorm(groups, ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class BottleneckAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(ch, 4 * ch),
            nn.GELU(),
            nn.Linear(4 * ch, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        s = s + self.attn(self.norm1(s), self.norm1(s), self.norm1(s), need_weights=False)[0]
        s = s + self.ff(self.norm2(s))
        return s.reshape(B, H, W, C).permute(0, 3, 1, 2)


class UNet_ASPP_Attn(nn.Module):
    """MS-TMO-UNet: U-Net with ASPP + self-attention bottleneck."""

    def __init__(self, in_ch: int = 1, out_ch: int = 25, base: int = 64, attn_heads: int = 4) -> None:
        super().__init__()
        b = base
        self.inc = ConvBlock(in_ch, b)
        self.d1 = Down(b, 2 * b)
        self.d2 = Down(2 * b, 4 * b)
        self.d3 = Down(4 * b, 8 * b)

        self.mid = ConvBlock(8 * b, 8 * b)
        self.aspp = ASPP(8 * b)
        self.attn = BottleneckAttention(8 * b, heads=attn_heads)

        self.u3 = Up(8 * b, 4 * b)
        self.u2 = Up(4 * b, 2 * b)
        self.u1 = Up(2 * b, b)
        self.out = nn.Conv2d(b, out_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)

        m = self.mid(x4)
        m = self.aspp(m)
        m = self.attn(m)

        y = self.u3(m, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)
        return self.out(y)


# ============================================================================
# Loss helpers (same as Obj1 fairness rules)
# ============================================================================


def make_weight_log_linear(y_log: torch.Tensor, alpha: float, y_bg: float, y_cap: float) -> torch.Tensor:
    denom = (y_cap - y_bg) if (y_cap - y_bg) != 0 else 1.0
    s = (y_log - y_bg) / denom
    s = torch.clamp(s, 0.0, 1.0)
    return 1.0 + alpha * s


def timestep_weights(T: int, mode: str = "quad", w_min: float = 0.5, w_max: float = 1.5) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, steps=T)
    s = t * t if mode == "quad" else t
    return w_min + (w_max - w_min) * s


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    hub = F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
    return (weight * hub).mean()


# ============================================================================
# Config
# ============================================================================


@dataclass
class CFG:
    # Paths
    main_data_root: str = str(REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL")
    extra_data_root: str = str(REPO_ROOT / "simulation" / "datasets" / "obj3_calibration")
    split_json: str = str(REPO_ROOT / "splits" / "param_split_obj3_ood.json")
    stats_json: str = str(REPO_ROOT / "metadata" / "obj3_train_stats.json")

    out_root: str = str(REPO_ROOT / "experiments" / "obj3" / "conference" / "runs")
    run_name: str = ""
    seed: int = 0
    deterministic: bool = False
    legacy_crop_sampling: bool = False

    # Training
    patch_size: int = 320
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Input normalisation
    use_logK: bool = True
    eps_k: float = 1e-6
    eps_c: float = 1e-12

    # Architecture
    base_channels: int = 64
    attn_heads: int = 4

    # Loss
    alpha: float = 3.0
    y_bg: float = -12.0
    y_cap: float = -2.0
    huber_delta: float = 1.0
    pred_clamp_min: float = -12.0
    pred_clamp_max: float = 2.0
    t_weight_mode: str = "quad"
    t_w_min: float = 0.5
    t_w_max: float = 1.5
    lambda_temp: float = 0.05

    # Engineering (fairness-locked)
    use_flip: bool = False
    resume: bool = False
    # Realization identifiers to train on. None means all five, the original protocol.
    realizations: 'list[int] | None' = None


def filter_realizations(files, realizations):
    """Keep only files whose real_XXX index is in `realizations`. None keeps everything."""
    if not realizations:
        return files
    keep = {int(r) for r in realizations}
    return [f for f in files if int(Path(f).stem.split("_")[-1]) in keep]


def parse_args() -> CFG:
    ap = argparse.ArgumentParser(description="Train MS-TMO-UNet on Obj3 OOD split")
    ap.add_argument("--main_data_root", type=str, default=CFG.main_data_root)
    ap.add_argument("--extra_data_root", type=str, default=CFG.extra_data_root)
    ap.add_argument("--split_json", type=str, default=CFG.split_json)
    ap.add_argument("--stats_json", type=str, default=CFG.stats_json)
    ap.add_argument("--realizations", type=int, nargs="*", default=None,
                    help="Restrict to these realization numbers, e.g. 1 2 3. Default: all. Used by the realization-disjoint twin control, since amplitude twins share a realization id.")
    ap.add_argument("--out_root", type=str, default=CFG.out_root)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--legacy_crop_sampling", action="store_true",
                    help="Use the crop-origin convention of the reported checkpoints")

    ap.add_argument("--patch_size", type=int, default=CFG.patch_size)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    ap.add_argument("--grad_clip_norm", type=float, default=CFG.grad_clip_norm)
    ap.add_argument("--num_workers", type=int, default=CFG.num_workers)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--base_channels", type=int, default=CFG.base_channels)
    ap.add_argument("--attn_heads", type=int, default=CFG.attn_heads)
    ap.add_argument("--alpha", type=float, default=CFG.alpha)
    ap.add_argument("--lambda_temp", type=float, default=CFG.lambda_temp)
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()
    cfg = CFG()
    for field in cfg.__dataclass_fields__:
        if hasattr(args, field):
            setattr(cfg, field, getattr(args, field))

    cfg.amp = True
    if args.no_amp:
        cfg.amp = False

    return cfg


# ============================================================================
# Training loop
# ============================================================================


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, cfg: CFG) -> dict:
    model.train()
    total_loss = total_main = total_temp = 0.0
    n = 0
    n_skipped = 0

    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.amp):
            pred = model(K)

        pred = torch.nan_to_num(pred.float(), nan=0.0,
                                posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
        Y_f = Y.float()
        predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)

        w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        T = Y_f.shape[1]
        w_t = timestep_weights(T, cfg.t_weight_mode, cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, T, 1, 1)
        w = w_y * w_t

        loss_main = weighted_huber(predL, Y_f, w, delta=cfg.huber_delta)
        d1 = predL[:, 1:] - predL[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        loss_temp = d2.abs().mean()
        loss = loss_main + cfg.lambda_temp * loss_temp

        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=False)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        total_loss += float(loss.item())
        total_main += float(loss_main.item())
        total_temp += float(loss_temp.item())
        n += 1

    if n_skipped > 0:
        logger.warning("%d/%d batches skipped (NaN grad)", n_skipped, n + n_skipped)

    return {"loss": total_loss / max(n, 1), "main": total_main / max(n, 1),
            "temp": total_temp / max(n, 1), "skipped": n_skipped}


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, cfg: CFG) -> dict:
    model.eval()
    total_loss = total_main = total_temp = 0.0
    n = 0

    for batch in loader:
        K = batch["K"].to(cfg.device, non_blocking=True)
        Y = batch["C_log"].to(cfg.device, non_blocking=True)

        pred = model(K)
        pred = torch.nan_to_num(pred.float(), nan=0.0,
                                posinf=cfg.pred_clamp_max, neginf=cfg.pred_clamp_min)
        Y_f = Y.float()
        predL = torch.clamp(pred, cfg.pred_clamp_min, cfg.pred_clamp_max)

        w_y = make_weight_log_linear(Y_f, cfg.alpha, cfg.y_bg, cfg.y_cap)
        T = Y_f.shape[1]
        w_t = timestep_weights(T, cfg.t_weight_mode, cfg.t_w_min, cfg.t_w_max).to(Y_f.device).view(1, T, 1, 1)
        w = w_y * w_t

        loss_main = weighted_huber(predL, Y_f, w, delta=cfg.huber_delta)
        d1 = predL[:, 1:] - predL[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        loss_temp = d2.abs().mean()
        loss = loss_main + cfg.lambda_temp * loss_temp

        if torch.isfinite(loss):
            total_loss += float(loss.item())
            total_main += float(loss_main.item())
            total_temp += float(loss_temp.item())
            n += 1

    return {"loss": total_loss / max(n, 1), "main": total_main / max(n, 1),
            "temp": total_temp / max(n, 1)}


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = parse_args()
    cfg.runtime = set_seed(cfg.seed, deterministic=cfg.deterministic)

    if not cfg.run_name:
        cfg.run_name = f"obj3_conf_ms_tmo_det_full_seed{cfg.seed}"

    out_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Load data
    splits = load_obj3_split(cfg.split_json)
    with open(cfg.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    train_files = collect_npz_files(splits["train"], cfg.main_data_root, cfg.extra_data_root)
    val_files = collect_npz_files(splits["val_calib"], cfg.main_data_root, cfg.extra_data_root)
    realizations = getattr(cfg, "realizations", None)
    if realizations:
        train_files = filter_realizations(train_files, realizations)
        val_files = filter_realizations(val_files, realizations)
        logger.info("Realization filter %s applied", sorted(realizations))

    logger.info("Train: %d files | Val: %d files", len(train_files), len(val_files))

    train_ds = Obj3PatchDataset(train_files, stats, patch_size=cfg.patch_size,
                                use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c,
                                legacy_crop_sampling=cfg.legacy_crop_sampling)
    val_ds = Obj3PatchDataset(val_files, stats, patch_size=cfg.patch_size,
                              use_logK=cfg.use_logK, eps_k=cfg.eps_k, eps_c=cfg.eps_c,
                              legacy_crop_sampling=cfg.legacy_crop_sampling)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = UNet_ASPP_Attn(in_ch=1, out_ch=25, base=cfg.base_channels,
                           attn_heads=cfg.attn_heads).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val = float("inf")
    start_epoch = 1
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")

    if cfg.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if cfg.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        logger.info("[RESUME] From epoch %d, best_val=%.6f", ckpt["epoch"], best_val)

    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_main",
                                    "val_main", "train_temp", "val_temp", "lr", "skipped"])

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "cfg": cfg.__dict__, "stats": stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if cfg.amp else None,
            "best_val": best_val,
        }, last_path)

        improved = ev["loss"] < best_val
        if improved:
            best_val = ev["loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                         "cfg": cfg.__dict__, "stats": stats, "best_val": best_val}, best_path)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, tr["loss"], ev["loss"], tr["main"],
                                    ev["main"], tr["temp"], ev["temp"], lr_now, tr["skipped"]])

        skip_str = f" | skip {tr['skipped']}" if tr["skipped"] > 0 else ""
        logger.info("Epoch %03d | train %.6f  val %.6f | lr %.2e | best %.6f %s%s",
                     epoch, tr["loss"], ev["loss"], lr_now, best_val,
                     "*" if improved else "", skip_str)

    logger.info("Training complete. Best: %s", best_path)


if __name__ == "__main__":
    main()
