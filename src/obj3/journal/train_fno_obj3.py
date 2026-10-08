"""Train an FNO backbone on the Obj3 OOD split (Tier 2 multi-backbone reliability).

Second architecture for the reliability-taxonomy generality check: is the
input-blind / uncertainty-catches-transport finding specific to MS-TMO-UNet, or
does it hold for a spectral (FNO) surrogate too? Reuses the EXACT Obj3
deterministic training loop, data, and fairness settings; only the model changes.

FNO precision = fp32 (AMP OFF) — complex spectral weights are incompatible with
GradScaler (project fairness rule).

Smoke:
    python -m src.obj3.journal.train_fno_obj3 --seed 0 --epochs 1 --batch_size 4 \
        --max_train_files 8 --max_val_files 4 --run_name obj3_jnl_fno_smoke_seed0
Full:
    python -m src.obj3.journal.train_fno_obj3 --seed 0 --epochs 200 --batch_size 16
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.shared.utils.seed import set_seed
from src.obj3.conference.data_obj3_ood import Obj3PatchDataset, collect_npz_files, load_obj3_split
from src.obj3.conference.train_ms_tmo_obj3 import filter_realizations
from src.obj3.conference.train_ms_tmo_obj3 import CFG, train_one_epoch, eval_one_epoch
from src.obj1.journal.models.fno_replicate import FNOReplicate2D

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> CFG:
    ap = argparse.ArgumentParser(description="Train FNO on Obj3 OOD split (fp32)")
    ap.add_argument("--main_data_root", default=CFG.main_data_root)
    ap.add_argument("--extra_data_root", default=CFG.extra_data_root)
    ap.add_argument("--split_json", default=CFG.split_json)
    ap.add_argument("--stats_json", default=CFG.stats_json)
    ap.add_argument("--out_root", default=str(REPO_ROOT / "experiments/obj3/journal/runs"))
    ap.add_argument("--run_name", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--legacy_crop_sampling", action="store_true")
    ap.add_argument("--patch_size", type=int, default=CFG.patch_size)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    ap.add_argument("--grad_clip_norm", type=float, default=CFG.grad_clip_norm)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--fno_width", type=int, default=64)
    ap.add_argument("--fno_modes", type=int, default=20)
    ap.add_argument("--fno_depth", type=int, default=4)
    ap.add_argument("--max_train_files", type=int, default=0)
    ap.add_argument("--max_val_files", type=int, default=0)
    ap.add_argument("--realizations", type=int, nargs="*", default=None,
                    help="Restrict to these realization numbers, e.g. 1 2 3. Default: all. Used by the realization-disjoint twin control, since amplitude twins share a realization id.")
    a = ap.parse_args()
    cfg = CFG()
    for f in cfg.__dataclass_fields__:
        if hasattr(a, f):
            setattr(cfg, f, getattr(a, f))
    cfg.amp = False  # FNO -> fp32 (fairness rule)
    cfg._fno = dict(width=a.fno_width, modes1=a.fno_modes, modes2=a.fno_modes, depth=a.fno_depth)
    cfg._max_train = a.max_train_files
    cfg._max_val = a.max_val_files
    return cfg


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = parse_args()
    cfg.runtime = set_seed(cfg.seed, deterministic=cfg.deterministic)
    if not cfg.run_name:
        cfg.run_name = f"obj3_jnl_fno_seed{cfg.seed}"
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
    print(f"[fno-obj3] Train {len(train_files)} | Val {len(val_files)} files (fp32)", flush=True)

    train_ds = Obj3PatchDataset(train_files, stats, patch_size=cfg.patch_size,
                               legacy_crop_sampling=cfg.legacy_crop_sampling)
    val_ds = Obj3PatchDataset(val_files, stats, patch_size=cfg.patch_size,
                             legacy_crop_sampling=cfg.legacy_crop_sampling)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    model = FNOReplicate2D(in_ch=1, out_ch=25, **cfg._fno).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    best_val = float("inf")
    best_path = os.path.join(out_dir, "best.pt"); last_path = os.path.join(out_dir, "last.pt")
    log_csv = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, scaler, cfg)
        ev = eval_one_epoch(model, val_loader, cfg)
        sched.step()
        lr_now = sched.get_last_lr()[0]
        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__, "stats": stats,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": None, "best_val": best_val}, last_path)
        if ev["loss"] < best_val:
            best_val = ev["loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__,
                        "stats": stats, "best_val": best_val}, best_path)
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr["loss"], ev["loss"], lr_now])
        print(f"[fno-obj3] Epoch {epoch:03d} | train {tr['loss']:.5f} val {ev['loss']:.5f} | lr {lr_now:.2e} | best {best_val:.5f}", flush=True)
    if cfg.epochs <= 2 and os.path.exists(best_path):
        print("OBJ3_FNO_SMOKE_PASS", flush=True)
    print(f"[fno-obj3] done best={best_path}", flush=True)


if __name__ == "__main__":
    main()
