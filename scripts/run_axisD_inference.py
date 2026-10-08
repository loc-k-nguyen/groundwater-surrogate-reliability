"""Axis D ensemble inference (local GPU) — reuse existing trained seeds, NO training.

Bypasses the Obj3 id-range config resolver (which does not know Axis D id 9200) by
building the file list explicitly, then reuses the EXACT Obj3FullFieldDataset +
export_split internals (same preprocessing, sliding-window inference, clamp, save)
so outputs are consistent with the official ensemble preds.

Output npz keys: y_true, y_pred, y_pred_std, c_phys  (same schema as official preds).
"""
from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path
import argparse, glob, json
from pathlib import Path

import torch
from src.obj3.conference.export_predictions import load_models, export_split
from src.obj3.conference.data_obj3_ood import Obj3FullFieldDataset

REPO = DEFAULT_ROOT
DEF_STATS = REPO / "metadata/obj3_train_stats.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_dir", required=True, help="dir with real_*.npz for the Axis D config")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats_json", default=str(DEF_STATS))
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    files = sorted(Path(p) for p in glob.glob(str(Path(a.param_dir) / "real_*.npz")))
    if not files:
        raise FileNotFoundError(f"no real_*.npz under {a.param_dir}")
    with open(a.stats_json) as f:
        stats = json.load(f)
    models = load_models(a.checkpoints, a.device)
    print(f"[axisD-infer] {len(models)} seeds x {len(files)} realisations on {a.device}", flush=True)
    ds = Obj3FullFieldDataset(files, stats)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    export_split(models, ds, out, a.device, a.patch_size, a.stride)
    print(f"[axisD-infer] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
