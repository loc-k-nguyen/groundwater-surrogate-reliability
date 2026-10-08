from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.obj3.conference.data_obj3_ood import Obj3FullFieldDataset, collect_npz_files, load_obj3_split
from src.obj3.conference.export_predictions import export_split, load_models
from src.obj3.journal.aci import plume_residual_scores
from src.obj3.journal.crc import sample_plume_mae


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SOURCE_PREDS = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_ensemble5_full" / "preds" / "calibration_preds.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "obj3" / "journal" / "calibration"
DEFAULT_SPLIT_JSON = REPO_ROOT / "splits" / "param_split_obj3_ood.json"
DEFAULT_MAIN_ROOT = REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL"
DEFAULT_EXTRA_ROOT = REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"
DEFAULT_STATS_JSON = REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"


def _derive_calibration_artifacts(calibration_npz: Path, output_dir: Path) -> None:
    payload = np.load(calibration_npz, allow_pickle=False)
    pixel_scores = plume_residual_scores(payload["y_true"], payload["y_pred"], payload["c_phys"])
    plume_mae = sample_plume_mae(payload["y_true"], payload["y_pred"], payload["c_phys"])
    np.save(output_dir / "calibration_scores.npy", pixel_scores)
    np.save(output_dir / "calibration_plume_mae.npy", plume_mae)


def _export_from_checkpoints(args: argparse.Namespace, target_npz: Path) -> None:
    if not args.checkpoints:
        raise FileNotFoundError(
            "Conference calibration_preds.npz is unavailable and no --checkpoints were provided "
            "for a fallback export."
        )

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    split_data = load_obj3_split(args.split_json)
    calib_ids = split_data["val_calib"] + split_data["extra_calib"]
    calib_files = collect_npz_files(calib_ids, args.main_data_root, args.extra_data_root)
    if not calib_files:
        raise FileNotFoundError("Combined calibration set resolved to zero files.")

    dataset = Obj3FullFieldDataset(calib_files, stats)
    models = load_models(args.checkpoints, args.device)
    export_split(
        models=models,
        dataset=dataset,
        output_path=target_npz,
        device=args.device,
        patch_size=args.patch_size,
        stride=args.stride,
        limit=args.limit,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare journal calibration prediction artifacts for ACI and CRC.")
    ap.add_argument("--source_preds", type=str, default=str(DEFAULT_SOURCE_PREDS))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT_JSON))
    ap.add_argument("--main_data_root", type=str, default=str(DEFAULT_MAIN_ROOT))
    ap.add_argument("--extra_data_root", type=str, default=str(DEFAULT_EXTRA_ROOT))
    ap.add_argument("--stats_json", type=str, default=str(DEFAULT_STATS_JSON))
    ap.add_argument("--patch_size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="Limit calibration samples for smoke tests.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_preds = Path(args.source_preds)
    target_npz = output_dir / "calibration_preds.npz"

    if not target_npz.exists():
        if source_preds.exists():
            shutil.copy2(source_preds, target_npz)
        else:
            _export_from_checkpoints(args, target_npz)

    _derive_calibration_artifacts(target_npz, output_dir)


if __name__ == "__main__":
    main()
