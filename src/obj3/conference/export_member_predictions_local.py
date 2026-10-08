"""Contract-gated local export of per-member Obj3 ensemble predictions.

This script is intentionally limited to the local RTX 3090 Ti. It performs a
five-member, four-sample smoke before allowing the full IID/OOD export and
never overwrites an existing per-member artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.obj3.conference.data_obj3_ood import (
    Obj3FullFieldDataset,
    collect_npz_files,
    load_obj3_split,
)
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

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
MEMBERS = tuple(range(5))
SPLITS = ("iid_test", "ood_test")
SMOKE_LIMIT = 4
POOL_TOLERANCE = 2e-5
TRUTH_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def expected_outputs(output_dir: Path) -> list[Path]:
    return [
        output_dir / f"preds_member_seed{seed}_{split.removesuffix('_test')}.npz"
        for seed in MEMBERS
        for split in SPLITS
    ]


def assert_no_overwrite(output_dir: Path) -> None:
    collisions = [path for path in expected_outputs(output_dir) if path.exists()]
    wildcard_hits = sorted(output_dir.glob("preds_member_seed*.npz")) if output_dir.exists() else []
    collisions = sorted(set(collisions + wildcard_hits))
    if collisions:
        joined = "\n".join(str(path) for path in collisions)
        raise FileExistsError(f"No-overwrite gate failed; existing member outputs:\n{joined}")


def require_local_3090_ti() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local RTX 3090 Ti is required")
    name = torch.cuda.get_device_name(0)
    if "3090 Ti" not in name:
        raise RuntimeError(f"Wrong GPU: expected RTX 3090 Ti, found {name!r}")
    return name


def load_dataset(split_name: str, args: argparse.Namespace, limit: int = 0) -> Obj3FullFieldDataset:
    split = load_obj3_split(args.split_json)
    files = collect_npz_files(split[split_name], args.main_data_root, args.extra_data_root)
    expected = 110 if split_name == "iid_test" else 270
    if len(files) != expected:
        raise RuntimeError(f"{split_name} resolved {len(files)} files; expected {expected}")
    with Path(args.stats_json).open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if limit:
        files = files[:limit]
    return Obj3FullFieldDataset(files, stats)


def predict_member(
    checkpoint: Path,
    dataset: Obj3FullFieldDataset,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = load_models([str(checkpoint)], device)[0]
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    c_phys: list[np.ndarray] = []
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            k_field = sample["K"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = sliding_window_inference(model, k_field, 320, 160, device)
                pred = torch.clamp(pred, -12.0, 2.0)
            y_true.append(sample["C_log"].numpy())
            y_pred.append(pred.squeeze(0).cpu().numpy())
            c_phys.append(sample["C_phys"].numpy())
            LOGGER.info("%s: %d/%d", checkpoint.parent.name, index + 1, len(dataset))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return np.stack(y_true), np.stack(y_pred), np.stack(c_phys)


def smoke(args: argparse.Namespace, checkpoints: list[Path], status_dir: Path) -> dict:
    smoke_dir = status_dir / "smoke"
    if smoke_dir.exists() and not args.diagnosed_retry:
        raise FileExistsError(f"Smoke output already exists: {smoke_dir}")
    smoke_dir.mkdir(parents=True, exist_ok=args.diagnosed_retry)
    member_predictions: list[np.ndarray] = []
    reference_truth: np.ndarray | None = None
    reference_phys: np.ndarray | None = None

    for seed, checkpoint in zip(MEMBERS, checkpoints):
        smoke_path = smoke_dir / f"preds_member_seed{seed}_iid_smoke4.npz"
        if args.diagnosed_retry:
            if not smoke_path.exists():
                raise FileNotFoundError(f"Diagnosed retry requires existing smoke output: {smoke_path}")
            with np.load(smoke_path, allow_pickle=False) as saved:
                y_true = saved["y_true"]
                y_pred = saved["y_pred"]
                c_phys = saved["c_phys"]
        else:
            dataset = load_dataset("iid_test", args, SMOKE_LIMIT)
            y_true, y_pred, c_phys = predict_member(checkpoint, dataset, args.device)
        if reference_truth is None:
            reference_truth, reference_phys = y_true, c_phys
        elif not np.allclose(reference_truth, y_true, rtol=0.0, atol=TRUTH_TOLERANCE) or not np.array_equal(reference_phys, c_phys):
            raise RuntimeError("Smoke ground truth differs across member exports")
        member_predictions.append(y_pred)
        if not args.diagnosed_retry:
            np.savez_compressed(smoke_path, y_true=y_true, y_pred=y_pred, c_phys=c_phys)

    ensemble_mean = np.mean(np.stack(member_predictions, axis=0), axis=0, dtype=np.float32)
    with np.load(args.pooled_iid, allow_pickle=False) as pooled:
        pooled_mean = pooled["y_pred"][:SMOKE_LIMIT]
        pooled_truth = pooled["y_true"][:SMOKE_LIMIT]
        pooled_phys = pooled["c_phys"][:SMOKE_LIMIT]
    truth_max_abs = float(np.max(np.abs(reference_truth - pooled_truth)))
    if truth_max_abs > TRUTH_TOLERANCE or not np.array_equal(reference_phys, pooled_phys):
        raise RuntimeError("Smoke data order/schema does not match the pooled IID cache")

    absolute = np.abs(ensemble_mean - pooled_mean)
    max_abs = float(np.max(absolute))
    mean_abs = float(np.mean(absolute))
    if not np.isfinite(max_abs) or max_abs > POOL_TOLERANCE:
        raise RuntimeError(
            f"Pooled-mean gate failed: max_abs={max_abs:.8g} exceeds {POOL_TOLERANCE:.8g}"
        )

    plume_scores: list[float] = []
    mass_errors: list[float] = []
    for sample_index in range(SMOKE_LIMIT):
        timestep_scores: list[float] = []
        timestep_mass: list[float] = []
        for timestep in range(ensemble_mean.shape[1]):
            gt_phys = np.clip(reference_phys[sample_index, timestep], 0.0, None)
            score, _ = compute_plume_ssim(
                gt_phys,
                ensemble_mean[sample_index, timestep],
                EPS_C,
                PLUME_THRESH,
                PLUME_PAD,
                PLUME_MIN_PIXELS,
                DATA_RANGE,
            )
            if np.isfinite(score):
                timestep_scores.append(float(score))
            pred_phys = np.clip(10.0 ** ensemble_mean[sample_index, timestep] - EPS_C, 0.0, None)
            timestep_mass.append(
                abs(float(pred_phys.sum()) - float(gt_phys.sum())) / max(float(gt_phys.sum()), 1e-12)
            )
        plume_scores.append(float(np.mean(timestep_scores)))
        mass_errors.append(float(np.mean(timestep_mass)))

    if not all(np.isfinite(plume_scores)) or not all(-1.0 <= value <= 1.0 for value in plume_scores):
        raise RuntimeError(f"Smoke plume SSIM sanity gate failed: {plume_scores}")
    if not all(np.isfinite(mass_errors)):
        raise RuntimeError(f"Smoke mass sanity gate failed: {mass_errors}")

    result = {
        "status": "PASS_DIAGNOSED_RETRY" if args.diagnosed_retry else "PASS",
        "phase": "smoke",
        "samples": SMOKE_LIMIT,
        "members": len(MEMBERS),
        "pooled_tolerance": POOL_TOLERANCE,
        "pooled_max_abs": max_abs,
        "pooled_mean_abs": mean_abs,
        "truth_tolerance": TRUTH_TOLERANCE,
        "truth_max_abs": truth_max_abs,
        "mean_plume_ssim": float(np.mean(plume_scores)),
        "mean_relative_mass_error": float(np.mean(mass_errors)),
    }
    write_json(status_dir / "smoke_status.json", result)
    return result


def save_full_member(
    checkpoint: Path,
    dataset: Obj3FullFieldDataset,
    output_path: Path,
    device: str,
    status_dir: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"No-overwrite gate failed: {output_path}")
    partial_path = output_path.with_suffix(".partial.npz")
    if partial_path.exists():
        raise FileExistsError(f"Partial output requires diagnosis before retry: {partial_path}")

    first = dataset[0]
    shape = (len(dataset),) + tuple(first["C_log"].shape)
    temp_dir = status_dir / "tmp" / output_path.stem
    if temp_dir.exists():
        raise FileExistsError(f"Temporary output requires diagnosis before retry: {temp_dir}")
    temp_dir.mkdir(parents=True)
    y_true = np.lib.format.open_memmap(temp_dir / "y_true.npy", mode="w+", dtype=np.float32, shape=shape)
    y_pred = np.lib.format.open_memmap(temp_dir / "y_pred.npy", mode="w+", dtype=np.float32, shape=shape)
    c_phys = np.lib.format.open_memmap(temp_dir / "c_phys.npy", mode="w+", dtype=np.float32, shape=shape)

    model = load_models([str(checkpoint)], device)[0]
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            k_field = sample["K"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = sliding_window_inference(model, k_field, 320, 160, device)
                pred = torch.clamp(pred, -12.0, 2.0)
            y_true[index] = sample["C_log"].numpy()
            y_pred[index] = pred.squeeze(0).cpu().numpy()
            c_phys[index] = sample["C_phys"].numpy()
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                LOGGER.info("%s %s: %d/%d", checkpoint.parent.name, output_path.stem, index + 1, len(dataset))
        y_true.flush()
        y_pred.flush()
        c_phys.flush()
        np.savez_compressed(partial_path, y_true=y_true, y_pred=y_pred, c_phys=c_phys)
        partial_path.replace(output_path)
    finally:
        del model, y_true, y_pred, c_phys
        gc.collect()
        torch.cuda.empty_cache()

    for temp_file in temp_dir.glob("*.npy"):
        temp_file.unlink()
    temp_dir.rmdir()


def full_export(args: argparse.Namespace, checkpoints: list[Path], status_dir: Path) -> dict:
    smoke_path = status_dir / "smoke_status.json"
    smoke_status = json.loads(smoke_path.read_text(encoding="utf-8")).get("status", "") if smoke_path.exists() else ""
    if not smoke_status.startswith("PASS"):
        raise RuntimeError("Full export requires a passing smoke_status.json")
    assert_no_overwrite(Path(args.output_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for split_name in SPLITS:
        dataset = load_dataset(split_name, args)
        for seed, checkpoint in zip(MEMBERS, checkpoints):
            output_path = Path(args.output_dir) / f"preds_member_seed{seed}_{split_name.removesuffix('_test')}.npz"
            save_full_member(checkpoint, dataset, output_path, args.device, status_dir)

    outputs = expected_outputs(Path(args.output_dir))
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"Export finished with missing outputs: {missing}")
    result = {
        "status": "PASS",
        "phase": "full",
        "outputs": [{"path": str(path), "bytes": path.stat().st_size} for path in outputs],
    }
    write_json(status_dir / "full_status.json", result)
    return result


def checkpoint_paths() -> list[Path]:
    return [
        REPO_ROOT
        / "experiments"
        / "obj3"
        / "conference"
        / "runs"
        / f"obj3_conf_ms_tmo_det_full_seed{seed}"
        / "best.pt"
        for seed in MEMBERS
    ]


def validate_inputs(checkpoints: Iterable[Path], args: argparse.Namespace) -> dict:
    paths = list(checkpoints)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    for required in (args.split_json, args.stats_json, args.main_data_root, args.pooled_iid):
        if not Path(required).exists():
            raise FileNotFoundError(required)
    return {
        "gpu": require_local_3090_ti(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "split_json": str(Path(args.split_json).resolve()),
        "split_sha256": sha256(Path(args.split_json)),
        "checkpoints": [str(path.resolve()) for path in paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--diagnosed-retry", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "obj3"
            / "conference"
            / "runs"
            / "obj3_conf_ms_tmo_ensemble5_full"
            / "preds"
        ),
    )
    parser.add_argument(
        "--status-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "obj3"
            / "conference"
            / "runs"
            / "obj3_conf_ms_tmo_ensemble5_full"
            / "per_member_export_2026_07_13"
        ),
    )
    parser.add_argument("--split-json", default=str(REPO_ROOT / "splits" / "param_split_obj3_ood.json"))
    parser.add_argument(
        "--main-data-root",
        default=str(
            REPO_ROOT
            / "Obj1"
            / "obj1_surrogate_conference"
            / "data"
            / "T25_TSTEP_OVERRIDE_FINAL"
            / "T25_TSTEP_OVERRIDE_FINAL_FLIPPED"
        ),
    )
    parser.add_argument(
        "--extra-data-root",
        default=str(REPO_ROOT / "simulation" / "datasets" / "obj3_calibration"),
    )
    parser.add_argument(
        "--stats-json",
        default=str(REPO_ROOT / "experiments" / "obj3" / "conference" / "configs" / "obj3_train_stats.json"),
    )
    parser.add_argument(
        "--pooled-iid",
        default=str(
            REPO_ROOT
            / "experiments"
            / "obj3"
            / "conference"
            / "runs"
            / "obj3_conf_ms_tmo_ensemble5_full"
            / "preds"
            / "iid_test_preds.npz"
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_paths()
    try:
        manifest = validate_inputs(checkpoints, args)
        manifest["phase"] = args.phase
        manifest["output_dir"] = str(Path(args.output_dir).resolve())
        manifest_name = "manifest_diagnosed_retry.json" if args.diagnosed_retry else "manifest.json"
        write_json(status_dir / manifest_name, manifest)
        assert_no_overwrite(Path(args.output_dir))
        result = smoke(args, checkpoints, status_dir) if args.phase == "smoke" else full_export(args, checkpoints, status_dir)
        LOGGER.info("%s", json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        LOGGER.exception("Per-member export failed")
        write_json(
            status_dir / f"{args.phase}_failure.json",
            {"status": "FAIL", "phase": args.phase, "error_type": type(exc).__name__, "error": str(exc)},
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
