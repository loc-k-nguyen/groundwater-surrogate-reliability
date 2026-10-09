"""
Evaluate heteroscedastic Obj3 journal checkpoints on official splits and Axis D.

This is evaluation-only. It does not train models, modify split JSON files, or
write into conference output roots.
"""

from __future__ import annotations
from package_paths import DEFAULT_ROOT, asset_path, metadata_path, configure_script_paths

if __name__ == "__main__":
    DEFAULT_ROOT = configure_script_paths()

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

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
)
from src.obj3.journal.train_ms_tmo_obj3_hetero import HeteroUNet
from src.shared.eval.plume_ssim import compute_plume_ssim

logger = logging.getLogger(__name__)

REPO_ROOT = DEFAULT_ROOT
DEFAULT_MAIN_ROOT = asset_path("data", REPO_ROOT / "Obj1" / "obj1_surrogate_conference" / "data" / "T25_TSTEP_OVERRIDE_FINAL")
DEFAULT_EXTRA_ROOT = asset_path("calibration", REPO_ROOT / "simulation" / "datasets" / "obj3_calibration")
DEFAULT_SPLIT = REPO_ROOT / "splits" / "param_split_obj3_ood.json"
DEFAULT_STATS = REPO_ROOT / "metadata" / "obj3_train_stats.json"
DEFAULT_AXISD_D1 = (
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


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def mean_or_none(values: Iterable[Any]) -> float | None:
    vals = []
    for value in values:
        if value is None:
            continue
        if np.isfinite(value):
            vals.append(float(value))
    return float(np.mean(vals)) if vals else None


def std_or_none(values: Iterable[Any]) -> float | None:
    vals = []
    for value in values:
        if value is None:
            continue
        if np.isfinite(value):
            vals.append(float(value))
    return float(np.std(vals)) if vals else None


def ensure_clean_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory or pass --force intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_models(checkpoints: Sequence[str], device: str) -> List[torch.nn.Module]:
    models: List[torch.nn.Module] = []
    for checkpoint in checkpoints:
        ckpt = torch.load(checkpoint, map_location=device)
        cfg = ckpt.get("cfg", {})
        model = HeteroUNet(
            base=cfg.get("base_channels", 64),
            attn_heads=cfg.get("attn_heads", 4),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
        logger.info("Loaded hetero checkpoint: %s epoch=%s", checkpoint, ckpt.get("epoch"))
    return models


def patch_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError(
            f"patch_size and stride must be positive, got patch_size={patch_size}, stride={stride}"
        )
    if patch_size >= length:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    end_position = length - patch_size
    if positions[-1] != end_position:
        positions.append(end_position)
    return positions


def sliding_window_hetero(
    model: torch.nn.Module,
    k_full: torch.Tensor,
    patch_size: int,
    stride: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, h, w = k_full.shape
    out_channels = 25

    hann_h = torch.hann_window(patch_size, periodic=False).to(device)
    hann_w = torch.hann_window(patch_size, periodic=False).to(device)
    hann_2d = (hann_h.unsqueeze(1) * hann_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    hann_2d = torch.clamp(hann_2d, min=1e-3)

    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    k_padded = F.pad(k_full, [0, pad_w, 0, pad_h], mode="reflect")
    h_pad, w_pad = k_padded.shape[2], k_padded.shape[3]

    mean_padded = torch.zeros(1, out_channels, h_pad, w_pad, device=device)
    logvar_padded = torch.zeros(1, out_channels, h_pad, w_pad, device=device)
    weight_padded = torch.zeros(1, 1, h_pad, w_pad, device=device)

    row_positions = patch_positions(h_pad, patch_size, stride)
    col_positions = patch_positions(w_pad, patch_size, stride)
    for r in row_positions:
        for c in col_positions:
            patch = k_padded[:, :, r : r + patch_size, c : c + patch_size]
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                mean_patch, logvar_patch = model(patch)
            mean_patch = mean_patch.float()
            logvar_patch = logvar_patch.float()
            mean_padded[:, :, r : r + patch_size, c : c + patch_size] += mean_patch * hann_2d
            logvar_padded[:, :, r : r + patch_size, c : c + patch_size] += logvar_patch * hann_2d
            weight_padded[:, :, r : r + patch_size, c : c + patch_size] += hann_2d

    weight = weight_padded[:, :, :h, :w]
    if torch.any(weight <= 0):
        raise RuntimeError(
            f"Sliding-window coverage failure for field={(h, w)}, patch={patch_size}, "
            f"stride={stride}, rows={row_positions}, cols={col_positions}"
        )
    mean = mean_padded[:, :, :h, :w] / weight
    logvar = logvar_padded[:, :, :h, :w] / weight
    return mean, logvar


def evaluate_sample(mean_log: np.ndarray, total_std: np.ndarray, aleatoric_std: np.ndarray, epistemic_std: np.ndarray, sample: Dict[str, object]) -> Dict[str, Any]:
    c_log_gt = sample["C_log"].numpy()
    c_phys_gt = sample["C_phys"].numpy()
    timestep_metrics: List[Dict[str, Any]] = []

    for t in range(mean_log.shape[0]):
        gt_log_t = c_log_gt[t]
        pred_log_t = mean_log[t]
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

        timestep_metrics.append(
            {
                "t": int(t),
                "global_ssim": global_ssim,
                "plume_ssim": finite_or_none(plume_ssim_val),
                "mass_error": mass_error,
                "plume_pixels": int(plume_pixels),
                "mean_total_std_log": float(np.mean(total_std[t])),
                "mean_aleatoric_std_log": float(np.mean(aleatoric_std[t])),
                "mean_epistemic_std_log": float(np.mean(epistemic_std[t])),
                "plume_total_std_log": float(np.mean(total_std[t][plume_mask])) if np.any(plume_mask) else None,
                "plume_aleatoric_std_log": float(np.mean(aleatoric_std[t][plume_mask])) if np.any(plume_mask) else None,
                "plume_epistemic_std_log": float(np.mean(epistemic_std[t][plume_mask])) if np.any(plume_mask) else None,
            }
        )

    return {
        "param_id": sample["param_id"],
        "real_id": sample["real_id"],
        "path": sample["path"],
        "mean_plume_ssim": mean_or_none(m["plume_ssim"] for m in timestep_metrics),
        "mean_global_ssim": mean_or_none(m["global_ssim"] for m in timestep_metrics),
        "mean_mass_error": mean_or_none(m["mass_error"] for m in timestep_metrics),
        "mean_total_std_log": mean_or_none(m["mean_total_std_log"] for m in timestep_metrics),
        "mean_aleatoric_std_log": mean_or_none(m["mean_aleatoric_std_log"] for m in timestep_metrics),
        "mean_epistemic_std_log": mean_or_none(m["mean_epistemic_std_log"] for m in timestep_metrics),
        "plume_total_std_log": mean_or_none(m["plume_total_std_log"] for m in timestep_metrics),
        "plume_aleatoric_std_log": mean_or_none(m["plume_aleatoric_std_log"] for m in timestep_metrics),
        "plume_epistemic_std_log": mean_or_none(m["plume_epistemic_std_log"] for m in timestep_metrics),
        "timestep_metrics": timestep_metrics,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = [
        "mean_plume_ssim",
        "mean_global_ssim",
        "mean_mass_error",
        "mean_total_std_log",
        "mean_aleatoric_std_log",
        "mean_epistemic_std_log",
        "plume_total_std_log",
        "plume_aleatoric_std_log",
        "plume_epistemic_std_log",
    ]
    out: Dict[str, Any] = {"n_samples": len(rows)}
    for field in fields:
        out[f"{field}_mean"] = mean_or_none(row[field] for row in rows)
        out[f"{field}_std"] = std_or_none(row[field] for row in rows)
    return out


def rank_auc(negative_scores: Sequence[float], positive_scores: Sequence[float]) -> float | None:
    neg = [float(x) for x in negative_scores if np.isfinite(x)]
    pos = [float(x) for x in positive_scores if np.isfinite(x)]
    if not neg or not pos:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / (len(pos) * len(neg)))


def maybe_mannwhitney(negative_scores: Sequence[float], positive_scores: Sequence[float]) -> Dict[str, Any]:
    neg = [float(x) for x in negative_scores if np.isfinite(x)]
    pos = [float(x) for x in positive_scores if np.isfinite(x)]
    result = {
        "n_negative": len(neg),
        "n_positive": len(pos),
        "auc_positive_gt_negative": rank_auc(neg, pos),
        "mannwhitneyu_stat": None,
        "mannwhitneyu_p_one_sided_positive_greater": None,
    }
    try:
        from scipy.stats import mannwhitneyu

        stat = mannwhitneyu(pos, neg, alternative="greater")
        result["mannwhitneyu_stat"] = float(stat.statistic)
        result["mannwhitneyu_p_one_sided_positive_greater"] = float(stat.pvalue)
    except Exception as exc:  # pragma: no cover - optional dependency path
        result["mannwhitneyu_note"] = f"scipy_unavailable_or_failed: {exc}"
    return result


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "target",
        "param_id",
        "real_id",
        "path",
        "mean_plume_ssim",
        "mean_global_ssim",
        "mean_mass_error",
        "mean_total_std_log",
        "mean_aleatoric_std_log",
        "mean_epistemic_std_log",
        "plume_total_std_log",
        "plume_aleatoric_std_log",
        "plume_epistemic_std_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: finite_or_none(row.get(name)) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heteroscedastic Obj3 UQ checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_json", default=str(DEFAULT_SPLIT))
    parser.add_argument("--main_data_root", default=str(DEFAULT_MAIN_ROOT))
    parser.add_argument("--extra_data_root", default=str(DEFAULT_EXTRA_ROOT))
    parser.add_argument("--stats_json", default=str(DEFAULT_STATS))
    parser.add_argument("--splits", nargs="+", default=["iid_test", "ood_test"])
    parser.add_argument("--axisd_d1_root", default=str(DEFAULT_AXISD_D1))
    parser.add_argument("--include_axisd_d1", action="store_true")
    parser.add_argument("--patch_size", type=int, default=320)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pred_clamp_min", type=float, default=-12.0)
    parser.add_argument("--pred_clamp_max", type=float, default=2.0)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_clean_output_dir(output_dir, args.force)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    models = load_models(args.checkpoints, args.device)
    split_data = load_obj3_split(args.split_json)

    targets: Dict[str, List[Path]] = {}
    for split_name in args.splits:
        files = collect_npz_files(split_data[split_name], args.main_data_root, args.extra_data_root)
        if args.limit > 0:
            files = files[: args.limit]
        targets[split_name] = files
    if args.include_axisd_d1:
        axisd_files = sorted(Path(args.axisd_d1_root).glob("real_*.npz"))
        if args.limit > 0:
            axisd_files = axisd_files[: args.limit]
        targets["axisD_d1"] = axisd_files

    all_rows: List[Dict[str, Any]] = []
    target_summaries: Dict[str, Any] = {}
    npz_outputs: Dict[str, str] = {}

    for target_name, files in targets.items():
        if not files:
            raise FileNotFoundError(f"No files resolved for target {target_name}")
        dataset = Obj3FullFieldDataset(files, stats)
        rows: List[Dict[str, Any]] = []
        saved_mean: List[np.ndarray] = []
        saved_total_std: List[np.ndarray] = []
        saved_c_log: List[np.ndarray] = []
        saved_c_phys: List[np.ndarray] = []
        logger.info("Evaluating target=%s files=%d checkpoints=%d", target_name, len(dataset), len(models))

        for i in range(len(dataset)):
            sample = dataset[i]
            k_full = sample["K"].unsqueeze(0).to(args.device)
            member_means: List[np.ndarray] = []
            member_vars: List[np.ndarray] = []
            for model in models:
                mean, logvar = sliding_window_hetero(model, k_full, args.patch_size, args.stride, args.device)
                mean = torch.clamp(mean, args.pred_clamp_min, args.pred_clamp_max)
                member_means.append(mean.squeeze(0).cpu().numpy())
                member_vars.append(torch.exp(logvar.squeeze(0).cpu()).numpy())

            means_np = np.stack(member_means, axis=0)
            vars_np = np.stack(member_vars, axis=0)
            mean_log = means_np.mean(axis=0)
            epistemic_var = means_np.var(axis=0)
            aleatoric_var = vars_np.mean(axis=0)
            total_std = np.sqrt(np.maximum(aleatoric_var + epistemic_var, 0.0))
            aleatoric_std = np.sqrt(np.maximum(aleatoric_var, 0.0))
            epistemic_std = np.sqrt(np.maximum(epistemic_var, 0.0))

            row = evaluate_sample(mean_log, total_std, aleatoric_std, epistemic_std, sample)
            row["target"] = target_name
            rows.append(row)
            all_rows.append(row)
            if args.save_npz:
                saved_mean.append(mean_log)
                saved_total_std.append(total_std)
                saved_c_log.append(sample["C_log"].numpy())
                saved_c_phys.append(sample["C_phys"].numpy())

            logger.info(
                "[%s %d/%d] %s/%s plume=%s total_std=%s",
                target_name,
                i + 1,
                len(dataset),
                sample["param_id"],
                sample["real_id"],
                row["mean_plume_ssim"],
                row["mean_total_std_log"],
            )

        target_summaries[target_name] = summarize(rows)
        if args.save_npz:
            npz_path = output_dir / f"{target_name}_hetero_preds.npz"
            np.savez_compressed(
                str(npz_path),
                y_true=np.stack(saved_c_log, axis=0),
                y_pred=np.stack(saved_mean, axis=0),
                y_pred_total_std=np.stack(saved_total_std, axis=0),
                c_phys=np.stack(saved_c_phys, axis=0),
                paths=np.array([str(p) for p in files]),
                checkpoints=np.array([str(p) for p in args.checkpoints]),
            )
            npz_outputs[target_name] = str(npz_path)

    comparisons: Dict[str, Any] = {}
    if "iid_test" in targets and "ood_test" in targets:
        iid_scores = [row["mean_total_std_log"] for row in all_rows if row["target"] == "iid_test"]
        ood_scores = [row["mean_total_std_log"] for row in all_rows if row["target"] == "ood_test"]
        comparisons["ood_vs_iid_total_std"] = maybe_mannwhitney(iid_scores, ood_scores)
    if "iid_test" in targets and "axisD_d1" in targets:
        iid_scores = [row["mean_total_std_log"] for row in all_rows if row["target"] == "iid_test"]
        axis_scores = [row["mean_total_std_log"] for row in all_rows if row["target"] == "axisD_d1"]
        comparisons["axisD_d1_vs_iid_total_std"] = maybe_mannwhitney(iid_scores, axis_scores)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "obj3_journal_heteroscedastic_uq_eval",
        "checkpoints": [str(p) for p in args.checkpoints],
        "output_dir": str(output_dir),
        "targets": {name: [str(p) for p in files] for name, files in targets.items()},
        "protocol": {
            "data_range": DATA_RANGE,
            "plume_thresh": PLUME_THRESH,
            "plume_pad": PLUME_PAD,
            "plume_min_pixels": PLUME_MIN_PIXELS,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "pred_clamp_min": args.pred_clamp_min,
            "pred_clamp_max": args.pred_clamp_max,
            "uncertainty_space": "log10 concentration",
            "total_var": "mean(exp(logvar)) + var(seed_mean)",
        },
        "summaries": target_summaries,
        "comparisons": comparisons,
        "npz_outputs": npz_outputs,
        "per_sample": [
            {k: finite_or_none(v) for k, v in row.items() if k != "timestep_metrics"}
            for row in all_rows
        ],
    }
    metrics_path = output_dir / "hetero_uq_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    timestep_path = output_dir / "hetero_uq_timestep_metrics.json"
    with timestep_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": report["created_utc"],
                "per_sample_timestep_metrics": [
                    {
                        "target": row["target"],
                        "param_id": row["param_id"],
                        "real_id": row["real_id"],
                        "timestep_metrics": row["timestep_metrics"],
                    }
                    for row in all_rows
                ],
            },
            f,
            indent=2,
        )
    write_csv(all_rows, output_dir / "hetero_uq_per_sample.csv")
    with (output_dir / "hetero_uq_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": report["created_utc"],
                "script": str(Path(__file__).resolve()),
                "metrics": str(metrics_path),
                "timestep_metrics": str(timestep_path),
                "per_sample_csv": str(output_dir / "hetero_uq_per_sample.csv"),
                "npz_outputs": npz_outputs,
            },
            f,
            indent=2,
        )
    logger.info("Saved hetero metrics: %s", metrics_path)
    logger.info("Summaries: %s", target_summaries)


if __name__ == "__main__":
    main()
