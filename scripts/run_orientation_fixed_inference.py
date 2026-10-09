"""Isolated evaluation repair using explicit paired row alignment; no training."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import zipfile

from package_paths import enable_package_imports

if __name__ == "__main__":
    enable_package_imports()

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoints(root, arch):
    if arch == "unet_det":
        parent = root / "experiments/obj3/conference/runs"
        prefix = "obj3_conf_ms_tmo_det_full_seed"
    else:
        parent = root / "experiments/obj3/journal/runs"
        prefix = {"hetero": "obj3_jnl_ms_tmo_hetero_seed", "fno": "obj3_jnl_fno_seed",
                  "deeponet": "obj3_jnl_deeponet_seed"}[arch]
    paths = [parent / f"{prefix}{seed}/best.pt" for seed in range(5)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--adapter-path", type=Path, default=REPO / "src/obj3/journal/orientation.py")
    parser.add_argument("--evaluator-path", type=Path, default=REPO / "scripts/eval_ensemble_uq.py")
    parser.add_argument("--stats-json", type=Path, default=REPO / "metadata/obj3_train_stats.json")
    parser.add_argument("--split-json", type=Path, default=REPO / "splits/param_split_obj3_ood.json")
    parser.add_argument("--interp-split-json", type=Path, default=REPO / "splits/param_split_obj3_interp.json")
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--interpolation-root", type=Path)
    parser.add_argument("--transport-root", type=Path)
    parser.add_argument("--check-sources", action="store_true", help="CPU-only source import/hash check; no data or inference")
    parser.add_argument("--mode", choices=("smoke", "calibration", "transport", "interpolation"))
    parser.add_argument("--arch", choices=("unet_det", "hetero", "fno", "deeponet"), default="unet_det")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_sources:
        adapter = module_at("orientation_adapter", args.adapter_path.resolve())
        evaluator = module_at("orientation_evaluator", args.evaluator_path.resolve())
        if not all(callable(fn) for fn in (adapter.OrientationAlignedFullFieldDataset,
                                          evaluator.build_models, evaluator.sliding, evaluator.eval_sample)):
            raise ValueError("Source interface differs")
        print(json.dumps({"status": "SOURCE_IMPORTS_PASSED_ONLY", "data_checked": False,
                          "inference_performed": False,
                          "source_sha256": {key: sha256_file(path) for key, path in
                                            (("adapter", args.adapter_path), ("evaluator", args.evaluator_path))}}))
        return
    if args.mode is None or args.output_dir is None:
        parser.error("Inference requires --mode and --output-dir")
    root, out = args.repo_root.resolve(), args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this runner never falls back to local CPU inference")
    device = "cuda"
    torch.zeros(1, device=device)
    adapter = module_at("orientation_adapter", args.adapter_path)
    evaluator_path = args.evaluator_path.resolve()
    evaluator = module_at("orientation_evaluator", evaluator_path)
    main_root = (args.native_root or root / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL").resolve()
    extra = (args.calibration_root or root / "simulation/datasets/obj3_calibration").resolve()
    interp = (args.interpolation_root or root / "simulation/datasets/obj3_interpolation").resolve()
    axis = (args.transport_root or root / "experiments/obj3/journal/upgrade_2026_07_01/tier1_multi_axis_ood/axisD_transport_shift/sim/dataset_axisD").resolve()
    stats_path = args.stats_json.resolve()
    stats = json.loads(stats_path.read_text())
    split_path = args.split_json.resolve()
    split = json.loads(split_path.read_text())
    roots = {"native_roots": [main_root], "reverse_roots": [extra, interp, axis]}
    started = time.monotonic()
    architectures = ("unet_det", "hetero", "fno", "deeponet") if args.mode == "smoke" else (args.arch,)
    if args.mode in ("calibration", "interpolation") and args.arch != "unet_det":
        raise ValueError("Cache exports preserve the existing U-Net ensemble protocol")
    checkpoint_records = {}
    for arch in architectures:
        checkpoint_records[arch] = []
        for path in checkpoints(root, arch):
            config = path.parent / "config.json"
            if not config.is_file():
                raise FileNotFoundError(config)
            checkpoint_records[arch].append({"path": str(path), "bytes": path.stat().st_size,
                                             "sha256": sha256_file(path),
                                             "config": str(config), "config_sha256": sha256_file(config)})
    records = {"mode": args.mode, "architectures": list(architectures), "patch": 320, "stride": 160,
               "seeds": list(range(5)), "training_performed": False,
               "checkpoints": checkpoint_records,
               "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                           "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
               "orientation": {"native": str(main_root), "reversed": [str(extra), str(interp), str(axis)]},
               "source_hashes": {str(path): sha256_file(path)
                                 for path in (Path(__file__), args.adapter_path, evaluator_path, stats_path, split_path,
                                              args.interp_split_json.resolve(),
                                              REPO / "src/obj3/conference/data_obj3_ood.py",
                                              REPO / "src/obj3/conference/eval_deterministic.py")}}

    def dataset(files):
        return adapter.OrientationAlignedFullFieldDataset(files, stats, **roots)

    def aligned_sample(ds, index):
        sample = ds[index]
        for key in ("K", "C_log", "C_phys"):
            if not torch.isfinite(sample[key]).all():
                raise ValueError(f"Nonfinite {key}: {sample['path']}")
        source_row = np.unravel_index(np.argmax(sample["C_phys"][0].numpy()), (600, 400))[0]
        if source_row != 200:
            raise ValueError(f"Training-frame source anchor mismatch: {sample['path']}")
        return sample

    def predict(models, sample, arch, export_protocol=False):
        k = sample["K"].unsqueeze(0).to(device)
        means, variances = [], []
        for model in models:
            if export_protocol:
                from src.obj3.conference.eval_deterministic import sliding_window_inference
                mean = sliding_window_inference(model, k, 320, 160, device)
                variance = None
            else:
                mean, variance = evaluator.sliding(model, k, 320, 160, device, arch == "hetero", False)
            mean = torch.clamp(mean, -12., 2.).squeeze(0).cpu().numpy()
            means.append(mean)
            if variance is not None:
                variances.append(variance.squeeze(0).cpu().numpy())
        members = np.stack(means, axis=0)
        mean = members.mean(axis=0)
        epistemic = members.var(axis=0)
        aleatoric = np.stack(variances, axis=0).mean(axis=0) if variances else np.zeros_like(epistemic)
        total = np.sqrt(np.maximum(aleatoric + epistemic, 0.))
        if mean.shape != (25, 600, 400) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(total)):
            raise ValueError("Prediction geometry/finite-value gate failed")
        return mean, total, np.sqrt(np.maximum(aleatoric, 0.)), np.sqrt(np.maximum(epistemic, 0.))

    out.mkdir(parents=True)
    (out / "RUNNING.json").write_text(json.dumps(records, indent=2))
    if args.mode == "smoke":
        native_file = main_root / "param_128/real_001.npz"
        files = [native_file, extra / "param_200/real_001.npz", interp / "param_276/real_001.npz",
                 axis / "param_9210/real_001.npz"]
        ds = dataset(files)
        original = evaluator.Obj3FullFieldDataset([native_file], stats)[0]
        keys = ("K", "C_log", "C_phys")
        if not all(torch.equal(original[key], ds[0][key]) for key in keys):
            raise ValueError("Native input equality gate failed")
        paired_checks = []
        for i, path in enumerate(files):
            sample = aligned_sample(ds, i)
            source_row = int(np.unravel_index(np.argmax(sample["C_phys"][0].numpy()), (600, 400))[0])
            raw = evaluator.Obj3FullFieldDataset([path], stats)[0]
            equal = all(torch.equal(sample[key], raw[key] if i == 0 else torch.flip(raw[key], (-2,)))
                        for key in keys)
            if source_row != 200 or not equal:
                raise ValueError(f"Paired geometry gate failed: {path}")
            paired_checks.append({"path": str(path), "sha256": sha256_file(path),
                                  "reversed": i != 0, "source_row": source_row,
                                  "paired_tensor_equality": equal})
        records["geometry_checks"] = paired_checks
        records["native_tensor_equality"] = True
        summaries = []
        for arch in ("unet_det", "hetero", "fno", "deeponet"):
            paths = checkpoints(root, arch)
            models = evaluator.build_models(arch, [str(p) for p in paths], device)
            for i in range(4):
                sample = aligned_sample(ds, i)
                mean, total, aleatoric, epistemic = predict(models, sample, arch)
                row = evaluator.eval_sample(mean, total, aleatoric, epistemic, sample)
                row["arch"] = arch
                summaries.append(row)
                print("SMOKE", arch, sample["path"], row["mean_plume_ssim"], flush=True)
            if arch == "unet_det":
                predict(models, aligned_sample(ds, 1), arch, export_protocol=True)
                records["unet_amp_export_check"] = {"passed": True, "path": str(files[1]),
                                                     "shape": [25, 600, 400]}
            del models
            torch.cuda.empty_cache()
        records["status"] = "ORIENTATION_SMOKE_DONE"
        records["cases"] = summaries
    elif args.mode == "transport":
        arch = args.arch
        paths = checkpoints(root, arch)
        models = evaluator.build_models(arch, [str(p) for p in paths], device)
        files = [p for identifier in [*range(9210, 9216), *range(9220, 9238)]
                 for p in sorted((axis / f"param_{identifier}").glob("real_*.npz"))]
        if len(files) != 48:
            raise ValueError(f"Expected 48 ladder cases, got {len(files)}")
        ds = dataset(files)
        rows = []
        for i in range(len(ds)):
            sample = aligned_sample(ds, i)
            mean, total, aleatoric, epistemic = predict(models, sample, arch)
            row = evaluator.eval_sample(mean, total, aleatoric, epistemic, sample)
            row["target"] = "axisD_" + sample["param_id"]
            rows.append(row)
            print("TRANSPORT", arch, i + 1, len(ds), flush=True)
        records["status"] = "ORIENTATION_TRANSPORT_DONE"
        records["per_sample"] = rows
        with (out / "per_sample.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        models = evaluator.build_models("unet_det", [str(p) for p in checkpoints(root, "unet_det")], device)
        from src.obj3.conference.data_obj3_ood import collect_npz_files
        if args.mode == "calibration":
            ids = split["val_calib"] + split["extra_calib"]
            targets = {"calibration": collect_npz_files(ids, main_root, extra)}
            if len(targets["calibration"]) != 225:
                raise ValueError("Calibration count mismatch")
        else:
            interp_split = json.loads(args.interp_split_json.read_text())
            targets = {name: collect_npz_files(ids, main_root, extra, interp)
                       for name, ids in interp_split.items() if isinstance(ids, list)}
            if {name: len(files) for name, files in targets.items()} != {
                    "interp_03": 100, "interp_07": 89, "interp_13": 95, "interp_17": 90}:
                raise ValueError("Frozen interpolation group/count gate failed")
        cache_records = {}
        for name, files in targets.items():
            ds = dataset(files)
            cache_dir = out / name
            cache_dir.mkdir()
            fields = ("y_true", "y_pred", "y_pred_std", "c_phys")
            handles = {key: (cache_dir / (key + ".npy")).open("wb") for key in fields}
            try:
                for handle in handles.values():
                    np.lib.format.write_array_header_1_0(handle, {"descr": "<f4", "fortran_order": False,
                                                                 "shape": (len(ds), 25, 600, 400)})
                for i in range(len(ds)):
                    sample = aligned_sample(ds, i)
                    mean, total, _, _ = predict(models, sample, "unet_det", export_protocol=True)
                    values = {"y_true": sample["C_log"].numpy(), "c_phys": sample["C_phys"].numpy(),
                              "y_pred": mean, "y_pred_std": total}
                    for key, value in values.items():
                        handles[key].write(np.asarray(value, dtype="<f4", order="C").tobytes())
                    print("EXPORT", name, i + 1, len(ds), flush=True)
            finally:
                for handle in handles.values():
                    handle.close()
            target = out / (name + "_preds.npz")
            with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for key in fields:
                    archive.write(cache_dir / (key + ".npy"), key + ".npy")
            cache_records[name] = {"file": target.name, "cases": len(ds), "order": [str(p) for p in files],
                                   "shape": [len(ds), 25, 600, 400], "bytes": target.stat().st_size,
                                   "sha256": sha256_file(target)}
        records["status"] = "ORIENTATION_EXPORT_DONE"
        records["caches"] = cache_records
    records["elapsed_seconds"] = time.monotonic() - started
    (out / "COMPLETED.json").write_text(json.dumps(records, indent=2))
    print(records["status"], flush=True)


if __name__ == "__main__":
    main()
