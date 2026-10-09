"""Frozen-checkpoint evaluation supplement; no training or prediction-cache export."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from package_paths import enable_package_imports

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    enable_package_imports()

import numpy as np
import torch

from src.obj3.journal.physical_quality import quality_by_time


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=PACKAGE_ROOT,
                   help="External restricted-asset project root, not the source-code root")
    p.add_argument("--run-root", type=Path, help="External certified correction-run folder")
    p.add_argument("--runner-path", type=Path, default=PACKAGE_ROOT / "scripts/run_orientation_fixed_inference.py")
    p.add_argument("--evaluator-path", type=Path, default=PACKAGE_ROOT / "scripts/eval_ensemble_uq.py")
    p.add_argument("--adapter-path", type=Path, default=PACKAGE_ROOT / "src/obj3/journal/orientation.py")
    p.add_argument("--stats-json", type=Path, default=PACKAGE_ROOT / "metadata/obj3_train_stats.json")
    p.add_argument("--native-root", type=Path)
    p.add_argument("--calibration-root", type=Path)
    p.add_argument("--interpolation-root", type=Path)
    p.add_argument("--transport-root", type=Path)
    p.add_argument("--check-sources", action="store_true", help="CPU-only source import/hash check; no data or inference")
    p.add_argument("--arch", choices=("unet_det", "hetero", "fno", "deeponet"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not a.check_sources and (a.output is None or (not a.smoke and a.arch is None)):
        p.error("Evaluation requires --output and either --arch or --smoke")
    if a.check_sources:
        runner = module_at("quality_orientation_runner", a.runner_path.resolve())
        evaluator = module_at("quality_existing_evaluator", a.evaluator_path.resolve())
        adapter = module_at("quality_orientation_adapter", a.adapter_path.resolve())
        if not all(callable(fn) for fn in (runner.checkpoints, evaluator.build_models,
                                          evaluator.sliding, evaluator.eval_sample,
                                          adapter.OrientationAlignedFullFieldDataset, quality_by_time)):
            raise ValueError("Source interface differs")
        print(json.dumps({"status": "SOURCE_IMPORTS_PASSED_ONLY", "data_checked": False,
                          "inference_performed": False,
                          "source_sha256": {key: runner.sha256_file(path) for key, path in
                                            (("runner", a.runner_path), ("evaluator", a.evaluator_path),
                                             ("adapter", a.adapter_path),
                                             ("metric", PACKAGE_ROOT / "src/obj3/journal/physical_quality.py"))}}))
        return
    if a.output.exists():
        raise FileExistsError("Use a new isolated output")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requires allocated CUDA")
    root = a.repo_root.resolve()
    run = (a.run_root or root / "experiments/obj3/journal/runs/obj3_orientationfix_20261009").resolve()
    runner = module_at("quality_orientation_runner", a.runner_path.resolve())
    evaluator = module_at("quality_existing_evaluator", a.evaluator_path.resolve())
    adapter = module_at("quality_orientation_adapter", a.adapter_path.resolve())
    certificate = run / "cache_certification/COMPLETED.json"
    inputs_file = run / "cache_certification/input_archives.json"
    if runner.sha256_file(certificate) != "fd011bb76d00829df0c17a87847927d3c4c5271f5a43d7c15d22db88baf89637":
        raise ValueError("Cache certificate changed")
    if runner.sha256_file(inputs_file) != "532e45c72562e11083cd0141cc1e6cd42ca6931600767a136a55340a56481012":
        raise ValueError("Input archive certificate changed")
    cert = json.loads(certificate.read_text())
    if cert["status"] != "ORIENTATION_CACHES_CERTIFIED":
        raise ValueError("Full source/cache certification did not pass")
    inputs = json.loads(inputs_file.read_text())
    native = (a.native_root or root / "Obj1/obj1_surrogate_conference/data/T25_TSTEP_OVERRIDE_FINAL").resolve()
    extra = (a.calibration_root or root / "simulation/datasets/obj3_calibration").resolve()
    interp = (a.interpolation_root or root / "simulation/datasets/obj3_interpolation").resolve()
    axis = (a.transport_root or root / "experiments/obj3/journal/upgrade_2026_07_01/tier1_multi_axis_ood/axisD_transport_shift/sim/dataset_axisD").resolve()
    groups = {c["name"]: [Path(f) for f in c["order"]] for c in cert["caches"] if c["name"] in ("iid_test", "ood_test")}
    # Preserve the certificate's exact names; the original-cache certificate uses native prefixes.
    if not groups:
        groups = {c["name"]: [Path(f) for f in c["order"]] for c in cert["caches"] if c["shape"][0] in (110, 270)}
    if sorted(len(v) for v in groups.values()) != [110, 270]:
        raise ValueError("Reference/variance inventory differs")
    groups["transport_ladder"] = [axis / f"param_{i}" / f"real_{r:03d}.npz"
                                   for i in [*range(9210, 9216), *range(9220, 9238)] for r in (1, 2)]
    if any(str(f) not in inputs for paths in groups.values() for f in paths):
        raise ValueError("Input not covered by full archive certificate")
    stats = json.loads(a.stats_json.read_text())
    original_smoke = json.loads((run / "smoke/COMPLETED.json").read_text())
    archs = ("unet_det", "hetero", "fno", "deeponet") if a.smoke else (a.arch,)
    if archs == (None,):
        p.error("--arch required for full evaluation")
    a.output.mkdir(parents=True)
    (a.output / "RUNNING.json").write_text(json.dumps({"architectures": archs, "training": False}))
    reports = {}
    for arch in archs:
        paths = runner.checkpoints(root, arch)
        pinned = original_smoke["checkpoints"][arch]
        for f, pin in zip(paths, pinned):
            if runner.sha256_file(f) != pin["sha256"] or runner.sha256_file(f.parent / "config.json") != pin["config_sha256"]:
                raise ValueError("Checkpoint/config hash changed")
        models = evaluator.build_models(arch, [str(f) for f in paths], "cuda")
        reports[arch] = {"checkpoint_pins": pinned, "splits": {}}
        selected_groups = {"native_check": [native / "param_128/real_001.npz"],
                           "transport_check": [axis / "param_9210/real_001.npz"]} if a.smoke else groups
        for name, files in selected_groups.items():
            dataset = adapter.OrientationAlignedFullFieldDataset(files, stats, native_roots=[native], reverse_roots=[extra, interp, axis])
            rows = []
            for i, f in enumerate(files):
                if runner.sha256_file(f) != inputs[str(f)]["sha256"]:
                    raise ValueError(f"Input archive changed: {f}")
                sample = dataset[i]
                if np.unravel_index(np.argmax(sample["C_phys"][0].numpy()), (600, 400))[0] != 200:
                    raise ValueError("Source-anchor alignment failed")
                k = sample["K"].unsqueeze(0).to("cuda")
                means, variances = [], []
                torch.cuda.synchronize()
                start = time.perf_counter()
                for model in models:
                    mean, var = evaluator.sliding(model, k, 320, 160, "cuda", arch == "hetero", False)
                    means.append(torch.clamp(mean, -12, 2).squeeze(0).cpu().numpy())
                    if var is not None:
                        variances.append(var.squeeze(0).cpu().numpy())
                members = np.stack(means)
                mean, epi = members.mean(0), members.var(0)
                ale = np.stack(variances).mean(0) if variances else np.zeros_like(epi)
                total = np.sqrt(np.maximum(epi + ale, 0))
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                if mean.shape != (25, 600, 400) or not np.isfinite(mean).all() or not np.isfinite(total).all():
                    raise ValueError("Prediction schema/finite gate failed")
                row = evaluator.eval_sample(mean, total, np.sqrt(np.maximum(ale, 0)), np.sqrt(np.maximum(epi, 0)), sample)
                if a.smoke and name == "native_check":
                    original = next(r for r in original_smoke["cases"] if r["arch"] == arch and r["param_id"] == sample["param_id"])
                    for metric in ("mean_plume_ssim", "mean_global_ssim", "plume_total_std_log", "plume_epistemic_std_log", "mean_total_std_log"):
                        if not np.isclose(row[metric], original[metric], rtol=0, atol=1e-9):
                            raise ValueError(f"Native metric parity failed: {arch}/{metric}")
                row["quality_per_time"] = quality_by_time(sample["C_log"].numpy(), mean, sample["C_phys"].numpy())
                row["five_member_inference_and_transfer_seconds"] = seconds
                row["timing_warmup_case"] = i == 0
                rows.append(row)
                if (i + 1) % 25 == 0 or a.smoke:
                    print(arch, name, i + 1, len(files), flush=True)
            reports[arch]["splits"][name] = rows
        del models
        torch.cuda.empty_cache()
    report_file = a.output / "quality.json"
    report_file.write_text(json.dumps(reports, indent=2, allow_nan=False))
    (a.output / "COMPLETED.json").write_text(json.dumps({
        "status": "QUALITY_SMOKE_DONE" if a.smoke else "QUALITY_EVALUATION_DONE", "training_performed": False,
        "quality_sha256": runner.sha256_file(report_file), "script_sha256": runner.sha256_file(__file__),
        "metric_sha256": runner.sha256_file(PACKAGE_ROOT / "src/obj3/journal/physical_quality.py"),
        "source_dependencies": {key: runner.sha256_file(path) for key, path in
                                (("runner", a.runner_path), ("evaluator", a.evaluator_path),
                                 ("adapter", a.adapter_path), ("normalization", a.stats_json))},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0), "evaluation_amp": False,
                    "seeds": list(range(5)), "patch": 320, "stride": 160},
        "certificate_sha256": runner.sha256_file(certificate), "input_manifest_sha256": runner.sha256_file(inputs_file),
        "timing_scope": "Five sequential members including sliding-window assembly and host transfer, excluding source I/O and metric computation; omit first case per split as warmup. Not simulator speedup."}, indent=2))


if __name__ == "__main__":
    main()
