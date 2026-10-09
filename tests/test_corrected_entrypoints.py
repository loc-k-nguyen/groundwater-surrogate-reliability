"""CPU-only packaging checks; these never certify or execute restricted inference."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
NAMES = ("evaluate_four_family_quality_v5", "run_orientation_fixed_inference",
         "recompute_orientation_statistics_v5")


class CorrectedEntrypointTests(unittest.TestCase):
    def command(self, name, arguments, cwd):
        return subprocess.run([sys.executable, "-B", str(SCRIPTS / (name + ".py"))] + arguments,
                              cwd=cwd, capture_output=True, text=True, timeout=60)

    def test_help_from_unrelated_directory_needs_no_data(self):
        for name in NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                result = self.command(name, ["--help"], directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cpu_source_checks_use_shipped_dependencies_without_assets(self):
        paths = {"runner": SCRIPTS / "run_orientation_fixed_inference.py",
                 "evaluator": SCRIPTS / "eval_ensemble_uq.py",
                 "adapter": ROOT / "src/obj3/journal/orientation.py",
                 "metric": ROOT / "src/obj3/journal/physical_quality.py"}
        for name in NAMES[:2]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                result = self.command(name, ["--check-sources", "--repo-root", directory], directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["status"], "SOURCE_IMPORTS_PASSED_ONLY")
                self.assertFalse(report["data_checked"])
                self.assertFalse(report["inference_performed"])
                self.assertGreaterEqual(len(report["source_sha256"]), 2)
                for label, digest in report["source_sha256"].items():
                    self.assertEqual(digest, hashlib.sha256(paths[label].read_bytes()).hexdigest())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_imports_preserve_host_arguments_and_environment(self):
        with patch.object(sys, "path", [str(SCRIPTS), str(ROOT)] + sys.path), \
                patch.object(sys, "argv", ["host", "--repo-root", "missing", "--output", "never"]):
            for name in NAMES:
                with self.subTest(name=name):
                    before = (list(sys.argv), dict(os.environ), list(sys.path))
                    importlib.import_module(name)
                    self.assertEqual((sys.argv, dict(os.environ), sys.path), before)

    def test_existing_outputs_fail_before_cuda_or_private_asset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "preserved"
            out.mkdir()
            marker = out / "marker.txt"
            marker.write_bytes(b"original")
            commands = (
                (NAMES[0], ["--arch", "unet_det", "--output", str(out)]),
                (NAMES[1], ["--mode", "calibration", "--output-dir", str(out)]),
                (NAMES[2], ["--certificate", "missing", "--split", "missing", "--output", str(out)]),
            )
            for name, arguments in commands:
                with self.subTest(name=name):
                    result = self.command(name, arguments, directory)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("FileExistsError", result.stderr)
                    self.assertNotIn("CUDA", result.stderr)
                    self.assertEqual(marker.read_bytes(), b"original")
                    self.assertEqual(list(out.iterdir()), [marker])

    def test_missing_dependency_fails_without_creating_an_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "must-not-exist"
            result = self.command(NAMES[0], ["--check-sources", "--evaluator-path", "missing.py",
                                             "--output", str(out)], directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
