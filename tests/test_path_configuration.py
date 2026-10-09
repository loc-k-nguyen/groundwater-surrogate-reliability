"""Path setup is explicit, preserves native flags and never runs on import."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class PathConfigurationTests(unittest.TestCase):
    def test_helper_import_preserves_process_state(self):
        with patch.object(sys, "path", [str(SCRIPTS)] + sys.path), \
                patch.object(sys, "argv", ["host", "--repo-root", "missing", "--data-root", "fields"]):
            before = (list(sys.argv), dict(os.environ), list(sys.path))
            import package_paths
            importlib.reload(package_paths)
            self.assertEqual((sys.argv, dict(os.environ), sys.path), before)

    def test_entry_point_imports_do_not_consume_host_arguments(self):
        names = ("eval_ensemble_uq", "eval_hetero_uq", "axisD_d1_inference_eval",
                 "evaluate_matched_input_screening")
        with patch.object(sys, "path", [str(SCRIPTS), str(ROOT)] + sys.path), \
                patch.object(sys, "argv", ["host", "--repo-root", "missing", "--data-root", "fields"]):
            for name in names:
                with self.subTest(script=name):
                    before = (list(sys.argv), dict(os.environ), list(sys.path))
                    importlib.import_module(name)
                    self.assertEqual((sys.argv, dict(os.environ), sys.path), before)

    def test_explicit_launch_resolves_overrides_and_preserves_native_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "splits").mkdir()
            argv = ["script", "--repo-root", str(root), "--data-root", str(root / "fields"),
                    "--calibration-root", str(root / "extra"), "--data_root", "native-field", "--output", "new"]
            with patch.object(sys, "path", [str(SCRIPTS), str(root), str(ROOT)] + sys.path), \
                    patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True):
                import package_paths
                self.assertEqual(package_paths.configure_script_paths(), root.resolve())
                self.assertEqual(Path(sys.path[0]), ROOT)
                self.assertEqual(sys.argv, ["script", "--data_root", "native-field", "--output", "new"])
                self.assertEqual(package_paths.asset_path("data", root / "fallback"), root / "fields")
                self.assertEqual(package_paths.asset_path("calibration", root / "fallback"), root / "extra")

    def test_invalid_root_fails_before_mutating_state(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "path", [str(SCRIPTS)] + sys.path), \
                patch.object(sys, "argv", ["script", "--repo-root", directory, "--data-root", "fields"]):
            import package_paths
            before = (list(sys.argv), dict(os.environ), list(sys.path))
            with self.assertRaises(SystemExit) as caught:
                package_paths.configure_script_paths()
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual((sys.argv, dict(os.environ), sys.path), before)

    def test_launch_help_without_restricted_assets(self):
        names = ("eval_ensemble_uq", "eval_hetero_uq", "axisD_d1_inference_eval",
                 "evaluate_matched_input_screening")
        for name in names:
            with self.subTest(script=name), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-B", str(SCRIPTS / (name + ".py")),
                                         "--repo-root", str(ROOT), "--help"], cwd=directory,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_superseded_interfaces_and_simulator_overlay_are_excluded(self):
        for path in (SCRIPTS / "build_axisD_taxonomy_table.py", SCRIPTS / "run_axisD_inference.py",
                     SCRIPTS / "build_v4_analysis.py", ROOT / "metadata/config_obj3_axisD_matched.py"):
            self.assertFalse(path.exists(), path.name)

    def test_inference_rejects_existing_output_before_loading_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "preserved"
            out.mkdir()
            marker = out / "evidence.txt"
            marker.write_bytes(b"unchanged")
            commands = (
                ["eval_ensemble_uq.py", "--arch", "unet_det", "--checkpoints", "missing",
                 "--output_dir", str(out)],
            )
            for command in commands:
                with self.subTest(script=command[0]):
                    result = subprocess.run([sys.executable, "-B", str(SCRIPTS / command[0])] + command[1:],
                                            cwd=directory, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("Output already exists", result.stderr)
                    self.assertEqual(marker.read_bytes(), b"unchanged")


if __name__ == "__main__":
    unittest.main()
