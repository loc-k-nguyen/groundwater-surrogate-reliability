import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


class ReleaseTests(unittest.TestCase):
    def test_raw_audit_help_needs_no_restricted_assets(self):
        scripts = ("audit_conductivity_twins_obj3", "audit_conductivity_twins_obj3_breakdown",
                   "audit_rd_near_twins", "conformal_unit_and_calibration_sensitivity",
                   "calibration_nonphysical_sensitivity")
        for name in scripts:
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / (name + ".py")), "--help"],
                                        cwd=ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--output", result.stdout)

    def test_manifest_uses_posix_string_order(self):
        path = ROOT / "scripts/make_manifest.py"
        spec = importlib.util.spec_from_file_location("manifest_v5", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as directory:
            mod.ROOT = Path(directory)
            mod.MANIFEST = mod.ROOT / "MANIFEST_CODE_REVIEW.csv"
            for name in ("z.txt", "A.txt", "a.txt", "Z.txt"):
                # Case-only variants cannot coexist on Windows.
                if not (mod.ROOT / name).exists():
                    (mod.ROOT / name).write_text(name)
            self.assertEqual([r[0] for r in mod.rows()], sorted(p.name for p in mod.ROOT.iterdir()))

    def test_oracle_variance_is_not_full_field(self):
        from src.obj3.conference.eval_ensemble import summarize_variance
        var = np.array([1, 4, 9, 16], dtype=np.float32)
        cp = np.array([0, 1e-9, 1, 1], dtype=np.float32)
        result = summarize_variance(var, cp)
        self.assertEqual(result["mean_ensemble_var"], 7.5)
        self.assertEqual(result["plume_ensemble_var"], 12.5)
        self.assertIsNone(summarize_variance(var, np.zeros(4))["plume_ensemble_var"])

    def test_full_field_all_channels_include_boundaries(self):
        import torch
        from src.obj3.conference.eval_deterministic import sliding_window_inference

        class IdentityChannels(torch.nn.Module):
            def forward(self, x):
                return x.repeat(1, 25, 1, 1)

        # Strictly positive coordinates reveal uncovered pixels, including every edge.
        x = torch.arange(600 * 400, dtype=torch.float32).reshape(1, 1, 600, 400) / 240000 + 1
        result = sliding_window_inference(IdentityChannels(), x, 320, 160, "cpu")
        self.assertEqual(tuple(result.shape), (1, 25, 600, 400))
        torch.testing.assert_close(result, x.repeat(1, 25, 1, 1), rtol=1e-6, atol=1e-6)

    def test_orientation_reverses_input_and_all_targets_before_shared_crop(self):
        import torch
        from src.obj3.journal.orientation import reverse_sample_rows

        rows = torch.arange(600, dtype=torch.float32).reshape(1, 600, 1)
        cols = torch.arange(400, dtype=torch.float32).reshape(1, 1, 400) / 1000
        k = rows + cols
        times = torch.arange(25, dtype=torch.float32).reshape(25, 1, 1) * 1000
        sample = {"K": k, "C_log": k + times, "C_phys": k + times + 100,
                  "file": "synthetic-coordinate-fixture"}
        aligned = reverse_sample_rows(sample)
        self.assertEqual(aligned["file"], sample["file"])
        for key in ("K", "C_log", "C_phys"):
            torch.testing.assert_close(aligned[key], torch.flip(sample[key], dims=(-2,)))
            torch.testing.assert_close(reverse_sample_rows(aligned)[key], sample[key])
        for r, c in ((0, 0), (280, 80)):
            crop = (slice(None), slice(r, r + 320), slice(c, c + 320))
            torch.testing.assert_close(aligned["C_log"][crop] - times,
                                       aligned["K"][crop].expand(25, -1, -1), rtol=0, atol=0.002)
            torch.testing.assert_close(aligned["C_phys"][crop] - 100, aligned["C_log"][crop])
        torch.testing.assert_close(sample["K"], rows + cols)


if __name__ == "__main__":
    unittest.main()
