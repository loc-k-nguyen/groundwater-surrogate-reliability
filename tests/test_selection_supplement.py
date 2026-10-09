"""Exact-tie selection, portable reconstruction and output-preservation checks."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_selection_supplement import build_supplement, retained_mean
from src.obj3.journal.crc import solve_crc_lambda, evaluate_crc_solution


class SelectionSupplementTests(unittest.TestCase):
    def test_partial_tie_matches_exhaustive_selection_and_permutation(self):
        scores, values = np.array([1., 1., 1., 2.]), np.array([.2, .4, .9, .8])
        result = retained_mean(scores, values, 2)
        attainable = [np.mean(c) for c in itertools.combinations(values[:3], 2)]
        self.assertAlmostEqual(result["expected_mean_plume_ssim"], np.mean(attainable))
        self.assertAlmostEqual(result["attainable_min"], min(attainable))
        self.assertAlmostEqual(result["attainable_max"], max(attainable))
        permutation = [2, 3, 0, 1]
        self.assertEqual(result, retained_mean(scores[permutation], values[permutation], 2))

    def test_full_retention_and_invalid_inputs(self):
        result = retained_mean(np.array([1., 2., 2.]), np.array([.2, .4, .9]), 3)
        self.assertAlmostEqual(result["expected_mean_plume_ssim"], .5)
        self.assertEqual(result["attainable_min"], result["attainable_max"])
        self.assertIsNone(result["partial_tie"])
        for scores, values, count in (([1], [1], 0), ([1], [1], 2),
                                     ([np.nan], [1], 1), ([1, 2], [1], 1)):
            with self.assertRaises(ValueError):
                retained_mean(scores, values, count)

    def test_exact_shipped_summary_and_local_source_hashes(self):
        document = build_supplement()
        expected = json.loads((ROOT / "results/orientation_v5/selection_crc_baselines.json").read_text())
        self.assertEqual(document, expected)
        self.assertEqual(len(document["sources"]), 5)
        for source in document["sources"]:
            relative = Path(source["path"])
            self.assertFalse(relative.is_absolute())
            self.assertTrue((ROOT / relative).resolve().is_relative_to(ROOT))
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), source["sha256"])

    def test_cli_external_directory_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "supplement.json"
            command = [sys.executable, "-B", str(ROOT / "scripts/build_selection_supplement.py"),
                       "--output", str(output)]
            first = subprocess.run(command, cwd=folder, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            second = subprocess.run(command, cwd=folder, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Preserve existing outputs", second.stderr)
            self.assertEqual(output.read_bytes(), original)
            inside = subprocess.run(command[:-1] + [str(ROOT / "README.md")],
                                    cwd=folder, capture_output=True, text=True)
            self.assertNotEqual(inside.returncode, 0)
            self.assertIn("outside the package", inside.stderr)

    def test_calibration_attainment_is_not_test_budget_attainment(self):
        solution = solve_crc_lambda(np.zeros(100, dtype=np.float32), .2)
        self.assertTrue(solution.attained)
        evaluation = evaluate_crc_solution(np.ones(2, dtype=np.float32), solution)
        self.assertGreater(evaluation["excess_risk_mean"], solution.risk_budget)
        self.assertIn("not test risk", solve_crc_lambda.__doc__)

    def test_direct_image_verifier_dependency_is_declared(self):
        for name in ("requirements.txt", "requirements-analysis-lock.txt"):
            self.assertIn("Pillow==11.1.0", (ROOT / name).read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
