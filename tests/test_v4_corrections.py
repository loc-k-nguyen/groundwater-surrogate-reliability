import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("current_analysis", ROOT / "scripts/build_v5_analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
from src.obj3.conference.data_obj3_ood import Obj3PatchDataset
from src.shared.utils.seed import set_seed


class Corrections(unittest.TestCase):
    def test_ties_are_order_invariant_and_bound_recall(self):
        scores = np.array([3, 2, 2, 2, 1], dtype=float)
        failed = np.array([False, True, False, True, False])
        expected, low, high, blocks = analysis.tie_curve(scores, failed)
        self.assertAlmostEqual(expected[2], 1/3)
        np.testing.assert_array_equal(low[2:4], [0, .5])
        np.testing.assert_array_equal(high[2:4], [.5, 1])
        perm = np.array([4, 2, 0, 3, 1])
        for actual, wanted in zip(analysis.tie_curve(scores[perm], failed[perm])[:3],
                                  (expected, low, high)):
            np.testing.assert_array_equal(actual, wanted)
        self.assertEqual(sum(b["n_cases"] for b in blocks), len(scores))

    def test_last_crop_origin_is_available_and_legacy_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real_001.npz"
            np.savez(path, K=np.ones((5, 4)), C=np.ones((25, 5, 4)))
            for legacy, bounds in [(False, [3, 2]), (True, [2, 1])]:
                data = Obj3PatchDataset([path], {"k_mean":0,"k_std":1},
                                       patch_size=3, legacy_crop_sampling=legacy)
                calls = []
                def final_origin(low, high):
                    calls.append(high)
                    return high - 1
                with patch("numpy.random.randint", side_effect=final_origin):
                    sample = data[0]
                self.assertEqual(calls, bounds)
                self.assertEqual(tuple(sample["K"].shape), (1,3,3))

    def test_deterministic_mode_records_and_controls_flags(self):
        record = set_seed(0, deterministic=True)
        first = torch.rand(4)
        set_seed(0, deterministic=True)
        self.assertTrue(torch.equal(first, torch.rand(4)))
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(record["deterministic"])
        set_seed(0, deterministic=False)


if __name__ == "__main__":
    unittest.main()
