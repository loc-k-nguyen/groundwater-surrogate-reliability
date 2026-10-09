import importlib.util
from pathlib import Path
import unittest


class SyntheticNumerics(unittest.TestCase):
    def test_orientation_inference_metrics_all_times_and_edges(self):
        path = Path(__file__).resolve().parents[1] / "scripts/synthetic_demo.py"
        spec = importlib.util.spec_from_file_location("synthetic_demo", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.demonstrate()
        self.assertEqual(result["shape"], [25, 600, 400])
        self.assertEqual(result["log_target_plume_cutoff"], -8.)
        self.assertLess(result["max_log_mae"], 2e-6)


if __name__ == "__main__":
    unittest.main()
