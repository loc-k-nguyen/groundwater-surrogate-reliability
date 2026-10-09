import unittest

import numpy as np

from src.obj3.uq.conformal import SplitConformalPredictor


class ConformalScaleTests(unittest.TestCase):
    def test_scale_is_required(self):
        with self.assertRaises(TypeError):
            SplitConformalPredictor()

    def test_log_plume_regression(self):
        model = SplitConformalPredictor(target_scale="log10")
        model.fit(np.array([0.1, 0.2, 0.3]))
        report = model.coverage_report(np.array([-9., -7.]), np.array([-9., -7.]))["alpha_0.10"]
        self.assertEqual(report["plume_coverage"], 1.)
        self.assertEqual(report["plume_cutoff_in_target_units"], -8.)

    def test_physical_and_shifted_log_masks_agree(self):
        c = np.array([1e-9, 1e-8, 1e-7], dtype=np.float32)
        for offset in (0., 1e-12):
            physical = SplitConformalPredictor(target_scale="physical")
            logged = SplitConformalPredictor(target_scale="log10", log_offset=offset)
            physical.fit(np.array([1e-10]))
            logged.fit(np.array([0.01]))
            prediction = c.copy()
            prediction[-1] = 1e-6
            p = physical.coverage_report(c, prediction)["alpha_0.10"]
            q = logged.coverage_report(np.log10(c + offset), np.log10(prediction + offset))["alpha_0.10"]
            self.assertEqual(p["plume_coverage"], q["plume_coverage"])

    def test_empty_plume_is_explicit_nan(self):
        model = SplitConformalPredictor(target_scale="log10")
        model.fit(np.ones(4))
        self.assertTrue(np.isnan(model.coverage_report(np.array([-9.]), np.array([-9.]))["alpha_0.10"]["plume_coverage"]))

    def test_invalid_scale_threshold_offset(self):
        for options in ({"target_scale": "unknown"}, {"target_scale": "log10", "plume_thresh": 0},
                        {"target_scale": "log10", "log_offset": -1}, {"target_scale": "physical", "log_offset": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SplitConformalPredictor(**options)

    def test_invalid_targets(self):
        model = SplitConformalPredictor(target_scale="log10")
        model.fit(np.ones(4))
        for truth, prediction in (([np.nan], [0]), ([], []), ([0, 1], [0])):
            with self.assertRaises(ValueError):
                model.coverage_report(np.array(truth), np.array(prediction))


if __name__ == "__main__":
    unittest.main()
