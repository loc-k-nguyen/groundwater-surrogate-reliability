"""Validate portable summary hashes, privacy, counts and scalar round trips."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1] / "results/orientation_v5"


class PortableSummaryTests(unittest.TestCase):
    def test_manifest_and_private_path_absence(self):
        manifest = json.loads((ROOT / "MANIFEST.json").read_text())
        self.assertEqual(manifest["status"], "PORTABLE_SCALAR_EXPORT_DONE")
        for name, digest in manifest["files"].items():
            path = ROOT / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertLess(path.stat().st_size, 5 * 1024 * 1024)
            self.assertIsNone(re.search(r"/nfs/|/home/|/local/scratch|[A-Za-z]:[/\\]|\.claude", path.read_text()))

    def test_all_times_and_case_mean_summaries(self):
        stats = json.loads((ROOT / "statistics.json").read_text())
        for family in ("unet_det", "hetero", "fno", "deeponet"):
            folder = ROOT / "quality" / family
            quality = json.loads((folder / "quality.json").read_text())[family]
            self.assertEqual({k: len(v) for k, v in quality["splits"].items()},
                             {"iid_test": 110, "ood_test": 270, "transport_ladder": 48})
            self.assertEqual([r["seed"] for r in quality["checkpoint_pins"]], list(range(5)))
            with (folder / "per_time.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 428 * 25)
            grouped = {}
            for row in rows:
                key = (row["split"], row["param_id"], row["real_id"])
                grouped.setdefault(key, []).append(row)
            self.assertEqual(len(grouped), 428)
            for group in grouped.values():
                self.assertEqual([int(r["timestep"]) for r in group], list(range(25)))
            for split in ("iid_test", "ood_test", "transport_primary_200m"):
                selected = [group for key, group in grouped.items()
                            if key[0] == split or (split == "transport_primary_200m" and key[0] == "transport_ladder"
                                                  and 9210 <= int(key[1].split("_")[-1]) <= 9215)]
                expected = stats["quality"][family]["splits"][split]
                self.assertEqual(len(selected), expected["n_cases"])
                for metric, summary in expected["case_mean"].items():
                    means = []
                    for group in selected:
                        values = [float(r[metric]) for r in group if r[metric] != ""]
                        if values:
                            means.append(sum(values) / len(values))
                    # NumPy and Python use different float64 reduction orders.
                    self.assertTrue(math.isclose(math.fsum(means) / len(means), summary["mean"],
                                                 rel_tol=1e-13, abs_tol=1e-12), metric)


if __name__ == "__main__":
    unittest.main()
