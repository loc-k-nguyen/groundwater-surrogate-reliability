"""CPU numerical fixture, not a trained surrogate or scientific validation dataset."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.obj3.conference.eval_deterministic import sliding_window_inference
from src.obj3.journal.orientation import reverse_sample_rows
from src.obj3.journal.physical_quality import quality_by_time
from src.obj3.uq.conformal import SplitConformalPredictor


class AnalyticChannels(torch.nn.Module):
    def forward(self, x):
        times = torch.arange(25, dtype=x.dtype, device=x.device)[None, :, None, None]
        return x.repeat(1, 25, 1, 1) + times * .05


def demonstrate():
    torch.set_num_threads(1)
    rows = torch.linspace(-9., -7., 600)[:, None]
    cols = torch.linspace(0., .2, 400)[None, :]
    k = (rows + cols)[None]
    truth = k.repeat(25, 1, 1) + torch.arange(25)[:, None, None] * .05
    physical = (10. ** truth.double() - 1e-12).clamp(min=0)
    sample = {"K": k, "C_log": truth, "C_phys": physical}
    aligned = reverse_sample_rows(sample)
    torch.testing.assert_close(reverse_sample_rows(aligned)["C_log"], truth)
    pred = sliding_window_inference(AnalyticChannels(), aligned["K"][None], 320, 160, "cpu")[0]
    torch.testing.assert_close(pred, aligned["C_log"], rtol=1e-6, atol=2e-6)
    metrics = quality_by_time(aligned["C_log"].numpy(), pred.numpy(), aligned["C_phys"].numpy())
    maximum = max(row["log_mae"] for row in metrics)
    assert maximum < 2e-6
    probe = np.array([-9., -7.])
    cp = SplitConformalPredictor(target_scale="log10")
    cp.fit(np.zeros(20))
    coverage = cp.coverage_report(probe, probe, (.1,))["alpha_0.10"]
    assert coverage["plume_coverage"] == 1.
    return {"status": "SYNTHETIC_NUMERICAL_FIXTURE_PASS", "shape": [25, 600, 400],
            "max_log_mae": maximum, "physical_cutoff": 1e-8,
            "log_target_plume_cutoff": coverage["plume_cutoff_in_target_units"],
            "all_times_and_terminal_edges_checked": True,
            "scope": "Analytic mapping and synthetic arrays; no trained checkpoint, simulator field or raw-study reproduction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        if output.exists() or output == ROOT or ROOT in output.parents:
            raise FileExistsError("Use a new file outside the package")
    record = demonstrate()
    encoded = json.dumps(record, indent=2) + "\n"
    if args.output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
