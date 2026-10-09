"""Reconstruct descriptive selection, excess-risk and native baseline summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def retained_mean(scores: np.ndarray, values: np.ndarray, count: int) -> dict:
    """Expected retained mean and attainable extrema under exact score ties."""
    scores, values = np.asarray(scores, dtype=float), np.asarray(values, dtype=float)
    if scores.ndim != 1 or scores.shape != values.shape or not np.isfinite(scores).all() or not np.isfinite(values).all():
        raise ValueError("Finite aligned vectors are required")
    if not 1 <= count <= len(scores):
        raise ValueError("Invalid retained count")
    expected = lower = upper = 0.0
    remaining = count
    partial = None
    for score in np.unique(scores):
        block = np.sort(values[scores == score])
        take = min(remaining, len(block))
        if not take:
            break
        expected += float(block.mean()) * take
        lower += float(block[:take].sum())
        upper += float(block[-take:].sum())
        if take < len(block):
            partial = {"score": float(score), "block_size": len(block), "retained_from_block": take}
        remaining -= take
    return {"expected_mean_plume_ssim": expected / count,
            "attainable_min": lower / count, "attainable_max": upper / count,
            "partial_tie": partial}


def build_supplement(root: Path = ROOT) -> dict:
    """Use manifest-pinned scalar records only, without private assets or inference."""
    with (root / "MANIFEST_CODE_REVIEW.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    manifest = {record["path"]: record["sha256"] for record in records}
    if len(manifest) != len(records):
        raise ValueError("Duplicate manifest paths")
    sources = []

    def load(relative: str) -> dict:
        raw = (root / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != manifest[relative]:
            raise ValueError("Source manifest mismatch: " + relative)
        sources.append({"path": relative, "sha256": digest})
        return json.loads(raw)

    statistics = load("results/orientation_v5/statistics.json")
    results = {}
    for split, expected_count in (("iid_test", 110), ("ood_test", 270)):
        metrics = load("results/postfix_final/ensemble/eval/ensemble_metrics_" + split + ".json")
        rows = metrics["per_sample"]
        if len(rows) != expected_count or len({(r["param_id"], r["real_id"]) for r in rows}) != expected_count:
            raise ValueError("Native case identity/count mismatch")
        scores = np.asarray([r["mean_ensemble_var"] for r in rows])
        values = np.asarray([r["mean_plume_ssim"] for r in rows])
        if not np.isclose(values.mean(), metrics["aggregate"]["plume_ssim_mean"], rtol=0, atol=1e-14):
            raise ValueError("Native aggregate mismatch")
        post = statistics["coverage"][split]
        results[split] = {
            "n_cases": expected_count,
            "n_distinct_scores": len(np.unique(scores)),
            "selection_rule": "Ascending full-field mean ensemble variance; uniform selection within exact ties",
            "curves": [{"retention": fraction, "n_retained": round(expected_count * fraction),
                        **retained_mean(scores, values, round(expected_count * fraction))}
                       for fraction in (1.0, .9, .8, .7, .6, .5)],
            "crc": post["crc_descriptive_only_unbounded_loss"],
            "predicted_extra175": post["coverage"]["extra_only/predicted/scp/0.1"]["predicted"],
            "plume_selection": post["selection"],
        }
    baselines = {}
    for family in ("mcdrop", "deterministic"):
        document = load("results/orientation_v5/native_baselines/" + family + ".json")
        baselines[family] = {}
        for split in ("iid_test", "ood_test"):
            rows = [r for r in document["per_seed"] if r["split"] == split]
            if sorted(r["seed"] for r in rows) != list(range(10)):
                raise ValueError("Baseline seed inventory mismatch")
            summaries = [r for r in document["summaries"] if r["split"] == split and
                         r["seed_set"] == "seeds_0_9" and r["metric"] == "plume_ssim_mean"]
            if len(summaries) != 1:
                raise ValueError("Nonunique baseline summary")
            values = np.asarray([r["plume_ssim_mean"] for r in rows])
            summary = summaries[0]
            if not (np.isclose(values.mean(), summary["mean"], rtol=0, atol=1e-14) and
                    np.isclose(values.std(ddof=0), summary["std"], rtol=0, atol=1e-14)):
                raise ValueError("Baseline aggregate mismatch")
            baselines[family][split] = {
                "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                "std_definition": "descriptive checkpoint spread; ddof=0", "n_seeds": len(values),
            }
    return {
        "status": "DESCRIPTIVE_SCALAR_RECONSTRUCTION_COMPLETE",
        "results": results, "baselines": baselines, "sources": sources,
        "scope": "Native AMP selection and baseline summaries, with certified corrected excess-risk and region sensitivities. Attainable tie extrema are not confidence intervals. Calibration-budget attainment is not test-budget attainment; unbounded loss and shared geological draws preclude a prospective risk certificate. No inference or independent physical validation is performed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file outside the package")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("Select an output outside the package")
    if output.exists():
        raise FileExistsError("Preserve existing outputs")
    document = build_supplement()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
    print("Reconstructed two native curves, four baseline summaries and corrected descriptive excess risk")


if __name__ == "__main__":
    main()
