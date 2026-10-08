from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SEED0 = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_mcdrop_full_seed0"
DEFAULT_SEED1 = REPO_ROOT / "experiments" / "obj3" / "journal" / "runs" / "obj3_jnl_ms_tmo_mcdrop_seed1"
DEFAULT_SEED2 = REPO_ROOT / "experiments" / "obj3" / "journal" / "runs" / "obj3_jnl_ms_tmo_mcdrop_seed2"
DEFAULT_FALLBACK1 = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_mcdrop_full_seed1"
DEFAULT_FALLBACK2 = REPO_ROOT / "experiments" / "obj3" / "conference" / "runs" / "obj3_conf_ms_tmo_mcdrop_full_seed2"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "experiments" / "obj3" / "journal" / "reports" / "analysis" / "mcdrop_3seed_summary.json"


def _resolve_run(primary: Path, fallback: Path | None = None) -> Path:
    iid_json = primary / "mcdrop_metrics_iid_test.json"
    ood_json = primary / "mcdrop_metrics_ood_test.json"
    if iid_json.exists() and ood_json.exists():
        return primary
    if fallback is not None:
        iid_json = fallback / "mcdrop_metrics_iid_test.json"
        ood_json = fallback / "mcdrop_metrics_ood_test.json"
        if iid_json.exists() and ood_json.exists():
            return fallback
    raise FileNotFoundError(f"MC Dropout metrics not found in {primary} or fallback {fallback}.")


def _load_metrics(run_dir: Path, split: str) -> dict[str, float]:
    with open(run_dir / f"mcdrop_metrics_{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)["aggregate"]


def _summary(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "per_seed": [float(v) for v in arr.tolist()]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Obj3 journal 3-seed MC Dropout summary.")
    ap.add_argument("--seed0_dir", type=str, default=str(DEFAULT_SEED0))
    ap.add_argument("--seed1_dir", type=str, default=str(DEFAULT_SEED1))
    ap.add_argument("--seed2_dir", type=str, default=str(DEFAULT_SEED2))
    ap.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    args = ap.parse_args()

    seed_dirs = [
        _resolve_run(Path(args.seed0_dir)),
        _resolve_run(Path(args.seed1_dir), DEFAULT_FALLBACK1),
        _resolve_run(Path(args.seed2_dir), DEFAULT_FALLBACK2),
    ]

    iid = [_load_metrics(path, "iid_test") for path in seed_dirs]
    ood = [_load_metrics(path, "ood_test") for path in seed_dirs]

    iid_plume = [entry["plume_ssim_mean"] for entry in iid]
    ood_plume = [entry["plume_ssim_mean"] for entry in ood]
    iid_var = [entry["mc_var_mean"] for entry in iid]
    ood_var = [entry["mc_var_mean"] for entry in ood]
    var_increase = [ood_var[idx] - iid_var[idx] for idx in range(len(seed_dirs))]

    payload = {
        "seeds": [0, 1, 2],
        "iid_plume_ssim": _summary(iid_plume),
        "ood_plume_ssim": _summary(ood_plume),
        "iid_variance_mean": _summary(iid_var),
        "ood_variance_mean": _summary(ood_var),
        "ood_variance_increase": _summary(var_increase),
        "resolved_run_dirs": [str(path) for path in seed_dirs],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
