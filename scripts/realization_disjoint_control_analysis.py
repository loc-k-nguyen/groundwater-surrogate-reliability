"""Analyse the realization-disjoint twin control for the deep ensemble (CPU only).

The control models were trained and validated on realizations 1-3 only. Because amplitude twins
share a realization identifier, realizations 4 and 5 have no twin anywhere in training (maximum
train-to-test standardized correlation 0.0877; see audit_rd_near_twins.py). This script reports:

1. Twin-free result. On realizations 4-5: plume SSIM on both pools, beside the main-protocol value
   on the same fields, and the shifted-versus-reference AUROC of both uncertainty scores with an
   descriptive AUROC without setting-level inference.

2. Twin-free paired variance test. Shifted fields whose exact twin sits in the reference set pair up
   within realizations 4-5, so the same-pattern comparison is repeated with models that saw neither
   member's pattern.

3. Seen-pattern effect with identical checkpoints. For each shifted configuration, the gap between
   realizations 1-3 (pattern seen in training, exactly or as a near-twin at r >= 0.97) and 4-5
   (pattern unseen, r < 0.09). Realizations 1-3 and 4-5 are different fields, so the same gap is
   computed under the main protocol, which trained on all five and therefore saw every pattern;
   that gap is subtracted as a descriptive difference in differences.

4. Pattern-novelty AUROC. With the control checkpoints, how well each uncertainty score separates
   unseen-pattern fields from seen-pattern fields within each pool.

Read-only. Writes one JSON report.
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np

import argparse

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results/matched_pool"
# Family to analyse: det_ensemble (default, the first control), hetero, fno or deeponet.
parser = argparse.ArgumentParser(description="Finite-design small-draw control; no setting inference")
parser.add_argument("family", nargs="?", choices=("det_ensemble", "hetero", "fno", "deeponet"), default="det_ensemble")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
FAMILY = args.family
if FAMILY == "det_ensemble":
    RD = RUNS / "obj3_jnl_rd_det_eval_20260927" / "rd_det_ensemble_per_sample.csv"
else:
    RD = RUNS / f"obj3_jnl_rd_{FAMILY}_eval_20260928" / f"rd_{FAMILY}_per_sample.csv"
MAIN = sorted(RUNS.glob(f"*{FAMILY}*axisDmatched_boundaryfix*/*per_sample.csv"))[0]
BREAK = REPO / "results/twin_audit/obj3_conductivity_twin_breakdown.json"
OUT = args.output_dir / (
    "obj3_realization_disjoint_control.json" if FAMILY == "det_ensemble"
    else f"obj3_realization_disjoint_control_{FAMILY}.json")
REG = {int(r["param_id"]): r for r in csv.DictReader(open(REPO / "metadata/param_registry_master.csv"))}
SCORES = {"oracle": "plume_total_std_log", "deployable": "mean_total_std_log"}
METRICS = {"plume_ssim": "mean_plume_ssim", **SCORES}
UNSEEN, SEEN = (4, 5), (1, 2, 3)


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["target"], int(str(r["param_id"]).split("_")[-1]), int(str(r["real_id"]).split("_")[-1]))] = r
    return out


def setting(pid):
    r = REG[pid]
    return (r["anisotropy"], r["correlation_length"], r["sigma2Y"])


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(sum(1.0 if a > b else .5 if a == b else 0. for a, b in product(pos, neg)) / (pos.size * neg.size))


rd, main = load(RD), load(MAIN)
rep = {"checkpoints": f"{FAMILY}, 5 members, trained and validated on realizations 1-3",
       "unseen_realizations": list(UNSEEN)}

# 1. twin-free pooled result -----------------------------------------------------------------
t1 = {}
for pool in ("iid_test", "ood_test"):
    keys = [k for k in rd if k[0] == pool and k[2] in UNSEEN]
    t1[pool] = {"n": len(keys),
                "plume_ssim": float(np.mean([float(rd[k]["mean_plume_ssim"]) for k in keys])),
                "plume_ssim_main_protocol_same_fields":
                    float(np.mean([float(main[k]["mean_plume_ssim"]) for k in keys if k in main]))}
for name, col in SCORES.items():
    neg, pos = defaultdict(list), defaultdict(list)
    for (t, pid, rid), r in rd.items():
        if rid in UNSEEN:
            (neg if t == "iid_test" else pos)[setting(pid)].append(float(r[col]))
    t1[f"auroc_{name}"] = {"auroc": auc([x for v in neg.values() for x in v], [x for v in pos.values() for x in v]),
                           "settings": [len(neg), len(pos)], "inference": "descriptive only"}
rep["twin_free_pooled"] = t1

# 2. twin-free paired variance test ----------------------------------------------------------
pairs = [p for p in json.load(open(BREAK))["fields"]
         if p["twin"] and p["best_pool"] == "iid_test" and p["real"] in UNSEEN]
t2 = {}
for name, col in METRICS.items():
    d = np.array([float(rd[("ood_test", p["param_id"], p["real"])][col])
                  - float(rd[("iid_test", p["best_param"], p["best_real"])][col]) for p in pairs])
    t2[name] = {"n_pairs": int(d.size), "mean_change": float(d.mean()), "n_increase": int((d > 0).sum())}
rep["twin_free_paired_variance"] = t2


# 3. seen-pattern effect, difference in differences ------------------------------------------
def per_config_gap(src, col):
    by = defaultdict(lambda: {"seen": [], "unseen": []})
    for (t, pid, rid), r in src.items():
        if t == "ood_test":
            by[pid]["seen" if rid in SEEN else "unseen"].append(float(r[col]))
    return {pid: np.mean(v["seen"]) - np.mean(v["unseen"]) for pid, v in by.items() if v["seen"] and v["unseen"]}


t3 = {}
for name, col in METRICS.items():
    g_rd, g_main = per_config_gap(rd, col), per_config_gap(main, col)
    common = sorted(set(g_rd) & set(g_main))
    did = np.array([g_rd[p] - g_main[p] for p in common])
    t3[name] = {"n_configs": len(common),
                "control_gap_seen_minus_unseen": float(np.mean([g_rd[p] for p in common])),
                "main_protocol_gap_difficulty_only": float(np.mean([g_main[p] for p in common])),
                "seen_pattern_effect_net": float(did.mean()),
                "n_configs_positive": int((did > 0).sum())}
rep["seen_pattern_effect"] = t3

# 4. pattern-novelty AUROC --------------------------------------------------------------------
t4 = {}
for pool in ("ood_test", "iid_test"):
    for name, col in SCORES.items():
        seen = [float(r[col]) for (t, _, rid), r in rd.items() if t == pool and rid in SEEN]
        uns = [float(r[col]) for (t, _, rid), r in rd.items() if t == pool and rid in UNSEEN]
        t4[f"{pool}_{name}"] = {"auroc_unseen_vs_seen": auc(seen, uns), "n_seen": len(seen), "n_unseen": len(uns)}
rep["pattern_novelty_auroc"] = t4

# 4b. Null baseline for 4: the same realization 4-5 versus 1-3 AUROC under the main protocol, which
# trained on all five realizations, so no pattern is novel there. Any departure from 0.5 in that
# baseline is realization-level difficulty, not novelty, and the control's AUROC is read against it.
t4b = {}
for pool in ("ood_test", "iid_test"):
    for name, col in SCORES.items():
        seen = [float(r[col]) for (t, _, rid), r in main.items() if t == pool and rid in SEEN]
        uns = [float(r[col]) for (t, _, rid), r in main.items() if t == pool and rid in UNSEEN]
        t4b[f"{pool}_{name}"] = {"auroc_r45_vs_r123_all_seen": auc(seen, uns), "n_seen": len(seen), "n_unseen": len(uns)}
rep["pattern_novelty_auroc_null_baseline_main_protocol"] = t4b
rep["independent_units_note"] = ("Spatial fields share random draw identifiers across settings. "
                                 "Only two evaluation draws are held out; results are descriptive.")

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    raise FileExistsError(OUT)
json.dump(rep, open(OUT, "w"), indent=1)
print(json.dumps(rep, indent=1))
