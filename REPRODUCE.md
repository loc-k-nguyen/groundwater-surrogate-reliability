# Reproduction guide

Run commands from the code-folder root. Summary analyses require no raw fields or GPU.

## Current analyses and figures

```bash
python -m pip install -r requirements-analysis-lock.txt
python scripts/build_v4_analysis.py --output-dir outputs/reproduction
python scripts/build_workflow_v4.py --output-dir outputs/reproduction
python scripts/verify_repository.py --reproduced-dir outputs/reproduction
```

The tested environment is Python 3.9.25, NumPy 1.26.4, SciPy 1.13.1, scikit-image 0.24.0, and Matplotlib 3.9.4. No package installation was performed for the v4 checks. The source environment also contains PyTorch 2.1.0 with CUDA build 11.8 and cuDNN 8.7.0, used for CPU tests and source parameter counting.

The descriptive generator reads the matched-pool CSVs, transport-ladder CSVs, corrected ensemble JSONs, and reference-derived threshold in results/reporting_statistics/. It computes pairwise AUROC with one-half credit for equal scores, and expected review recall under uniform selection within exact uncertainty-score ties. Min/max tie-order bounds are computed analytically. No setting resampling is performed.

| Output in figures/v4_final/ | Manuscript item |
|---|---|
| fig1_workflow_v4.pdf and .svg | Figure 1; vector conceptual workflow |
| fig2_transport_ladder_v4.pdf | Figure 2 |
| fig3_taxonomy_v4.pdf | Figure 3; descriptive taxonomy table |
| fig4_distributions_v4.pdf | Figure 4; all simulation-case points |
| fig5_triage_v4.pdf | Figure 5; tie-aware illustrative review |
| v4_analysis.json | Taxonomy, ladder, triage, and ensemble accuracy |
| model_capacity_v4.json | Source parameter counts |

The JSON records SHA-256 hashes of the numerical source files. For PDF comparison use rendered pixels rather than file hashes, because PDF metadata can include creation timestamps.

## Other manuscript sources

The four-family descriptive control can also be regenerated without the withdrawn permutations:

```bash
python scripts/realization_disjoint_control_analysis.py det_ensemble --output-dir outputs/reproduction/rd
python scripts/realization_disjoint_control_analysis.py hetero --output-dir outputs/reproduction/rd
python scripts/realization_disjoint_control_analysis.py fno --output-dir outputs/reproduction/rd
python scripts/realization_disjoint_control_analysis.py deeponet --output-dir outputs/reproduction/rd
```

- results/twin_audit/obj3_realization_disjoint_control{,_hetero,_fno,_deeponet}.json: four-family small-draw control. Existing p-values inside these archived records are withdrawn from inference.
- results/twin_audit/obj3_twin_paired_shift_effect{,_full}.json: paired amplitude changes.
- results/postfix_final/conformal/conformal_unit_and_calibration_sensitivity.json: 175-only and realization-level oracle-region calibration.
- results/postfix_final/tier0/tier0_results_boundaryfix.csv: original 225-case oracle and predicted-region coverage.
- results/postfix_final/conformal/calibration_nonphysical_sensitivity.json: flagged calibration IDs and quantile sensitivity.
- results/postfix_final/aci/aci_alpha_0.10.json: post-hoc adaptive sensitivity.
- results/postfix_final/selective_prediction_curve.json: retention summaries.
- results/postfix_final/crc/crc_results.json: historical unnormalized Hoeffding sensitivity, with no risk-control guarantee.
- results/postfix_final/ensemble/eval/: full-field SSIM, concentration-sum error, and oracle Gaussian CRPS.

## Focused checks and capacity

```bash
python -m unittest discover -s tests -v
python scripts/model_capacity.py --output outputs/reproduction/model_capacity_v4.json
python scripts/make_manifest.py --check
```

The manifest verifies sizes and SHA-256 digests. It does not replace an archival DOI or approved license.

## Raw scripts

The twelve repaired raw-data scripts use scripts/package_paths.py. Their common root options are --repo-root, --data-root, and --calibration-root, or environment variables SURROGATE_REPO_ROOT, SURROGATE_DATA_ROOT, and SURROGATE_CALIBRATION_ROOT. Package-relative defaults find src/, splits/, and metadata/ from the shipped folder rather than an assumed ancestor depth. The corresponding script's existing checkpoint/cache and output options must still point to restricted assets and a new output location.

For example:

```bash
python scripts/evaluate_matched_input_screening.py --repo-root . --data-root /path/to/fields --matched-root /path/to/transport-fields --matched-audit /path/to/transport-audit.json --output-dir /path/to/new-audit
python scripts/eval_ensemble_uq.py --help
```

Calibration sensitivity scripts additionally accept the original project layout through --repo-root because their cache paths refer to the certified run structure. Missing raw assets prevent execution; root configuration does not create data or make restricted assets available. Their numerical execution has not been retested with raw assets.

## Training entry points

All four families are present:

```bash
python -m src.obj3.conference.train_ms_tmo_obj3 --help
python -m src.obj3.journal.train_ms_tmo_obj3_hetero --help
python -m src.obj3.journal.train_fno_obj3 --help
python -m src.obj3.journal.train_deeponet_obj3 --help
```

Each command accepts --main_data_root, --extra_data_root, --split_json, --stats_json, --out_root, --run_name, --seed, --epochs, --batch_size, --deterministic, and --legacy_crop_sampling. Raw roots and the original training-normalization JSON must be supplied explicitly. Use the fixed split at splits/param_split_obj3_ood.json, isolated output paths, and seeds 0,1,2,3,4 for each ensemble. A one-epoch, batch-size-four, seed-zero smoke must precede any new full training.

Example restricted-data smoke for DeepONet:

```bash
python -m src.obj3.journal.train_deeponet_obj3 --main_data_root /path/to/main-fields --extra_data_root /path/to/calibration-fields --split_json splits/param_split_obj3_ood.json --stats_json metadata/obj3_train_stats.json --out_root /path/to/new-runs --run_name deeponet_smoke_seed0 --seed 0 --epochs 1 --batch_size 4 --max_train_files 8 --max_val_files 4 --legacy_crop_sampling
```

Training writes config.json, epoch metrics, and checkpoints beneath the selected run directory. The common budget is AdamW (1e-4, zero decay), cosine schedule, 200 epochs, batch 16, patch 320, clip 1.0, and no augmentation; FNO uses fp32 and the other families fp16. Historical runs used the legacy crop convention and no deterministic-kernel requirement. A corrected-crop or deterministic retraining is a different implementation protocol, not a regeneration of already reported weights. GPU training and heavy inference require a suitable external compute environment; these stages were not executed during repository preparation.

## Availability

See RELEASE_SCOPE.md and CODE_AVAILABILITY.md. Original project code uses MIT; restricted assets remain excluded. This private preparation snapshot supports summary reproduction, not an assertion that all training assets are publicly available. An archival release DOI will be recorded only after verification.
