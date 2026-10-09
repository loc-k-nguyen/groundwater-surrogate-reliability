# Historical Training and Restricted Assets

These commands are for users with authorized external data and suitable GPU resources. No data, checkpoint or simulator permission is supplied by the software license. They are source/CLI checked, not rerun to prepare v6. Historical GPU kernels were not deterministic; repeating a configuration does not promise identical weights.

## Fixed Configuration Ledger

| Family | Module after `python -m` | Precision | Architecture options |
|---|---|---|---|
| U-Net | `src.obj3.conference.train_ms_tmo_obj3` | FP16 AMP | `--base_channels 64 --attn_heads 4 --alpha 3 --lambda_temp 0.05 --amp` |
| Heteroscedastic U-Net | `src.obj3.journal.train_ms_tmo_obj3_hetero` | FP16 AMP | `--base_channels 64 --attn_heads 4 --amp` |
| Fourier operator | `src.obj3.journal.train_fno_obj3` | FP32, fixed in source | `--fno_width 64 --fno_modes 20 --fno_depth 4` |
| DeepONet | `src.obj3.journal.train_deeponet_obj3` | FP16 AMP, fixed in source | `--branch_width 128 --branch_latent 384 --trunk_width 384 --basis_rank 96` |

One setting per family is recorded. Full earlier architecture/loss search histories and historical GPU-hours are unavailable. Shared budgets do not establish equal tuning effort or an optimal accuracy ranking. Each reported ensemble uses seeds 0 through 4. The ten-checkpoint MC-dropout parity summary is separate and excludes the transport pool.

## Command Template

Replace example external paths with authorized locations and give every family/seed a new run name:

```bash
python -m src.obj3.conference.train_ms_tmo_obj3 \
  --main_data_root ../approved_assets/native_fields \
  --extra_data_root ../approved_assets/calibration_fields \
  --split_json splits/param_split_obj3_ood.json \
  --stats_json metadata/obj3_train_stats.json \
  --out_root ../new_training --run_name unet_seed0 --seed 0 \
  --epochs 200 --batch_size 16 --patch_size 320 \
  --lr 0.0001 --weight_decay 0 --grad_clip_norm 1 --num_workers 0 \
  --legacy_crop_sampling --base_channels 64 --attn_heads 4 \
  --alpha 3 --lambda_temp 0.05 --amp
```

For another family, replace the module, unique run name and architecture options with the table entry. Do not pass `--amp` to FNO or DeepONet: their precision is set in source. The shared schedule is cosine annealing over 200 epochs without warmup; geometric augmentation is disabled. The historical sampler excludes the final crop origin, hence `--legacy_crop_sampling`. Omitting it uses the repaired sampler and is not the reported training protocol. `--deterministic` is available for new studies, not enabled retrospectively. The realization-disjoint control adds `--realizations 1 2 3` for fitting and checkpoint selection; it holds out draws 4 and 5.

## Producer and Asset Matrix

| Stage | Required inputs | Outputs / producer |
|---|---|---|
| Fitting | Authorized NPZ fields, split and normalization | `new_training/RUN/config.json`, `best.pt`, `last.pt`, `train_log.csv` |
| Corrected raw inference | Five-member checkpoint/config hashes, original certificates/inventory, declared field roots | `run_orientation_fixed_inference.py`; external isolated caches |
| FP32 physical quality | Same approved assets and certificates | `evaluate_four_family_quality_v5.py`; `quality.json`, `per_time.csv` |
| Calibration | Certified AMP caches, declared unit and mask | Dedicated statistics/conditional scripts, not generic conformal API |
| Summary reproduction | Public CSV/JSON only | `build_v5_analysis.py`, `build_selection_supplement.py`, `build_physical_diagnostics_v6.py` |

All five checkpoint/config fingerprints per family are in `results/orientation_v5/quality/FAMILY/quality.json`. Fingerprints establish lineage, not access to absent weights. Raw execution needs original certification records and frozen inventory paths; source checks do not numerically validate assets on a new machine. Training interfaces can write into existing run directories: choose new paths. No blanket no-overwrite claim is made for historical training interfaces.
