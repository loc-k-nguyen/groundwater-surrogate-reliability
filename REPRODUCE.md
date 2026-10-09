# Reproduction

## Environment

Summary reconstruction was tested with Python 3.9.25, NumPy 1.26.4, SciPy 1.13.1, scikit-image 0.24.0 and Matplotlib 3.9.4. The local software tests use PyTorch 2.1.0. The reported four-family GPU quality run used PyTorch 2.1.2/CUDA 11.8 on an A100 40 GB in FP32. Calibration and native review inputs retain their distinct AMP protocol. These environments are not claimed bitwise equivalent for GPU execution.

requirements-analysis-lock.txt records the tested scalar-analysis dependencies. requirements.txt adds the local tested PyTorch stack. The shipped source does not import scikit-learn; it is not required by these commands. Installing those packages is a separate user action; this snapshot did not install dependencies or execute training.

## Scalar Results and Figures

Run from the package root and select a new output outside it:

```bash
python scripts/make_manifest.py --check
python scripts/verify_repository.py
python -m unittest discover -s tests -v
python scripts/build_v5_analysis.py --output-dir ../reproduced_v5
python scripts/verify_repository.py --reproduced-dir ../reproduced_v5
python scripts/build_selection_supplement.py --output ../selection_crc_baselines.json
python scripts/verify_repository.py --supplement ../selection_crc_baselines.json
```

The generator uses results/orientation_v5/quality/, two unchanged native AMP accuracy files and a reduced JSON containing only the unchanged reference-derived review threshold. Current outputs are compared to figures/v5_corrected/v5_analysis_portable.json. Numerical scientific sections must match exactly; source fingerprints differ intentionally for the reduced threshold file. Generated source paths and hashes must resolve inside this package. PNG comparisons use pixels, not timestamp-dependent PDF bytes.

The package includes corrected calibration, conditional and paired scalar summaries in results/orientation_v5/. It does not re-execute private-cache calibration through the figure command. Historical scalar controls under results/twin_audit/ are retained only as finite-design controls; archived p-values and setting-level confidence intervals are withdrawn from population inference. See [results/README.md](results/README.md) for masks and protocol distinctions.

The separate selection command reconstructs results/orientation_v5/selection_crc_baselines.json from five manifest-pinned scalar inputs. It uses uniform selection within exact uncertainty-score ties; attainable extrema are not confidence intervals. It preserves ten-checkpoint baseline spreads with ddof=0 and corrected descriptive excess-risk/region summaries. CRC calibration-criterion attainment is not test-budget attainment or a prospective risk guarantee. This command performs no inference and refuses existing outputs or outputs inside the package.

The workflow source and shipped PDF/SVG are provided. Regenerating the manual workflow requires the authors' diagrams.net layout process; the scalar command reproduces only quantitative figures 2-5. It neither regenerates simulation panels nor revises their spatial orientation.

## CPU-Only Source Checks

```bash
python scripts/evaluate_four_family_quality_v5.py --check-sources
python scripts/run_orientation_fixed_inference.py --check-sources
python scripts/recompute_orientation_statistics_v5.py --help
```

The first two return SOURCE_IMPORTS_PASSED_ONLY with data_checked=false and inference_performed=false. These checks import bundled sources and report hashes; they do not validate fields, model weights, full caches or numerical output.

Actual corrected inference requires external approved assets, the original certificates and frozen absolute inventory paths, all five checkpoint/config hashes per family, declared native/reversed field roots, verified normalization and CUDA. Source overrides are --runner-path, --evaluator-path and --adapter-path; asset roots are explicitly configurable. The quality runner also accepts --run-root for original certification records. No silent inventory relocation is supported. Existing outputs are preserved. No training, raw inference or cache download is authorized by a source-check command.

## Historical Training Protocol

All four training modules expose --help, --legacy_crop_sampling and --deterministic. Reported checkpoints used the legacy crop sampler and did not enforce deterministic GPU kernels. The repaired sampler or deterministic execution changes that implementation protocol; it is not a regeneration of reported weights. Fixed seeds do not imply bitwise historical GPU repeatability.

Repeating training requires approved fields and normalization, unchanged splits, explicit new output paths and the appropriate server execution process. Raw assets are not supplied by this package. No new training or simulation was performed to prepare this snapshot.
