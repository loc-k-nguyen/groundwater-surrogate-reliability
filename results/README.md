# Result Inventory

orientation_v5/ contains corrected scalar statistics, conditional/paired summaries, four-family case scores and all 42,800 case-time metrics. MANIFEST.json checksums preserve export identity and original source fingerprints. No full concentration or prediction array is included.

postfix_final/ensemble/eval/ contains only the two unchanged certified native AMP accuracy/review inputs. reporting_statistics/reporting_statistics.json contains only their unchanged reference-derived review threshold, its original source hash and a scope label. These native inputs are not substituted for the FP32 four-family quality protocol.

orientation_v5/native_baselines/ preserves the original native Monte Carlo dropout and deterministic scalar JSON summaries, including all ten checkpoint seeds. orientation_v5/selection_crc_baselines.json is the current tie-aware selective/descriptive excess-risk supplement. Its five relative source paths and hashes link it to the included inputs; scripts/build_selection_supplement.py reproduces it exactly. The original scalar-export MANIFEST.json retains its original scope; the package-wide manifest also covers these additional files. No historical arbitrary-order selective curve is included as a current result.

twin_audit/ contains the unchanged native controls. Archived p-values and setting-level confidence intervals do not provide population inference: configurations share geological draws, and the disjoint control has only two held-out draw identifiers. These numerical records are not a claim of independent replication.

## Masks and Calibration

Oracle masks require true concentration above 1e-8 and are retrospective. Predicted-positive masks use predicted log concentration above -8; they are not full-field uncertainty scores. Union and full-domain coverage, selected-region coverage and missed true-plume fractions have distinct denominators and must not be merged.

The streaming statistics routine calibrates NEC floors separately by region. The conditional routine uses a common oracle-calibrated floor for its region sensitivities. Their predicted-region NEC values are different protocols, not interchangeable estimates. The historical nec_recal key denotes nearest-regime NEC, not global scalar recalibration; deployable is a historical predicted-positive-region label, not evidence of deployability. ACI step size was selected post hoc using mean within-case shifted coverage, not pooled pixel coverage.

Corrected interval widths/coverage do not restore prospective conformal safety under shift. Full-field and oracle uncertainty results are separate; paired fixed-input scores remain invariant. Unpaired transport AUROC does not establish omitted-control detection or active overconfidence. Integrated concentration errors are concentration proxies, not mass-balance residuals.
