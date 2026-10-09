# Scientific Snapshot History

## ems-v6-2026-10-10-r1

Increase paired/time-profile figure typography for manuscript-scale reading, move legends off data, wrap long labels and show exact ladder levels. All scalar/source records remain byte-identical. The first v6 tag is preserved; no experiment, numerical summary or model implementation changed in this presentation revision.

## ems-v6-2026-10-10

- Require explicit physical/log10 scale in the generic conformal API; fix physical plume masking and test invalid/empty inputs. Dedicated manuscript calibration values are unchanged.
- Add an analytic full-domain CPU numerical fixture, fast/full thread-bounded test runners and measured reports.
- Reconstruct paired transverse-ratio groups and time-resolved physical diagnostics from unchanged certified scalars. Retain all negative results and historical summaries.
- Document reproduction levels, external-asset requirements, historical training options and checkpoint producer mappings. No new training, simulator data or restricted-asset publication.

## ems-v5-2026-10-10-r1

This snapshot contains the corrected analyses associated with the current manuscript, paired input/target orientation and full-field sliding-window boundary handling. Frozen checkpoints and splits are unchanged. Earlier repository snapshots contain superseded orientation-dependent calibration and transport summaries; they must not be mixed with the current figures or used as current findings.

Transport full-field uncertainty rankings in the corrected unpaired pools differ from earlier summaries. Fixed-input dispersivity comparisons show unchanged full-field uncertainty scores. These findings do not establish detection of omitted controls. Conformal summaries distinguish oracle, prediction-selected and full-domain regions; the physical concentration cutoff is applied in the correct logarithmic units. Reported coverage is descriptive for a small shared-geology design, not a deployment or risk guarantee.

The package retains lightweight scalar summaries and reproduction scripts, not simulation fields, model setup files, executables, checkpoints or full prediction caches. Original project code is MIT-licensed, with third-party notices retained. Software tests and figure reproduction do not certify historical GPU determinism or reproduce simulations without the excluded assets.
