# Provenance and corrections

This preparation snapshot follows manuscript revision 4 dated 8 October 2026. It retains lightweight corrected numerical inputs and their certification records. Those records document the excluded source caches; including a cache checksum does not make the cache available.

The sliding-window boundary correction covers terminal positions and rejects zero-coverage pixels. Original restricted checkpoints and data were not changed by this repository preparation. Historical training used nondeterministic cuDNN benchmarking and the legacy crop-origin convention. The revised source exposes a corrected sampler and an explicit deterministic option for future runs.

The analysis withdraws setting-bootstrap intervals and setting-permutation p-values because generator draws are shared across settings. Historical JSON fields remain as provenance, not current inferential evidence. Figure sources identify numerical input files by SHA-256. Triage bounds describe possible tie orderings, not sampling uncertainty.

Simulator command fields were removed from five metadata files. All other fields were checked for equality; `metadata_transformation.json` records input and output hashes. A simulator configuration overlay, unrelated temporal evaluation scripts, and an unverified historical simulator-speed benchmark are not part of this repository.

AI-assisted tools were used in code review, analysis/figure tooling, and documentation editing. Their use does not replace numerical verification or author responsibility. Scientific correction records and limitations are retained rather than removed for presentation.

Method families and shared implementations are identified in the source. A file-level origin and license review remains required before public distribution; do not assume a top-level license replaces third-party obligations.
