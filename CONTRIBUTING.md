# Contributing

Use a short-lived branch and describe the behavioral change and its tests. Do not add restricted data, checkpoints, simulator setup files, credentials, or local infrastructure paths.

Preserve the fixed split and numerical source files. Changes to statistical interpretation, scoring regions, crop conventions, or seed handling must be documented; do not mix a changed protocol with historical results.

Run the focused tests, repository audit, and reproduction comparison before proposing a change. Regenerate the manifest after intentional tracked-file changes, then verify it. Use a new output directory for every reproduction run.

Issues should include the command, environment versions, and a minimal non-sensitive example. Restricted raw files should not be attached. Public releases require a verified paper-to-commit mapping and a fresh-clone check.
