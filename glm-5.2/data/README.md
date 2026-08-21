# GLM-5.2 publication data

The CSV files in this directory contain only accepted measurements. Empty CSVs
retain their schema but deliberately contain no `pending` or fabricated-zero
rows. `publication-manifest.json` distinguishes absent and rejected topology
runs.

`evidence/` contains compact benchmark and full natural-output audit JSON.
`runtime/` contains hashes plus the exact container command arrays and pinned
provenance extracted from the much larger retained runtime logs. Run
`../recipes/publish_results.sh` to regenerate all of these files.

[`deep-study/`](deep-study/) contains separately frozen, checksummed increments
from the broader optimization matrix. Accepted TP2+EP2 CuTeDSL measurements
are kept separate from backend incompatibilities, capacity failures, and
pre-request audit-harness failures. Excluded and harness-only starts contain no
throughput values.

When publishing new measurements, retain the source benchmark JSON, startup
logs, and complete natural-output audit alongside the normalized tables.
