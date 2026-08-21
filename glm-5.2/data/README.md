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
throughput values. P8 separately retains the immutable raw PP2 correctness
harness plus a transparent corrected validator/result; it has no performance
row. P9 records the full-envelope 93% PP2 capacity exclusion. P10 is the first
accepted PP2 request-bearing performance increment and includes a
P0-versus-P10 configuration comparison, retained quality, network, capacity,
and startup evidence. P11–P13 add a one-knob TP2 prefill sweep at fixed 4K,
8K, and 16K maximum batched-token chunks, compared with the frozen P0 32K
control. The new arms are prefill-only and publish request/cache sanity rather
than new retained semantic-quality results.

When publishing new measurements, retain the source benchmark JSON, startup
logs, and complete natural-output audit alongside the normalized tables.
