# GLM-5.3-Flash data

- `external-rtx-pro-6000.csv` contains the user-supplied 4× RTX PRO 6000 MTP5
  decode and prefill values. It is explicitly external and not independently
  validated. Exact revision/runtime/parallelism fields absent from the handoff
  are stored as `NOT_SUPPLIED`, never inferred from the DGX lane.
- `qualification.csv` records smoke, failed-timing, and pending dispositions.
- `diagnostic-throughput.csv` contains the measured native-FP8 decode cells
  from three sealed runs that failed the whole-run publication gate. Every row
  is explicitly non-rankable and retains occupancy/failure fields plus the raw
  result SHA-256.
- `diagnostic-prefill.csv` contains exact, multi-sample native-FP8 prefill
  cells from two of those failed parent runs. They are useful diagnostics but
  are not promoted independently of their failed run.
- `throughput.csv` and `prefill.csv` are accepted-only GB300 tables. They are
  header-only because no complete timed profile has passed publication gates.
- `checkpoint.json` contains audited model and checkpoint facts.

Missing measurements remain empty. Failed, unsupported, or invalid cells are
never represented by zero and are not interpolated. Official FP8,
`LibertAIDAI` NVFP4, and `dealignai` uncensored NVFP4 rows retain distinct model
IDs and revisions.
