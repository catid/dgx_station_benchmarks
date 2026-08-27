# GLM-5.3-Flash data

- `throughput.csv` contains the three measured 2× DGX Station decode series,
  including effective concurrency, maximum active requests, saturation,
  latency, MTP, and source-hash fields.
- `prefill.csv` contains all three measured prefill series with client rates,
  prompt lengths, sample counts, available server rates, and source hashes.
- `external-rtx-pro-6000.csv` contains the supplied 4× RTX PRO 6000 MTP5
  comparison values. Fields absent from that handoff remain `NOT_SUPPLIED`.
- `diagnostic-throughput.csv` and `diagnostic-prefill.csv` preserve the fuller
  acquisition records from which the compact published CSVs were selected.
- `qualification.csv` records measured lanes, pending work, the current
  unmeasured NVFP4 candidate, and historical attempt metadata.
- `checkpoint.json` contains audited model and checkpoint facts.

Requested concurrency and observed active concurrency are separate fields.
Saturation is reported as measured behavior; throughput is not replaced with
zero or interpolated. Official FP8, current and historical `LibertAIDAI`
NVFP4 revisions, and `dealignai` uncensored NVFP4 rows remain distinct.
