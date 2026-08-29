# GLM-5.3 data

- `throughput.csv` contains every working fixed 8K-to-1K DFlash2 path measured
  so far. Rows exist only for measured cells; requested and observed
  concurrency are separate fields.
- `prefill.csv` contains the two exact 65,536-token cold-prefill results.
- `checkpoint.json` records the exact target, draft, runtime, topology, memory,
  and source-artifact pins.

The compact rows are transcribed from the retained validation records under
`/home/catid/frontier-bench/results/glm53-full-dflash2/` and retain the SHA-256
of each source JSON. The `publication_status` field separates matched headline
measurements from working-path diagnostics. No missing result is represented as
zero.
