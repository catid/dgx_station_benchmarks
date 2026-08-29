# GLM-5.3 data

- `throughput.csv` contains every working fixed 8K-to-1K DFlash2 path measured
  so far, including the vLLM PP2 K4/K5/K7 sweep. Rows exist only for measured
  cells; requested and observed concurrency are separate fields.
- `prefill.csv` contains the retained cold-prefill results, including matched
  five-sample PP2/AR medians at exact 8K, 64K, and 128K.
- `pp2-prefill-samples.json` records all 15 PP2/AR sweep samples, unique prompt
  hashes, source hashes, validation hashes, and launch provenance.
- `checkpoint.json` records the exact target, draft, runtime, topology, memory,
  and source-artifact pins.
- `pp2-dflash2-provenance.json` pins the vLLM image/source overlay, PP cohort
  patch, launch profile, autotune keys, and evidence hashes for K4/K5/K7.
- `headline-per-user.csv` keeps PP2 and TP2+EP2 as separate topology series
  and stores `aggregate output tok/s / offered C`. Missing cells stay absent.
- `rtx-pro-6000-comparison.csv` contains the separately labeled 4× RTX PRO
  6000 Blackwell / EXL3 3.25 bpw derivative / native-MTP3 comparison points.
  It is not merged with the exact Inco NVFP4 + DFlash2 result set.

The compact rows are transcribed from the retained validation records under
`/home/catid/frontier-bench/results/glm53-full-dflash2/` and retain the SHA-256
of each source JSON. The PP2/AR CSV rows instead retain the SHA-256 of
`pp2-prefill-samples.json`, which lists all 15 source and validation hashes.
The `publication_status` field separates headline measurements from
working-path diagnostics. No missing result is represented as zero.
