# GLM-5.3 data

- `throughput.csv` contains every working fixed 8K-to-1K DFlash2 path measured
  so far, including the vLLM PP2 K4/K5/K7 sweep. Rows exist only for measured
  cells; requested and observed concurrency are separate fields.
- `prefill.csv` contains the three exact 65,536-token cold-prefill results,
  including the five-sample PP2/AR median.
- `pp2-prefill-samples.json` records each PP2/AR sample, its unique prompt,
  source hash, validation hash, and launch provenance.
- `checkpoint.json` records the exact target, draft, runtime, topology, memory,
  and source-artifact pins.
- `pp2-dflash2-provenance.json` pins the vLLM image/source overlay, PP cohort
  patch, launch profile, autotune keys, and evidence hashes for K4/K5/K7.
- `rtx-pro-6000-comparison.csv` contains the separately labeled 4× RTX PRO
  6000 Blackwell / EXL3 3.25 bpw derivative / native-MTP3 comparison points.
  It is not merged with the exact Inco NVFP4 + DFlash2 result set.

The compact rows are transcribed from the retained validation records under
`/home/catid/frontier-bench/results/glm53-full-dflash2/` and retain the SHA-256
of each source JSON. The PP2/AR CSV row instead retains the SHA-256 of
`pp2-prefill-samples.json`, which lists all five source and validation hashes.
The `publication_status` field separates headline measurements from
working-path diagnostics. No missing result is represented as zero.
