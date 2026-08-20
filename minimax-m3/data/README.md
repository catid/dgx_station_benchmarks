# MiniMax M3 data

`model-provenance.json` records the exact official repositories, revisions,
download sizes, licenses, and selected runtime. The normalized CSVs contain
complete one-station NVIDIA NVFP4 and two-station MiniMax MXFP8 PP2 base-model
milestones.

Artifacts:

- [`throughput.csv`](throughput.csv): fixed 8K/1K decode by thinking mode,
  quantization, and topology. NVFP4 C1–C16 and MXFP8 C1–C64 are measured for
  all three thinking policies; larger cells are explicit capacity rows with
  blank performance fields rather than invented zeros. MXFP8 C64 is a measured
  eager-path row beyond the captured C32 graph tier.
- [`prefill.csv`](prefill.csv): cold exact-token 8K, 64K, and 128K prefill for
  both measured checkpoints/topologies.
- [`natural-quality.csv`](natural-quality.csv): NVFP4 and MXFP8 retained-output
  audit summaries for C1, C64, and C128 across all three thinking policies.
- [`wikitext2-perplexity.csv`](wikitext2-perplexity.csv): measured NVFP4 and
  MXFP8 document-level WikiText-2 word/byte perplexity plus matched BF16-KV
  single-stream decode.
- [`evidence/`](evidence/): sanitized authoritative `llm-inference-bench` JSON,
  the disclosed selected reruns, retained natural samples, both WikiText-2
  results, all MXFP8 thinking modes, the superseded eager-C32 result, and
  SHA-256 hashes for every audited output.

Run `recipes/package_results.py` against a private raw-results directory to
regenerate the normalized CSV and sanitized evidence. The source host name is
replaced with `benchmark-host`; local addresses are reduced to `loopback`.

Do not insert zeros for failed, skipped, capacity-limited, or unmeasured cells.
