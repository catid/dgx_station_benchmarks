# MiniMax M3 data

`model-provenance.json` records the exact official repositories, revisions,
download sizes, licenses, and selected runtime. The normalized CSVs contain
the complete one-station NVIDIA NVFP4 milestone plus the provisional
two-station MiniMax MXFP8 disabled-thinking throughput/prefill milestone.

Artifacts:

- [`throughput.csv`](throughput.csv): fixed 8K/1K decode by thinking mode,
  quantization, and topology. NVFP4 C1–C16 and MXFP8 C1–C32 are measured;
  larger cells are explicit capacity rows with blank performance fields rather
  than invented zeros. MXFP8 C32 is marked provisional because it ran beyond
  the captured graph tiers.
- [`prefill.csv`](prefill.csv): cold exact-token 8K, 64K, and 128K prefill for
  both measured checkpoints/topologies.
- [`natural-quality.csv`](natural-quality.csv): retained-output audit summaries
  for C1, C64, and C128 across all three thinking policies.
- [`wikitext2-perplexity.csv`](wikitext2-perplexity.csv): measured NVFP4
  document-level WikiText-2 word/byte perplexity and BF16-KV single-stream
  decode; MXFP8 quality remains pending.
- [`evidence/`](evidence/): sanitized authoritative `llm-inference-bench` JSON,
  the disclosed selected reruns, retained natural samples, NVFP4 WikiText-2
  result, MXFP8 disabled-thinking benchmark, and SHA-256 hashes for every
  audited output.

Run `recipes/package_results.py` against a private raw-results directory to
regenerate the normalized CSV and sanitized evidence. The source host name is
replaced with `benchmark-host`; local addresses are reduced to `loopback`.

Do not insert zeros for failed, skipped, capacity-limited, or unmeasured cells.
