# MiniMax M3 data

`model-provenance.json` records the exact official repositories, revisions,
download sizes, licenses, and selected runtime. The CSV files contain headers
only until measurements pass the safety, capacity, error, and quality gates.

Planned artifacts:

- `throughput.csv`: fixed 8K/1K decode by topology, thinking mode, and C1–C128;
- `prefill.csv`: cold exact-token 8K, 64K, and 128K prefill;
- `natural-quality.csv`: retained-output audit summaries;
- `wikitext2-perplexity.csv`: WikiText-2 word/byte PPL and bits/byte;
- `evidence/`: raw `llm-inference-bench`, server metadata, logs, and retained
  responses once collected.

Do not insert zeros for failed, skipped, capacity-limited, or unmeasured cells.

