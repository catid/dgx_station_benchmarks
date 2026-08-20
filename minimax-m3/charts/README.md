# MiniMax M3 charts

Published figures:

- `nvfp4-decode.png`: aggregate and per-stream 8K/1K decode across thinking
  policies and C1–C16;
- `nvfp4-prefill.png`: client-observed cold-prefill throughput and TTFT at 8K,
  64K, and 128K;
- `nvfp4-natural-quality.png`: maximum repeated 8-gram fraction plus retained
  output completeness/manual review;
- `mxfp8-decode.png`: PP2 aggregate and per-stream 8K/1K decode across all
  three thinking policies through measured C64;
- `mxfp8-natural-quality.png`: MXFP8 retained-output completeness and
  repetition audit;
- `prefill-comparison.png`: one-station NVFP4 versus two-station PP2 MXFP8
  cold prefill through 128K;
- `nvfp4-wikitext2.png`: document-level WikiText-2 perplexity and the matched
  BF16-KV single-stream NVFP4 result;
- `wikitext2-comparison.png`: matched NVFP4-versus-MXFP8 WikiText-2 word PPL
  and BF16-KV single-stream decode.

Regenerate them from package-local CSV files:

```bash
python3 -m venv .chart-venv
.chart-venv/bin/pip install -r minimax-m3/recipes/render-requirements.txt
.chart-venv/bin/python minimax-m3/recipes/render_charts.py
```

The renderer rejects incomplete measured NVFP4 C1–C16 series, request errors,
capacity-limited selected rows, missing C1/C64/C128 quality cells, and a failed
manual degeneration review. It also requires every MXFP8 thinking mode through
measured C64, both checkpoints' 8K/64K/128K prefill rows, both natural-output
audits, and both WikiText-2 rows. The graph explicitly distinguishes graph-C32
from the measured C64 eager path.
