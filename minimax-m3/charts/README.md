# MiniMax M3 charts

Published figures:

- `nvfp4-decode.png`: aggregate and per-stream 8K/1K decode across thinking
  policies and C1–C16;
- `nvfp4-prefill.png`: client-observed cold-prefill throughput and TTFT at 8K,
  64K, and 128K;
- `nvfp4-natural-quality.png`: maximum repeated 8-gram fraction plus retained
  output completeness/manual review;
- `mxfp8-decode.png`: provisional PP2 aggregate and per-stream 8K/1K decode;
- `prefill-comparison.png`: one-station NVFP4 versus two-station PP2 MXFP8
  cold prefill through 128K;
- `nvfp4-wikitext2.png`: document-level WikiText-2 perplexity and the matched
  BF16-KV single-stream decode result.

Regenerate them from package-local CSV files:

```bash
python3 -m venv .chart-venv
.chart-venv/bin/pip install -r minimax-m3/recipes/render-requirements.txt
.chart-venv/bin/python minimax-m3/recipes/render_charts.py
```

The renderer rejects incomplete measured NVFP4 C1–C16 series, request errors,
capacity-limited selected rows, missing C1/C64/C128 quality cells, and a failed
manual degeneration review. It also requires the measured MXFP8 C1–C32 and
8K/64K/128K prefill rows plus the NVFP4 WikiText-2 row. The graph notes preserve
the provisional MXFP8 C32 interpretation rather than presenting it as tuned.
