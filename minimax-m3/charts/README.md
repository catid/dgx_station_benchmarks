# MiniMax M3 charts

Charts will be rendered only after the corresponding raw evidence and
normalized CSV rows are present. Planned figures are:

- aggregate decode tokens/s versus concurrency, faceted by thinking mode;
- per-stream decode tokens/s versus concurrency;
- cold prefill tokens/s and TTFT at 8K, 64K, and 128K;
- NVFP4 one-station results and a separately labeled MXFP8 PP2 capacity row, if collected;
- WikiText-2 perplexity and BF16-KV decode throughput.

The rendering recipe will read only package-local CSV files so every committed
figure can be regenerated without access to the benchmark hosts.
