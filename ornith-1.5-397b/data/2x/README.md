# Two-station PP2 and TP2 data

This directory contains measured two-DGX-Station results over one active
400GbE RoCE rail. `PP2` means pipeline parallel size 2. `TP2` means tensor
parallel size 2 with expert parallelism enabled.

Files:

- `decode-throughput.csv`: the complete 30-second C1–C128 sustained matrix for
  both topologies. The C128 rows are retained as short-window diagnostics.
- `c128-stability.csv`: paired 30-second and 60-second C128 results. Rows with
  `role=headline_stable` are the values used for topology headlines and charts.
- `prefill.csv`: repeated standalone cold-prefill results at canonical 8K, 64K,
  and 128K tokenizer targets.
- `quality-audit-summary.json`: compact repetition statistics for four full
  normal-EOS-policy outputs per topology.
- `raw/`: benchmark JSON, correct natural-output audits, container commands,
  model endpoint captures, and the pinned benchmark commit. Large console and
  server logs are intentionally excluded.
- `wikitext2-perplexity.csv`: header-only. Two-station PPL was not rerun because
  topology does not change weights; the canonical BF16-KV result is under
  `../1x/`.

All decode rows used exact 8,192-token input, 1,024 output tokens, temperature
zero, EOS ignored, and FP8 E4M3 KV. The harness marked PP2 and TP2 rows at C16
and higher as capacity-limited. They remain visible with measured scheduler
occupancy; offered concurrency must not be mistaken for fully resident
concurrency.

In the decode CSVs, `per_request_tps` is aggregate throughput divided by offered
concurrency. `ttft_seconds` and `itl_seconds` are p50 values; averages remain in
the raw benchmark JSON.

The stable C128 values are 3,799.6 aggregate output tok/s for PP2 and 2,447.1
for TP2+EP. TP2's short 30-second cell reached 3,966.5 tok/s but did not persist
over 60 seconds, so it is not used as a headline.
