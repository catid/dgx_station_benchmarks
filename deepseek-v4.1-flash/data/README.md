# DeepSeek-V4.1-Flash data

Every table is produced by [`build_data.py`](build_data.py) from the manifest
[`sources.json`](sources.json); nothing is typed by hand. Rerun it with
`--source-root` pointing at the private benchmark working directory, then
`python3 data/build_data.py --check` to validate the committed files offline.

- `prefill.csv` — accepted prefill rows only (`publication_status=accepted`,
  `rankable=true`): one row per lane × prompt length × concurrency with the
  lane's `lane_label`/`series_order` (the label and order every chart and
  README table uses), wall
  time, aggregate prompt tok/s, TTFT mean/p50/p99, the server
  `prompt_tokens_total` counter delta, token-count parity, rank-0 GPU
  utilisation/power, the run id, and the SHA-256 of the source JSONL. A lane is
  accepted only when its full 16K/32K/64K/128K × C1/C4/C16 grid completed with
  zero request errors. Four lanes are accepted: the replay-off SGLang Data
  Direct lane (`swa_bounded_replay=false`, the numerically exact full-prefill
  reference), the SGLang SWA bounded replay lane (`swa_bounded_replay=true`,
  DeepSeek's documented deployment technique, faster and not bit-identical),
  the vLLM TP1 × PP2 lane (text-only, two local source patches; its GPU
  columns were sampled on node0 = pipeline stage 0), and the vLLM TP2 lane
  (same image and settings; launched with rank 0 on node1, so its GPU columns
  describe TP rank 1). Both vLLM lanes take `server_prompt_tokens_delta` from
  the `vllm:prompt_tokens_total` counter.
- `diagnostic-prefill.csv` — the same columns plus `diagnostic_status`, for
  tuning-ladder steps and spot checks that are never ranked
  (`publication_status=diagnostic`, `rankable=false`) but are still drawn as
  their own chart series. Rows with request errors are dropped, not zeroed.
- `throughput.csv` — accepted `llm-inference-bench` decode rows: 8,192-token
  input, 1,024 forced output tokens, `5 × C` requests after `C` warm-ups (a
  lane may merge several run directories, such as the SGLang DSpark C64 cell
  measured after its C1–C32 sweep; `run_id` names the run behind each row and
  `qualification.csv` lists every run behind an accepted lane, `;`-separated), with
  per-user p50 rate, TTFT, ITL, effective concurrency, and the DSpark accept
  length/rate for speculative lanes (`mtp_accept_length` stays empty: V4.1
  has no classic MTP head), plus `lane_label`/`series_order`.
  `engine_steps_per_second` is filled only where the server reports a
  positive step counter (SGLang DSpark); SGLang AR reports zero and vLLM has
  no such counter, and neither is ever published as a zero. Accepted and
  diagnostic decode lanes share this table; `publication_status` tells them apart.
- `fabric.csv` — nccl-tests `all_reduce_perf` rows (out-of-place and in-place)
  per rail configuration and message size, with the run average, the
  out-of-bounds verdict, and whether NCCL logged the Data Direct DMA interface.
- `profile.csv` — GPU kernel time by category from one torch-profiler trace
  per node and profile (`tools/analyze_trace.py` summaries), with the summed
  kernel time and kernel count of the capture.
- `qualification.csv` — the lane ledger: one row per lane and metric with its
  disposition (`PASS_RANKABLE_*`, `DIAGNOSTIC_UNRANKED_*`, `PENDING`,
  `NOT_MEASURED`, `NOT_ATTEMPTED`) and the evidence run id. Only
  `PASS_RANKABLE_*` rows are rankable.
- `checkpoint.json` — audited model and checkpoint facts (`config.json`
  shape, `model.safetensors.index.json` size and entry counts, on-disk shard
  bytes, quantization block format).
- `evidence/` (optional, written with `--copy-evidence`) — sanitized copies of
  the source JSONL/JSON/log/txt artifacts (decode `c*.json` files under their
  run-directory name) with a `SHA256SUMS` list; every copy must pass the same
  redaction rules as the tables or the build fails.

`engine` and `runtime` carry the same value; `runtime` is kept for schema
parity with the other sections. GPU utilisation and power columns come from
the benchmark client's 1 Hz `nvidia-smi` sampling of the rank-0 station only.
Hostnames are published as `node0`/`node1`; private paths, management
addresses, and GPU UUIDs are rejected by the converter.

Missing measurements remain empty. Failed, unsupported, or invalid cells are
never represented by zero and are not interpolated.
