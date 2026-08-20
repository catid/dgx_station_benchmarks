# Hy3 data schema

Only verified measurements belong in this directory. Rows marked
`provisional_tuning` passed structural and zero-error checks but are not final
publication cells. Rows marked `accepted` additionally require the complete
C1-C128 matrix, canonical runtime evidence, and a natural-output audit.

## `capacity.csv`

- `per_gpu_reported_hbm_mib`: exact `nvidia-smi` usable-memory report, not marketing capacity.
- `weight_bytes`: aggregate size of the 101 indexed safetensors shards.
- `weight_only_fit`: whether the weight bytes fit aggregate HBM, before runtime allocations.
- `native_runtime_status`: `not_launchable` for one GB300 or
  `validated_pp2_and_tp2_ep` for the completed two-node matrix.
- `cpu_offload_included`: always false for comparable GPU-native tables.

## `throughput.csv`

One row represents one validated fixed-length decode cell.

- `publication_status`: `provisional_tuning` or `accepted`.
- `topology`: `pp2` for TP1/PP2 or `tp2` for TP2/PP1.
- `expert_parallel`: false for PP2 and required/true for TP2 on this checkpoint.
- `mtp_tokens`: `0`, `1`, or `2`.
- `mtp_compile_fix`: whether the local vLLM image contains `@support_torch_compile` on `HYV3MTP`.
- `gpu_memory_utilization`: exact runtime fraction; accepted baseline cells use
  0.956, MTP2 uses the guarded 0.958 speculative profile, and the preserved
  tuning pass uses 0.92.
- `flashinfer_autotune`: false for PP2 because distributed autotuning deadlocks
  across unequal pipeline stages. FlashInfer kernels themselves remain enabled.
- `concurrency`: offered client concurrency.
- `aggregate_output_tokens_per_second`: total completed output tokens divided by benchmark wall time.
- `effective_concurrency`, `max_running_requests`, `capacity_limited`: retain the benchmark's resolved scheduling fields; never infer residency from offered concurrency.
- `prompt_tokens`, `max_tokens`, `temperature`: workload identity. The fixed
  matrix is 8,192 / 1,024 / 0.
- `measurement_seconds`: measured wall time for the duration-mode cell.
- `server_spec_draft_tokens`, `server_spec_accepted_tokens`, and
  `speculative_acceptance_rate`: server-side MTP counters; blank for MTP0.
- `headline_eligible` and `headline_exclusion_reason`: explicit publication
  policy. TP2/MTP1 C128 and TP2/MTP2 C32-C128 are valid measurements retained
  for diagnosis but excluded from performance headlines.
- `result_file`: path to the compact, checksummed `llm-inference-bench` evidence.

Validate a duration-mode row only when `num_errors` is zero and every completed
request has the exact requested token counts. Requests still in flight at the
30-second cutoff are expected; preserve both submitted and completed counts.
The server evidence must identify the expected model revision, topology, cache
dtype, and runtime image.

The preserved `hy3-fp8-pp2-mtp0-092-provisional` run ends at C64 and has no
quality artifact. It remains provisional even though its cells are internally
valid. Each supported canonical directory contains C1-C128. PP2/MTP1 and
PP2/MTP2 are unsupported by the pinned draft model, not pending data.

## `prefill.csv`

One row is one standalone cold-prefill sample. Record 8K, 64K, and 128K
separately. `actual_prompt_tokens` retains the two template tokens added by the
API. `prompt_tokens_per_second` is client-observed prompt tokens divided by
TTFT; retain `ttft_seconds` as the primary latency measurement. Do not copy
decode throughput into this file.

## Raw artifacts

Store accepted raw files beneath the following untracked/generated layout while measuring, then add only the compact artifacts selected for publication:

```text
results/
  hy3-fp8-pp2-mtp0-092-provisional/
  hy3-fp8-pp2-mtp0/
  hy3-fp8-pp2-mtp1/
  hy3-fp8-pp2-mtp2/
  hy3-fp8-tp2-mtp0/
  hy3-fp8-tp2-mtp1/
  hy3-fp8-tp2-mtp2/
  hy3-fp8-tp2-mtp1-c128-confirm60-util0958/
```

Each canonical configuration directory contains the benchmark JSON,
natural-quality JSON, and a `runtime/` directory captured before the server is
stopped. The one preserved tuning directory instead contains `runtime-092/`;
the extractor recognizes that exception only under its exact provisional name.

Do not add model weights, caches, container layers, Python bytecode, multi-gigabyte logs, or incomplete temporary downloads to this repository.

Natural output JSON must validate against [`quality-audit.schema.json`](quality-audit.schema.json). Automatic repetition statistics supplement, but do not replace, manual coherence review.

## `quality-summary.csv`

The extractor groups natural outputs by topology, MTP depth, and reasoning effort. `flagged_outputs` applies the public audit thresholds; it is not a semantic score. `manual_review_status` and notes come from [`manual-quality-review.json`](manual-quality-review.json), which must be edited only after reading every preserved output for that configuration.

## `stability-confirmation.csv`

This contains the separately launched 60-second TP2/MTP1 C128 confirmation at
utilization 0.958. It retains client/server throughput, full-residency and error
counters, speculative acceptance, and the fixed KV guard. It confirms the
canonical C128 cliff; it does not replace the 30-second canonical row.

## `kv-capacity.csv`

One row is one accepted, provisional, confirmation, or rejected server profile.
`observed_global_kv_tokens` is parsed from the runtime log,
`required_global_kv_tokens` is the predeclared 1,179,904-token guard, and
`margin_tokens` is observed minus required. The file retains log-backed
capacity misses, including a rejected profile that did not meet the guard.

## Compact evidence and runtime metadata

`recipes/extract-results.py` writes accepted benchmark/quality evidence under `evidence/` and filtered container/server metadata under `runtime/`. The evidence retains complete natural response text and benchmark cells while omitting bulky event streams, request samples, and full server logs. Each compact file records the SHA-256 of its source artifact so it can be traced back to the original result directory.

## Failed-attempt evidence

[`failure-evidence.json`](failure-evidence.json) records hashes and required log
markers for the PP2 distributed-autotune deadlock and TP2-without-EP OOM. These
failures are operational results, not throughput rows. The extractor regenerates
the manifest when the preserved raw failure directories are available.

The manifest also includes a guarded 0.956 PP2 attempt that reached health but
did not meet the recorded KV-capacity requirement. Its compact runtime record
is [`runtime/pp2-mtp0-attempt3-kv602176.json`](runtime/pp2-mtp0-attempt3-kv602176.json)
and is backed by retained server logs and the KV-capacity record.

An ownerless-residual-HBM observation is retained only as a generic operational
lesson because its raw terminal capture was not preserved. It is not included
in this checksummed evidence manifest. Use
[`../recipes/preflight-idle-hbm.sh`](../recipes/preflight-idle-hbm.sh) before
each memory-tight distributed launch.

The same manifest proves the MTP/PP incompatibility and the clean-idle MTP1 and
MTP2 capacity rejections. Failed attempts never emit throughput or prefill rows.
