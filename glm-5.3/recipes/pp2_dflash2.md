# vLLM PP2 + DFlash2 recipe

This is the reproducible recipe behind the PP2 K4/K5/K7 proposal sweep. It
also records the backend and scheduler experiments that are intentionally kept
out of the headline page.

## Winning uses

| Objective | Configuration | Result |
| --- | --- | ---: |
| Interactive code | PP2 K4 | 154.9 output tok/s at C1 |
| Interactive prose | PP2 K4 | 106.3 output tok/s at C1 |
| C2 code | PP2 K5 | 270.6 aggregate tok/s |
| C4 code | PP2 K5 | 409.9 aggregate tok/s |
| C8 code | PP2 K7 | 553.8 aggregate tok/s |
| C16 code | PP2 K7 batch | 742.0 aggregate tok/s |
| C32 code | PP2 K7 batch | 1,065.0 aggregate tok/s |
| C64 code | PP2 K7 batch | 1,093.8 aggregate tok/s |

K4 is the measured interactive winner. K7 is the measured batch winner. K5 is
a narrow middle point rather than a universal winner.

All winning cells, acceptance rates, model-load sizes, HBM headroom, and KV
envelopes on this page were measured with the historical target below.

## Measured target and runtime

| Item | Pin |
| --- | --- |
| Target | `incoai/GLM-5.3-NVFP4@54e52520606f96b3d9fc84088ad22882a61648ac` |
| Draft | `incoai/GLM-5.3-DFlash2@425aa615ce320caac34400208b30808c8f14f76c` |
| vLLM base source | `c01b50e390e6d3d0019aa53f41ff1198c8105e5a` |
| Image | `sha256:343c67ff9c28dca5ae22385804d6276d6c4e45aecdf17f581d583e87000ed482` |
| PP DFlash overlay commit | `895c5d5c531f13351284133846e2b5c643d744f0` |
| Overlay tree | `cd08d251f9fe82ff72ce1df0e21bfe529e8edbf2` |
| Overlay manifest SHA-256 | `5921028ebd11487c511625a0eb7623e66a42c05b72206741ba5bd2fc7a7f8d2a` |
| FlashInfer | 0.6.17 |
| Benchmark | `llm-inference-bench` 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |

## Current target compatibility

The launch skeleton now pins
[`local-inference-lab/GLM-5.3-NVFP4@cca10d1`](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4/tree/cca10d1586255195d3279785fc85577bfc1e9227).
The exact static audit is in [`current-target.json`](current-target.json). Its
non-quantization model geometry, tokenizer/chat assets, and indexed tensor-name
set match the measured target. The draft still targets the same 78-layer GLM
geometry and is statically compatible.

The pinned vLLM ModelOpt loader already supports the replacement's flat
quantization config and wildcard exclusions, and the PP2+DFlash2 overlay is
architecture-based rather than repository-name-based. No overlay, parser,
quantization flag, or draft-config change is warranted. The quantization
producer version, exclusion representation, calibrated weights, and auxiliary
scale assets differ from the measured target. The replacement also omits
the previous checkpoint's KV-cache scheme metadata, so the explicit
`--kv-cache-dtype fp8` is required. This is not a runtime or performance
validation: rerun output correctness, load/HBM checks, draft acceptance, and
the K4/K5/K7 sweep before promoting a profile.

The overlay is opt-in through `VLLM_PP_DFLASH_DECODE_PARTITIONS=2`. Its default
value is one, which preserves upstream scheduling. With two partitions, active
decode requests are divided into two PP cohorts and alternated so stage 0 can
work on one cohort while stage 1 works on the other. The patch also separates
newly admitted work from resumed preempted work; this fixes the draft-token ID
handoff that previously serialized or blocked C2+ PP decoding.

## Serving profile

| Setting | Value |
| --- | --- |
| Topology | 2 hosts, TP1 / PP2 / EP1, one GB300 per host |
| Layer split | 42 / 36 |
| Target sparse attention | `FLASHINFER_MLA_SPARSE` |
| Target MoE | `FLASHINFER_TRTLLM` |
| Draft attention | `FLASH_ATTN` dispatching FlashAttention 4 |
| Target / draft KV | FP8 / BF16 |
| Hosted context | 500,000 tokens |
| Maximum sequences | 16 interactive / 64 batch |
| Scheduler batch-token budget | 8,192; not a request-context limit |
| HBM fraction | 0.93 interactive / 0.95 batch |
| KV block size | 64 |
| Prefix cache | Disabled |
| Async scheduling | Disabled |
| CUDA graph sizes | Through 80 interactive / through 128 batch |
| FlashInfer autotune | Enabled; private cache per runtime configuration |

`FLASHINFER_TRTLLM` is FlashInfer's TensorRT-LLM-derived NVFP4 MoE kernel
selected by **vLLM**. These runs did not use the standalone TensorRT-LLM
serving engine.

| Profile | Stage 0 / stage 1 model load | Stage 0 / stage 1 KV headroom | Coordinated KV tokens |
| --- | ---: | ---: | ---: |
| K4 | 217.11 / 205.30 GiB | 11.61 / 22.98 GiB | 480,832 |
| K5 | 217.11 / 205.30 GiB | 11.56 / 22.91 GiB | 478,656 |
| K7 interactive | 217.24 / 205.42 GiB | 11.40 / 22.77 GiB | 472,320 |
| K7 C16/C32/C64 | 217.11 / 205.30 GiB | 16.51 / 27.85 GiB | 640,896 |

## Launch skeleton

Apply the complete, source-only
[`vLLM overlay`](vllm-overlay/) to the pinned base commit. Its patch includes
all 21 source files that were bind-mounted into the measured containers, and
its `SHA256SUMS` is byte-identical to the deployed overlay manifest named by
[`pp2-dflash2-provenance.json`](../data/pp2-dflash2-provenance.json). Start the
worker rank first, and substitute only the node rank, addresses, CDI GPU, local
offline model paths, and proposal count.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1'
export NCCL_IB_MERGE_NICS=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DMABUF_ENABLE=1
export NCCL_SOCKET_IFNAME=enP1p3s0f0np0
export GLOO_SOCKET_IFNAME=enP1p3s0f0np0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=8192
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_PP_DFLASH_DECODE_PARTITIONS=2
export VLLM_PP_DFLASH_TRACE_STEPS=0
export VLLM_PP_LAYER_PARTITION=42,36
export TARGET_MODEL_ID='local-inference-lab/GLM-5.3-NVFP4'
export TARGET_REVISION='cca10d1586255195d3279785fc85577bfc1e9227'

vllm serve /model \
  --revision "$TARGET_REVISION" \
  --served-model-name "$TARGET_MODEL_ID" \
  --trust-remote-code \
  --model-impl vllm \
  --dtype bfloat16 \
  --quantization modelopt_fp4 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --kv-cache-dtype fp8 \
  --max-model-len 500000 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.93 \
  --no-enable-prefix-caching \
  --moe-backend flashinfer_trtllm \
  --enable-flashinfer-autotune \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 80 \
  --block-size 64 \
  --no-async-scheduling \
  --per-request-spec-decode-metrics summary \
  --speculative-config \
    '{"method":"dflash","model":"/draft","revision":"425aa615ce320caac34400208b30808c8f14f76c","num_speculative_tokens":K,"draft_tensor_parallel_size":1,"kv_cache_dtype":"bfloat16","attention_backend":"FLASH_ATTN"}' \
  --host "$SERVE_HOST" \
  --port 30000
```

The real launches bind-mounted the verified 21-file overlay into the pinned
image. The retained `runtime/node{0,1}/launch-command.json` files are the
authoritative full Docker commands, including all mounted source files and CDI
device IDs.
The C16/C32/C64 profile changes the skeleton to `--max-num-seqs 64`,
`--gpu-memory-utilization 0.95`, K7, and CUDA graph sizes
`1 2 4 8 16 32 64 128`.

The checkpoint declares a native 1,048,576-token context. A guarded PP2 +
DFlash2 launch at that value needed 25.31 GiB of KV memory on the limiting
rank, where 22.82 GiB was available, and reported an estimated 945,408-token
ceiling. The hosted recipe therefore advertises 500,000 tokens. The published
throughput runs deliberately used exact 8,192-token input and 1,024-token
output; those workload lengths are not a production client cap.
`--max-num-batched-tokens 8192` controls scheduler work per iteration and does
not reduce the advertised request context. Verify the deployed KV envelope
before relying on one request occupying the entire window.

`VLLM_PP_DFLASH_TRACE_STEPS=0` keeps the overlay's optional tensor diagnostic
disabled. Nonzero values introduce device-to-host copies and verbose logging;
they are for correctness debugging only and were not used for any published
throughput result.

## OpenCode

Use [`opencode-1m.jsonc`](opencode-1m.jsonc) as the custom-provider overlay.
It points at the same OpenAI-compatible endpoint without setting a client-side
context or output limit; the server-side recipe advertises 500,000 tokens. Do
not copy a short benchmark acquisition envelope into the OpenCode model
configuration.

## Proposal sweep details

| K | C | Aggregate tok/s | Median TTFT | Median ITL | Accepted proposals / verify | Effective / max active | Queue fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 code | 154.918 | 497.7 ms | 5.931 ms | 2.427 | 0.9 / 1 | 0.0% |
| 4 | 1 prose | 106.259 | 498.9 ms | 8.705 ms | 1.261 | 0.9 / 1 | 0.0% |
| 4 | 2 | 270.566 | 519.3 ms | 6.697 ms | 2.326 | 1.9 / 2 | 0.0% |
| 4 | 4 | 399.929 | 518.6 ms | 8.973 ms | 2.322 | 3.7 / 4 | 0.0% |
| 4 | 8 | 542.526 | 519.5 ms | 13.411 ms | 2.260 | 7.6 / 8 | 2.7% |
| 4 | 16 screen | 725.588 | 978.8 ms | 19.461 ms | 2.274 | 14.4 / 16 | 8.9% |
| 5 | 1 code | 152.581 | 502.5 ms | 6.068 ms | 2.573 | 0.9 / 1 | 0.0% |
| 5 | 1 prose | 104.080 | 502.7 ms | 9.212 ms | 1.329 | 0.9 / 1 | 0.0% |
| 5 | 2 | 270.605 | 524.3 ms | 6.580 ms | 2.507 | 1.9 / 2 | 0.0% |
| 5 | 4 | 409.908 | 524.5 ms | 9.032 ms | 2.607 | 3.8 / 4 | 0.0% |
| 5 | 8 | 538.949 | 524.5 ms | 13.605 ms | 2.532 | 7.6 / 8 | 3.9% |
| 5 | 16 | 723.614 | 574.1 ms | 20.416 ms | 2.571 | 15.0 / 16 | 3.6% |
| 7 | 1 code | 146.042 | 499.9 ms | 6.040 ms | 2.746 | 0.9 / 1 | 0.0% |
| 7 | 1 prose | 92.703 | 498.9 ms | 10.195 ms | 1.249 | 0.9 / 1 | 0.0% |
| 7 | 2 | 267.044 | 526.3 ms | 6.744 ms | 2.802 | 1.9 / 2 | 0.0% |
| 7 | 4 | 405.715 | 525.6 ms | 9.342 ms | 2.962 | 3.9 / 4 | 0.0% |
| 7 | 8 | 553.836 | 527.5 ms | 13.470 ms | 2.894 | 7.6 / 8 | 2.7% |
| 7 | 16 | 741.994 | 568.6 ms | 19.803 ms | 2.876 | 15.1 / 16 | 3.6% |
| 7 | 32 | 1,064.979 | 676.6 ms | 27.664 ms | 2.873 | 29.9 / 32 | 9.2% |
| 7 | 64 | 1,093.793 | 903.5 ms | 54.904 ms | 2.858 | 60.2 / 64 | 11.1% |

Accepted proposals excludes the guaranteed target/anchor token. K4 C16 used
32 measured requests (`2×C`) under successive halving; every other row used
`5×C`. All rows completed the exact request count with zero request errors.
The batch-profile validator sets `capacity_limited=true` when it observes
scheduler queueing. C16, C32, and C64 all have `underfilled=false`, reached the
full offered maximum active count, and completed 80/80, 160/160, and 320/320
requests respectively. The prior 0.93-memory K7 envelope reached 726.849 tok/s
at C16; it remains in the matched patch A/B below rather than the headline.

| Batch cell | Result JSON SHA-256 | Validation JSON SHA-256 |
| ---: | --- | --- |
| C16 | `fce317d9bf20f848ca25da19f3579c02e213a3c8a5e42230130ced0ea8245cb6` | `e56712376c4e42b682cf1b350029a90cabeab155c42f7f0ac47a72935b36063f` |
| C32 | `2fd9d1947d277acf94b3f325ede2e8ac2706bf92234c9aaff3e52ee42d4b5e0e` | `5e9c9dc9acf57a7dd5eb4677268d644875015b77c2789dfda2a2efc366960beb` |
| C64 | `a9a1613c9ff0377745153821d6a2aedc7da9eb4ce845922eca3967251c8855d9` | `2813a15544b40b52b919345f27e89bef75ade23d99e8e2e034983c257336fb0a` |

## Per-user headline curve

The per-user chart selects the highest validated headline aggregate rate at
each offered concurrency and computes:

```text
output tok/s/user = aggregate output tok/s / offered concurrency
```

This intentionally uses offered concurrency, not average scheduler residency.
The plotted 60 tok/s reference crosses the straight line segment on the
log2-concurrency axis between measured C8 and C16 at approximately C10.6. That
is an interpolated visual crossover, not a measured C10 or C11 benchmark.

## Patch A/B

The matched K7 comparison below uses the same `glmfullbench` code prompt and
request counts. The patched lane changed the PP cohort scheduler and created a
new FlashInfer autotune fingerprint.

| C | Before patch | Cohort patch | Change |
| ---: | ---: | ---: | ---: |
| 1 | 147.095 | 146.042 | -0.72% |
| 2 | 256.900 | 267.044 | +3.95% |
| 4 | 357.527 | 405.715 | +13.48% |
| 8 | 508.541 | 553.836 | +8.91% |
| 16 | 697.190 | 726.849 | +4.25% |

## Backend screens

| Screen | C1 code | C1 prose | C16 code | Disposition |
| --- | ---: | ---: | ---: | --- |
| FlashInfer sparse target + FlashInfer TRT-LLM MoE | 147.095 | 97.585 | 697.190 | K7 batch control |
| FlashMLA target + FlashInfer TRT-LLM MoE | 154.875 | 99.936 | 624.315 | Better C1, worse batch |
| FlashInfer sparse target + FlashInfer CUTLASS MoE | 135.346 | 85.630 | 584.817 | Slower across the screen |

FlashInfer autotuning materially improved CUTLASS engine-step rate in its
screen, but did not close the gap to the TRT-LLM-derived MoE kernel. For the
TRT-LLM-derived kernel, C1 engine-step rate was effectively neutral; the final
profiles retain autotuning because it helped the batch-oriented configurations
and the cache is reproducibly keyed per runtime configuration.

## Benchmark contract

- Exact 8,192-token prompt and 1,024-token output per request.
- Temperature 0, top-p 1, EOS ignored, `enable_thinking=true`, and
  `reasoning_effort=low` in the benchmark request.
- C warmup requests and `5×C` measured requests for full cells.
- Unique prefix namespaces; prefix caching disabled at the server.
- Aggregate throughput is completed OpenAI-usage output tokens divided by the
  client monotonic measurement interval.
- Fixed arithmetic, JSON, code, tool-call, and prose battery passed before the
  sweep.
- Backend logs proved FlashInfer sparse target attention, FlashInfer TRT-LLM
  NVFP4 MoE, and FlashAttention 4 draft attention.
- Both dual-rail RoCE HCAs were pinned for NCCL; teardown verified clean idle
  HBM on both hosts.

## Non-headline paths

- The original vLLM TP2+EP2 DFlash2 path reached only 43.8 tok/s at C1 because
  it accepted 0.248 of seven proposals per verification cycle. It is retained
  in `throughput.csv` as diagnostic evidence, not a recipe.
- SGLang with FA4 draft attention reached 124.4 tok/s at C1 and 396.5 at C16,
  behind its TRTLLM-MHA draft path.
- The FlashInfer 0.6.18rc10 SGLang screen reached 155.6 tok/s at C1 but did not
  clear the complete invocation gate; it was not expanded.
- PP2 K4 C16 is a validated screen, not a five-wave finalist row.
- The older SGLang TP2+EP2 C32 value of 570.0 tok/s averaged 26.1 active and
  reached only 29 maximum active requests. Its TP2+EP1 C64 value of 566.9
  tok/s averaged 25.3 active and also peaked at 29. Both are retained as
  capacity-limited detail rows, not headline winners.
- Use the separate SGLang PP2/AR 40/38 profile for long prefill.

Machine-readable compact rows are in
[`throughput.csv`](../data/throughput.csv). Exact runtime and evidence hashes
are in
[`pp2-dflash2-provenance.json`](../data/pp2-dflash2-provenance.json).
