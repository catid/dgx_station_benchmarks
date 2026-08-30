# Reproducing the full GLM-5.3 benchmark

This page contains the exact model, runtime, launch, workload, and capacity
details kept out of the headline README and graph labels. The patched vLLM
pipeline-parallel series has its own
[PP2 + DFlash2 recipe](pp2_dflash2.md), including the complete
[21-file vLLM source overlay](vllm-overlay/).

## Measured target and runtime

Every benchmark result and resource figure in this section is attributed to
the measured target below.

| Item | Pin |
| --- | --- |
| Target | `incoai/GLM-5.3-NVFP4@54e52520606f96b3d9fc84088ad22882a61648ac` |
| Target architecture | `GlmMoeDsaForCausalLM` (`glm_moe_dsa`) |
| Target payload | 87 safetensors, 464,822,872,912 bytes |
| Target quantization | ModelOpt NVFP4; BF16 compute |
| Target KV cache | FP8 E4M3 |
| Draft | `incoai/GLM-5.3-DFlash2@425aa615ce320caac34400208b30808c8f14f76c` |
| Draft payload | 1 safetensor, 4,918,859,112 bytes |
| Draft precision / KV | BF16 / BF16 |
| DFlash2 | Block size 8; seven proposals per verification cycle |
| SGLang source | `4d78d59e516f96b1a86e1dd1a458fda2664427d6` |
| Image | `sha256:e73ae9252ba7cd877b8ff98cddba11e65dcd6b8ff6817c7b680622cca7fa64b2` |
| FlashInfer | 0.6.17 |
| vLLM challenger source | `c01b50e390e6d3d0019aa53f41ff1198c8105e5a` |
| vLLM challenger image | `sha256:343c67ff9c28dca5ae22385804d6276d6c4e45aecdf17f581d583e87000ed482` |
| Benchmark client | `llm-inference-bench` 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |

## Current serving target — unmeasured

The launchers and OpenCode overlay now select
[`local-inference-lab/GLM-5.3-NVFP4@cca10d1`](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4/tree/cca10d1586255195d3279785fc85577bfc1e9227).
Its exact metadata audit is recorded in
[`current-target.json`](current-target.json). The repository has no model card
or published evaluation at that revision, so this publication makes no quality
or speed claim for the replacement.

The architecture, all non-quantization config fields, tokenizer/chat assets,
and 232,385 indexed tensor names match the measured target. The new flat
ModelOpt configuration uses wildcard exclusions supported by both pinned
runtimes, but its producer version, exclusion representation, calibrated
weights, and auxiliary scale assets differ. It does not declare a KV-cache
scheme, so the explicit
`--kv-cache-dtype fp8_e4m3` SGLang setting and `--kv-cache-dtype fp8` vLLM
setting are required and must not be removed. No parser, draft, or source
overlay change is needed from this static audit. Runtime correctness, HBM
headroom, draft acceptance, proposal count, and throughput must be requalified.

The server is one distributed endpoint across two DGX Stations, one GB300 per
host. Both 400 GbE ConnectX-8 RoCE rails carried NCCL traffic through
GPUDirect RDMA.

## Working paths

| Run | Serving engine | Topology | Draft attention | Target MoE | FlashInfer | Measured cells |
| --- | --- | --- | --- | --- | --- | --- |
| `g53-b2-tep2-df2-low-bs32-r1` | SGLang | TP2 / EP2 | TRTLLM-MHA | FlashInfer TRTLLM | 0.6.17 | C1 code/prose headline; C16 detail |
| `g53-b4-tep2-df2-low-bs32-schema5-r1` | SGLang | TP2 / EP2 | TRTLLM-MHA | FlashInfer TRTLLM | 0.6.17 | Capacity-limited C32/C64 and 64K prefill detail |
| `g53-b1-tp2-df2-low-bs32-r1` | SGLang | TP2 / EP1 | TRTLLM-MHA | FlashInfer TRTLLM | 0.6.17 | C1 code/prose, C16, C32, offered C64, 64K prefill diagnostic |
| `g53-s1-tep2-df2-fa4-fi0617-r1` | SGLang | TP2 / EP2 | FA4 | FlashInfer TRTLLM | 0.6.17 | C1 code/prose and C16 diagnostic |
| `g53-b3-tep2-df2-fi0618rc10-r1` | SGLang | TP2 / EP2 | TRTLLM-MHA | FlashInfer TRTLLM | 0.6.18rc10 | C1 code diagnostic |
| `g53-v2-tep2-df2-cutlass-r1` | vLLM | TP2 / EP2 | FA4 | FlashInfer CUTLASS | 0.6.17 | Valid C1 code challenger |
| `g53-p1-pp2-ar-prefill-sweep-r1` | SGLang | PP2 40/38 | N/A (AR) | FlashInfer TRTLLM | 0.6.17 | Five exact cold-prefill samples each at 8K/64K/128K |
| `vpp2df-k4-fi-trt-p2-r31` | vLLM + PP cohort patch | PP2 42/36 | FA4, K4 | FlashInfer TRTLLM | 0.6.17 | C1–C8 full cells and C16 screen |
| `vpp2df-k5-fi-trt-p2-r30` | vLLM + PP cohort patch | PP2 42/36 | FA4, K5 | FlashInfer TRTLLM | 0.6.17 | C1–C16 full cells |
| `vpp2df-k7-fi-trt-ef25-p2-r29b` | vLLM + PP cohort patch | PP2 42/36 | FA4, K7 | FlashInfer TRTLLM | 0.6.17 | C1–C8 full cells and prior C16 envelope |
| `vpp2df-k7-c64-p2-r32` | vLLM + PP cohort patch | PP2 42/36 | FA4, K7 | FlashInfer TRTLLM | 0.6.17 | C16/C32/C64 full cells; batch leader |

The headline charts select the competitive cells and never fill in missing
concurrencies.
The TP2+EP1 run used the earlier schema-v3 framing check, which did not reject
raw reasoning tags in API content. The FA4 screen preceded the low-reasoning
launch default and used the template's implicit maximum reasoning setting, so
neither is a matched operational headline. For FlashInfer 0.6.18rc10, both
measured 1,024-token responses were tag-clean and complete, but the unmeasured
64-token compile warmup leaked a raw reasoning tag, causing the invocation to
be rejected. Acceptance counters were not normalized for that row. The vLLM
challenger passed its schema-v5 C1 cell, but accepted only 0.248 of seven draft
proposals per verification cycle (3.54%); its 43.8 tok/s is a valid measurement
of that acceptance collapse, not a competitive recipe.

## 500K hosted context

The serving launchers advertise a 500,000-token request context. Their 65,536
maximum-prefill and 8,192 chunk/batch settings are scheduler work budgets, not
client context limits. The checkpoint's native context is 1,048,576 tokens,
but a guarded PP2+DFlash2 1M launch required 25.31 GiB of KV memory on the
limiting rank with only 22.82 GiB available. The resulting estimated ceiling
was 945,408 tokens, so the production recipe uses a conservative 500,000.

Run `serve.sh` on the worker first (`NODE_RANK=1`), then on the controller
(`NODE_RANK=0`). Set the node-specific GPU CDI device and the exact local model
paths. The script runs in Hugging Face offline mode and uses the immutable image
digest.

For OpenCode, copy [`opencode-1m.jsonc`](opencode-1m.jsonc), replace the
controller address, and select the listed model. The overlay sets no client-side
context or output limit; the server advertises 500,000 tokens.

The published TP2+EP2 decode measurements used a smaller acquisition envelope:
`mem-fraction-static=0.93`, 32 maximum running requests, 65,536 maximum prefill
tokens, 8,192-token prefill chunks, and CUDA graph capture through batch 32.
Those retained settings describe the benchmark, not the production client
limit.

## Long-prefill profile

Run `serve-prefill-pp2.sh` on the worker first and then the controller for the
long-prefill leader. It uses TP1 / PP2 / EP1, a 40/38 transformer-layer split,
AR, one maximum running request, a 500,000-token advertised context, 65,536
maximum prefill tokens, 8,192-token chunks, and CUDA graphs disabled. The
published run used the measured target above; the current launcher replaces
only that target. It otherwise uses the same SGLang commit, image, FP8 KV
cache, FlashInfer TRTLLM MoE, target DSA backends, and dual-rail GPUDirect RDMA
transport as the decode profile.

Prefill itself does not use speculative decoding. DFlash2 is unavailable with
PP2 in this pinned SGLang runtime, so this profile has no draft model or
draft-attention backend. The separately documented vLLM source overlay enables
PP2 + DFlash2 for decode. Switching between the SGLang prefill and vLLM decode
profiles requires a restart; the publication treats them as separate product
profiles.

## Workload

- Exact 8,192-token input and 1,024-token output per decode request.
- Temperature 0, top-p 1, EOS ignored, `enable_thinking=true`, and low reasoning
  effort. Under this pinned template, that setting preserved the same rendered
  model-input bytes while letting the GLM parser separate reasoning from final
  content.
- Unique cache namespaces prevent cross-request prefix-cache reuse.
- C1 used one warmup plus five measured requests.
- C16 used 16 warmups plus 32 measured requests.
- C32 used 32 warmups plus 160 measured requests.
- Offered C64 used 64 warmups plus one 64-request measured wave.
- Aggregate output rate is completed OpenAI-usage output tokens divided by the
  client monotonic measurement interval.
- The DFlash2 cold-prefill rows used one exact 65,536-token unique prompt and
  one output token.
- The PP2/AR headline is the median of five independently cache-isolated exact
  65,536-token prompts with one output token.

## SGLang observed concurrency detail

| Offered C | Average / maximum active | Average / maximum queued | Queue fraction |
| ---: | ---: | ---: | ---: |
| 1 code | 0.8 / 1 | 0.0 / 0 | 0.0% |
| 1 prose | 0.9 / 1 | 0.0 / 0 | 0.0% |
| 16 code | 13.0 / 16 | 1.6 / 14 | 21.9% |
| 32 code | 26.1 / 29 | 4.2 / 30 | 93.3% |
| 64 code | 22.6 / 29 | 23.6 / 62 | 86.4% |

C16, C32, and C64 are completed offered-load measurements; they do not mean
every offered request remained resident for the full interval. The C64 client
used a 589,824-token client-side capacity override so it would offer the cell;
the server's actual 311,680-token KV pool did not change. Every measured request
completed with zero errors. The public graph uses offered concurrency on the
x-axis. C32 and C64 are capacity-limited detail rows and are not headline
winners.

## Prefill note

The matched PP2/AR 40/38 sweep used five distinct cache namespaces per exact
prompt size:

| Prompt | Median prompt tok/s | Median client TTFT |
| ---: | ---: | ---: |
| 8K | 16,425 | 0.499 s |
| 64K | 25,854 | 2.535 s |
| 128K | 25,249 | 5.191 s |

The TP2+EP2 DFlash2 comparison row is the isolated rerun: 65,536 prompt tokens,
8.174-second client TTFT, and 8,018 prompt tok/s. An earlier 8,036 tok/s sample
overlapped the C64 client and is excluded.
