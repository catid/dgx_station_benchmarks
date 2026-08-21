# P8: vLLM PP2 serves correctly with block64 and eager execution

P8 is a correctness and compatibility milestone, not a performance result. It
supersedes P7's heterogeneous 64/32-token KV-layout failure by explicitly
setting a common 64-token block size. Both pipeline stages reached the API,
initialized a coordinated 402,688-token KV cache, and completed four retained
long-context generations without visible corruption or degeneration.

| Pin or setting | Value |
| --- | --- |
| Checkpoint | [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Topology | TP1/PP2, EP1; explicit 40/38 layer partition; no CPU offload |
| Backend | FlashInfer CuTeDSL NVFP4 MoE; autotuning and speculation off |
| KV | FP8 E4M3; `--block-size 64`; 402,688 coordinated tokens |
| Correctness envelope | 16,384 max length; 4 sequences; 16,384 batched tokens; 90% memory utilization |
| Execution | `--enforce-eager`; compilation and CUDA graphs disabled |

| Stage | Assigned layers | Model memory | Weight load | Total model load | Local KV memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| PP0 | 40 | 206.41 GiB | 56.99 s | 70.80 s | 12.81 GiB |
| PP1 | 38 | 209.61 GiB | 57.71 s | 71.49 s | 8.65 GiB |

The harness built a benchmark-style prompt at exactly 8,192 tokens through
`/tokenize`. It then issued two forced 1,024-token and two forced 4,608-token
greedy generations sequentially. Direct API usage reported exactly 8,192
prompt tokens on all four requests.

| Output | Characters | Words | Repeated 8-grams | Max character run | Max word run | Text flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 tokens, repeat 1 | 4,398 | 558 | 2.359% | 3 | 1 | 0 |
| 1,024 tokens, repeat 2 | 4,443 | 528 | 0.960% | 3 | 2 | 0 |
| 4,608 tokens, repeat 1 | 18,953 | 2,667 | 0.338% | 4 | 2 | 0 |
| 4,608 tokens, repeat 2 | 19,169 | 2,738 | 0.659% | 3 | 2 | 0 |

All four outputs passed every retained check for empty text, insufficient text,
printability, ASCII/alphanumeric ratios, replacement or control characters,
identical-character and identical-word runs, and repeated 8-grams. Full text is
preserved in [`raw-harness.json`](raw-harness.json) for direct inspection.

The two greedy pairs were coherent but not byte-identical: the 1,024-token pair
shared a 326-character prefix and the 4,608-token pair shared a 636-character
prefix before their wording diverged. Both hash mismatches are retained as
nondeterminism; neither pair exhibited the garbage or DSA-corruption signature
this smoke test was designed to detect.

The immutable raw harness says `failed`. It had borrowed the 8,194-token count
seen in a different standalone-prefill path and also treated byte-identical
greedy repeats as mandatory. The measured direct chat path and `/tokenize`
agree at exactly 8,192, and byte nondeterminism is not by itself text
corruption. [`corrected-result.json`](corrected-result.json) revalidates the
unchanged raw output and records these two distinctions explicitly;
[`structural-validation.json`](structural-validation.json) independently
recomputes and accepts that correction.

Quiet before/after fabric telemetry recorded no health-counter deltas. This was
a correctness window, so [`network-delta.json`](network-delta.json) is evidence,
not a bandwidth or throughput headline. Graceful teardown completed, the
current-boot kernel scan was clean, and idle HBM returned to 2 / 7 MiB. No
retry was made.

[`runtime-summary.json`](runtime-summary.json) contains stage, capacity,
configuration, teardown, and private-raw-source hashes. The exact launch profile
is in [`resolved-plan.json`](resolved-plan.json); `SHA256SUMS` covers every
published file in this increment.
