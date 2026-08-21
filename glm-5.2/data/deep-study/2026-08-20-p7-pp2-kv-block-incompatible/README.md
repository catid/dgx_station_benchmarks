# P7: balanced vLLM PP2 loads, then fails on heterogeneous KV block sizes

P7 is pipeline-partition and compatibility evidence, not a performance result.
The audited 40/38 layer split loaded successfully with a small stage-memory
spread, no expert parallelism, no speculation, and FlashInfer distributed
autotuning disabled. Pinned vLLM 0.27.1 then failed before API readiness while
constructing a common KV-cache layout across the two different pipeline stages.

| Pin or setting | Value |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Topology | TP1/PP2, EP1; explicit 40/38 pipeline partition; no CPU offload |
| Backend | FlashInfer CuTeDSL NVFP4 MoE; FP8 E4M3 KV |
| Bootstrap envelope | 8,192 max length; 16 sequences; 4,096 batched tokens; CUDA graphs through 16 |
| Speculation / autotune | Off / off |

| Stage | Assigned layers | Runtime rank | Model memory | Weight load | Compile | Warmup |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| PP0 | 40 | TP0 / EP0 | 206.32 GiB | 154.18 s | 43.00 s | 18.60 s |
| PP1 | 38 | TP0 / EP0 | 209.52 GiB | 163.72 s | 34.81 s | 18.38 s |

The measured model-memory spread was 3.20 GiB. This validates the checkpoint
payload audit that selected 40/38 instead of a nominal 39/39 layer count; it
does not validate request serving.

During CUDA-graph memory profiling, PP0 selected a 64-token block for
`DEEPSEEK_V32_INDEXER`, while PP1 selected a 32-token block for
`FLASHINFER_MLA_SPARSE`. PP1 failed in `select_common_block_size` with:

```text
ValueError: No common block size for 32.
```

PP0 had estimated 0.46 GiB of CUDA-graph memory and 12.53 GiB of locally
available KV memory, but PP1 failed before it reported local KV memory and
before vLLM emitted a coordinated token capacity. The API never became healthy.
No bootstrap request, full P0-envelope run, throughput measurement, quality
output, or request-bearing network measurement was attempted.

Both ranks completed distributed initialization, stage-specific load, compile,
and warmup before the incompatibility surfaced. That proves the two ranks could
coordinate startup; it is not a fabric-performance result. Graceful teardown,
the current-boot kernel scan, and the idle-HBM gate passed, returning to 2 / 7
MiB. No retry was made.

[`failure-summary.json`](failure-summary.json) retains the exact stage,
startup, failure, and raw-source hashes. The exact resolved profile is in
[`resolved-plan.json`](resolved-plan.json); `SHA256SUMS` covers every published
file in this increment.
