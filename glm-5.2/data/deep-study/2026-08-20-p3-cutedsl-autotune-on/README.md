# P3: accepted CuTeDSL + FlashInfer autotune

P3 is a single-variable A/B against accepted P0. It retains the same pinned
checkpoint and image, TP2 + EP2 topology, FP8 E4M3 KV cache, full 135,168-token
profile, C1–C128 workload, and no CPU offload. Only FlashInfer autotuning changes
from disabled to enabled.

| Pin | Value |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Fixed profile | TP2/PP1 + EP2, FP8 E4M3 KV, 135,168-token context, 93% HBM utilization, no CPU offload |

| Metric | P0 autotune off | P3 autotune on | Delta |
| --- | ---: | ---: | ---: |
| 8K prefill tok/s | 6,223 | 6,359 | +2.185% |
| 64K prefill tok/s | 7,461 | 7,624 | +2.185% |
| 128K prefill tok/s | 7,179 | 7,318 | +1.936% |
| C1 decode tok/s | 68.164 | 67.745 | -0.615% |
| C2 decode tok/s | 117.942 | 115.110 | -2.401% |
| C4 decode tok/s | 218.289 | 211.107 | -3.290% |
| C8 decode tok/s | 360.975 | 357.387 | -0.994% |
| C16 decode tok/s | 539.652 | 546.642 | +1.295% |
| C32 decode tok/s | 903.688 | 886.530 | -1.899% |
| C64 decode tok/s | 1,252.236 | 1,252.079 | -0.013% |
| C128 decode tok/s | 2,013.851 | 1,997.099 | -0.832% |

Across all eight decode cells, the unweighted mean delta was -1.094%; only C16
improved materially in this single pass. Prefill improved by an unweighted mean
of +2.102%. This supports a small prefill lift, not a decode win; repeats would
be required to separate small effects from run-to-run variance.

Cold compile took 127.71 / 131.17 seconds by rank. FlashInfer then spent about
28.09 seconds autotuning and wrote 23 new configurations. The fixed capacity
gate passed with 147,264 GPU KV tokens, but that is substantially less than
P0's 261,952-token cache. All four natural outputs finished normally with no
automatic flags, RoCE health counters stayed clean, and teardown returned both
GPUs to the idle baseline.

| Cost or validation | P0 autotune off | P3 autotune on |
| --- | ---: | ---: |
| Checkpoint load by rank | 923.64 / 997.60 s | 69.48 / 72.22 s |
| Compile by rank | 2.46 / 2.73 s | 127.71 / 131.17 s |
| Autotune | Disabled | 28.087 s; 23 new configs |
| GPU KV capacity | 261,952 tokens | 147,264 tokens |
| Natural-output quality | 4/4 stop; 0 flags | 4/4 stop; 0 flags |
| RoCE matrix average | 26.55 Gb/s each direction | 26.62 Gb/s each direction |
| RoCE health-counter deltas | None | None |
| Post-teardown idle HBM | 2 / 2 MiB | 2 / 4 MiB |

The load and compile timings expose different cache states and are startup
costs, not an inference-performance comparison. The network values are quiet
before/after whole-matrix averages, not peak-link measurements.

[`runtime-summary.json`](runtime-summary.json) is the compact result and raw
source-hash index. [`benchmark.json`](benchmark.json) and
[`quality-audit.json`](quality-audit.json) retain the complete measured output;
[`validation.json`](validation.json) and
[`network-delta.json`](network-delta.json) retain acceptance and fabric
evidence. [`autotune-configs.json`](autotune-configs.json) is the exact
23-entry FlashInfer cache artifact. `SHA256SUMS` covers every published file in
this increment.
