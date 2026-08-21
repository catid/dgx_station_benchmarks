# P9: full-context PP2 Inductor does not fit at 93% HBM utilization

P9 tested the reportable PP2 performance path after P8 established that the
explicit 64-token KV block layout serves correct text in eager mode. It kept
the 40/38 pipeline split, CuTeDSL backend, FP8 E4M3 KV, speculation-off and
autotune-off controls, enabled vLLM's Inductor compilation, and explicitly set
CUDA graph mode to `NONE`. Both pipeline stages loaded, compiled, and completed
their profiling warmup, but the limiting stage had too little memory left for
the declared 135,168-token context. The API never became ready.

| Pin or setting | Value |
| --- | --- |
| Checkpoint | [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Topology | TP1/PP2, EP1; explicit 40/38 layer partition; no CPU offload |
| Backend | FlashInfer CuTeDSL NVFP4 MoE; autotuning and speculation off |
| KV | FP8 E4M3; `--block-size 64` |
| Execution | Inductor enabled; CUDA graph mode `NONE`; eager execution off |
| Full envelope | 135,168 max length; 128 sequences; 32,768 batched tokens; 93% memory utilization |
| Planned workload | Exact 8K input/up-to-1K output at C1–C128; 8K/64K/128K prefill |

| Stage | Assigned layers | Model memory | Weight load | Total model load | Compile | Profiling warmup | Available KV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PP0 | 40 | 206.54 GiB | 58.51 s | 72.36 s | 49.76 s | 19.21 s | 7.60 GiB |
| PP1 | 38 | 209.74 GiB | 58.29 s | 72.04 s | 40.81 s | 18.35 s | 0.28 GiB |

The limiting PP1 stage needed 2.9 GiB of KV memory to admit one request at the
declared maximum length, but exposed only 0.28 GiB. vLLM estimated a maximum
model length of 12,864 tokens on that stage and failed closed during KV-cache
initialization. It emitted no coordinated KV-token capacity.

The startup log independently proves PP0/PP1 ownership, the CuTeDSL NVFP4 MoE
backend, the explicit 64-token block size, `enforce_eager=False`, Inductor's
`VLLM_COMPILE` mode, and CUDA graph mode `NONE`. vLLM's message about compiling
an Inductor graph is not CUDA-graph capture; no CUDA graph capture was enabled.

The API, retained 512/4,097-token correctness gate, headline matrix, natural
output audit, and request-bearing network capture did not run. P9 therefore has
zero throughput, prefill, quality, or bandwidth rows; this capacity exclusion
must not be plotted as zero performance. The failure occurred during startup,
not during a collective, so it is not evidence of a network bottleneck.

Both named containers were gracefully stopped and removed. The current-boot
kernel-danger scan remained clean, idle HBM returned to 5 / 8 MiB, and no retry
was made within this increment.

[`failure-summary.json`](failure-summary.json) retains the exact configuration,
stage measurements, failure disposition, and hashes of the private raw logs and
memory snapshots. The resolved launch profile is in
[`resolved-plan.json`](resolved-plan.json); `SHA256SUMS` covers every published
file in this increment.
