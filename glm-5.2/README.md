# GLM-5.2 NVFP4 on 2× NVIDIA GB300

**753B parameters · 40B active · two DGX Stations · 400 GbE RoCE · no CPU offload**

The recommended serving profile reaches **2,012 aggregate output tok/s at
C128** and **7,244 prompt tok/s at 128K** with NVIDIA's NVFP4 checkpoint.

## Headline results

<!-- BEGIN GENERATED:HEADLINES -->
| Decode C1<br><sub>output tok/s</sub> | Decode C64<br><sub>output tok/s</sub> | Decode C128<br><sub>output tok/s</sub> | Prefill 8K<br><sub>prompt tok/s / TTFT</sub> | Prefill 64K<br><sub>prompt tok/s / TTFT</sub> | Prefill 128K<br><sub>prompt tok/s / TTFT</sub> |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **68.0** | **1,261.0** | **2,012.4** | **6,262 / 1.309s** | **7,444 / 8.804s** | **7,244 / 18.095s** |
<!-- END GENERATED:HEADLINES -->

Decode uses an exact 8K input, up to 1K output, and a 30-second sustained
window. Prefill is single-request and client-timed.

![GLM-5.2 TP2 sustained decode throughput](charts/decode-throughput.png)

## Recommended configuration

| Setting | Recommendation |
| --- | --- |
| Hardware | 2× DGX Station, one GB300 each, dedicated 400 GbE RoCE |
| Runtime | vLLM 0.27.1, TP2 / PP1 with expert parallelism |
| MoE backend | FlashInfer CuTeDSL |
| FlashInfer autotuning | Off |
| KV cache | FP8 E4M3 with prefix caching |
| Serving envelope | 135,168 max model length, 128 sequences, 32,768 batched tokens |
| HBM / offload | 93% static utilization; no CPU or disk weight offload |

## Quality

<!-- BEGIN GENERATED:QUALITY -->
| KV cache | Word PPL ↓ | Byte PPL ↓ | Bits/byte ↓ | Natural outputs | Automatic flags | Manual review |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bfloat16 | **3.3524** | 1.253844 | 0.326357 | 4/4 finished naturally | 0 | clean |
<!-- END GENERATED:QUALITY -->

WikiText-2 and the natural-output audit use BF16 KV rather than the FP8 KV used
for performance. The output audit checks coherence and degeneration, not
factual accuracy.

## Checkpoint and runtime provenance

| Item | Pinned value |
| --- | --- |
| Checkpoint | [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| Revision | `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Architecture | 753B total / 40B active MoE; documented 1M-token context ceiling |
| Quantization | NVFP4 expert linear layers; shared expert and selected tensors remain at higher precision |
| Indexed payload | 47 weight files; 464,823,042,096 bytes |
| License | [MIT, as declared by the pinned model card](https://huggingface.co/nvidia/GLM-5.2-NVFP4/blob/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa/README.md) |
| Runtime | vLLM 0.27.1 at `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Container | `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |

## Scope and caveats

- Decode is aggregate output throughput from vLLM's continuous usage counters.
  Streams still active at the 30-second boundary contribute observed tokens.
- The C128 result uses a shared 8K prefix with prefix caching; it is not a
  capacity claim for 128 unrelated 8K prompts.
- Prefill reports actual prompt tokens divided by client TTFT. The measured
  135,168-token serving envelope is not a benchmark of the full 1M context.
- The pinned benchmark and evaluator commits, exact metric definitions, sample
  counts, validation gates, and complete commands are in the recipes and data.

## Reproduce or inspect

- [Accepted serving recipe and benchmark methodology](recipes/)
- [Optimization recipe and experiment log](recipes/deep-study/)
- [Normalized results and compact evidence](data/)
- [Checksummed deep-study increments](data/deep-study/)
