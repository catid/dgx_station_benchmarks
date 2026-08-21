# P5: split-backend native MTP1 maps correctly but does not fit

This is backend-compatibility success and capacity-failure provenance, not a
performance result. P5 used vLLM's stock per-draft MoE override: the NVFP4
target retained CuTeDSL while the unquantized native-MTP draft used FlashInfer
CUTLASS. The conservative bootstrap reduced context, sequences, batching, and
CUDA graph sizes before the single attempt.

| Pin or setting | Value |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Topology | TP2/PP1 + EP2; FP8 E4M3 KV; no CPU offload |
| Target / draft | CuTeDSL NVFP4 target / FlashInfer CUTLASS unquantized MTP1 draft |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=0` |
| Bootstrap envelope | 32,768 max length; 16 sequences; 16,384 batched tokens; CUDA graphs through 16 |

Both ranks independently selected the intended backends. Target weights loaded
in 66.86 / 71.14 seconds and the draft in 9.37 / 9.93 seconds. Total model
memory was 219.01 GiB per rank. Cold backbone compile took 126.91 / 129.16
seconds, draft compile took 11.47 / 11.53 seconds, and estimated CUDA graph
memory was 2.86 GiB per rank.

The limiting result was KV capacity: 3.42 GiB was available on rank 0 but only
0.20 GiB on rank 1. The pinned runtime required 1.48 GiB to serve a single
32,768-token sequence and estimated only a 4,352-token maximum on the limiting
rank. The API therefore never became healthy. No benchmark, acceptance,
quality, or network request was issued, and no retry or profile change was
attempted.

Graceful teardown, the current-boot kernel scan, and the idle-HBM gate passed.
[`failure-summary.json`](failure-summary.json) retains the exact mapping,
startup, capacity, and raw-source hashes. The exact resolved profile is in
[`resolved-plan.json`](resolved-plan.json); `SHA256SUMS` covers every published
file in this increment.
