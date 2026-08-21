# P4: native MTP1 is incompatible with the CuTeDSL target backend

This is compatibility-failure provenance, not a performance result. P4 held
the accepted P0 target checkpoint, image, TP2 + EP2 topology, full context,
memory settings, CuTeDSL backend, and autotune-off state fixed while enabling
one native MTP draft token.

| Pin | Value |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Fixed profile | TP2/PP1 + EP2, FP8 E4M3 KV, 135,168-token context, 93% HBM utilization, no CPU offload |
| Changed setting | Native MTP enabled with one speculative token |

Both target ranks loaded all 47 checkpoint shards in 67.32 / 70.46 seconds.
The subsequent draft-model construction failed identically on both ranks. The
target's NVFP4 MoE accepts `FLASHINFER_CUTEDSL`, but the MTP draft layer is an
unquantized MoE; the pinned runtime does not map CuTeDSL to an unquantized MoE
implementation.

The API never became healthy and no benchmark request was issued. Therefore
P4 has no throughput, draft-acceptance, quality, KV-capacity, or network row.
No retry, alternative backend, or reduced profile was attempted. Graceful
teardown, the current-boot kernel scan, and the idle-HBM gate all passed.

[`failure-summary.json`](failure-summary.json) retains the exact exception,
startup and teardown disposition, and hashes of the raw plan and logs. The
exact resolved profile is in [`resolved-plan.json`](resolved-plan.json);
`SHA256SUMS` covers every published file in this increment.
