# P2: vLLM-native CUTLASS is incompatible with EP2

This is compatibility-failure provenance, not a performance result. It held
the P0 checkpoint, image, topology, workload, context, and memory settings
fixed while selecting vLLM's native `VLLM_CUTLASS` NVFP4 MoE backend.

| Pin | Value |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Fixed profile | TP2/PP1 + EP2, FP8 E4M3 KV, 135,168-token context, 93% HBM utilization, no CPU offload |

Both ranks rejected the backend during model construction because the pinned
kernel does not support the required expert-parallel configuration: EP size 2,
`use_ep=True`, and `allgather_reducescatter`. No weights were loaded, the API
never became healthy, and no benchmark request was issued. No retry or profile
reduction was attempted. Graceful teardown, the current-boot kernel scan, and
the idle-HBM gate all passed.

[`failure-summary.json`](failure-summary.json) retains the structured failure,
teardown disposition, and hashes of the raw plan and logs. The exact resolved
profile is in [`resolved-plan.json`](resolved-plan.json); `SHA256SUMS` covers
every published file in this increment.
