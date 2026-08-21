# P6: short-context MTP1 reached capacity but the audit harness stopped both attempts

P6 is harness-failure provenance, not a performance result. The deliberately
short profile used the same pinned split backend as P5, raised static HBM
utilization to 95%, and reduced the maximum length to 9,216 tokens. Both
controlled starts reached a healthy API, loaded the CuTeDSL NVFP4 target plus
the FlashInfer CUTLASS unquantized MTP1 draft, and exposed enough coordinated
KV cache for the planned 8K-input C1/C2/C4/C8 workload. Neither start issued a
benchmark request because the pre-request audit itself failed closed.

| Pin or setting | Value |
| --- | --- |
| Checkpoint | `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa` |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Image | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Topology | TP2/PP1 + EP2; FP8 E4M3 KV; no CPU offload |
| Target / draft | CuTeDSL NVFP4 target / FlashInfer CUTLASS unquantized MTP1 draft |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=0` |
| Short envelope | 9,216 max length; 8 sequences; 16,384 batched tokens; CUDA graphs through 8 |
| Planned requests | Exactly 8K input / up to 1K output at C1, C2, C4, C8; 8K standalone prefill |

| Attempt | API | Available KV GiB by rank | Coordinated KV tokens | Audit disposition | Requests |
| --- | --- | ---: | ---: | --- | ---: |
| 1, cold compile | Healthy | 6.64 / 5.25 | 116,416 | SSH quoting split Docker's Go template; stopped | 0 |
| 2, warm compile cache | Healthy | 8.08 / 8.09 | 179,264 | Audit incorrectly required the global EngineCore token count in each worker log; stopped | 0 |

The declared pre-request minimum was 16,384 KV tokens, so both coordinated
capacity values cleared it. Those are capacity observations only. There are no
decode, prefill, speculative-acceptance, target-step, quality, or network
measurements from P6.

The first failure was corrected and verified offline before the one authorized
second attempt. That attempt exposed a separate logging-scope assumption:
vLLM reports rank-local available-KV memory in each worker log, but emits the
coordinated `GPU KV cache size` once from EngineCore on the API rank. The
published verifier now checks backend selection, MRv1, fatal markers, and local
available-KV memory on both ranks, then checks the single coordinated token
capacity on the API rank. No third attempt was made.

Both attempts used graceful teardown, passed the current-boot kernel-danger
scan, and returned to 5 / 6 MiB and 4 / 8 MiB idle HBM respectively.
[`failure-summary.json`](failure-summary.json) retains the exact startup,
capacity, audit, and raw-source hashes. The exact resolved profile is in
[`resolved-plan.json`](resolved-plan.json); `SHA256SUMS` covers every published
file in this increment.
