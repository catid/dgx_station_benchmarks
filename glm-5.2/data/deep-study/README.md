# GLM-5.2 deep-study data

Each subdirectory is an immutable publication increment with its own
`SHA256SUMS`. Accepted measurements and excluded starts are kept separate; a
failed start is never represented as zero throughput.

| Increment | Backend | API ready | Benchmark | Disposition |
| --- | --- | --- | --- | --- |
| [`2026-08-20-p0-cutedsl/`](2026-08-20-p0-cutedsl/) | FlashInfer CuTeDSL | Yes | 3 prefill + 8 decode rows; 4 natural outputs | Accepted |
| [`2026-08-20-p1-flashinfer-cutlass-cold-cache/`](2026-08-20-p1-flashinfer-cutlass-cold-cache/) | FlashInfer CUTLASS | No | Not run | Excluded: 0.43 / 0.43 GiB KV available, 6.0 GiB required |
| [`2026-08-20-p1-flashinfer-cutlass-warm-cache/`](2026-08-20-p1-flashinfer-cutlass-warm-cache/) | FlashInfer CUTLASS | No | Not run | Excluded: 2.19 / 10.39 GiB KV available, 6.0 GiB required per rank |
| [`2026-08-20-p2-vllm-cutlass-incompatible/`](2026-08-20-p2-vllm-cutlass-incompatible/) | vLLM CUTLASS | No | Not run | Excluded: pinned kernel does not support the required EP2 configuration |
| [`2026-08-20-p3-cutedsl-autotune-on/`](2026-08-20-p3-cutedsl-autotune-on/) | FlashInfer CuTeDSL + autotune | Yes | 3 prefill + 8 decode rows; 4 natural outputs | Accepted; +2.102% mean prefill, -1.094% mean decode vs P0 |
| [`2026-08-20-p4-mtp1-cutedsl-incompatible/`](2026-08-20-p4-mtp1-cutedsl-incompatible/) | CuTeDSL target + native MTP1 | No | Not run | Excluded: unquantized MTP draft MoE does not support CuTeDSL |
| [`2026-08-20-p5-mtp1-split-bootstrap-capacity/`](2026-08-20-p5-mtp1-split-bootstrap-capacity/) | CuTeDSL target + FlashInfer CUTLASS MTP1 draft | No | Not run | Mapping accepted; excluded at 3.42 / 0.20 GiB KV capacity before API |
| [`2026-08-20-p6-mtp1-short-context-harness-only/`](2026-08-20-p6-mtp1-short-context-harness-only/) | CuTeDSL target + FlashInfer CUTLASS MTP1 draft | Yes | Not run | Capacity passed at 116,416 / 179,264 coordinated KV tokens; two fail-closed audit-harness stops; 0 requests |
| [`2026-08-20-p7-pp2-kv-block-incompatible/`](2026-08-20-p7-pp2-kv-block-incompatible/) | FlashInfer CuTeDSL, TP1/PP2 40/38 | No | Not run | Balanced stages loaded at 206.32 / 209.52 GiB; excluded when 64/32 KV block sizes had no common layout |
| [`2026-08-21-p8-pp2-block64-eager-correctness/`](2026-08-21-p8-pp2-block64-eager-correctness/) | FlashInfer CuTeDSL, TP1/PP2 40/38, block64 eager | Yes | 4 retained correctness outputs; no performance rows | Accepted correctness smoke; 402,688 KV tokens, exact 8K direct API prompts, no corruption flags; greedy pairs not byte-identical |
| [`2026-08-21-p9-pp2-inductor-full-capacity/`](2026-08-21-p9-pp2-inductor-full-capacity/) | FlashInfer CuTeDSL, TP1/PP2 40/38, block64 Inductor, CUDA graphs off | No | Not run | Excluded at full 135,168-token envelope: 7.60 / 0.28 GiB KV available, 2.9 GiB required on the limiting stage |
| [`2026-08-21-p10-pp2-inductor-warm095/`](2026-08-21-p10-pp2-inductor-warm095/) | FlashInfer CuTeDSL, TP1/PP2 40/38, block64 Inductor, CUDA graphs off, 95% HBM | Yes | 3 prefill + 8 decode rows; 2 forced gate + 4 natural outputs | Accepted; 494,528 KV tokens; warm PP-specific AOT and page caches |

The accepted benchmark artifacts remove only private machine labels and
generic driver-reconfiguration suggestion fields. Measurement values are
unchanged; each `runtime-summary.json` records the SHA-256 hashes of retained
raw sources. Excluded starts publish compact failure summaries and raw-source
hashes, while machine-specific full logs remain outside the repository. P6 is
explicitly harness-only: API and capacity succeeded, but no benchmark request
ran, so it publishes no performance or acceptance row. P7 is likewise
compatibility-only and has no throughput row. P3 also retains the exact
23-entry FlashInfer autotune configuration artifact. P8 preserves its raw
harness failure unchanged beside a corrected result: direct API accounting was
8,192 rather than the harness's path-specific 8,194 expectation, and coherent
greedy hash mismatches are retained as nondeterminism rather than corruption.
P9 preserves the first full-envelope Inductor/no-CUDA-graphs capacity result:
both stages compiled, but the limiting rank exposed only 0.28 GiB KV and the
API rejected startup before the retained correctness gate or any benchmark
request. P10 changed declared HBM utilization from 93% to 95% after P9 had
populated its PP-specific AOT and checkpoint page caches. It passed capacity
and all request-bearing gates. Its P0 comparison is explicitly
configuration-level, because topology, EP, HBM utilization, CUDA-graph mode,
and cache state are not held constant. P10's long-prefill requests completed
despite nine recovered 2.25-GiB allocation failures on its limiting stage; the
result is not a memory-headroom claim.
