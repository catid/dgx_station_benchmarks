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

The accepted benchmark artifacts remove only private machine labels and
generic driver-reconfiguration suggestion fields. Measurement values are
unchanged; each `runtime-summary.json` records the SHA-256 hashes of retained
raw sources. Excluded starts publish compact failure summaries and raw-source
hashes, while machine-specific full logs remain outside the repository. P6 is
explicitly harness-only: API and capacity succeeded, but no benchmark request
ran, so it publishes no performance or acceptance row. P3 also retains the
exact 23-entry FlashInfer autotune configuration artifact.
