# GLM-5.2 deep-study data

Each subdirectory is an immutable publication increment with its own
`SHA256SUMS`. Accepted measurements and excluded starts are kept separate; a
failed start is never represented as zero throughput.

| Increment | Backend | API ready | Benchmark | Disposition |
| --- | --- | --- | --- | --- |
| [`2026-08-20-p0-cutedsl/`](2026-08-20-p0-cutedsl/) | FlashInfer CuTeDSL | Yes | 3 prefill + 8 decode rows; 4 natural outputs | Accepted |
| [`2026-08-20-p1-flashinfer-cutlass-cold-cache/`](2026-08-20-p1-flashinfer-cutlass-cold-cache/) | FlashInfer CUTLASS | No | Not run | Excluded: 0.43 / 0.43 GiB KV available, 6.0 GiB required |
| [`2026-08-20-p1-flashinfer-cutlass-warm-cache/`](2026-08-20-p1-flashinfer-cutlass-warm-cache/) | FlashInfer CUTLASS | No | Not run | Excluded: 2.19 / 10.39 GiB KV available, 6.0 GiB required per rank |

The P0 benchmark artifact removes only private machine labels and generic
driver-reconfiguration suggestion fields. Measurement values are unchanged;
`runtime-summary.json` records the SHA-256 hashes of the retained raw sources.
The P1 directories publish compact failure summaries and raw-source hashes,
while the machine-specific full logs remain outside the repository.
