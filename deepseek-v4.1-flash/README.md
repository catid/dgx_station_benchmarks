# DeepSeek-V4.1-Flash on 2× NVIDIA GB300 DGX Stations

The official [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint at revision `dba1be0a40aa45a94ad051997016db3960a90277` (48 native FP8-dense / FP4-expert shards, 510,286,023,000 bytes, more than one GB300 holds), served across two DGX Stations over dual 400GbE RoCE rails by SGLang (TP2+EP2) and vLLM (TP1 × PP2 and TP2), autoregressive and with DSpark speculative decoding.

[![vLLM PP2 social clip: one 128K request across two DGX Stations](assets/fly-vllm-preview.gif)](assets/fly-vllm-2xdgx.mp4)

**[Open/download the MP4](assets/fly-vllm-2xdgx.mp4)**

<video controls preload="metadata" src="assets/fly-vllm-2xdgx.mp4"></video>

*Social clip for the headline result: vLLM PP2, one 128K request, 55,992 prompt tok/s.*

## Headline

- **55,992 prompt tok/s** for a single 128K request (TTFT **2.340s**) and **65,966 aggregate prompt tok/s** at 64K / C16 (**63,048** at 128K / C16) — vLLM TP1 × PP2 over Data Direct RDMA, text-only, with the two local source patches this image needs to start under pipeline parallelism. At those three points that is +48.6% / +68.6% / +67.3% over SGLang's SWA-replay lane (37,674 / 39,133 / 37,693) and +129.5% / +157.3% / +158.2% over SGLang's exact full prefill (24,394 / 25,638 / 24,419).
- On the same vLLM image the pipeline split beats splitting every layer: PP2 delivers **55,992** versus **32,672 prompt tok/s** for one 128K request (+71.4%) and **65,966** versus **34,695** at 64K / C16 (+90.1%) over TP2, which pays a cross-node all-reduce in every layer while PP2 keeps both Engram layers on one stage.
- SGLang TP2+EP2 with SWA bounded replay (the deployment technique DeepSeek's V4.1 report describes; faster than full prefill but not bit-identical to it): **39,549 tok/s** for a single 16K request (TTFT **0.415s**; a 128K prompt in **3.480s**) and **37,693 prompt tok/s** at 128K / C16 — the fastest single 16K request of any lane.
- Exact full prefill (SGLang, replay off, the numerically exact reference): **24,419 prompt tok/s** at 128K / C16 and **26,316 tok/s** for a single 16K request (TTFT **0.623s**).
- Per-user decode at C1: vLLM PP2 DSpark **252.9 tok/s per user at C1** (2.57 accepted tokens per step; the TP1 × PP2 split running DSpark through a local five-file overlay that stock vLLM does not support) versus vLLM TP2 DSpark **201.0 tok/s** (2.56 accepted tokens per step), SGLang DSpark **180.0 tok/s** (2.80 accepted tokens per step), vLLM PP2 autoregressive **141.3 tok/s**, and SGLang autoregressive **106.8 tok/s**; PP2 DSpark leads per user and in aggregate at C1 and from C4 to C32 (+25.8% per user at C1 over TP2 DSpark, +79.0% over its own AR server), TP2 DSpark at C2 and C64.
- Peak decode: **3,401.6 aggregate tok/s** (vLLM TP2 · DSpark, C64) versus **3,257.9** for vLLM PP2 · DSpark (+4.4%), **2,805.9** for vLLM PP2 · AR (+21.2%), **2,392.3** for SGLang DSpark (+42.2%) and **2,236.7** for SGLang AR · Data Direct (+52.1%) at C64; among the autoregressive lanes PP2 leads, and SGLang DSpark still beats SGLang AR at C64 (+7.0%).
- Data Direct RDMA lifted the same SGLang server from **16,621** to **25,638 prompt tok/s** (+54.2%) at 64K prompts, C16, and SWA bounded replay took it to **39,133** (+135.4% versus stock, +52.6% versus Data Direct alone); vLLM PP2 reaches **65,966** at that point (+296.9% versus stock, +68.6% versus the best SGLang step); NCCL fell from **54.8%** to **29.2%** of GPU kernel time in a 32K prefill.
- Dual-rail NCCL all-reduce: **93.7 GB/s** bus bandwidth at 2 GiB versus **48.7 GB/s** on one rail.

## Best numbers

| Configuration | Prefill 128K · C1 | Prefill 64K · C16 | Decode C1 · per user | Decode C64 · aggregate |
| --- | ---: | ---: | ---: | ---: |
| SGLang AR · stock RDMA† | 16,098 | 16,621 | — | — |
| SGLang AR · Data Direct | 24,394 | 25,638 | 106.8 | 2,236.7 |
| SGLang AR + SWA replay | 37,674 | 39,133 | — | — |
| SGLang DSpark† | — | — | 180.0 | 2,392.3 |
| vLLM PP2 · AR | **55,992** | **65,966** | 141.3 | 2,805.9 |
| vLLM TP2 · AR | 32,672 | 34,695 | 102.3 | 1,964.5 |
| vLLM TP2 · DSpark | — | — | 201.0 | **3,401.6** |
| vLLM PP2 · DSpark | — | — | **252.9** | 3,257.9 |

*Prompt tok/s for one 128K request and aggregate at 64K with 16 requests in flight; output tok/s per user at C1 and aggregate at C64. Bold is the best accepted value in the column, † a diagnostic lane that is drawn but never ranked, — a point outside the lane's measured grid. Every measured configuration is its own series on every chart; accepted lanes rank, and the replay-off SGLang lane is the numerically exact reference. One station was not attempted. Full per-lane grids with TTFT, ITL, and accept lengths: [notes/](notes/).*

## Prefill throughput

![DeepSeek-V4.1-Flash prefill throughput versus prompt length](charts/prefill-throughput.png)

*Aggregate prompt tok/s versus prompt length, one series per configuration (solid at its best concurrency, faint at the others); unique random-id prompts, one output token, 16K–128K × C1/C4/C16. vLLM PP2 leads at every length; SWA replay is not bit-identical to the exact replay-off series.*

## Prefill by concurrency

![DeepSeek-V4.1-Flash prefill throughput at 64K by concurrency](charts/prefill-concurrency.png)

*64K prompts with 1, 4, and 16 requests in flight. The two-stage vLLM pipeline needs more than one request to fill (its C1 bar is 81% of its C16 bar); the SGLang lanes and vLLM TP2 are flat across C1–C16.*

| Configuration | 64K · C1 | 64K · C4 | 64K · C16 | 128K · C1 | 128K · C4 | 128K · C16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang AR · stock RDMA† | 16,599 | 16,638 | 16,621 | 16,098 | 16,112 | 16,108 |
| SGLang AR · Data Direct | 25,571 | 25,654 | 25,638 | 24,394 | 24,417 | 24,419 |
| SGLang AR + SWA replay | 39,014 | 39,139 | 39,133 | 37,674 | 37,711 | 37,693 |
| SGLang DSpark† | 25,357 | — | — | — | — | — |
| vLLM PP2 · AR | **53,515** | **65,338** | **65,966** | **55,992** | **62,750** | **63,048** |
| vLLM TP2 · AR | 34,270 | 34,682 | 34,695 | 32,672 | 32,992 | 33,010 |

*Aggregate prompt tok/s; bold is the best accepted value per column, † a diagnostic lane, — outside the lane's grid. The 16K and 32K columns and every TTFT are in [notes/](notes/).*

## Time to first token

![DeepSeek-V4.1-Flash single-request TTFT versus prompt length](charts/prefill-ttft.png)

*TTFT p50 for one request in flight, which with one output token is the prefill time. SGLang with SWA replay is fastest for a lone 16K request; vLLM PP2 is fastest from 32K up.*

## Decode throughput

![DeepSeek-V4.1-Flash aggregate decode throughput versus concurrency](charts/decode-throughput.png)

*Aggregate output tok/s versus concurrency, AR and DSpark on both engines; 8,192-token input, 1,024 forced output tokens, temperature 0, `C` warm-ups then `5 × C` measured requests. vLLM PP2 DSpark (five-file overlay, adaptive verification off) leads at C1 and from C4 to C32 (+10.8% over vLLM TP2 DSpark at C32, +16.1% over vLLM PP2 AR at C64); vLLM TP2 DSpark leads at C2 and C64 (+4.4% at C64). Both speculative vLLM lanes beat every AR lane at every concurrency; vLLM TP2 AR trails PP2 AR throughout, and SGLang DSpark beats SGLang AR at every concurrency but trails vLLM PP2 AR at C4 and from C16 upward.*

| Configuration | C1 | C4 | C16 | C32 | C64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang AR · Data Direct | 104.2 | 352.4 | 906.8 | 1,410.3 | 2,236.7 |
| SGLang DSpark | 169.2 | 416.6 | 946.2 | 1,418.3 | 2,392.3 |
| vLLM PP2 · AR | 140.6 | 431.1 | 1,033.4 | 1,878.0 | 2,805.9 |
| vLLM TP2 · AR | 101.9 | 343.5 | 803.7 | 1,272.0 | 1,964.5 |
| vLLM TP2 · DSpark | 201.7 | 525.3 | 1,275.5 | 2,116.8 | **3,401.6** |
| vLLM PP2 · DSpark | **248.5** | **612.6** | **1,677.7** | **2,346.4** | 3,257.9 |

## Per-user decode speed

![DeepSeek-V4.1-Flash per-user decode speed versus concurrency](charts/decode-per-user.png)

*Median per-request output tok/s at each concurrency: vLLM PP2 DSpark leads at C1 (2.57 accepted tokens per step) and from C4 to C32, vLLM TP2 DSpark at C2 and C64 (2.56 accepted tokens per step at C1); among the rest SGLang DSpark leads at C1–C2 (2.80 accepted tokens per step) and vLLM PP2 AR from C4 upward. The accept length per cell is kept in [data/throughput.csv](data/throughput.csv).*

| Configuration | C1 | C4 | C16 | C32 | C64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang AR · Data Direct | 106.8 | 91.6 | 59.0 | 45.7 | 36.5 |
| SGLang DSpark | 180.0 | 108.4 | 61.3 | 45.5 | 40.4 |
| vLLM PP2 · AR | 141.3 | 122.2 | 72.8 | 60.2 | 43.9 |
| vLLM TP2 · AR | 102.3 | 87.5 | 53.2 | 41.5 | 31.6 |
| vLLM TP2 · DSpark | 201.0 | 134.9 | 82.0 | 68.6 | **55.3** |
| vLLM PP2 · DSpark | **252.9** | **156.6** | **108.9** | **76.2** | 52.6 |

## SGLang prefill tuning ladder

![DeepSeek-V4.1-Flash SGLang prefill tuning ladder](charts/tuning-ladder.png)

*The same TP2+EP2 server at 64K prompts, C16 after each cumulative step: +54.2% from Data Direct RDMA, +135.4% with SWA bounded replay. The hatched vLLM PP2 bar is another engine at the same point, a comparison rather than a tuning step.*

## Where the GPU time goes

![DeepSeek-V4.1-Flash GPU kernel time by category](charts/kernel-time.png)

*Summed GPU kernel time by category for one profiled 32K prefill on each node. The Data Direct path cut the NCCL share from 54.8% to 29.2%, leaving the hyper-connection mixing-statistics Triton kernel as the largest single item.*

## NCCL all-reduce bus bandwidth

![DeepSeek-V4.1-Flash NCCL all-reduce bus bandwidth between the stations](charts/fabric.png)

*Host-side nccl-tests `all_reduce_perf` between the two stations, 64 MiB–2 GiB: dual rail with the tuned 8-channel ring/simple setting reaches 93.7 GB/s at 2 GiB versus 48.7 GB/s on one rail; 16 channels and NCCL auto are both slower.*

Details: [notes/](notes/) · [data/](data/) · [recipes/](recipes/)

Return to the [repository overview](../).
