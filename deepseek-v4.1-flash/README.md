# DeepSeek-V4.1-Flash on 2× NVIDIA GB300 DGX Stations

The official [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint at revision `dba1be0a40aa45a94ad051997016db3960a90277` (48 native FP8-dense / FP4-expert shards, 510,286,023,000 bytes), served across two DGX Stations over dual 400GbE RoCE rails by SGLang (TP2+EP2) and vLLM (PP2 and TP2), autoregressive and with DSpark speculative decoding. Every measured configuration is its own series on every chart; accepted lanes rank, and the replay-off SGLang lane is the numerically exact reference.

[![vLLM PP2 social clip: one 128K request across two DGX Stations](assets/fly-vllm-preview.gif)](assets/fly-vllm-2xdgx.mp4)

**[Open/download the MP4](assets/fly-vllm-2xdgx.mp4)**

<video controls preload="metadata" src="assets/fly-vllm-2xdgx.mp4"></video>

*Social clip for the headline result: vLLM PP2, one 128K request, 55,992 prompt tok/s.*

## Headline

- **55,992 prompt tok/s** for a single 128K request (TTFT **2.340s**) and **65,966 aggregate prompt tok/s** at 64K / C16 (**63,048** at 128K / C16) — vLLM PP2 over Data Direct RDMA, full prefill and the fastest accepted prefill lane.
- **37,693 prompt tok/s** at 128K / C16 and **39,549 tok/s** for a single 16K request (TTFT **0.415s**; a 128K prompt in **3.480s**) — SGLang TP2+EP2 over Data Direct RDMA with SWA bounded replay, the deployment technique DeepSeek's V4.1 report describes; faster than full prefill but not bit-identical to it.
- Exact full prefill on SGLang (replay off, the numerically exact reference): **24,419 prompt tok/s** at 128K / C16 and **26,316 tok/s** for a single 16K request (TTFT **0.623s**).
- Data Direct RDMA lifted the same SGLang server from **16,621** to **25,638 prompt tok/s** (+54.2%) at 64K prompts, C16, and SWA bounded replay took it to **39,133** (+135.4% versus stock, +52.6% versus Data Direct alone); NCCL fell from **54.8%** to **29.2%** of GPU kernel time in a 32K prefill.
- DSpark: **180.0 tok/s per user at C1** (2.80 accepted tokens per step) versus **106.8 tok/s** autoregressive on SGLang; the two lanes meet by C16–C32. vLLM PP2 autoregressive decodes at **141.3 tok/s** per user at C1.
- Peak decode: **2,805.9 aggregate tok/s** (vLLM PP2 · AR, C64) versus **2,236.7** for SGLang AR · Data Direct at C64.
- vLLM PP2 versus SGLang TP2+EP2 prefill at 128K / C16: **63,048** versus **37,693** (SWA replay) and **24,419 prompt tok/s** (exact).
- Dual-rail NCCL all-reduce: **93.7 GB/s** bus bandwidth at 2 GiB versus **48.7 GB/s** on one rail.

## Prefill throughput

![DeepSeek-V4.1-Flash prefill throughput versus prompt length](charts/prefill-throughput.png)

*Aggregate prompt tok/s versus prompt length, one series per configuration: solid at its best concurrency, faint at the others. Unique random-id prompts, one output token, `/flush_cache` before every point; 16K–128K × C1/C4/C16 (the stock-path baseline also ran 8K). The SWA replay series is DeepSeek's documented bounded-replay prefill: accepted and ranked, but not bit-identical to the replay-off series, which is the exact reference.*

| Prompt length | SGLang AR · stock RDMA† | SGLang AR · Data Direct | SGLang AR + SWA replay | SGLang DSpark† | vLLM PP2 · AR | vLLM TP2 · AR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 17,082 | 26,702 | 40,358 | 25,954 | **61,518** | {{VLLM_TP2_PREFILL_16K_BEST}} |
| 32K | 16,910 | 26,300 | 39,967 | — | **64,935** | {{VLLM_TP2_PREFILL_32K_BEST}} |
| 64K | 16,638 | 25,654 | 39,139 | 25,357 | **65,966** | {{VLLM_TP2_PREFILL_64K_BEST}} |
| 128K | 16,112 | 24,419 | 37,711 | — | **63,048** | {{VLLM_TP2_PREFILL_128K_BEST}} |

*Best concurrency per cell; bold is the best accepted (rankable) value in the row, † a diagnostic lane that is drawn but never ranked, — a point outside that lane's measured grid (the DSpark prefill check is 16K/64K at C1 only). vLLM PP2 and both SGLang AR lanes are accepted: vLLM PP2 and SGLang Data Direct (replay off) are full prefill, the latter the numerically exact reference, and SWA replay is not bit-identical to it (see [notes/](notes/)). Full grids with TTFT and server-counter parity: [`data/prefill.csv`](data/prefill.csv), [`data/diagnostic-prefill.csv`](data/diagnostic-prefill.csv).*

## Prefill scaling with concurrency

![DeepSeek-V4.1-Flash prefill throughput at 64K by concurrency](charts/prefill-concurrency.png)

*Aggregate prompt tok/s for 64K prompts with 1, 4, and 16 requests held in flight; one bar per configuration with a 64K cell.*

| Lane | C1 | C4 | C16 |
| --- | ---: | ---: | ---: |
| SGLang AR · stock RDMA† | 16,599 | 16,638 | 16,621 |
| SGLang AR · Data Direct | 25,571 | 25,654 | 25,638 |
| SGLang AR + SWA replay | 39,014 | 39,139 | 39,133 |
| SGLang DSpark† | 25,357 | — | — |
| vLLM PP2 · AR | **53,515** | **65,338** | **65,966** |
| vLLM TP2 · AR | {{VLLM_TP2_PREFILL_64K_C1}} | {{VLLM_TP2_PREFILL_64K_C4}} | {{VLLM_TP2_PREFILL_64K_C16}} |

## Time to first token

![DeepSeek-V4.1-Flash single-request TTFT versus prompt length](charts/prefill-ttft.png)

*TTFT p50 for one request in flight (C1) versus prompt length, one line per configuration; with one output token this is the prefill time.*

| Prompt length | SGLang AR · stock RDMA† | SGLang AR · Data Direct | SGLang AR + SWA replay | SGLang DSpark† | vLLM PP2 · AR | vLLM TP2 · AR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 0.969s | 0.623s | **0.415s** | 0.632s | 0.458s | {{VLLM_TP2_TTFT_16K_C1}} |
| 32K | 1.947s | 1.257s | 0.827s | — | **0.710s** | {{VLLM_TP2_TTFT_32K_C1}} |
| 64K | 3.948s | 2.563s | 1.680s | 2.585s | **1.224s** | {{VLLM_TP2_TTFT_64K_C1}} |
| 128K | 8.142s | 5.373s | 3.480s | — | **2.340s** | {{VLLM_TP2_TTFT_128K_C1}} |

## Decode throughput

![DeepSeek-V4.1-Flash aggregate decode throughput versus concurrency](charts/decode-throughput.png)

*Aggregate output tok/s versus request concurrency, one line per configuration (AR and DSpark, both engines); 8,192-token input, 1,024 forced output tokens, temperature 0, `C` warm-ups then `5 × C` measured requests.*

| C | SGLang AR · Data Direct | SGLang DSpark | vLLM PP2 · AR | vLLM TP2 · AR | vLLM TP2 · DSpark |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 104.2 | **169.2** | 140.6 | {{VLLM_TP2_DECODE_C1}} | {{VLLM_TP2_DSPARK_DECODE_C1}} |
| 4 | 352.4 | 416.6 | **431.1** | {{VLLM_TP2_DECODE_C4}} | {{VLLM_TP2_DSPARK_DECODE_C4}} |
| 16 | 906.8 | 946.2 | **1,033.4** | {{VLLM_TP2_DECODE_C16}} | {{VLLM_TP2_DSPARK_DECODE_C16}} |
| 32 | 1,410.3 | 1,418.3 | **1,878.0** | {{VLLM_TP2_DECODE_C32}} | {{VLLM_TP2_DSPARK_DECODE_C32}} |
| 64 | 2,236.7 | — | **2,805.9** | {{VLLM_TP2_DECODE_C64}} | — |

*Both SGLang decode lanes and the vLLM PP2 lane are accepted; DSpark lanes were requested at C1–C32 only, so their C64 cell does not exist. Every cell, with TTFT, ITL, and accept length, is in [`data/throughput.csv`](data/throughput.csv).*

## Per-user decode speed

![DeepSeek-V4.1-Flash per-user decode speed versus concurrency](charts/decode-per-user.png)

*Median per-request output tok/s at each concurrency; the DSpark legend carries its C1 accept length, and the per-cell accept length is retained in [`data/throughput.csv`](data/throughput.csv).*

| C | SGLang AR · Data Direct | SGLang DSpark | vLLM PP2 · AR | vLLM TP2 · AR | vLLM TP2 · DSpark |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 106.8 | **180.0** | 141.3 | {{VLLM_TP2_USER_C1}} | {{VLLM_TP2_DSPARK_USER_C1}} |
| 4 | 91.6 | 108.4 | **122.2** | {{VLLM_TP2_USER_C4}} | {{VLLM_TP2_DSPARK_USER_C4}} |
| 16 | 59.0 | 61.3 | **72.8** | {{VLLM_TP2_USER_C16}} | {{VLLM_TP2_DSPARK_USER_C16}} |
| 32 | 45.7 | 45.5 | **60.2** | {{VLLM_TP2_USER_C32}} | {{VLLM_TP2_DSPARK_USER_C32}} |
| 64 | 36.5 | — | **43.9** | {{VLLM_TP2_USER_C64}} | — |

## SGLang prefill tuning ladder

![DeepSeek-V4.1-Flash SGLang prefill tuning ladder](charts/tuning-ladder.png)

*Aggregate prompt tok/s of the same TP2+EP2 server at 64K prompts, C16 after each cumulative tuning step; the superseded stock-path step is a non-rankable diagnostic, steps 2 and 3 are both accepted (step 3 is not bit-identical to step 2). The hatched vLLM PP2 bar is a comparison at the same point, not a tuning step.*

| Step | Configuration | Change | Prompt tok/s | vs step 1 |
| ---: | --- | --- | ---: | ---: |
| 1 | SGLang AR · stock RDMA† | Stock container RDMA path (image rdma-core 50, no Data Direct) | 16,621 | — |
| 2 | SGLang AR · Data Direct | Host rdma-core overlay → ConnectX-8 Data Direct RDMA | 25,638 | +54.2% |
| 3 | SGLang AR + SWA replay | + SWA bounded replay (`--enable-decoder-swa-bounded-replay`; +52.6% over step 2) | **39,133** | **+135.4%** |
| — | vLLM PP2 · AR | Other engine at the same point (comparison, not a tuning step) | 65,966 | +296.9% |

## Where the GPU time goes

![DeepSeek-V4.1-Flash GPU kernel time by category](charts/kernel-time.png)

*Summed GPU kernel time by category for one profiled 32K random-token prefill request on each node; the NCCL all-reduce share is what the RDMA path changes. After the fix the largest single non-NCCL kernel is the hyper-connection mixing-statistics Triton kernel (about 245 ms per 32K prefill, in "Other"; see [notes/](notes/)).*

| Profile | Node | NCCL | Attention | GEMM | Norm/RoPE/quant | Other | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Data Direct RDMA · 32K prefill | node0 | **354 ms** · 29.2% | 122 ms | 140 ms | 147 ms | 452 ms | 1,215 ms |
| Data Direct RDMA · 32K prefill | node1 | **359 ms** · 29.6% | 121 ms | 141 ms | 147 ms | 442 ms | 1,210 ms |
| Stock RDMA path · 32K prefill | node0 | **1,046 ms** · 54.8% | 122 ms | 141 ms | 147 ms | 453 ms | 1,908 ms |
| Stock RDMA path · 32K prefill | node1 | **1,053 ms** · 55.2% | 121 ms | 142 ms | 147 ms | 443 ms | 1,906 ms |

## NCCL all-reduce bus bandwidth

![DeepSeek-V4.1-Flash NCCL all-reduce bus bandwidth between the stations](charts/fabric.png)

*nccl-tests `all_reduce_perf` between the two stations (one GB300 each), out-of-place bus bandwidth in GB/s from 64 MiB to 2 GiB; single rail versus dual rail, tuned ring/simple versus NCCL auto.*

| Configuration | 64 MiB | 512 MiB | 2 GiB | Average |
| --- | ---: | ---: | ---: | ---: |
| Dual rail · NCCL auto | 79.5 | 89.5 | 89.7 | 86.4 |
| Dual rail · tuned (16 ch, 4 QP) | 77.9 | 83.8 | 84.2 | 82.7 |
| Dual rail · tuned (8 ch, 4 QP) | **81.2** | **91.3** | **93.7** | **88.7** |
| Single rail · tuned (8 ch, 4 QP) | 44.5 | 47.9 | 48.7 | 47.2 |

## What was measured

| Engine | Topology | Mode | Prefill 16K–128K × C1/4/16 | Decode C1–C64 |
| --- | --- | --- | --- | --- |
| SGLang `dev-dsv41` | TP2+EP2 | AR · stock RDMA | diagnostic only: superseded tuning step (8K–128K × C1/4/16, 0 errors) | not measured |
| SGLang `dev-dsv41` | TP2+EP2 | AR · Data Direct | accepted (12/12 points, 0 errors); exact full-prefill reference | accepted (C1–C64, 0 errors) |
| SGLang `dev-dsv41` | TP2+EP2 | AR + SWA replay | accepted (12/12 points, 0 errors); not bit-identical to full prefill | not measured (prefill-only flag) |
| SGLang `dev-dsv41` | TP2+EP2 | DSpark | diagnostic only: 16K/64K C1 spot check (0 errors) | accepted (C1–C32 as requested, 0 errors) |
| vLLM `deepseekv41-flash-0909` | PP2 | AR | accepted (12/12 points, 0 errors); full prefill on the patched image | accepted (C1–C64, 0 errors) |
| vLLM `deepseekv41-flash-0909` | TP2 | AR | {{STATUS_VLLM_TP2_PREFILL}} | {{STATUS_VLLM_TP2_DECODE}} |
| vLLM `deepseekv41-flash-0909` | TP2 | DSpark | not measured | {{STATUS_VLLM_TP2_DSPARK_DECODE}} |
| — | 1× GB300 | — | not attempted | not attempted |

Pending, failed, unsupported, and unmeasured cells never appear as numeric zeroes; diagnostic lanes (†) are drawn on every chart but are never ranked. The lane ledger is [`data/qualification.csv`](data/qualification.csv).

Details: [recipes/](recipes/) · [notes/](notes/) · [data/](data/)

Return to the [repository overview](../).
