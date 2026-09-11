# DeepSeek-V4.1-Flash notes

Working notes kept out of the headline deck: the detailed tables behind every
chart, how the two-station SGLang and vLLM servers were tuned, what the
profiles showed, what went wrong, and what the numbers do not claim. Generic
names only: `node0` is rank 0 (API host), `node1` is rank 1. Placeholders in
`{{...}}` are listed in [`PLACEHOLDERS.md`](PLACEHOLDERS.md).

## Detailed tables

One table per chart in [the deck](../README.md), same numbers and rounding
rules: bold is the best accepted (rankable) value in the row, † a diagnostic
lane that is drawn but never ranked, — a point outside that lane's measured
grid (never a placeholder for a pending run). Every cell rounds from the CSV
tables in [`../data/`](../data/), which the section tests enforce.

### Prefill throughput

*Best concurrency per cell, aggregate prompt tok/s; unique random-id prompts, one output token, a cache flush before every point; 16K–128K × C1/C4/C16 (the stock-path baseline also ran 8K). Both SGLang AR lanes are accepted: Data Direct (replay off) is the numerically exact full-prefill reference, and SWA replay is not bit-identical to it (see Tuning story). The vLLM PP2 lane is text-only on the default 20/20 layer split with two local source patches; the vLLM TP2 lane is the same image and settings with `--tensor-parallel-size 2` (rank 0 on node1). On both vLLM lanes the cache-flush endpoint answered 404 and the prefill-token counter shows no work was skipped (see Incidents). The DSpark prefill check is 16K/64K at C1 only. Full grids with TTFT and server-counter parity: [`prefill.csv`](../data/prefill.csv), [`diagnostic-prefill.csv`](../data/diagnostic-prefill.csv).*

| Prompt length | SGLang AR · stock RDMA† | SGLang AR · Data Direct | SGLang AR + SWA replay | SGLang DSpark† | vLLM PP2 · AR | vLLM TP2 · AR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 17,082 | 26,702 | 40,358 | 25,954 | **61,518** | 35,982 |
| 32K | 16,910 | 26,300 | 39,967 | — | **64,935** | 35,551 |
| 64K | 16,638 | 25,654 | 39,139 | 25,357 | **65,966** | 34,695 |
| 128K | 16,112 | 24,419 | 37,711 | — | **63,048** | 33,010 |

### Prefill scaling with concurrency

*Aggregate prompt tok/s for 64K prompts with 1, 4, and 16 requests held in flight; one row per configuration with a 64K cell. The two-stage vLLM pipeline needs more than one request in flight to fill: its C1 cell is 81% of its C16 cell, while the SGLang lanes and vLLM TP2 are flat across C1–C16.*

| Lane | C1 | C4 | C16 |
| --- | ---: | ---: | ---: |
| SGLang AR · stock RDMA† | 16,599 | 16,638 | 16,621 |
| SGLang AR · Data Direct | 25,571 | 25,654 | 25,638 |
| SGLang AR + SWA replay | 39,014 | 39,139 | 39,133 |
| SGLang DSpark† | 25,357 | — | — |
| vLLM PP2 · AR | **53,515** | **65,338** | **65,966** |
| vLLM TP2 · AR | 34,270 | 34,682 | 34,695 |

### Time to first token

*TTFT p50 for one request in flight (C1) versus prompt length; with one output token this is the prefill time. SGLang with SWA replay is fastest for a lone 16K request; vLLM PP2 is fastest from 32K up.*

| Prompt length | SGLang AR · stock RDMA† | SGLang AR · Data Direct | SGLang AR + SWA replay | SGLang DSpark† | vLLM PP2 · AR | vLLM TP2 · AR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 0.969s | 0.623s | **0.415s** | 0.632s | 0.458s | 0.466s |
| 32K | 1.947s | 1.257s | 0.827s | — | **0.710s** | 0.936s |
| 64K | 3.948s | 2.563s | 1.680s | 2.585s | **1.224s** | 1.912s |
| 128K | 8.142s | 5.373s | 3.480s | — | **2.340s** | 4.011s |

### Decode throughput

*Aggregate output tok/s versus request concurrency (AR and DSpark, both engines); 8,192-token input, 1,024 forced output tokens, temperature 0, `C` warm-ups then `5 × C` measured requests. All six decode lanes are accepted. vLLM PP2 DSpark (the PP2 server with `--speculative-config` dspark through the local five-file overlay: 5 speculative tokens, block rejection sampling, adaptive verification off) leads every lane at C1 and from C4 to C32: 248.5 aggregate tok/s at C1 and 2,346.4 at C32, +23.2% and +10.8% over vLLM TP2 DSpark, +76.7% and +24.9% over vLLM PP2 AR, with 2.53–2.61 accepted tokens per step across the sweep. vLLM TP2 DSpark (`--speculative-config` dspark, adaptive verification on) leads at C2 (328.5 versus 320.9) and at C64 (3,401.6 versus 3,257.9, +4.4%), where it is also +21.2% over vLLM PP2 AR and +73.2% over its own AR server, with 2.23–2.56 accepted tokens per step. vLLM TP2 AR trails PP2 AR at every concurrency (102.3 versus 141.3 tok/s per user at C1; 1,964.5 versus 2,805.9 aggregate tok/s at C64): under TP2 every decode step pays the cross-node all-reduce in each layer, while PP2 only hands activations between stages once per step. The SGLang DSpark C64 cell comes from a follow-up run on the same server profile and contract; at C64 DSpark (2,392.3) beats SGLang AR (2,236.7, +7.0%) with 2.54 accepted tokens per step but trails vLLM PP2 AR (2,805.9, −14.7%). The C2 and C8 cells and every TTFT, ITL, and accept length are in [`throughput.csv`](../data/throughput.csv).*

| C | SGLang AR · Data Direct | SGLang DSpark | vLLM PP2 · AR | vLLM TP2 · AR | vLLM TP2 · DSpark | vLLM PP2 · DSpark |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 104.2 | 169.2 | 140.6 | 101.9 | 201.7 | **248.5** |
| 4 | 352.4 | 416.6 | 431.1 | 343.5 | 525.3 | **612.6** |
| 16 | 906.8 | 946.2 | 1,033.4 | 803.7 | 1,275.5 | **1,677.7** |
| 32 | 1,410.3 | 1,418.3 | 1,878.0 | 1,272.0 | 2,116.8 | **2,346.4** |
| 64 | 2,236.7 | 2,392.3 | 2,805.9 | 1,964.5 | **3,401.6** | 3,257.9 |

### Per-user decode speed

*Median per-request output tok/s at each concurrency; the per-cell DSpark accept length is retained in [`throughput.csv`](../data/throughput.csv).*

| C | SGLang AR · Data Direct | SGLang DSpark | vLLM PP2 · AR | vLLM TP2 · AR | vLLM TP2 · DSpark | vLLM PP2 · DSpark |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 106.8 | 180.0 | 141.3 | 102.3 | 201.0 | **252.9** |
| 4 | 91.6 | 108.4 | 122.2 | 87.5 | 134.9 | **156.6** |
| 16 | 59.0 | 61.3 | 72.8 | 53.2 | 82.0 | **108.9** |
| 32 | 45.7 | 45.5 | 60.2 | 41.5 | 68.6 | **76.2** |
| 64 | 36.5 | 40.4 | 43.9 | 31.6 | **55.3** | 52.6 |

### SGLang prefill tuning ladder

*Aggregate prompt tok/s of the same TP2+EP2 server at 64K prompts, C16 after each cumulative tuning step; the superseded stock-path step is a non-rankable diagnostic, steps 2 and 3 are both accepted (step 3 is not bit-identical to step 2). The hatched vLLM PP2 bar on the chart is another engine at the same point: a comparison, not a tuning step. Bold marks the largest value in the column.*

| Step | Configuration | Change | Prompt tok/s | vs step 1 |
| ---: | --- | --- | ---: | ---: |
| 1 | SGLang AR · stock RDMA† | Stock container RDMA path (image rdma-core 50, no Data Direct) | 16,621 | — |
| 2 | SGLang AR · Data Direct | Host rdma-core overlay → ConnectX-8 Data Direct RDMA | 25,638 | +54.2% |
| 3 | SGLang AR + SWA replay | + SWA bounded replay (`--enable-decoder-swa-bounded-replay`; +52.6% over step 2) | 39,133 | +135.4% |
| — | vLLM PP2 · AR | Other engine at the same point (comparison, not a tuning step; +68.6% over step 3) | **65,966** | **+296.9%** |

### Where the GPU time goes

*Summed GPU kernel time by category for one profiled 32K random-token prefill request on each node; the NCCL all-reduce share is what the RDMA path changes. After the fix the largest single non-NCCL kernel is the hyper-connection mixing-statistics Triton kernel (about 245 ms per 32K prefill, in "Other"; see Profiles).*

| Profile | Node | NCCL | Attention | GEMM | Norm/RoPE/quant | Other | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Data Direct RDMA · 32K prefill | node0 | **354 ms** · 29.2% | 122 ms | 140 ms | 147 ms | 452 ms | 1,215 ms |
| Data Direct RDMA · 32K prefill | node1 | **359 ms** · 29.6% | 121 ms | 141 ms | 147 ms | 442 ms | 1,210 ms |
| Stock RDMA path · 32K prefill | node0 | **1,046 ms** · 54.8% | 122 ms | 141 ms | 147 ms | 453 ms | 1,908 ms |
| Stock RDMA path · 32K prefill | node1 | **1,053 ms** · 55.2% | 121 ms | 142 ms | 147 ms | 443 ms | 1,906 ms |

### NCCL all-reduce bus bandwidth

*nccl-tests `all_reduce_perf` between the two stations (one GB300 each), out-of-place bus bandwidth in GB/s from 64 MiB to 2 GiB; single rail versus dual rail, tuned ring/simple versus NCCL auto.*

| Configuration | 64 MiB | 512 MiB | 2 GiB | Average |
| --- | ---: | ---: | ---: | ---: |
| Dual rail · NCCL auto | 79.5 | 89.5 | 89.7 | 86.4 |
| Dual rail · tuned (16 ch, 4 QP) | 77.9 | 83.8 | 84.2 | 82.7 |
| Dual rail · tuned (8 ch, 4 QP) | **81.2** | **91.3** | **93.7** | **88.7** |
| Single rail · tuned (8 ch, 4 QP) | 44.5 | 47.9 | 48.7 | 47.2 |

### What was measured

| Engine | Topology | Mode | Prefill 16K–128K × C1/4/16 | Decode C1–C64 |
| --- | --- | --- | --- | --- |
| SGLang `dev-dsv41` | TP2+EP2 | AR · stock RDMA | diagnostic only: superseded tuning step (8K–128K × C1/4/16, 0 errors) | not measured |
| SGLang `dev-dsv41` | TP2+EP2 | AR · Data Direct | accepted (12/12 points, 0 errors); exact full-prefill reference | accepted (C1–C64, 0 errors) |
| SGLang `dev-dsv41` | TP2+EP2 | AR + SWA replay | accepted (12/12 points, 0 errors); not bit-identical to full prefill | not measured (prefill-only flag) |
| SGLang `dev-dsv41` | TP2+EP2 | DSpark | diagnostic only: 16K/64K C1 spot check (0 errors) | accepted (C1–C64, 0 errors; C64 from a follow-up run) |
| vLLM `deepseekv41-flash-0909` | TP1 × PP2 | AR · text-only, two local patches | accepted (12/12 points, 0 errors) | accepted (C1–C64, 0 errors) |
| vLLM `deepseekv41-flash-0909` | TP2 | AR · text-only | accepted (12/12 points, 0 errors); rank 0 on node1 | accepted (C1–C64, 0 errors); rank 0 on node1 |
| vLLM `deepseekv41-flash-0909` | TP2 | DSpark | not measured | accepted (C1–C64, 0 errors); rank 0 on node1 |
| vLLM `deepseekv41-flash-0909` | TP1 × PP2 | DSpark · five-file overlay (experiment) | not measured | accepted (C1–C64, 0 errors); greedy-equivalence check against the PP2 AR reference pending |
| — | 1× GB300 | — | not attempted | not attempted |

Pending, failed, unsupported, and unmeasured cells never appear as numeric
zeroes; diagnostic lanes (†) are drawn on every chart but are never ranked.
Stock vLLM's DSpark runner rejects pipeline parallelism; the PP2 DSpark lane
runs through the local five-file overlay described under the vLLM narrative
below (its greedy-equivalence check against the PP2 AR reference is pending).
One station was not attempted: 510,286,023,000 checkpoint bytes exceed
one GB300. The lane ledger is [`qualification.csv`](../data/qualification.csv).

## Tuning story

### Fabric first: pick the NCCL configuration before touching a model

- Host-side nccl-tests `all_reduce_perf` between the two GB300s, 64 MiB–2 GiB,
  20 iterations after 5 warm-ups, out-of-bounds check clean on every run.
- Dual rail with tuned ring/simple, 8 channels, 4 QPs per connection averaged
  **88.7 GB/s** bus bandwidth (93.7 GB/s at 2 GiB); 16 channels regressed to
  82.7 GB/s; NCCL auto reached 86.4 GB/s; a single rail averaged 47.2 GB/s.
- The tuned 8-channel dual-rail setting is what `config.env` exports
  (`NCCL_TUNING=tuned`, `NCCL_HCAS=mlx5_0,mlx5_1`).

### Step 1 — stock container RDMA path

- The preview image ships Ubuntu rdma-core 50. Its `libmlx5` lacks
  `mlx5dv_get_data_direct_sysfs_path`, so NCCL inside the container cannot
  open the ConnectX-8 Data Direct DMA device and logs
  `dlvsym failed on mlx5dv_get_data_direct_sysfs_path ... undefined symbol`.
- The same rank logs also carried `dlvsym failed on mlx5dv_reg_dmabuf_mr ...
  undefined symbol`, so DMA-BUF memory registration was unavailable on the
  stock path as well, not only Data Direct: the image's user-space RDMA stack
  was behind the host's on two fronts.
- Baseline SGLang TP2+EP2 prefill was flat at roughly 16.1K–17.2K aggregate
  prompt tok/s across 8K–128K and C1–C16 (`data/diagnostic-prefill.csv`,
  `ladder_step=1`): the cross-node all-reduce, not the GPUs, set the ceiling.
- Profile of one 32K prefill: NCCL all-reduce was 54.8% (node0) / 55.2%
  (node1) of summed GPU kernel time.

### Step 2 — host rdma-core overlay → Data Direct RDMA

- `tools/stage_host_rdma.sh` copies the host's MOFED `libibverbs.so.1`,
  `libmlx5.so.1`, `librdmacm.so.1`; `serve_node.sh`/`serve_vllm_node.sh`
  bind-mount them **over the image's real library files** (names resolved from
  the image at launch) plus `/usr/lib/aarch64-linux-gnu/libibverbs` and
  `/etc/libibverbs.d`.
- A first attempt that only placed the host libraries beside the image's did
  not work: NCCL still resolved the image `libmlx5` and logged the `dlvsym`
  failure above. Overlaying the real files makes every process pick them up,
  including workers spawned with a scrubbed environment.
- Evidence of the fast path: `NCCL INFO NET/IB: Data Direct DMA Interface is
  detected for device mlx5_0` (and `mlx5_1`) in the fabric logs, and
  `[send] via NET/IB/2/GDRDMA` in the server logs.
- C1 spot check (`diagnostic-prefill.csv`, run `quick-datadirect`): 26.3K
  prompt tok/s at 16K and 24.4K at 128K, versus 16.9K / 16.1K on the stock
  path. Full-grid gain: +54.2% at 64K prompts, C16 (16,621 → 25,638 aggregate
  prompt tok/s, `data/prefill.csv` versus `data/diagnostic-prefill.csv`).
- Profile of one 32K prefill after the overlay: NCCL share fell to
  29.2% (node0) / 29.6% (node1); every
  other category was unchanged within noise.
- This replay-off lane (`sglang_tp2_ep2_ar`) is the section's **numerically
  exact reference**: every prompt token goes through full prefill, and its
  decode lanes (AR and DSpark) were measured on this profile.

### Step 3 — SWA bounded replay

- `SWA_BOUNDED_REPLAY=1` adds `--enable-decoder-swa-bounded-replay` and
  forces `--cuda-graph-backend-prefill disabled` (the build refuses the two
  together). Everything else (Data Direct overlay, chunk 16,384, mem fraction
  0.80, same image) is identical to step 2.
- Result at 64K prompts, C16: **39,133** aggregate prompt tok/s (+135.4% vs
  step 1, +52.6% vs step 2). The full grid is 39.0K–40.4K at 16K–64K and
  37.7K at 128K (`data/prefill.csv`, lane `sglang_tp2_ep2_ar_replay`, 12/12
  points, 0 errors); C1 TTFT 0.415s / 0.827s / 1.680s / 3.480s at
  16K / 32K / 64K / 128K versus 0.623s / 1.257s / 2.563s / 5.373s replay-off.
- **Published as an accepted, rankable lane with a caveat.** Bounded replay of
  the sliding-window attention decoder is the deployment technique DeepSeek's
  V4.1 technical report describes for its own serving, and LMSYS validated it
  on this branch (AIME pass@1 unchanged, 453/480). It is nevertheless **not
  bit-identical** to full prefill: the replay-off Data Direct lane remains the
  numerically exact reference, both lanes stay separate series on every chart
  and separate columns in every table, and no number from one is ever
  substituted for the other. The headline quotes both.
- Decode was not re-measured on this profile: the flag changes prefill only,
  and the accepted decode lanes come from the replay-off server.

### Other knobs that were fixed rather than swept

- `--cuda-graph-max-bs-decode 64`: the derived default OOMs during decode
  graph capture (sglang#38839). Decode is measured with this cap.
- `--enforce-disable-flashinfer-allreduce-fusion`: the auto-enable assumes
  MNNVL on Blackwell; this pair has RoCE only.
- Engram host table `private` layout, pinned: the 189 GiB of n-gram tables do
  not fit next to the weights in two GB300s; `shared` is a memfd in one PID
  namespace and cannot span two hosts.
- `--mem-fraction-static 0.80` for the Data Direct sweeps: 0.85 left about
  37 GB of headroom and produced allocator OOM warnings at 32K–128K × C16
  (and preceded the SIGBUS below); 0.80 keeps about 50 GB for transient
  prefill buffers. The stock-path baseline (ladder step 1) ran at 0.85.
- `--context-length 262144` so 128K prompts plus one output token fit.
- vLLM: PP2 default (both Engram layers on stage 0, no cross-node traffic for
  them); TP2 run for parity; stock DSpark requires TP (PP is rejected by the
  DSpark runner), and the PP2 DSpark lane lifts that with a local overlay
  (`VLLM_PATCH_PP_DSPARK=1`, `recipes/patches/vllm-pp2-dspark/`, see below).
  FlashInfer autotune off, custom all-reduce off. PP2 needs
  the two local source patches described under Incidents
  (`VLLM_PATCH_PP=1`, `recipes/patches/vllm/`).

### vLLM TP1 × PP2: the other engine at the same points

- Server: `vllm/vllm-openai:deepseekv41-flash-0909` (image id
  `sha256:00d577a6…`), `--tensor-parallel-size 1 --pipeline-parallel-size 2`
  on vLLM's default even 20/20 layer split, `--language-model-only`,
  `--engram-config '{"cpu_offload": true}'`, `--max-num-batched-tokens 16384`,
  `--max-num-seqs 64`, `--max-model-len 262144`,
  `--gpu-memory-utilization 0.90`, prefix caching on (irrelevant for unique
  prompts), FlashInfer autotune off, custom all-reduce off, the same Data
  Direct overlay and NCCL tuning as SGLang, and the two local source patches.
  Rank 0 ran on node0 for this lane.
- Prefill (`data/prefill.csv`, lane `vllm_pp2_ar`, 12/12 points, 0 errors):
  16K 35,872 / 60,072 / 61,518; 32K 46,129 / 63,826 / 64,935; 64K 53,515 /
  65,338 / 65,966; 128K 55,992 / 62,750 / 63,048 aggregate prompt tok/s at
  C1 / C4 / C16; C1 TTFT 0.458 s / 0.710 s / 1.224 s / 2.340 s. At 64K / C16
  that is +68.6% over SGLang with SWA replay (39,133) and +157.3% over exact
  full prefill (25,638); for a single 128K request 55,992 versus 37,674 and
  24,394. The one point where SGLang is faster is a lone 16K request (replay
  39,549 versus 35,872): a single 16K request is one chunk, so it crosses the
  two pipeline stages one after the other, whereas a 128K request (eight
  chunks) or C4+ keeps both stages busy; vLLM's C1 cells climb from 35.9K at
  16K to 56.0K at 128K while the SGLang lanes are flat in prompt length.
- Decode (`data/throughput.csv`, C1–C64, 0 errors, none underfilled): 140.6
  aggregate / 141.3 per-user tok/s at C1 (ITL p50 7.08 ms) versus 106.8
  per-user for SGLang AR and 180.0 for SGLang DSpark; 2,805.9 aggregate tok/s
  at C64 versus 2,236.7 for SGLang AR (+25.4%). vLLM exposes no engine-step
  counter, so `engine_steps_per_second` is empty for this lane (and for
  SGLang AR, whose counter reads zero); it is never published as zero.
- Cache flush: the client's per-point flush call (`/reset_prefix_cache` for
  vLLM) answered HTTP 404 on this image. Every point still ran on fresh
  prompts (unique random ids seeded per request), and the server's
  `vllm:request_prefill_kv_computed_tokens_sum` delta equals the prompt-token
  total at all 12 points (`server_prompt_tokens_delta` in `prefill.csv`), so
  no prefill work was skipped by the prefix cache.
- Power: the GPU utilisation/power columns are sampled on node0, which under
  PP2 is pipeline stage 0. That GPU averaged 1,092–1,219 W at the C4/C16
  points (peaks up to 1,254 W) at 100% utilisation, against 733–879 W mean for
  the SGLang TP2+EP2 lanes, which spend part of every layer in the cross-node
  all-reduce. Stage 1 was not sampled.
- Sanity: the lane's chat checks answered 17 × 19 = 323 and the 3:40 pm to
  6:05 pm trip as 145 minutes (`logs/chat-vllm-pp2-ar.txt` in the private
  working directory).
- **TP2 on the same image** (lane `vllm_tp2_ar`, `--tensor-parallel-size 2
  --pipeline-parallel-size 1`, otherwise identical settings; launched with
  `SWAP_RANKS=1`, so vLLM rank 0 ran on node1): 16K 35,134 / 35,958 / 35,982; 32K 35,011 / 35,530 / 35,551; 64K 34,270 / 34,682 / 34,695; 128K 32,672 / 32,992 / 33,010 aggregate prompt tok/s
  at C1 / C4 / C16 (12/12 points, 0 errors); C1 TTFT 0.466 s / 0.936 s / 1.912 s / 4.011 s. Like the
  SGLang tensor-parallel lanes it is flat in concurrency, because every layer
  ends in a cross-node all-reduce, so the pipeline split wins by +71.4% for a
  single 128K request (55,992 versus 32,672) and +90.1% at 64K / C16 (65,966
  versus 34,695). TP2 also trails SGLang's SWA-replay lane (−11.3% at
  64K / C16, −13.3% for one 128K request) while beating exact full prefill
  (+35.3% and +33.9%). The sampled GPU (node0, TP rank 1 in this launch) drew
  972–1,039 W mean at the C4/C16 points. Its AR decode sweep (run
  20260910-180632, C1–C64, 0 errors) trails PP2 AR at every concurrency:
  102.3 versus 141.3 tok/s per user at C1 and 1,964.5 versus 2,805.9
  aggregate tok/s at C64 (−27.6% and −30.0%), and it also trails SGLang AR ·
  Data Direct at every concurrency (106.8 per user at C1, 2,236.7 aggregate
  at C64). Each decode step under TP2 ends every layer in a cross-node
  all-reduce, the same cost that keeps its prefill flat in concurrency.
- **TP2 DSpark on the same image** (lane `vllm_tp2_dspark`, the TP2 server
  restarted with `--speculative-config` dspark: 5 speculative tokens, block
  rejection sampling, adaptive verification on; run 20260910-221511, C1–C64,
  0 errors) is the fastest decode lane of the section at every concurrency:
  201.7 / 525.3 / 1,275.5 / 2,116.8 / 3,401.6 aggregate tok/s and
  201.0 / 134.9 / 82.0 / 68.6 / 55.3 tok/s per user at C1 / C4 / C16 / C32 /
  C64, with 2.56 accepted tokens per step at C1 and 2.23–2.43 from C4 up.
  Against the same server without speculation that is +96.5% per user at C1
  and +73.2% aggregate at C64; against vLLM PP2 AR, +42.3% and +21.2%; against
  SGLang DSpark, +11.7% per user at C1 and +42.2% aggregate at C64. ITL p50
  fell from 9.78 ms (TP2 AR) to 4.97 ms at C1 and from 31.68 ms to 18.07 ms at
  C64. Stock vLLM's DSpark runner rejects pipeline parallelism, so this is
  the only speculative vLLM lane the image supports as shipped; prefill was
  not measured on this server profile.
- **PP2 DSpark through a local overlay** (lane `vllm_pp2_dspark`, the TP1 × PP2
  server of the PP2 AR lane restarted with `--speculative-config` dspark: 5
  speculative tokens, block rejection sampling, adaptive verification **off**;
  `VLLM_PATCH_PP_DSPARK=1`, rank 0 on node0). Stock vLLM refuses DSpark under
  pipeline parallelism in three places (the model runner's aux-hidden-state
  guard, the draft loader, and config validation of the draft's parallel
  config), and even with those gates lifted the last stage never relayed the
  draft block it proposed to the first stage, so stage 0 would have embedded
  token id 0 at every draft position. The five-file overlay in
  `recipes/patches/vllm-pp2-dspark/` (each file beside its `.orig` and
  `.diff`, `PLAN.md` with the file-and-line analysis) relays the draft block
  from the last stage to the first over the existing PP sampled-token side
  channel, pads the sampled-token broadcast to a fixed `K + 1` width so a
  batch without drafts cannot mismatch the receiver, lets the draft load its
  own `embed.weight` on the last stage instead of aliasing the target's
  placeholder embedding there, and gives the draft a PP1 parallel config so
  validation accepts it. It requires async scheduling (the recipe default) and
  runs with adaptive verification off, because vLLM's validator rejects it
  under PP (confidences and cost curves exist only on the last stage), so this
  profile is fixed-K=5 DSpark where the TP2 DSpark lane ran adaptive
  verification on. It is an experiment on top of the two PP2 patches,
  GPU-tested only on this pair with this image and checkpoint; its CPU-only
  self-check in the image passes (36 checks). Decode (run 20260910-225453, C1–C64, 0 errors, none underfilled, the PP2 AR lane's server settings otherwise unchanged): 248.5 / 612.6 / 1,677.7 / 2,346.4 / 3,257.9 aggregate tok/s and 252.9 / 156.6 / 108.9 / 76.2 / 52.6 tok/s per user at C1 / C4 / C16 / C32 / C64, with 2.57 accepted tokens per step at C1 and 2.53–2.61 across the sweep (accept rate 0.31 of the 5 drafted tokens at every cell; the spec-decode counters reported 5 completed requests, 5 measured and 1 warm-up, at C1). It is the fastest lane of the section at C1 and from C4 to C32: +25.8% per user at C1 over vLLM TP2 DSpark and +79.0% over its own AR server (PP2 AR 141.3), +10.8% aggregate at C32 over TP2 DSpark; TP2 DSpark stays ahead at C2 (328.5 versus 320.9 aggregate) and at C64 (3,401.6 versus 3,257.9, -4.2%), so the peak-throughput headline keeps the TP2 lane. ITL p50 fell from 7.08 ms (PP2 AR) to 3.96 ms at C1 and from 22.77 ms to 18.99 ms at C64 (TP2 DSpark: 4.97 ms and 18.07 ms). vLLM exposes no engine-step counter, so `engine_steps_per_second` is empty. Sanity checks on the same server, all passed before the sweep: a greedy 256-token probe (the Fibonacci prompt, temperature 0, 256 completion tokens, 0 reasoning tokens), four concurrent chats (Hamlet summaries of 40–160 words, all coherent), a 24,019-token prompt that crossed the 16,384-token chunked-prefill boundary (a non-final chunk posts no relay) and answered correctly, an abort two seconds into a 3,000-word essay followed by an immediate `OK` on the next request (the freed-row path of the slot ring), spec-decode counters (`vllm:spec_decode_num_accepted_tokens_total` / `num_draft_tokens_total`) advancing on every cell, and the section's quality probe (17 × 19 arithmetic, the 3:40 pm to 6:05 pm reasoning question with thinking on, the Fibonacci code, a needle retrieved from a ~35K-token document at 23,838 prompt tokens, and a JSON array), all PASS. **Pending:** the token-for-token greedy diff of the 256-token probe against a PP2 autoregressive reference on the same image (block rejection sampling at temperature 0 is lossless, so any divergence would point at wrong drafts reaching the first stage or a wrong embedding); until that lands the lane is published on the strength of the sanity runs and the accept-rate counters only.

## Profiles

- Method: `tools/profile_prefill.sh <isl> <label>` starts the SGLang torch
  profiler, sends one 32K random-token `/generate` request, stops it, and
  summarises each rank's trace with `tools/analyze_trace.py` (categories by
  kernel-name regex; summed kernel time, not wall time).
- `data/profile.csv` holds both nodes for the stock path and the Data Direct
  path; `charts/kernel-time.png` stacks them.
- Reading: after the overlay the remaining large items are the mHC mixing
  statistics kernels (`_hc_mix_stats_partial_kernel`, "other"), the sparse
  attention forward, block-scaled dense GEMMs, and norm/quant elementwise
  work.
- **Next bottleneck.** With NCCL down to 29%, the largest single non-NCCL
  kernel in the trace is the hyper-connection (mHC) mixing-statistics Triton
  kernel: about 245 ms per 32K prefill over 240 launches (160 prefill calls
  of ~1.5 ms each plus 80 tiny decode calls), roughly 20% of summed kernel
  time. It is a split-K skinny GEMM (K = 20,480, 80 K-slices, 32-row tiles,
  tf32x3) whose 255 registers per thread limit it to two CTAs per SM; it moves
  under 1 GB per call and runs about 13× below the HBM bandwidth floor, so it
  is occupancy-bound, not bandwidth-bound. V4.1 routes every layer through the
  "pre from previous sublayer" scheme, which always takes these batch-invariant
  Triton kernels, so **no env var or server flag in this build changes it**:
  `SGLANG_OPT_USE_TILELANG_MHC_PRE`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM`, and
  `SGLANG_OPT_FUSE_MHC_POST_PRE` select code paths V4.1 never executes. One
  useful consequence: `SGLANG_OPT_USE_TILELANG_MHC_PRE=0` skips the ~285 s
  cold prewarm of 23 TileLang/DeepGEMM `mhc_pre` buckets that V4.1 does not
  run (startup only; prefill unchanged). A fix needs a small kernel change
  (fewer K-slices or a single-pass bandwidth-bound kernel) and a bitwise
  re-baseline; not attempted for this section. Two attention-side copies (the
  64-head padding of `q` for FlashMLA and the `wo_a` output re-layout, ~57 ms
  per 32K) are the next items and also have no toggles.

## Incidents

- **Retained HBM on node1 before the session.** 68,394 MiB used on the GB300
  with no process attached (boot id recorded in the private log). Per the
  station guide no GPU reset was attempted; the operator rebooted node1.
- **First full Data Direct sweep lost its server.** After the 32K/C4 point the
  API stopped answering; the client recorded 221 failed requests; the partial
  JSONL was set aside and none of it is published. Root cause: rank 1 (node1)
  died with a SIGBUS (`Fatal Python error: Bus error`) inside the FlashInfer
  `mxfp8_quantize` kernel launch during a 32K × C16 prefill, about three
  minutes after the CUDA caching allocator on that rank had warned that a
  7.5 GB allocation failed with 3.5 GB free (`--mem-fraction-static 0.85`),
  and shortly after a torch-profiler capture had been taken on the same server
  instance. Rank 0 only saw the detokenizer heart-beat stop. The combination
  was not reproduced after moving every SGLang lane to 0.80 and never
  profiling a production instance again; the accepted grids come from those
  relaunches. Recovery: containers removed on both hosts, idle-HBM gate
  re-run, node1 rebooted (see next item), relaunch.
- **node1 retains HBM after every teardown.** After each SGLang teardown node1
  (never node0, until the vLLM PP2 crash below) kept GB300 memory with no
  process attached: 4–5 GiB after a graceful `docker stop` (4,224 and
  4,135 MiB), 35–69 GiB after a `docker rm -f` teardown (34,890 / 35,087 /
  57,829 / 69,137 MiB), boot id unchanged, no compute process, no RDMA memory
  regions held. The retained memory is real: a launch that started with only
  4.5 GiB visibly retained lost 52 GB at weight-load time (rank 1 reported
  194.9 GB used versus 142.7 GB on rank 0) and failed the KV-budget check, so
  the "small" residue is not safe to ignore either. Recovery is a **normal OS
  reboot of node1 between lanes** (never a GPU reset): `stop_cluster.sh` now
  stops gracefully and waits for the driver, and `tools/ensure_clean_rank1.sh`
  reboots rank 1 when it still holds more than 512 MiB with no process and
  waits for a new boot id, docker, NFS, and ACTIVE rails. The threshold was
  lowered from tens of GiB to 512 MiB after the 4.5 GiB launch failure. After
  the vLLM PP2 crashes node0 also retained 82,292 and 66,258 MiB and needed
  operator reboots before the vLLM lanes could continue (see the vLLM items
  below).
- **vLLM PP2 first launch failed** with `DeepSeek V4 vision MoE routing
  requires input_ids`: vLLM's model runner hands non-first pipeline-parallel
  ranks `input_ids=None` (and none during the memory-profile dummy run), and
  the build treats that as fatal even though the ids are only used to find
  image-span tokens for the vision routing bias. Fix: a one-line text-only
  patch of `vllm/models/deepseek_v4/nvidia/model.py` (when the bias exists but
  `input_ids` is `None`, route every token as text), bind-mounted over the
  image file by `serve_vllm_node.sh` when `VLLM_PATCH_PP=1` (the default).
  Under `--language-model-only` there are no image tokens, so the patch
  changes no routing. The patched file and the unmodified `.orig` are in
  `recipes/patches/vllm/` with the same description in its README. The crash
  itself left node0 with 82,292 MiB retained (see above).
- **vLLM PP2 second launch died allocating the KV cache on stage 1**:
  `StopIteration` in `allocate_kv_cache` (`vllm/v1/worker/utils.py`) on
  `Worker_PP1`. Root cause: `_project_kv_cache_groups_to_worker` in
  `vllm/v1/core/kv_cache_utils.py` leaves the global, unfiltered
  `UniformTypeKVCacheSpecs` on a KV-cache group when a pipeline rank owns none
  of that group's layers. V4.1-Flash has exactly one such group, the three
  `CircularBufferSpec` compressor caches of kv-source layers 2, 8, and 14
  (`model.layers.{2,8,14}.attn.compressor.state_cache`), which all live on
  stage 0 under the even 20/20 split; `get_kv_cache_config_from_groups`
  therefore emitted tensors for stage-0 layers on stage 1, and the allocator
  found no group for them. Fix: the second local patch
  (`recipes/patches/vllm/kv_cache_utils.py`, unified diff beside it) emits
  tensors only for the layers named in the group's own `layer_names`; without
  PP every group already lists all of its layers, so the output is identical.
  The no-code alternative is an unbalanced split that gives both ranks a
  ratio-2 kv-source layer (`VLLM_PP_LAYER_PARTITION=14,26` or `8,32`); 20/20
  is the only balanced valid split because layers 21–39 must stay with
  kv-source layer 20. This crash left node0 with 66,258 MiB retained and the
  operator rebooted it; the relaunch with both patches produced the accepted
  lane.
- **vLLM rank 0 strands HBM on its host at every teardown, crash or not**:
  82,292 MiB after the `input_ids` crash, 66,258 MiB after the KV-allocation
  crash, and 51,470 MiB after the clean `docker stop` that ended the accepted
  PP2 lane (node1, which ran rank 1, showed 363 MiB). node0 has no automated
  reboot (only node1's is preauthorized), so the later vLLM lanes were queued
  with `SWAP_RANKS=1`, which starts vLLM rank 0 on node1 and rank 1 on node0
  (`config.env` swaps rail addresses and checkpoint paths and binds the API
  on the remote node; every script then uses `API_URL`). The first swapped
  launch failed before any GPU work because the remote rank re-sourced
  `config.env` with the already-swapped values and swapped them back
  (checkpoint index reported missing); the `SWAP_APPLIED` guard now applies
  the swap once, and the relaunch with it produced the accepted TP2 prefill
  lane. To rule out the transport, a seven-configuration host-side
  NCCL bisect (`results/fabric/bisect/*.log`: `all_reduce_perf` 256 MiB–1 GiB
  with dual/single rail, DMA-BUF on/off, GDR on/off, cuMem on/off, Data Direct
  on/off, 10 iterations each) changed idle HBM by at most 23 MiB per run on
  either host, so the retention is tied to the engine process teardown, not to
  the RDMA settings. The same bisect reproduced the transport ladder: 93.5 GB/s
  with Data Direct, DMA-BUF and GDR on; 27.8 GB/s with any one of DMA-BUF,
  GDR, or Data Direct off; 48.5 GB/s on a single rail.
- **Cold launch time.** First launch took about 11 minutes: 376 s weight load
  of which 285 s was the TileLang mHC compile; compiler caches under
  `cache/` make later launches faster, and `SGLANG_OPT_USE_TILELANG_MHC_PRE=0`
  would skip the compile outright (see Profiles).
- **Overlay attempt without file overlay** (see step 2): container came up
  without Data Direct; fixed by bind-mounting over the real library files.

## Limitations

- Prompts are unique random token ids, not natural text; there is no
  prefix-cache benefit and no quality claim. No perplexity or repetition audit
  was run for this section; the SWA replay lane's accuracy evidence is
  DeepSeek's report and LMSYS's AIME check, not a measurement made here.
- Aggregate prefill throughput divides all prompt tokens by the wave's wall
  time, so the last wave's tail lowers C4/C16 numbers slightly; they are not
  interchangeable with single-request cold prefill cells elsewhere in the
  repository.
- GPU utilisation and power are sampled on node0 only; for the vLLM PP2 lane
  node0 is pipeline stage 0 (the stage that holds both Engram layers), whose
  1,092–1,219 W mean draw at C4/C16 says nothing about stage 1. For the vLLM
  TP2 lane, launched with `SWAP_RANKS=1`, node0 is TP rank 1, which does the
  same work as rank 0.
- vLLM's cache-flush endpoint answered 404 on this image; the prompts are
  unique per request and the server's prefill-token counter matched the client
  at every point, so no cached prefix was reused, but the flush itself did not
  happen.
- The SGLang image is a locally built preview (`dev-dsv41`) with no recorded
  git commit; V4.1 support comes from the unmerged `dsv4.1` branch
  (sgl-project/sglang PR #38798). The vLLM image is a dev build
  (`0.1.dev20904+g179dd0fa9`) from the pending V4.1 PR.
- Each grid was measured once; no repeat-run variance is published.
- One station was not attempted: 510,286,023,000 checkpoint bytes exceed one
  GB300 and no capacity run was retained.
- Fabric numbers are host-side nccl-tests, not in-container measurements.
- The vLLM PP2 DSpark lane runs through a local five-file overlay that stock
  vLLM does not support: an experiment on top of the two PP2 patches, tested
  on GPUs only on this pair with this image and checkpoint, async scheduling
  only, adaptive verification off. Its equivalence evidence is the greedy
  256-token diff against a PP2 autoregressive server and the sanity runs
  listed under the vLLM narrative, not a quality benchmark.

## Open items

- mHC mixing-statistics kernel: about 245 ms of every 32K prefill sits in one
  occupancy-bound Triton split-K kernel with no runtime toggle; a fewer-slice
  or single-pass variant (bit-identical or re-baselined) is the next SGLang
  prefill step, followed by the two attention-side copies.
- Retained HBM after teardown has no root cause yet: node1 after SGLang
  teardowns (driver 595.84; its kernel log carried NVRM
  `refcntRequestReference_IMPL: Failed to enter state 1` entries during the
  session, no Xid) and node0 after every vLLM rank-0 teardown (NCCL/RDMA
  settings excluded by the bisect above). The OS-reboot-between-lanes
  workaround and `SWAP_RANKS=1` are in the recipe; the vendor question is
  open.
- Every vLLM lane is accepted (PP2 AR prefill and decode, TP2 AR prefill and
  decode, TP2 DSpark decode, PP2 DSpark decode through the overlay); no
  placeholders remain. Still open for the PP2 DSpark overlay lane: the greedy
  token-for-token diff against the PP2 AR reference, a prefill sweep on the
  same server, and adaptive verification under PP (it would need the
  confidence probabilities and cost curves broadcast to the first stage).
