# DeepSeek-V4.1-Flash notes

Working notes kept out of the headline deck: how the two-station SGLang and
vLLM servers were tuned, what the profiles showed, what went wrong, and what
the numbers do not claim. Generic names only: `node0` is rank 0 (API host),
`node1` is rank 1. Placeholders in `{{...}}` are listed in
[`PLACEHOLDERS.md`](PLACEHOLDERS.md).

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
  them); TP2 measured for parity; DSpark requires TP (PP is rejected by the
  DSpark runner). FlashInfer autotune off, custom all-reduce off. PP2 needs
  the one-line text-only source patch described under Incidents
  (`VLLM_PATCH_PP=1`, `recipes/patches/vllm/`).

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
  the vLLM PP2 crash node0 also retained 82,292 MiB and needed an operator
  reboot before the vLLM lanes could continue.
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
  itself left node0 with 82 GiB retained (see above).
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
- GPU utilisation and power are sampled on node0 only.
- The SGLang image is a locally built preview (`dev-dsv41`) with no recorded
  git commit; V4.1 support comes from the unmerged `dsv4.1` branch
  (sgl-project/sglang PR #38798). The vLLM image is a dev build
  (`0.1.dev20904+g179dd0fa9`) from the pending V4.1 PR.
- Each grid was measured once; no repeat-run variance is published.
- One station was not attempted: 510,286,023,000 checkpoint bytes exceed one
  GB300 and no capacity run was retained.
- Fabric numbers are host-side nccl-tests, not in-container measurements.

## Open items

- mHC mixing-statistics kernel: about 245 ms of every 32K prefill sits in one
  occupancy-bound Triton split-K kernel with no runtime toggle; a fewer-slice
  or single-pass variant (bit-identical or re-baselined) is the next SGLang
  prefill step, followed by the two attention-side copies.
- node1 retained HBM after teardown has no root cause yet (driver 595.84;
  its kernel log carried NVRM `refcntRequestReference_IMPL: Failed to enter
  state 1` entries during the session, no Xid); the OS-reboot-between-lanes
  workaround is in the recipe, the vendor question is open.
- vLLM TP2 and TP2 DSpark lanes are pending: their prefill/decode grids fill
  the remaining placeholders once they complete error-free. The PP2 lane
  (launched with the text-only patch) is accepted and published.
