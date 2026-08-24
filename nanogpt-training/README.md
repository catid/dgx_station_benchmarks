# nanoGPT training on DGX Station GB300

This section measures two related language-model training workloads on one and
two directly connected DGX Station GB300 systems:

1. The full
   [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
   FineWeb speedrun at revision `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
   It trains through all 1,285 steps and reports time to the official ≤3.28
   validation-loss target. The upstream 8×H100 record is 1.23 minutes.
2. Classic [nanoGPT](https://github.com/karpathy/nanoGPT) GPT-2 124M at
   revision `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. Short one- and two-node
   throughput runs retain its 491,520-token global batch and will be compared
   with the published approximately four-day 8×A100 training reference.

## Completed results

The first result is the complete modded-nanogpt run on each station
independently. `train_time` is the upstream synchronized benchmark timer; it
includes the training steps and excludes compilation, warmup, and validation.
The batch schedule processes 338,821,120 training tokens.

| System | Train time | Final val loss | Training throughput | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| gemini1, 1× GB300 | 427.841 s | 3.2777 | 791,932 tok/s | 39,402 / 51,810 MiB |
| gemini2, 1× GB300 | 427.893 s | 3.2771 | 791,836 tok/s | 39,402 / 51,810 MiB |
| gemini1 + gemini2, 2× GB300 DDP | 225.081 s | 3.2764 | 1,505,330 tok/s | 35,321 / 48,930 MiB per rank |

Both independent runs met the nominal ≤3.28 loss target and differed by only
52 ms, or 0.012%, in measured training time. The mean is 427.867 seconds and
791,884 training tokens/s. That time is 5.80× the current 73.8-second 8×H100
record. This is a hardware/software comparison rather than an official record
submission: the GB300 port uses the adaptations below, and two runs alone do
not satisfy the upstream repository's stricter statistical submission rule.

Two-node DDP reduces training time to 225.081 seconds: 1.901× faster than the
one-node mean, with 95.0% strong-scaling efficiency. It is 3.05× the official
8×H100 time and its 3.2764 final loss also clears the upstream single-run
statistical threshold. NCCL used both ConnectX-8 interfaces with RoCE,
GDRDMA, merged NICs, and eight distributed channels.

Including one-time compilation/warmup and validation, process elapsed time was
approximately 14m17s on gemini2 and 14m20s on gemini1. Those operational wall
times are not comparable with the official training-only timer.

Completed modded-nanogpt measurements are retained in
[`data/modded-1x-20260824.json`](data/modded-1x-20260824.json) and
[`data/modded-2x-20260824.json`](data/modded-2x-20260824.json).

## Classic nanoGPT throughput

The classic GPT-2 124M run retains the upstream 491,520-token global batch.
The benchmark captures 201 optimizer steps and summarizes steps 20--199,
excluding initial compilation/evaluation and the final evaluation.

| System | Mean step | Throughput | 600k-step extrapolation | Step-200 val loss |
|---|---:|---:|---:|---:|
| gemini2, 1× GB300 | 529.134 ms | 928,914 tok/s | 317,480 s / 3.67 days | 6.6194 |
| gemini1 + gemini2, 2× GB300 DDP | 267.378 ms | 1,838,293 tok/s | 160,427 s / 1.86 days | 6.5857 |

The extrapolation is close to and about 8.1% shorter than the upstream
repository's approximate four-day 8×A100 result. It is a steady-state
throughput comparison, not a claim that the 200-step sample converged to the
published approximately 2.85 OpenWebText loss. Machine-readable measurements
are in [`data/classic-1x-20260824.json`](data/classic-1x-20260824.json) and
[`data/classic-2x-20260824.json`](data/classic-2x-20260824.json).

Two-node DDP is 1.979× faster than one node with 98.95% strong-scaling
efficiency, even though the fixed measurement window includes one 433 ms
outlier. The two-node 600,000-step extrapolation is 1.86 days.

## GB300 software port

The workload preserves model math, data, batch schedule, optimizer schedule,
step count, and loss target. It uses NVIDIA's CUDA 13.3 / PyTorch 2.13
development container and these platform adaptations:

- compile the custom CUDA cross-entropy kernel for the attached architecture
  instead of hardcoded H100 `sm_90`;
- use NVIDIA's Blackwell-built FlashAttention 2 because the Hub FA3 v1 branch
  has no Grace/aarch64 build;
- give equal Q/K sequence metadata distinct tensor identity for current Dynamo
  full-graph tracing;
- deduplicate PyTorch's custom-Triton SSA reachability traversal and
  conservatively disable the affected epilogue-fusion eligibility decision;
- split FP8 post conversion and exact `amax` from a persistent TMA MLP kernel
  because Triton 3.6 rejects its fused scalar reduction on `sm_103`;
- supply a stable logging-only run ID and persist the upstream logs.

An isolated compiled test checks that the split FP8 path produces exactly the
same FP8 tensor and global maximum as the BF16 reference. Exact source,
container, dataset, and adapted-file hashes are in
[`data/provenance.json`](data/provenance.json).

## Distributed transport and safety

The two-node launch selects both ConnectX-8 ports (`mlx5_0` and `mlx5_1`). NCCL
reports RoCE, GDRDMA, merged NICs, and eight distributed channels. Every launch
is gated by a kernel-log danger scan and an idle-HBM ownership check on both
hosts. Named containers are explicitly removed after a run. GPU resets are
prohibited on this platform.

The launcher in [`recipes/run_modded_rank.sh`](recipes/run_modded_rank.sh)
captures the exact container devices, networking, NCCL settings, and torchrun
topology used here.
