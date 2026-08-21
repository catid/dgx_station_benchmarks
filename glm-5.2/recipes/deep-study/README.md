# GLM-5.2 deep optimization study

This directory stages and runs a pinned, fail-closed comparison of GLM-5.2
NVFP4 on two one-GPU DGX Station GB300 systems. The first frozen increment is
published under [`../../data/deep-study/`](../../data/deep-study/): P0 passed
the structural, capacity, and natural-output gates; P1 FlashInfer CUTLASS did
not fit the unchanged long-context envelope and is retained only as excluded
startup evidence.

The target is
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) at
revision `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`. The official BF16 and FP8
checkpoints are too large for the combined HBM of two Stations. CPU offload is
outside this study.

## First measured increment

P0 reproduced the exact TP2+EP2 CuTeDSL profile with 261,952 GPU KV-cache
tokens. The full 8K-input/up-to-1K-output decode matrix was:

| Backend | C1 | C2 | C4 | C8 | C16 | C32 | C64 | C128 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FlashInfer CuTeDSL | 68.2 | 117.9 | 218.3 | 361.0 | 539.7 | 903.7 | 1,252.2 | 2,013.9 | Accepted |

P0 standalone prefill measured 6,223 tok/s at 8K, 7,461 tok/s at 64K, and
7,179 tok/s at 128K. All four natural outputs finished normally with zero
automatic flags.

| FlashInfer CUTLASS start | Available KV GiB by rank | Required per rank | Status |
| --- | ---: | ---: | --- |
| Cold backend compile | 0.43 / 0.43 | 6.0 | Excluded before API; no requests |
| Warm AOT cache hit | 2.19 / 10.39 | 6.0 | Excluded before API; no requests; no further retry |

These are capacity dispositions at the fixed 135,168-token profile, not
throughput comparisons. The accepted and excluded records, source hashes, and
per-directory checksums are in
[`../../data/deep-study/`](../../data/deep-study/). The remaining matrix below
is still pending and must pass the same gates before publication.

## What is pinned

[`manifests/software.json`](manifests/software.json) records the exact model,
runtime images, runtime source commits, FlashInfer versions, benchmark commit,
and evaluator commit. [`manifests/checkpoints.json`](manifests/checkpoints.json)
records the exact revisions and dispositions of the official checkpoint
variants considered. The principal runtime pins are:

| Runtime | Version / source | Executed image |
| --- | --- | --- |
| vLLM | 0.27.1 / `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` | `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| SGLang | 0.5.17 / `29481685462732237d80d86076d6563e1f658102` | ARM64 manifest `lmsysorg/sglang@sha256:9310c4b1c590393399a6e7dacfec48a140e29bff1fb97b59341b21f95223be50` |

The experiment grid is data, not hidden shell branching:
[`manifests/experiments.json`](manifests/experiments.json). Inspect a resolved
profile without touching hardware:

```bash
python3 resolve_plan.py --list
python3 resolve_plan.py vllm-tp2-exact
python3 resolve_plan.py vllm-tp2-mtp5 \
  --winner-backend flashinfer_cutedsl
python3 resolve_plan.py --validate-all
python3 test_offline.py
```

All launch commands dry-run unless both `--execute` and
`ALLOW_GPU_EXECUTION=YES` are supplied.

## Site configuration

Copy [`site.env.example`](site.env.example) to a private location and set the
two hosts' existing SSH alias, rail addresses, interface/HCA names, model path,
cache path, result path, and pinned tool checkouts. These recipes inspect the
existing fabric but never configure it. Copy this `deep-study/` directory to
the same absolute path on rank 1 before execution.

Before the first run, both nodes must already have the verified checkpoint and
the selected digest-pinned image, Docker with NVIDIA CDI, the existing RoCE
interface/HCA, and key-based SSH. Rank 0 also needs the pinned benchmark
checkout and its Python environment. The deep-study scripts intentionally do
not pull images, download models, or alter network configuration.

The normal checkpoint recipe writes `PINNED_REVISION`. If an older but
byte-identical local copy lacks that marker, an operator may set
`REQUIRE_REVISION_MARKER=no`; the verifier still requires all 47 shards, exact
weight and index byte counts, the expected architecture, and NVFP4 metadata.

```bash
export SITE_CONFIG=/private/path/glm52-deep-study.env
export ALLOW_GPU_EXECUTION=YES
```

Every launch first calls [`preflight_idle_hbm.sh`](preflight_idle_hbm.sh), or
the site-selected compatible preflight. It requires exactly one GB300 on each
host, no compute owner, and idle HBM below the declared ceiling. A failed
preflight stops the run; there is no reset or automatic retry path. Each
profile has a deterministic `glm52-deep-<profile>` container name and refuses
to overwrite an existing container or result directory.

## Ordered experiment matrix

### 1. Exact TP2 reproduction

This reproduces the accepted vLLM configuration: TP2/PP1, expert parallel,
FP8 E4M3 KV, FlashInfer CuTeDSL MoE, FlashInfer autotune off, 32,768 batched
tokens, 135,168 maximum model length, and 93% static HBM utilization.

```bash
bash run_headline.sh vllm-tp2-exact --execute
```

The headline workload is exact 8K input/up-to-1K output at C1–C128 plus cold
8K/64K/128K prefill. Its minimum 139,264-token KV gate is the workload's
shared-prefix requirement: 8,192 shared prompt tokens plus 128 × 1,024 output
tokens. It is not a claim about 128 unrelated 8K prompts.

An empty profile cache can make the first launch a nonreportable compiler-cache
priming pass. A cold AOT/CuTeDSL/Triton build may temporarily consume more
device memory than the subsequent cache-hit launch. Never relax the capacity
gate or raise utilization to rescue that attempt. Preserve its logs; after the
named containers exit, run the kernel and idle-HBM gates. If ownerless HBM is
above the clean-idle ceiling, stop GPU work and coordinate a normal reboot.
The generated on-disk compiler cache can be retained, but the next attempt is
still accepted only if the unchanged profile passes every capacity and quality
gate from a clean post-reboot baseline.

Record the driver and coherent-memory mode with every run. NVIDIA documents
that R610 and later default DGX Station GB300 to driver-managed coherent GPU
memory (CDMM), while older/legacy NUMA mode can place ordinary allocations in
HBM and is not recommended. A driver or memory-mode change is a platform A/B,
not a silent optimization inside a backend comparison. The memory snapshots
record `CoherentGPUMemoryMode`, memory-tier node lists, per-node `FilePages`,
and the boot ID so HBM accounting changes can be separated from kernel/backend
effects. See NVIDIA's
[mixed-coherency guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html)
and [optimization guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/optimization.html).

### 2. CuTeDSL versus CUTLASS

```bash
bash run_matrix.sh backends --execute
```

The three arms are `flashinfer_cutedsl`, `flashinfer_cutlass`, and `cutlass`.
The last name is vLLM's CUTLASS backend; it is not FlashInfer CUTLASS. Attention
is deliberately held on runtime auto-selection, so this isolates the MoE
runner. FlashInfer CUTLASS has an explicit `(M,K)` workspace. A failure to fit
the same workload envelope is a capacity result, not permission to shrink the
workload and publish a nominally comparable throughput number.

After selecting the winner from accepted, zero-error runs, record that choice
before looking at the autotune results and run:

```bash
bash run_matrix.sh autotune \
  --winner-backend flashinfer_cutedsl --execute
```

Substitute the actual winner. Autotune is never enabled for PP2.
If vLLM's own `cutlass` backend wins, record FlashInfer autotune as not
applicable instead of running the on arm; the resolver rejects that pairing.

### 3. Native MTP

```bash
bash run_matrix.sh mtp \
  --winner-backend flashinfer_cutedsl --execute
```

This runs native vLLM MTP at N=0,1,2,3,5 on TP2. It records the benchmark's
acceptance length, draft count, position acceptance, and target-step metrics.
The validator rejects an MTP run with missing acceptance telemetry. Compare
the four deterministic natural-output hashes exactly with MTP0:

```bash
python3 validate_quality.py \
  "$RESULT_ROOT/vllm-tp2-mtp5/quality/quality-audit.json" \
  --control-quality \
  "$RESULT_ROOT/vllm-tp2-mtp0/quality/quality-audit.json"
```

For WikiText-2, run the parent recipe against identically configured candidate
and control servers. Choose and record an acceptable relative PPL tolerance
before viewing the candidate, then pass that value explicitly:

```bash
python3 validate_quality.py CANDIDATE/quality/quality-audit.json \
  --control-quality CONTROL/quality/quality-audit.json \
  --wikitext-result CANDIDATE/wikitext2/results.json \
  --control-wikitext CONTROL/wikitext2/results.json \
  --max-relative-ppl-delta PREDECLARED_VALUE
```

The recipe intentionally supplies no default PPL tolerance.

### 4. Balanced PP2

The safetensor payload audit found that a default 39/39 layer count is not a
balanced byte split because GLM-5.2 begins with three much smaller dense
layers. The staged 40/38 split is approximately 205.58 GiB versus 208.75 GiB
after embedding/head assignment, before runtime buffers and repacking.

```bash
bash run_headline.sh vllm-pp2-balanced-bootstrap --execute
bash run_headline.sh vllm-pp2-balanced --execute
```

The bootstrap is explicitly nonreportable and proves only that partitioning
and startup work. PP2 always has speculation and distributed FlashInfer
autotuning disabled. The reportable profile restores the same headline
workload envelope.

### 5. SGLang 0.5.17

```bash
bash run_matrix.sh sglang --execute
```

The TP2 arm uses EP2. The PP2 arm uses EP1, `SGLANG_PP_LAYER_PARTITION=40,38`,
and `--disable-overlap-schedule`, which SGLang 0.5.17 requires for pipeline
parallelism. Speculative decoding is omitted from both stable SGLang arms.

### 6. Fixed prefill chunk sweep

```bash
bash run_matrix.sh prefill-vllm \
  --winner-backend flashinfer_cutedsl --execute
bash run_matrix.sh prefill-sglang --execute
```

The values are 4,096, 8,192, 16,384, and 32,768. vLLM varies
`--max-num-batched-tokens`; SGLang varies `--chunked-prefill-size`. These
profiles use `llm-inference-bench --prefill-only` at 8K/64K/128K. Dynamic
SGLang chunking is a later experiment after a fixed-size winner is known.

## Result gates

[`run_headline.sh`](run_headline.sh) saves the resolved plan before launch,
captures low-overhead RoCE counters immediately before and after the workload,
runs the benchmark and natural-output audit, preserves both container commands
and logs, sends SIGTERM to both ranks concurrently, waits for clean exits, and
then removes only the stopped named containers. It also records host
NUMA/cgroup/page-cache snapshots before, during, and after the run, repeats the
kernel-danger scan, enforces the post-teardown idle-HBM gate, and calls
[`validate_run.py`](validate_run.py). The validator rejects:

- a different checkpoint revision, image, topology, backend, KV type, or
  profile command;
- verbose NCCL logging in a headline run;
- CPU offload, PP without 40/38, PP plus speculation, or quarantined SGLang
  speculation;
- insufficient KV capacity, missing C cells, request errors, queued/capacity-
  limited cells, nonpositive measurements, or missing speculative telemetry;
- empty, truncated, mechanically degenerate, or otherwise flagged natural
  outputs.

Automatic checks do not replace reading the full saved natural outputs.

## Network diagnosis without perturbing headline runs

Headline containers set `NCCL_DEBUG=WARN`. Before/after snapshots read netdev,
RDMA, hardware, and `ethtool` counters; no profiler or verbose collective log
runs concurrently with a headline measurement. Review
`network/delta.json` for PFC/pause, retransmit, RNR, discard, timeout, and error
deltas.

Use a separate 5–15 second diagnostic at C1, C16, or C128:

```bash
bash run_nccl_diagnostic.sh vllm-tp2-exact 16 --mode info --execute
bash run_nccl_diagnostic.sh vllm-tp2-exact 16 --mode trace --execute
```

The INFO run records transport, GDR, algorithm/protocol, and channel selection.
The TRACE run is never reportable; it extracts collective operation/count/type
and payload bytes to `collective-sizes.tsv`. If the pinned NCCL log format does
not expose sizes, preserve the logs and use a 5–10 second Nsight Systems NCCL
capture as documented by NVIDIA rather than extending tracing across a full
benchmark.

Replay only the observed sizes with the official pinned `nccl-tests` build:

```bash
export MANAGEMENT_IFACE=YOUR_MANAGEMENT_INTERFACE
export ALLOW_NCCL_REPLAY=YES
bash replay_nccl_sizes.sh TRACE/collective-sizes.tsv TRACE/replay
```

Replay leaves NCCL algorithm, protocol, channels, and queue pairs on auto. The
large-message Ring/Simple tuning used to qualify a 2 GiB transfer is not copied
into model-serving baselines. The staged host `nccl-tests` build uses NCCL
2.31.2, while the pinned vLLM log reports NCCL 2.30.7+cuda13.3, so replay is a
size-matched fabric diagnostic rather than a runtime-identical model result.
Diagnose bandwidth only when NCCL spans dominate
the critical path and measured rail bytes approach the independently qualified
link rate. Low utilization with frequent serialized collectives indicates a
latency/launch problem; high GPU utilization with overlapped NCCL indicates a
compute or kernel limit. PP2 improving while critical network work falls is
stronger causal evidence than bulk-link throughput alone.

## Explicit exclusions

[`manifests/exclusions.json`](manifests/exclusions.json) is authoritative. In
particular:

- stock vLLM 0.27.1 does not have completed DSpark support
  ([issue](https://github.com/vllm-project/vllm/issues/50851),
  [open PR](https://github.com/vllm-project/vllm/pull/50694));
- the released NVFP4-target DSpark draft is not interchangeable with the FP8-
  target preview, and neither is enabled in this stable matrix
  ([NVFP4 draft](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark),
  [FP8 preview](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark-preview));
- the official DFlash catalog has no GLM-5.2 checkpoint
  ([DFlash docs](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md));
- no released compatible external EAGLE3 draft was found; native MTP is not
  relabeled as EAGLE3;
- PP2 plus speculation is excluded in pinned vLLM and rejected by pinned
  SGLang ([vLLM #49355](https://github.com/vllm-project/vllm/issues/49355),
  [vLLM #52069](https://github.com/vllm-project/vllm/issues/52069),
  [SGLang check](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/server_args.py#L8277-L8280));
- SGLang TP2 speculative decoding remains quarantined
  ([issue #30296](https://github.com/sgl-project/sglang/issues/30296),
  [open PR #30642](https://github.com/sgl-project/sglang/pull/30642));
- TP2+PCP2 needs four ranks, `flashinfer_b12x` targets SM120/121 rather than
  GB300 SM103, AMD MXFP4 targets ROCm, and `fp8_ds_mla` draft caching is not
  part of the baseline.

Primary backend and topology references are the
[vLLM CuTeDSL source](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py),
[vLLM FlashInfer CUTLASS source](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py),
[vLLM MTP guide](https://github.com/vllm-project/vllm/blob/v0.27.1/docs/features/speculative_decoding/mtp.md),
[SGLang GLM-5.2 cookbook](https://github.com/sgl-project/sglang/blob/v0.5.17/docs/cookbook/autoregressive/GLM/GLM-5.2.mdx), and
[SGLang PP guide](https://github.com/sgl-project/sglang/blob/v0.5.17/docs/docs/advanced_features/pipeline_parallelism.mdx).
