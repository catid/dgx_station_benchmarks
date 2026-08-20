# Reproducing the Hy3-FP8 experiment

This recipe verifies the official checkpoint, proves the one-GB300 capacity
result, and runs the two-node `llm-inference-bench` matrix. It assumes one
server-class GB300 per DGX Station and deliberately excludes the display GPU.

> **GB300 safety:** never run `nvidia-smi --gpu-reset`, PCI unbind/rescan, or
> unload/reload NVIDIA modules. A failed distributed launch is cleaned up by
> removing the named containers on both hosts. If the driver remains unhealthy,
> stop GPU work and coordinate a host reboot with the operator. Never reboot it
> automatically.

Run the commands from the experiment directory on node 0 unless a step says to repeat it on both nodes:

```bash
cd /absolute/path/to/dgx_station_benchmarks/hy3
```

## 1. Prerequisites

On both identical machines:

- NVIDIA driver 595.84 or a compatible newer server driver
- Docker with NVIDIA CDI and permission to run `docker` without an interactive prompt
- Python 3.12, Git, `curl`, `jq`, `rsync`, and roughly 330 GB free disk
- `enP1p3s0f0np0` configured with an isolated point-to-point `/30`; this recipe uses example addresses `192.168.200.1/30` on node 0 and `192.168.200.2/30` on node 1
- `mlx5_0` mapped to that interface and `/dev/infiniband/uverbs0` available
- Passwordless SSH from node 0 to node 1

Confirm the hardware and live rail:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
ibdev2netdev
ip -br address show enP1p3s0f0np0
ping -c 3 192.168.200.2
```

The measured systems report exactly 256,703 MiB usable HBM per GB300. Marketing capacity is not a substitute for this runtime-visible value.

## 2. Download and verify the pinned checkpoint

Create the same model path on both machines. This example keeps weights outside the Git checkout:

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install 'huggingface_hub[cli]'

export HF_BIN="$PWD/.download-venv/bin/hf"
export MODEL_DIR=/absolute/path/models/Hy3-FP8
recipes/download-and-verify.sh
```

The script pins revision `ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b`, asks Hugging Face to verify all remote checksums, and independently requires:

- 101 indexed and present shards
- 299,889,838,946 aggregate safetensors bytes
- no missing or empty shard

Repeat on node 1, or copy the verified directory over the 400-GbE rail and run `hf cache verify` plus `verify-checkpoint.py` there. Do not start a distributed server with partial weights or different revisions.

## 3. Reproduce the 1x capacity result

```bash
recipes/check-capacity.sh "$MODEL_DIR"
```

Exit status 2 means the complete weight set cannot fit. On the measured GB300 it prints a 28.608-GiB weight-only shortfall. This is the complete 1x result; do not replace it with a CPU-offload throughput number.

## 4. Install pinned runtime and benchmark

On both nodes:

```bash
docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

On node 0:

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git tools/llm-inference-bench
git -C tools/llm-inference-bench checkout 0b4185b5b435e948b199c9077a00b084864aa963
python3 -m venv .bench-venv
.bench-venv/bin/pip install httpx rich psutil matplotlib

export BENCH_DIR="$PWD/tools/llm-inference-bench"
export BENCH_PYTHON="$PWD/.bench-venv/bin/python"
```

## 5. Build the MTP compile-fix image

The pinned vLLM image leaves `HYV3MTP` eager. The included Dockerfile adds the same `@support_torch_compile` decorator used by vLLM's DeepSeek MTP implementation and changes no model weights or numerical settings.

Build it on both nodes:

```bash
recipes/build-mtp-image.sh
ssh node1 /absolute/path/to/repo/hy3/recipes/build-mtp-image.sh
```

The expected local tag is `hy3-vllm:0.27.1-mtp-compile`. Preserve the build log and inspect the two patched lines printed at the end.

## 6. Launch TP1/PP2 or TP2/PP1

Keep the repository and model at identical absolute paths on both nodes, then run from node 0:

`launch-cluster.sh` first runs `preflight-idle-hbm.sh` on both hosts. The
measured normal baseline is 0 MiB per GB300; the default tolerance is 2,048
MiB. Override the baseline only with a previously recorded healthy idle value,
not with the current allocation:

```bash
LOCAL_IDLE_HBM_BASELINE_MIB=0 REMOTE_IDLE_HBM_BASELINE_MIB=0 \
  recipes/preflight-idle-hbm.sh
```

If the preflight finds tens of GiB above baseline, the launch is blocked. While
the driver is known healthy, use compute-app queries, one `pmon` pass, `fuser`,
and `lsof` to identify an owner. If none exists, stop GPU work and coordinate a
normal reboot of both hosts. Never increase `GPU_MEMORY_UTILIZATION` to hide
retained HBM, and never reset, unbind, reload modules, or reboot automatically.

```bash
export MODEL_DIR=/absolute/path/models/Hy3-FP8
export REMOTE_HOST=node1
export REMOTE_RECIPE_DIR=/absolute/path/to/repo/hy3/recipes

# Compatibility baseline: TP1 / PP2, no speculative decoding.
PARALLEL_MODE=pp MTP_TOKENS=0 recipes/launch-cluster.sh
```

The worker launches first in headless mode. The API listens only on node 0 at `127.0.0.1:30000`. Follow both logs until healthy:

```bash
docker logs -f hy3-fp8-vllm
ssh "$REMOTE_HOST" 'docker logs -f hy3-fp8-vllm'
timeout 20m bash -c \
  'until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done'
```

PP2 is launched with `--no-enable-flashinfer-autotune`. In the pinned runtime,
the unequal pipeline stages enter different distributed tuning sequences: PP1
can finish while PP0 remains blocked, and the API never becomes healthy. The
flag disables the startup tuner, not FlashInfer kernels. Both ranks must log
`Skipping FlashInfer autotune because it is disabled`.

TP2 is launched with `--enable-expert-parallel`. Without it, the checkpoint
loads but replicated MoE experts exhaust both GB300s during warmup. Do not
remove that flag to chase a plain-TP comparison.

The default `GPU_MEMORY_UTILIZATION=0.956` is the accepted baseline profile.
It must be used only after the mandatory idle-HBM preflight succeeds on both
nodes. Preserve the exact value in runtime evidence:

```bash
GPU_MEMORY_UTILIZATION=0.956 PARALLEL_MODE=pp MTP_TOKENS=0 \
  recipes/launch-cluster.sh
```

If health times out, capture both logs and run `recipes/stop-cluster.sh`. Do not
reset the GPU to reclaim HBM.

A prior ownerless-residual-HBM observation is the reason for the mandatory
preflight, but its raw terminal capture was not retained and it is not published
as checksummed evidence. If idle HBM is materially above the healthy baseline
after the named containers exit and no userspace owner is found, stop; do not
increase utilization, reset a device, reload modules, perform PCI recovery, or
retry. Coordinate clean-baseline recovery with the operator.

On the first launch, verify the logs identify `mlx5_0` and the NCCL IB/GDR transport rather than a socket fallback:

```bash
docker logs hy3-fp8-vllm 2>&1 | grep -E 'mlx5_0|GDRDMA|Data Direct|NET/IB'
ssh "$REMOTE_HOST" 'docker logs hy3-fp8-vllm 2>&1' \
  | grep -E 'mlx5_0|GDRDMA|Data Direct|NET/IB'
```

`NCCL_TUNING=auto` pins the known-good interface/HCA while leaving NCCL's algorithm selection automatic. `NCCL_TUNING=tuned` additionally applies the bulk-transfer RING/SIMPLE/8-channel/4-QP settings measured in this repository's networking experiment. Treat it as an A/B test: settings that maximize a 2-GiB all-reduce may not minimize decode's small-collective latency.

## 7. Run the supported matrix

The pinned runtime supports four publishable configurations: PP2/MTP0 and
TP2+expert-parallel MTP0/MTP1/MTP2. PP2/MTP1 and PP2/MTP2 are unsupported:
`HYV3MTPModel` lacks the pipeline-parallel `SupportsPP` interface. The launcher
refuses those combinations rather than applying an unvalidated semantic patch.

For each supported configuration, launch the server, wait for health,
benchmark, audit natural output, capture runtime evidence, and stop both named
containers. Verify the clean idle-HBM baseline again before the next launch.

```bash
# Example: PP2 / MTP0
GPU_MEMORY_UTILIZATION=0.956 PARALLEL_MODE=pp MTP_TOKENS=0 \
  recipes/launch-cluster.sh
timeout 20m bash -c \
  'until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done'
recipes/benchmark.sh pp 0
"$BENCH_PYTHON" recipes/natural-quality-audit.py \
  --topology pp2 --mtp-tokens 0 \
  --output results/hy3-fp8-pp2-mtp0/natural-quality-audit.json
recipes/capture-runtime.sh pp 0
recipes/stop-cluster.sh

# Repeat the same benchmark/audit/capture/stop sequence for TP2 + EP.
GPU_MEMORY_UTILIZATION=0.956 PARALLEL_MODE=tp MTP_TOKENS=0 \
  recipes/launch-cluster.sh

# MTP1 selects the derived compile-fix image automatically.
GPU_MEMORY_UTILIZATION=0.956 PARALLEL_MODE=tp MTP_TOKENS=1 \
  recipes/launch-cluster.sh

# MTP2 needs the clean-idle speculative profile that passed the KV guard.
GPU_MEMORY_UTILIZATION=0.958 PARALLEL_MODE=tp MTP_TOKENS=2 \
  recipes/launch-cluster.sh
```

For each TP server, use benchmark labels `tp 0`, `tp 1`, or `tp 2`; pass the
same topology and MTP depth to the quality audit and `capture-runtime.sh`. The
launcher automatically adds expert parallel.

The fixed 1,179,904-token KV guard covers 128 full 8K+1K requests plus the
configured CUDA-graph and scheduling reserve. `benchmark.sh` parses the global
capacity from the healthy server log and exits before sending any request if it
misses the guard. MTP2 at 0.956 exposed 1,175,760 tokens, 4,144
below the guard; its one clean-idle 0.958 profile exposed 1,187,056. The 0.958
value is not a general recommendation and must never compensate for residual
HBM.

The fixed workload is:

- 8,192 exact input tokens
- 1,024 output tokens, temperature 0
- C1, C2, C4, C8, C16, C32, C64, and C128
- Standalone cold prefill at 8K, 64K, and 128K
- FP8 E4M3 KV cache
- Server maximum length 262,144, so 128K plus output is valid

### Reproduce the MTP1 C128 stability confirmation

The accepted 0.956 MTP1 matrix contains the measured C128 cliff. A separate
60-second run confirmed it under a clean-idle 0.958 profile. Start TP2/MTP1 at
0.958, then use a distinct directory so the canonical matrix is not overwritten:

```bash
GPU_MEMORY_UTILIZATION=0.958 PARALLEL_MODE=tp MTP_TOKENS=1 \
  recipes/launch-cluster.sh
timeout 20m bash -c \
  'until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done'
export RESULT_RUN_NAME=hy3-fp8-tp2-mtp1-c128-confirm60-util0958
CONCURRENCIES=128 DURATION=60 PREFILL_CONTEXTS=8k \
  recipes/benchmark.sh tp 1
"$BENCH_PYTHON" recipes/natural-quality-audit.py \
  --topology tp2 --mtp-tokens 1 \
  --output "results/$RESULT_RUN_NAME/natural-quality-audit.json"
recipes/capture-runtime.sh tp 1
recipes/stop-cluster.sh
unset RESULT_RUN_NAME
```

The confirmation is accepted only if all 128 requests are resident, errors are
zero, and the same fixed KV guard passes. It is retained in
`stability-confirmation.csv`, not substituted into the canonical 30-second row.

## 8. Validate before publishing

For every raw benchmark JSON:

- require zero request errors and exact token counts for every completed request;
- in duration mode, allow the requests still in flight at the 30-second cutoff, and
  retain both submitted and completed counts;
- preserve offered and effective concurrency, maximum running requests, queue metrics, and capacity-limited status;
- retain draft and accepted speculative-token counters, and calculate the
  acceptance rate from those counters rather than client text;
- retain the exact model revision, runtime image, topology, MTP depth, cache dtype, server log, and benchmark commit;
- confirm the natural audit contains coherent answers rather than repeated filler;
- manually inspect every natural output, with special attention to outputs
  flagged by repeated 8-grams, long identical-character runs, or repeated words;
- keep TP2/MTP1 C128 and TP2/MTP2 C32–C128 out of headline selection. These are
  accepted, zero-error measurements, but exhibit reproducible throughput cliffs.

The CSV field definitions and artifact layout are in [`../data/README.md`](../data/). Do not populate headline tables or charts from a failed, partial, or degenerate run.

## 9. Runtime cleanup

`stop-cluster.sh` removes only the named `hy3-fp8-vllm` containers on both
hosts. It does not remove model weights, caches, images, or result files.
Preserve compilation caches between comparable restarts unless the experiment
explicitly tests a cold runtime.

Never use GPU reset as cleanup. If container removal succeeds but the healthy
driver still reports a process, identify and stop that userspace process. If
the driver reports ATS-peer removal failures, `RmInitAdapter failed`,
`NV_ERR_INVALID_STATE`, an NVIDIA kernel oops, or an Xid that leaves the device
unavailable, issue no more NVIDIA ioctls and coordinate a reboot with the user.

## 10. Extract publication data and render charts

The extractor recognizes only these exact result directory names:

```text
hy3-fp8-pp2-mtp0-092-provisional
hy3-fp8-pp2-mtp0-attempt2-hung
hy3-fp8-pp2-mtp0-attempt3-kv602176
hy3-fp8-tp2-mtp0-attempt1-no-ep-oom
hy3-fp8-pp2-mtp1-attempt1-unsupported-mtp-pp
hy3-fp8-tp2-mtp1-c128-confirm60-attempt1-kv1179216
hy3-fp8-tp2-mtp2-attempt1-kv1175760-util0956
hy3-fp8-tp2-mtp1-c128-confirm60-util0958
hy3-fp8-pp2-mtp0  hy3-fp8-pp2-mtp1  hy3-fp8-pp2-mtp2
hy3-fp8-tp2-mtp0  hy3-fp8-tp2-mtp1  hy3-fp8-tp2-mtp2
```

The first name is the one intentional exception: it ingests the preserved 0.92
PP2/MTP0 C1-C64 run as `provisional_tuning` and requires `runtime-092/`. It has
no C128 or quality result, and the extractor never manufactures either.

The named failed attempts are evidence only. The extractor validates the
distributed-autotune deadlock, plain-TP2 OOM, PP+MTP incompatibility, and
rejected KV-capacity profiles from retained logs and settings. It never emits
performance rows for those attempts or incorporates an unretained teardown
summary into the evidence manifest.

The `...confirm60-util0958` directory is accepted only as a C128 stability
confirmation. It requires 60 seconds, C128 only, 8K prefill only, the 0.958
profile, full residency, zero errors, natural output, and complete runtime
evidence.

A canonical configuration is consumed as `accepted` only after it has a valid
complete C1-C128 `llm-inference-bench.json`, a natural-quality JSON, and the
canonical `runtime/` directory produced by `capture-runtime.sh`. Other tuning
attempts can remain in separately named directories because the extractor does
not search them. Run:

```bash
"$BENCH_PYTHON" recipes/extract-results.py \
  --results-root /absolute/path/to/frontier-bench/results
"$BENCH_PYTHON" charts/render-charts.py
```

The extractor rebuilds the performance CSVs from provisional and accepted
configurations and writes compact, checksummed evidence under `data/evidence/`
and `data/runtime/`. It prints the exact directories consumed, calls PP2
MTP1/MTP2 unsupported, and explains genuinely pending configurations. The
chart renderer gives provisional rows a dashed line and marks measured
high-concurrency cliffs with × rather than joining them into headline curves.

Before calling a quality result reviewed, read every preserved response in `data/evidence/<topology>-mtp<N>-quality.json`, then add a record to `data/manual-quality-review.json`:

```json
{
  "pp2-mtp0": {
    "status": "clean",
    "notes": "Read all 12 outputs; no incoherence or degeneration observed."
  }
}
```

Rerun extraction so the manual status enters `quality-summary.csv`. Never infer a clean manual review solely from zero automatic flags.
