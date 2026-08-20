# Reproducing the MiniMax H3 video benchmark

These recipes target one or two DGX Station GB300 systems running Linux/ARM64. They use the official FL2VA checkpoint, keep all model components resident, and do not use CPU or layerwise offload.

Install Docker with NVIDIA CDI support, the Hugging Face `hf` CLI, `curl`, GNU coreutils/findutils, `lsof`, and FFmpeg (`ffmpeg` plus `ffprobe`) before running the recipes. The media validator requires a build of FFmpeg that includes H.264, AAC, `blackdetect`, `freezedetect`, `silencedetect`, and `astats`.

## 0. License authorization

Read the pinned [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE) before downloading. Its default territory excludes the EU, UK, Republic of Korea, and United States. If the system is in an excluded territory, obtain separate MiniMax authorization first. Running the scripts is an acknowledgment that the operator has handled this prerequisite.

## 1. Host and driver preflight

Before each launch, set a clean-baseline value measured after a normal boot and run the executable gate on every system:

```bash
export EXPECTED_IDLE_HBM_MIB=2000  # replace with the measured healthy baseline
./preflight-idle-hbm.sh
```

Then confirm the following:

1. Confirm the NVIDIA driver is healthy and the GPU is at its normal idle-HBM baseline.
2. Confirm there are no compute, UVM, or device-file owners from an earlier run.
3. Stop if kernel logs show an NVIDIA Xid that left the device unavailable, failed ATS-peer removal, nonzero PMA use, `NV_ERR_INVALID_STATE`, `RmInitAdapter failed`, an NVIDIA kernel oops, or a soft lockup.
4. Never use an in-place GPU reset, PCI unbind/rescan, or NVIDIA driver-module reload as recovery. Use a coordinated normal host reboot if the driver or HBM accounting is unhealthy.
5. On two systems, clean up the named container explicitly on both ranks and compare both idle-HBM readings before relaunching.

The canonical launch needs at least 150 GB of free HBM for the model plus runtime headroom. A GB300's 288 GB HBM is sufficient for the one-partition resident plan, but measure the real workload peak.

## 2. Download the exact checkpoint

Install a current `huggingface_hub` CLI, then run:

```bash
export MODEL_DIR=/absolute/path/to/MiniMax-H3
./download-model.sh
```

The script downloads root metadata plus `FL2VA/*` at revision `42ed227ee7df40d41602854ae760620d6eb651fe`. It intentionally does not download Ref2VA or the duplicate Diffusers layout. Expect 144,051,182,625 bytes (134.16 GiB) under `FL2VA/`, including 29 safetensor files.

Copy that exact directory to the same absolute path on a second system if running Ring2. A direct high-speed link is preferable:

```bash
rsync -a --whole-file --info=progress2 \
  --exclude='.cache/' \
  "${MODEL_DIR}/" node1:"${MODEL_DIR}/"
```

## 3. Pull the pinned ARM64 runtime

On every system:

```bash
docker pull \
  lmsysorg/sglang@sha256:c3c427732dd726b6e1656dd3cb491bee3629a269c83c57496d26fe28b4d8c5ea
```

This ARM64 manifest was built from SGLang commit `c0b6474b43363c2f4bc60fe3d7817d393fb51d32`. Do not replace it with the mutable `dev` tag in a published run.

## 4. One-station BF16 resident server

After completing the safety preflight:

```bash
export MODEL_DIR=/absolute/path/to/MiniMax-H3
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
./serve-1x.sh
```

The first launch loads only FL2VA and uses Ulysses1/Ring1. It does not enable CPU offload, FSDP, quantization, Cache-DiT, or `torch.compile`. Wait for the health endpoint before benchmarking:

DGX Station also contains a smaller display GPU, so select the GB300 explicitly. Do not assume CUDA device 0 is the compute GPU. Resolve its CDI name during a healthy preflight with `nvidia-ctk cdi list`, correlate it with the GB300 UUID from the healthy driver inventory, and pass the full `nvidia.com/gpu=GPU-...` name.

```bash
curl --fail http://127.0.0.1:30010/health
```

## 5. Canonical performance run

Run one full 50-step warmup and at least three measured VBench prompts:

```bash
export OUTPUT_DIR=/absolute/path/to/results/1x-bf16
export SERVER_HOST=127.0.0.1
export NUM_PROMPTS=3
export MAX_CONCURRENCY=1
./benchmark-performance.sh
```

For publication, increase `NUM_PROMPTS` and preserve the output JSON, server log, and launch metadata. The benchmark fixes 1344×768, 5 seconds, 50 sigma-grid points, video flow shift 12, audio flow shift 3, and one output per prompt.

## 6. Generate and validate a fixed quality sample

Submit an asynchronous request using the pinned prompt and seed:

```bash
curl --fail-with-body -sS \
  -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"MiniMaxAI/MiniMax-H3",
    "prompt":"At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.",
    "seconds":5,
    "task":"t2va",
    "conditions":[],
    "target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0},
    "num_outputs_per_prompt":1,
    "num_inference_steps":50,
    "flow_shift":12.0,
    "audio_flow_shift":3.0,
    "seed":1101
  }'
```

SGLang's endpoint is asynchronous. Poll the returned job ID, download `/v1/videos/<id>/content`, and validate it:

```bash
./validate-output.sh /absolute/path/to/output.mp4 /absolute/path/to/validation
```

Inspect the warning logs as well as the machine-readable stream metadata. Black, frozen, or silent intervals can be legitimate for a prompt; they are flags for manual review rather than automatic failures.

## 7. Two-station modes

For aggregate throughput, run the one-station server independently on each system and send one request to each at the same time. This avoids inter-node communication and is the primary 2× throughput result.

For one-request latency scaling, use cross-node Ring sequence parallelism. Configure a routable high-speed interface on both systems, then run the same command on both with different ranks:

```bash
# node0
export MODEL_DIR=/absolute/path/to/MiniMax-H3
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
export NODE_RANK=0
export NODE0_ADDR=192.0.2.10
export FABRIC_IFACE=high_speed0
# Optional RDMA path, after verifying the interface/HCA/device mapping:
export NCCL_IB_HCA=mlx5_0
export RDMA_DEVICE=/dev/infiniband/uverbs0
./serve-2x-ring.sh

# node1
export MODEL_DIR=/absolute/path/to/MiniMax-H3
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
export NODE_RANK=1
export NODE0_ADDR=192.0.2.10
export FABRIC_IFACE=high_speed0
export NCCL_IB_HCA=mlx5_0
export RDMA_DEVICE=/dev/infiniband/uverbs0
./serve-2x-ring.sh
```

The example address and interface are placeholders. The launch uses world size 2, Ulysses1, Ring2, and replicated encoders. This exact 2×1 GB300 topology is experimental: validate media and same-seed quality before publishing speedup. SGLang's documented cross-node H3 validation used 2×8 H200, not two single-GPU stations.

The Ring2 recipe explicitly sets `--warmup-mode off`. In the pinned runtime,
the built-in FL2VA server warmup is not serialized correctly to the remote
rank: rank 1 treats the placeholder image as base64 and rejects it. This only
skips that two-step synthetic startup request. The benchmark below still runs
the same complete 50-step client warmup before recording any measurement.

Pure Ulysses2 is intentionally not used across the two systems. The pinned SGLang H3 runtime requires the cross-node boundary to use Ring; Ulysses is intended to stay within a node. With one GB300 per system, that yields Ulysses1 × Ring2. The DiT has 56 heads and the runtime's packed-sequence alignment both admit this topology.

For RDMA, set `NCCL_IB_HCA` and `RDMA_DEVICE` to the verified data-plane HCA and matching uverbs character device on both nodes. If they are omitted, the recipe forces NCCL sockets on `FABRIC_IFACE`. Do not guess an HCA from the management interface. Capture NCCL logs for the first run and confirm the intended transport is selected.

## 8. Follow-up optimizations

Apply one change at a time after the BF16 resident baseline:

1. Online FP8 DiT (`--quantization fp8`), with same-seed video and audio comparisons.
2. FP8 Qwen3-VL encoder, measured separately from DiT FP8.
3. Fixed-schedule AdaLN cache, which is lossless only for the exact covered schedule.
4. Cache-DiT or reduced-step LoRA only as explicitly approximate rows.

Do not combine FP8, Cache-DiT, reduced steps, and alternate attention in the first optimization run; compounded changes make quality regressions impossible to attribute.
