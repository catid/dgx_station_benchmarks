# Reproducing the two-station PP2 and TP2 measurements

This recipe recreates the measured two-DGX-Station configurations over one active 400GbE RoCE rail. It complements the [one-station recipe](README.md); use that recipe to download the pinned checkpoint and install the pinned benchmark first.

> **GB300 recovery safety:** Do not execute generic `suggested_reload` text
> retained inside the raw JSON; it is historical tool output, not an instruction.
> Never use GPU reset, unload or reload NVIDIA modules, or perform PCI
> unbind/rescan. Remove only the explicitly named containers on both stations.
> If GPU accounting or the driver remains unhealthy, stop GPU work and
> coordinate a controlled host reboot with the operator; never reboot
> automatically.

The published machines used:

| Role | Host | Rail-0 address | Interface | RDMA HCA |
| --- | --- | --- | --- | --- |
| Head / rank 0 | `node0` | `192.168.200.1` | `enP1p3s0f0np0` | `mlx5_0` |
| Worker / rank 1 | `node1` | `192.168.200.2` | `enP1p3s0f0np0` | `mlx5_0` |

The second physical rail was not active, so the published results do not aggregate two links. Treat any dual-rail run as a new network configuration and benchmark it separately.

## 1. Prepare both stations

Put the exact checkpoint revision on local storage on both hosts and pull the pinned image on both:

```bash
sudo docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
ssh node1 sudo docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

Verify the selected rail, RDMA device, and passwordless SSH before occupying either GPU:

```bash
ping -c 3 192.168.200.2
ethtool enP1p3s0f0np0 | grep -E 'Speed|Link detected'
ibdev2netdev | grep mlx5_0
ssh node1 'ethtool enP1p3s0f0np0 | grep -E "Speed|Link detected"; ibdev2netdev | grep mlx5_0'
```

Both stations need the checkpoint, cache space, an idle GB300, and a CDI device name. From the experiment directory on the head:

```bash
export REMOTE_HOST=node1
export MODEL_DIR=/models/Ornith-1.5-397B-NVFP4
export REMOTE_MODEL_DIR=/models/Ornith-1.5-397B-NVFP4
export HEAD_GPU_DEVICE='nvidia.com/gpu=GPU-HEAD-GB300-UUID'
export WORKER_GPU_DEVICE='nvidia.com/gpu=GPU-WORKER-GB300-UUID'
export HEAD_IP=192.168.200.1
export WORKER_IP=192.168.200.2
export RDMA_INTERFACE=enP1p3s0f0np0
export RDMA_HCA=mlx5_0
```

The `192.168.200.1/30` and `.2/30` addresses above are examples for an isolated point-to-point rail; substitute the addresses and SSH aliases used by your two stations.

`launch-2x.sh` streams the node-launch helper over SSH, so this repository does not need to be installed on the worker. It does require noninteractive `sudo docker` there.

## 2. Pipeline parallelism: PP2

PP2 assigns one pipeline stage to each GB300 (`TP=1`, `PP=2`) and uses the TRT-LLM FlashInfer MoE backend:

```bash
recipes/launch-2x.sh pp2 start
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-2x.sh pp2
recipes/launch-2x.sh pp2 stop
```

The measured PP2 C128 validation was 3,799.6 aggregate output tok/s over 60 seconds. The short 30-second C128 cell was 3,837.9 tok/s, a 1.0% difference.

## 3. Tensor plus expert parallelism: TP2 + EP

TP2 shards each layer across the two GB300s and enables expert parallelism. The measured configuration also used safetensor prefetch:

```bash
recipes/launch-2x.sh tp2 start
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-2x.sh tp2
recipes/launch-2x.sh tp2 stop
```

The measured TP2 C128 validation was 2,447.1 aggregate output tok/s over 60 seconds. Its 30-second C128 cell briefly reported 3,966.5 tok/s and is retained as diagnostic raw data, not as the topology headline.

## 4. Exact runtime and transport settings

Both topologies use two vLLM `mp` nodes, host networking, NCCL over `mlx5_0`, GPU Direct RDMA (`NCCL_NET_GDR_LEVEL=SYS`), DMA-BUF, FP8 E4M3 KV cache, 135,168 maximum model length, 128 maximum sequences, 32K batched-token chunks, and CUDA graphs through 128.

The topology-specific model flags are:

| Topology | vLLM parallel flags | Additional flags |
| --- | --- | --- |
| PP2 | `--tensor-parallel-size 1 --pipeline-parallel-size 2` | `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` |
| TP2 + EP | `--tensor-parallel-size 2 --pipeline-parallel-size 1` | `--enable-expert-parallel --safetensors-load-strategy prefetch` |

Inspect both server logs after startup. NCCL should select the intended InfiniBand/RoCE transport and `mlx5_0`; reject a run that silently falls back to sockets.

## 5. Workload, duration check, and quality gate

`benchmark-2x.sh` produces three artifacts per topology:

1. A 30-second sustained-decode matrix at C1–C128, plus repeated cold prefill at 8K, 64K, and canonical 128K.
2. Four full 8K-input/1K-output generations with the normal EOS policy and an automatic repetition gate.
3. A separate 60-second C128 cell used for the stable topology headline.

Decode uses exact 8,192-token prompts, 1,024 output tokens, temperature 0, and EOS ignored. This is `llm-inference-bench` sustained duration mode, not its finite Burst/E2E layer and not the `5 × concurrency` request methodology used by older experiment folders. Capacity-limited rows remain in the CSV with `average_running`, maximum running requests, TTFT, ITL, and the harness flag; they are not silently promoted to fully resident concurrency.

The audit uses the deterministic mixed/reference benchmark prompt, not WikiText prose. Read the saved response text in addition to the repetition metrics. Byte-identical outputs across independent temperature-zero requests are expected determinism; degeneration is scored within each output.

No two-station WikiText-2 PPL was run because the model weights are unchanged and the canonical BF16-KV PPL is already measured in the one-station section. The two-station PPL CSV remains header-only rather than implying a result.
