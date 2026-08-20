#!/usr/bin/env bash
set -euo pipefail

topology=${1:?usage: serve-2x-node.sh pp2|tp2 head|worker}
role=${2:?usage: serve-2x-node.sh pp2|tp2 head|worker}
: "${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory on this node}"
: "${GPU_DEVICE:?set GPU_DEVICE to the node GB300 CDI name}"

head_ip=${HEAD_IP:-192.168.200.1}
worker_ip=${WORKER_IP:-192.168.200.2}
rdma_interface=${RDMA_INTERFACE:-enP1p3s0f0np0}
rdma_hca=${RDMA_HCA:-mlx5_0}
master_port=${MASTER_PORT:-29503}
container=${CONTAINER_NAME:-ornith15-397b-dual-vllm}
cache_dir=${CACHE_DIR:-"${MODEL_DIR%/*}/.ornith15-dual-vllm-cache"}
image='vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967'

case "$role" in
  head)
    rank=0
    node_ip=$head_ip
    role_flags=(--host 127.0.0.1 --port 30000)
    ;;
  worker)
    rank=1
    node_ip=$worker_ip
    role_flags=(--headless)
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

case "$topology" in
  pp2)
    parallel_flags=(--tensor-parallel-size 1 --pipeline-parallel-size 2)
    load_flags=()
    topology_env=(-e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm)
    ;;
  tp2)
    parallel_flags=(--tensor-parallel-size 2 --pipeline-parallel-size 1 --enable-expert-parallel)
    load_flags=(--safetensors-load-strategy prefetch)
    topology_env=()
    ;;
  *)
    echo "unknown topology: $topology" >&2
    exit 2
    ;;
esac

test -f "$MODEL_DIR/model.safetensors.index.json"
mkdir -p "$cache_dir"
if sudo docker inspect "$container" >/dev/null 2>&1; then
  echo "container already exists on $node_ip: $container" >&2
  exit 1
fi

sudo docker run -d \
  --name "$container" \
  --device "$GPU_DEVICE" \
  --ipc=host \
  --network=host \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_HOST_IP="$node_ip" \
  -e GLOO_SOCKET_IFNAME="$rdma_interface" \
  -e NCCL_SOCKET_IFNAME="$rdma_interface" \
  -e NCCL_IB_HCA="$rdma_hca" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_NET_GDR_LEVEL=SYS \
  -e NCCL_DMABUF_ENABLE=1 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,NET \
  "${topology_env[@]}" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$cache_dir:/root/.cache" \
  "$image" \
  serve /model \
  "${load_flags[@]}" \
  --served-model-name Ornith-1.5-397B-NVFP4 \
  --trust-remote-code \
  --language-model-only \
  "${parallel_flags[@]}" \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$rank" \
  --master-addr "$head_ip" \
  --master-port "$master_port" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --moe-backend flashinfer_trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 135168 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 \
  --max-cudagraph-capture-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  "${role_flags[@]}"
