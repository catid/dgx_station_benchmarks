#!/usr/bin/env bash
set -euo pipefail

readonly node_rank="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
readonly master_addr="${MASTER_ADDR:-192.0.2.1}"
readonly master_port="${MASTER_PORT:-29502}"
readonly rank0_ip="${RANK0_IP:-192.0.2.1}"
readonly rank1_ip="${RANK1_IP:-192.0.2.2}"
readonly fabric_iface="${FABRIC_IFACE:-enP1p3s0f0np0}"
readonly fabric_hca="${FABRIC_HCA:-mlx5_0}"
readonly model_dir="${MODEL_DIR:-$PWD/models/GLM-5.2-NVFP4}"
readonly cache_dir="${CACHE_DIR:-$PWD/cache/glm52-vllm}"
readonly image="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
readonly container_name="${CONTAINER_NAME:-glm52-nvfp4-vllm}"
readonly parallel_mode="${PARALLEL_MODE:-tp}"
readonly nccl_tuning="${NCCL_TUNING:-auto}"
readonly moe_backend="${MOE_BACKEND:-flashinfer_cutedsl}"
readonly flashinfer_autotune="${FLASHINFER_AUTOTUNE:-disabled}"
readonly max_tokens_per_expert="${VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE:-32768}"

case "$node_rank" in
  0) readonly node_ip="$rank0_ip" ;;
  1) readonly node_ip="$rank1_ip" ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 1 ;;
esac
case "$parallel_mode" in
  pp) readonly tp_size=1; readonly pp_size=2 ;;
  tp) readonly tp_size=2; readonly pp_size=1 ;;
  *) echo "PARALLEL_MODE must be pp or tp" >&2; exit 1 ;;
esac
if [[ ! -s "$model_dir/model.safetensors.index.json" ]]; then
  echo "Checkpoint is incomplete at $model_dir" >&2
  exit 1
fi
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly first." >&2
  exit 1
fi

gpu_uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
  | awk -F ', ' '$2 ~ /GB300/ {print $1; exit}')"
readonly gpu_uuid
if [[ -z "$gpu_uuid" ]]; then
  echo "No NVIDIA GB300 found" >&2
  exit 1
fi

mkdir -p "$cache_dir"
docker_env=(
  --env "VLLM_HOST_IP=$node_ip"
  --env "GLOO_SOCKET_IFNAME=$fabric_iface"
  --env "NCCL_SOCKET_IFNAME=$fabric_iface"
  --env "NCCL_IB_HCA=$fabric_hca"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_NET_GDR_LEVEL=SYS"
  --env "NCCL_DMABUF_ENABLE=1"
  --env "NCCL_DEBUG=INFO"
  --env "NCCL_DEBUG_SUBSYS=INIT,NET"
  --env "VLLM_ALLREDUCE_USE_SYMM_MEM=0"
  --env "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=$max_tokens_per_expert"
)
if [[ "$parallel_mode" == "pp" ]]; then
  docker_env+=(--env "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm")
fi
if [[ "$nccl_tuning" == tuned ]]; then
  docker_env+=(
    --env "NCCL_ALGO=RING"
    --env "NCCL_PROTO=SIMPLE"
    --env "NCCL_MIN_NCHANNELS=8"
    --env "NCCL_MAX_NCHANNELS=8"
    --env "NCCL_IB_QPS_PER_CONNECTION=4"
    --env "NCCL_IB_SPLIT_DATA_ON_QPS=1"
  )
elif [[ "$nccl_tuning" != auto ]]; then
  echo "NCCL_TUNING must be auto or tuned" >&2
  exit 1
fi

vllm_args=(
  serve /model
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
  --served-model-name GLM-5.2-NVFP4
  --trust-remote-code
  --tensor-parallel-size "$tp_size"
  --pipeline-parallel-size "$pp_size"
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "$node_rank"
  --master-addr "$master_addr"
  --master-port "$master_port"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
  --max-model-len "${MAX_MODEL_LEN:-135168}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}"
  --enable-prefix-caching
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 128
  --max-cudagraph-capture-size 128
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --moe-backend "$moe_backend"
)
if [[ "$parallel_mode" == "pp" ]]; then
  # The two PP stages discover different kernel keys, so the distributed
  # FlashInfer autotuner can enter mismatched collectives and deadlock.
  vllm_args+=(--no-enable-flashinfer-autotune)
elif [[ "$flashinfer_autotune" == "disabled" ]]; then
  vllm_args+=(--enable-expert-parallel --no-enable-flashinfer-autotune)
elif [[ "$flashinfer_autotune" == "auto" ]]; then
  vllm_args+=(--enable-expert-parallel)
else
  echo "FLASHINFER_AUTOTUNE must be auto or disabled" >&2
  exit 1
fi
if [[ "$node_rank" == 0 ]]; then
  vllm_args+=(--host 127.0.0.1 --port "${API_PORT:-30000}")
else
  vllm_args+=(--headless)
fi

docker run --detach \
  --name "$container_name" \
  --device "nvidia.com/gpu=$gpu_uuid" \
  --device /dev/infiniband/uverbs0 \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir:/root/.cache" \
  "${docker_env[@]}" \
  --entrypoint vllm \
  "$image" "${vllm_args[@]}"

printf 'Started %s on rank %s (%s).\n' "$container_name" "$node_rank" "$parallel_mode"
