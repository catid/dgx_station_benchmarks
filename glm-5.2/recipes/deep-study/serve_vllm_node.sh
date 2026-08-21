#!/usr/bin/env bash
set -euo pipefail

[[ "${ALLOW_NODE_LAUNCH:-}" == YES ]] || {
  echo "This node launcher may only be called by launch_cluster.sh." >&2
  exit 3
}
readonly node_rank="${NODE_RANK:?Set NODE_RANK}"
case "$node_rank" in
  0) node_ip="${RANK0_IP:?}" ;;
  1) node_ip="${RANK1_IP:?}" ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac
readonly node_ip
readonly model_dir="${MODEL_DIR:?Set MODEL_DIR}"
readonly cache_profile_id="${CACHE_PROFILE_ID:-${PROFILE_ID:-$CONTAINER_NAME}}"
[[ "$cache_profile_id" =~ ^[a-z0-9][a-z0-9-]{2,80}$ ]] || {
  echo "Unsafe compiler-cache profile ID" >&2
  exit 2
}
readonly cache_dir="${CACHE_ROOT:?Set CACHE_ROOT}/vllm/$cache_profile_id"
readonly expected_image="vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"

[[ "${RUNTIME_IMAGE:?}" == "$expected_image" ]] || { echo "Unpinned vLLM image" >&2; exit 2; }
[[ "${CONTAINER_NAME:?}" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || { echo "Unsafe container name" >&2; exit 2; }
[[ -s "$model_dir/model.safetensors.index.json" ]] || { echo "Checkpoint index is missing" >&2; exit 2; }
docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && { echo "Container already exists: $CONTAINER_NAME" >&2; exit 4; }

gpu_uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
  | awk -F ', *' '$2 ~ /GB300/ {print $1}')"
[[ "$(wc -l <<<"$gpu_uuid")" -eq 1 && -n "$gpu_uuid" ]] || { echo "Expected exactly one GB300" >&2; exit 5; }
mkdir -p "$cache_dir"

docker_env=(
  --env "VLLM_HOST_IP=$node_ip"
  --env "GLOO_SOCKET_IFNAME=${FABRIC_IFACE:?}"
  --env "NCCL_SOCKET_IFNAME=${FABRIC_IFACE:?}"
  --env "NCCL_IB_HCA=${FABRIC_HCA:?}"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_NET_GDR_LEVEL=SYS"
  --env "NCCL_DMABUF_ENABLE=1"
  --env "VLLM_ALLREDUCE_USE_SYMM_MEM=0"
  --env "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=32768"
)
if [[ -n "${VLLM_USE_V2_MODEL_RUNNER:-}" ]]; then
  [[ "$VLLM_USE_V2_MODEL_RUNNER" == 0 ]] || {
    echo "VLLM_USE_V2_MODEL_RUNNER override must be 0" >&2
    exit 2
  }
  docker_env+=(--env "VLLM_USE_V2_MODEL_RUNNER=0")
fi
case "${NCCL_TRACE_MODE:-headline}" in
  headline) docker_env+=(--env "NCCL_DEBUG=WARN") ;;
  info) docker_env+=(--env "NCCL_DEBUG=INFO" --env "NCCL_DEBUG_SUBSYS=INIT,NET,TUNING") ;;
  trace) docker_env+=(--env "NCCL_DEBUG=TRACE" --env "NCCL_DEBUG_SUBSYS=INIT,NET,TUNING,COLL") ;;
  *) echo "Invalid NCCL_TRACE_MODE" >&2; exit 2 ;;
esac
if [[ "${PP_SIZE:?}" == 2 ]]; then
  [[ "${MTP_TOKENS:?}" == 0 && "${PP_LAYER_PARTITION:?}" == "40,38" ]] || {
    echo "PP2 requires 40,38 and speculation off" >&2
    exit 2
  }
  docker_env+=(--env "VLLM_PP_LAYER_PARTITION=40,38")
fi

read -r -a cudagraph_sizes <<<"${CUDAGRAPH_CAPTURE_SIZES:-}"
case "${ENFORCE_EAGER:?}" in
  yes)
    [[ "${#cudagraph_sizes[@]}" -eq 0 && "${MAX_CUDAGRAPH_CAPTURE_SIZE:?}" == 0 ]] || {
      echo "Eager mode must not declare CUDA graph captures" >&2
      exit 2
    }
    [[ "${BLOCK_SIZE:?}" == 64 ]] || { echo "Eager PP smoke requires block64" >&2; exit 2; }
    ;;
  no)
    [[ "${#cudagraph_sizes[@]}" -gt 0 ]] || { echo "Empty CUDA graph grid" >&2; exit 2; }
    [[ "${cudagraph_sizes[-1]}" == "${MAX_CUDAGRAPH_CAPTURE_SIZE:?}" ]] || {
      echo "CUDA graph maximum does not match capture grid" >&2
      exit 2
    }
    ;;
  *) echo "ENFORCE_EAGER must be yes or no" >&2; exit 2 ;;
esac

vllm_args=(
  serve /model
  --safetensors-load-strategy prefetch
  --served-model-name GLM-5.2-NVFP4
  --trust-remote-code
  --tensor-parallel-size "${TP_SIZE:?}"
  --pipeline-parallel-size "$PP_SIZE"
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "$node_rank"
  --master-addr "${MASTER_ADDR:?}"
  --master-port "${MASTER_PORT:?}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --kv-cache-dtype "${KV_CACHE_DTYPE:?}"
  --max-model-len "${MAX_MODEL_LEN:?}"
  --max-num-seqs "${MAX_NUM_SEQS:?}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:?}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:?}"
  --enable-prefix-caching
  --moe-backend "${MOE_BACKEND:?}"
)
if [[ -n "${BLOCK_SIZE:-}" ]]; then
  [[ "$BLOCK_SIZE" == 64 ]] || { echo "Only block64 is audited" >&2; exit 2; }
  vllm_args+=(--block-size "$BLOCK_SIZE")
fi
if [[ "$ENFORCE_EAGER" == yes ]]; then
  vllm_args+=(--enforce-eager)
else
  vllm_args+=(
    --cudagraph-capture-sizes "${cudagraph_sizes[@]}"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  )
fi
if [[ "${TP_SIZE:?}" == 2 ]]; then
  vllm_args+=(--enable-expert-parallel)
fi
case "${FLASHINFER_AUTOTUNE:?}" in
  off) vllm_args+=(--no-enable-flashinfer-autotune) ;;
  on)
    [[ "$PP_SIZE" == 1 ]] || { echo "Autotune is not allowed with PP2" >&2; exit 2; }
    vllm_args+=(--enable-flashinfer-autotune)
    ;;
  *) echo "FLASHINFER_AUTOTUNE must be on or off" >&2; exit 2 ;;
esac
if (( MTP_TOKENS > 0 )); then
  [[ "$TP_SIZE" == 2 && "$PP_SIZE" == 1 ]] || { echo "MTP is restricted to TP2" >&2; exit 2; }
  [[ "${MTP_DRAFT_MOE_BACKEND:?}" == flashinfer_cutlass ]] || {
    echo "MTP requires the audited FlashInfer CUTLASS draft override" >&2
    exit 2
  }
  vllm_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"moe_backend\":\"${MTP_DRAFT_MOE_BACKEND}\"}")
elif [[ -n "${MTP_DRAFT_MOE_BACKEND:-}" ]]; then
  echo "MTP=0 must not set a draft MoE backend" >&2
  exit 2
fi
if [[ "$node_rank" == 0 ]]; then
  vllm_args+=(--host 127.0.0.1 --port "${API_PORT:-30000}")
else
  vllm_args+=(--headless)
fi

docker run --detach \
  --name "$CONTAINER_NAME" \
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
  "$RUNTIME_IMAGE" "${vllm_args[@]}"
