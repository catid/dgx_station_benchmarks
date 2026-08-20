#!/usr/bin/env bash
set -euo pipefail

[[ "${ALLOW_NODE_LAUNCH:-}" == YES ]] || {
  echo "This node launcher may only be called by launch_cluster.sh." >&2
  exit 3
}
readonly node_rank="${NODE_RANK:?Set NODE_RANK}"
readonly model_dir="${MODEL_DIR:?Set MODEL_DIR}"
readonly cache_dir="${CACHE_ROOT:?Set CACHE_ROOT}/sglang/${PROFILE_ID:-$CONTAINER_NAME}"
readonly expected_image="lmsysorg/sglang@sha256:9310c4b1c590393399a6e7dacfec48a140e29bff1fb97b59341b21f95223be50"

[[ "$node_rank" == 0 || "$node_rank" == 1 ]] || { echo "NODE_RANK must be 0 or 1" >&2; exit 2; }
[[ "${RUNTIME_IMAGE:?}" == "$expected_image" ]] || { echo "Unpinned SGLang image" >&2; exit 2; }
[[ "${CONTAINER_NAME:?}" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || { echo "Unsafe container name" >&2; exit 2; }
[[ -s "$model_dir/model.safetensors.index.json" ]] || { echo "Checkpoint index is missing" >&2; exit 2; }
docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && { echo "Container already exists: $CONTAINER_NAME" >&2; exit 4; }

gpu_uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
  | awk -F ', *' '$2 ~ /GB300/ {print $1}')"
[[ "$(wc -l <<<"$gpu_uuid")" -eq 1 && -n "$gpu_uuid" ]] || { echo "Expected exactly one GB300" >&2; exit 5; }
mkdir -p "$cache_dir"

docker_env=(
  --env "GLOO_SOCKET_IFNAME=${FABRIC_IFACE:?}"
  --env "NCCL_SOCKET_IFNAME=${FABRIC_IFACE:?}"
  --env "NCCL_IB_HCA=${FABRIC_HCA:?}"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_NET_GDR_LEVEL=SYS"
  --env "NCCL_DMABUF_ENABLE=1"
)
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
  docker_env+=(--env "SGLANG_PP_LAYER_PARTITION=40,38")
fi

sglang_args=(
  -m sglang.launch_server
  --model-path /model
  --served-model-name GLM-5.2-NVFP4
  --trust-remote-code
  --tp-size "${TP_SIZE:?}"
  --pp-size "$PP_SIZE"
  --ep-size "${EP_SIZE:?}"
  --nnodes 2
  --node-rank "$node_rank"
  --dist-init-addr "${MASTER_ADDR:?}:${MASTER_PORT:?}"
  --quantization modelopt_fp4
  --kv-cache-dtype "${KV_CACHE_DTYPE:?}"
  --context-length "${MAX_MODEL_LEN:?}"
  --max-running-requests "${MAX_NUM_SEQS:?}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:?}"
  --mem-fraction-static "${GPU_MEMORY_UTILIZATION:?}"
  --moe-runner-backend "${MOE_BACKEND:?}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --host 127.0.0.1
  --port "${API_PORT:-30000}"
)
if [[ "$PP_SIZE" == 2 ]]; then
  sglang_args+=(--disable-overlap-schedule)
fi
[[ "${MTP_TOKENS:?}" == 0 ]] || { echo "SGLang speculation is quarantined" >&2; exit 2; }

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
  --entrypoint python3 \
  "$RUNTIME_IMAGE" "${sglang_args[@]}"
