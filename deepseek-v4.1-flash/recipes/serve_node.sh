#!/usr/bin/env bash
# Start one SGLang rank of the two-node DeepSeek-V4.1-Flash server on this host (node0 = rank 0, node1 = rank 1).
# Usage: NODE_RANK=0|1 ./serve_node.sh   (normally invoked by launch_cluster.sh)
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/config.env"

node_rank="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
case "$node_rank" in
  0) model_dir="$MODEL_DIR_RANK0" ;;
  1) model_dir="$MODEL_DIR_RANK1" ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac

[[ -s "$model_dir/model.safetensors.index.json" ]] || { echo "Checkpoint index missing at $model_dir" >&2; exit 1; }
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container $CONTAINER_NAME already exists on $(hostname); run stop_cluster.sh first." >&2
  exit 1
fi
for dev in /dev/infiniband/uverbs0 /dev/infiniband/uverbs1; do
  [[ -c "$dev" ]] || { echo "Missing RDMA device $dev" >&2; exit 1; }
done

gpu_uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', *' '$2 ~ /GB300/ {print $1}')"
[[ "$(wc -l <<<"$gpu_uuid")" -eq 1 && -n "$gpu_uuid" ]] || { echo "Expected exactly one GB300 on $(hostname)" >&2; exit 1; }

cache_dir="$CACHE_ROOT/sglang-dsv41"
mkdir -p "$cache_dir"/{root-cache,triton,tilelang,nv} "$script_dir/logs"

docker_env=(
  --env "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=$ENGRAM_HOST_TABLE"
  --env "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=$ENGRAM_LAYOUT"
  --env "SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=$ENGRAM_PIN"
  --env "NCCL_MNNVL_ENABLE=0"
  --env "NCCL_CUMEM_ENABLE=0"
  --env "GLOO_SOCKET_IFNAME=$FABRIC_IFACE"
  --env "NCCL_SOCKET_IFNAME=$FABRIC_IFACE"
  --env "NCCL_IB_HCA=$NCCL_HCAS"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_NET_GDR_LEVEL=SYS"
  --env "NCCL_DMABUF_ENABLE=1"
  --env "NCCL_DEBUG=$NCCL_DEBUG_LEVEL"
  --env "NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
)
extra_mounts=()
if [[ "$HOST_RDMA" == 1 ]]; then
  for f in libibverbs.so.1 libmlx5.so.1; do [[ -s "$HOST_RDMA_DIR/$f" ]] || { echo "Missing $HOST_RDMA_DIR/$f (run tools/stage_host_rdma.sh)" >&2; exit 1; }; done
  # Overlay the host libraries directly onto the image's real library files (the in-image symlinks
  # libibverbs.so.1 / libmlx5.so.1 keep pointing at these names), so every process picks them up even when
  # a worker is spawned with a scrubbed environment. Target names are resolved from the image at launch.
  ibv_real="$(docker run --rm --entrypoint readlink "$IMAGE" -f /usr/lib/aarch64-linux-gnu/libibverbs.so.1)"
  mlx5_real="$(docker run --rm --entrypoint readlink "$IMAGE" -f /usr/lib/aarch64-linux-gnu/libmlx5.so.1)"
  [[ -n "$ibv_real" && -n "$mlx5_real" ]] || { echo "Could not resolve rdma-core library names in $IMAGE" >&2; exit 1; }
  extra_mounts+=(
    --volume "$HOST_RDMA_DIR/libibverbs.so.1:$ibv_real:ro"
    --volume "$HOST_RDMA_DIR/libmlx5.so.1:$mlx5_real:ro"
    --volume /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
    --volume /etc/libibverbs.d:/etc/libibverbs.d:ro
  )
fi
if [[ "$NCCL_TUNING" == tuned ]]; then
  docker_env+=(
    --env "NCCL_ALGO=RING" --env "NCCL_PROTO=SIMPLE"
    --env "NCCL_MIN_NCHANNELS=8" --env "NCCL_MAX_NCHANNELS=8"
    --env "NCCL_IB_QPS_PER_CONNECTION=4" --env "NCCL_IB_SPLIT_DATA_ON_QPS=1"
  )
elif [[ "$NCCL_TUNING" != auto ]]; then
  echo "NCCL_TUNING must be tuned or auto" >&2; exit 2
fi
while IFS='=' read -r k v; do [[ -n "$k" ]] && docker_env+=(--env "$k=$v"); done <<<"${EXTRA_ENV:-}"

args=(
  --trust-remote-code
  --model-path /model
  --served-model-name "$SERVED_MODEL_NAME"
  --tp "$TP_SIZE" --ep-size "$EP_SIZE"
  --nnodes 2 --node-rank "$node_rank" --dist-init-addr "$RANK0_IP:$MASTER_PORT"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --context-length "$CONTEXT_LENGTH"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --reasoning-parser auto --tool-call-parser auto
  --enable-metrics
  --host "$API_HOST" --port "$API_PORT"
)
case "$MODE" in
  throughput) ;;
  low-latency) args+=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5) ;;
  *) echo "MODE must be throughput or low-latency" >&2; exit 2 ;;
esac
if [[ "$SWA_BOUNDED_REPLAY" == 1 ]]; then
  # Bounded replay is refused alongside the prefill CUDA graph (deepseek_v4_hook.py), so force it off.
  args+=(--enable-decoder-swa-bounded-replay --cuda-graph-backend-prefill disabled)
elif [[ -n "$PREFILL_GRAPH_BACKEND" ]]; then
  args+=(--cuda-graph-backend-prefill "$PREFILL_GRAPH_BACKEND")
fi
[[ "$LANGUAGE_MODEL_ONLY" == 1 ]] && args+=(--language-model-only)
[[ "$DISABLE_OVERLAP" == 1 ]] && args+=(--disable-overlap-schedule)
case "$ALLREDUCE_FUSION" in
  off) args+=(--enforce-disable-flashinfer-allreduce-fusion) ;;
  auto) ;;
  *) echo "ALLREDUCE_FUSION must be off or auto" >&2; exit 2 ;;
esac
[[ -n "$CUDA_GRAPH_MAX_BS_PREFILL" ]] && args+=(--cuda-graph-max-bs-prefill "$CUDA_GRAPH_MAX_BS_PREFILL")
[[ -n "$CUDA_GRAPH_MAX_BS_DECODE" ]] && args+=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE")
[[ -n "$KV_CACHE_DTYPE" ]] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

docker run --detach \
  --name "$CONTAINER_NAME" \
  --device "nvidia.com/gpu=$gpu_uuid" \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir/root-cache:/root/.cache" \
  --volume "$cache_dir/triton:/root/.triton" \
  --volume "$cache_dir/tilelang:/root/.tilelang" \
  --volume "$cache_dir/nv:/root/.nv" \
  "${extra_mounts[@]}" \
  "${docker_env[@]}" \
  --entrypoint python3 \
  "$IMAGE" -m sglang.launch_server "${args[@]}" >/dev/null

{
  echo "host=$(hostname) rank=$node_rank image=$IMAGE image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")"
  echo "env: ${docker_env[*]}"
  echo "args: ${args[*]}"
} > "$script_dir/logs/launch-rank$node_rank.txt"
echo "Started $CONTAINER_NAME rank $node_rank on $(hostname) (mode=$MODE tp=$TP_SIZE ep=$EP_SIZE hcas=$NCCL_HCAS replay=$SWA_BOUNDED_REPLAY engram=$ENGRAM_LAYOUT)"
