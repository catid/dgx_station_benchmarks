#!/usr/bin/env bash
set -euo pipefail

readonly model_dir="${MODEL_DIR:?Set MODEL_DIR to the verified MiniMax-M3-NVFP4 directory}"
readonly gpu_device="${GPU_DEVICE:?Set GPU_DEVICE to the GB300 CDI selector printed by preflight-idle-hbm.sh}"
readonly profile="${SERVER_PROFILE:-throughput}"
readonly thinking_mode="${THINKING_MODE:-disabled}"
readonly speculative_mode="${SPECULATIVE_MODE:-none}"
readonly container_name="${CONTAINER_NAME:-minimax-m3-vllm}"
readonly image="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
readonly cache_dir="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/dgx-station-benchmarks/minimax-m3-vllm}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir

case "$thinking_mode" in disabled|adaptive|enabled) ;; *) echo "THINKING_MODE must be disabled, adaptive, or enabled" >&2; exit 2 ;; esac
case "$speculative_mode" in none|eagle3-gqa) ;; *) echo "SPECULATIVE_MODE must be none or eagle3-gqa" >&2; exit 2 ;; esac
case "$profile" in
  throughput)
    readonly kv_dtype="${KV_CACHE_DTYPE:-fp8}"
    readonly max_model_len="${MAX_MODEL_LEN:-262144}"
    readonly max_num_seqs="${MAX_NUM_SEQS:-128}"
    readonly max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}"
    readonly default_graph_sizes="1,2,4,8,16,32,64,128"
    readonly default_max_graph_size=128
    ;;
  ppl)
    readonly kv_dtype="${KV_CACHE_DTYPE:-bfloat16}"
    readonly max_model_len="${MAX_MODEL_LEN:-4096}"
    readonly max_num_seqs="${MAX_NUM_SEQS:-4}"
    readonly max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}"
    readonly default_graph_sizes="1,2,4"
    readonly default_max_graph_size=4
    ;;
  *) echo "SERVER_PROFILE must be throughput or ppl" >&2; exit 2 ;;
esac
IFS=',' read -r -a graph_sizes <<<"${CUDA_GRAPH_CAPTURE_SIZES:-$default_graph_sizes}"
readonly -a graph_sizes
readonly max_graph_size="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$default_max_graph_size}"
(( ${#graph_sizes[@]} > 0 )) || { echo "CUDA_GRAPH_CAPTURE_SIZES cannot be empty" >&2; exit 2; }
for graph_size in "${graph_sizes[@]}"; do
  [[ "$graph_size" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CUDA graph size: $graph_size" >&2; exit 2; }
done
[[ "$max_graph_size" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MAX_CUDAGRAPH_CAPTURE_SIZE" >&2; exit 2; }

"$script_dir/verify_checkpoint.sh" "$model_dir"
[[ "$gpu_device" == nvidia.com/gpu=GPU-* ]] || { echo "GPU_DEVICE must select one GPU by CDI UUID" >&2; exit 2; }
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly before relaunching" >&2
  exit 1
fi
mkdir -p "$cache_dir"

draft_args=()
draft_mount=()
if [[ "$speculative_mode" == eagle3-gqa ]]; then
  readonly draft_model_dir="${DRAFT_MODEL_DIR:?Set DRAFT_MODEL_DIR for SPECULATIVE_MODE=eagle3-gqa}"
  "$script_dir/verify_draft.sh" "$draft_model_dir"
  draft_mount=(--volume "$draft_model_dir:/draft:ro")
  draft_args=(--speculative-config '{"method":"eagle3","model":"/draft","num_speculative_tokens":3,"attention_backend":"FLASH_ATTN"}')
fi

docker run --detach \
  --name "$container_name" \
  --device "$gpu_device" \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --env VLLM_FLOAT32_MATMUL_PRECISION=high \
  --env VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir:/root/.cache" \
  "${draft_mount[@]}" \
  --entrypoint vllm \
  "$image" serve /model \
    --served-model-name MiniMax-M3-NVFP4 \
    --host 127.0.0.1 \
    --port "${API_PORT:-30000}" \
    --trust-remote-code \
    --language-model-only \
    --block-size 128 \
    --kv-cache-dtype "$kv_dtype" \
    --attention_config.backend FLASHINFER \
    --attention_config.use_trtllm_attention true \
    --attention_config.indexer_kv_dtype fp8 \
    --attention_config.minimax_m3_msa_decode_backend cutlass \
    --tool-call-parser minimax_m3 \
    --enable-auto-tool-choice \
    --reasoning-parser minimax_m3 \
    --default-chat-template-kwargs "{\"thinking_mode\":\"$thinking_mode\"}" \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --enable-prefix-caching \
    --cudagraph-capture-sizes "${graph_sizes[@]}" \
    --max-cudagraph-capture-size "$max_graph_size" \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    "${draft_args[@]}"

echo "Started $container_name: profile=$profile thinking=$thinking_mode speculative=$speculative_mode; no CPU offload"
