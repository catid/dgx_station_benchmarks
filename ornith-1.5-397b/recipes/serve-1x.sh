#!/usr/bin/env bash
set -euo pipefail

profile=${1:?usage: serve-1x.sh throughput|prefill|wikitext2-bf16}
: "${MODEL_DIR:?set MODEL_DIR to the downloaded checkpoint directory}"
: "${GPU_DEVICE:?set GPU_DEVICE to the GB300 CDI name}"

image='vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967'
cache_dir=${CACHE_DIR:-"$PWD/.vllm-cache"}
model_name='Ornith-1.5-397B-NVFP4'
container_name="ornith15-397b-$profile"

test -f "$MODEL_DIR/model.safetensors.index.json"
mkdir -p "$cache_dir"

case "$profile" in
  throughput)
    kv_dtype=fp8_e4m3
    max_model_len=32768
    max_num_seqs=16
    memory_fraction=0.95
    graph_sizes=(1 2 4 8 16)
    ;;
  prefill)
    kv_dtype=fp8_e4m3
    max_model_len=135168
    max_num_seqs=1
    memory_fraction=0.96
    graph_sizes=(1)
    ;;
  wikitext2-bf16)
    kv_dtype=bfloat16
    max_model_len=32768
    max_num_seqs=4
    memory_fraction=0.95
    graph_sizes=(1 2 4)
    ;;
  *)
    echo "unknown profile: $profile" >&2
    exit 2
    ;;
esac

if sudo docker inspect "$container_name" >/dev/null 2>&1; then
  echo "container already exists: $container_name" >&2
  exit 1
fi

sudo docker run -d \
  --name "$container_name" \
  --device "$GPU_DEVICE" \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -v "$MODEL_DIR:/model:ro" \
  -v "$cache_dir:/root/.cache" \
  "$image" \
  serve /model \
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY:-prefetch}" \
  --served-model-name "$model_name" \
  --host 127.0.0.1 \
  --port 30000 \
  --trust-remote-code \
  --language-model-only \
  --moe-backend flashinfer_cutedsl \
  --kv-cache-dtype "$kv_dtype" \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization "$memory_fraction" \
  --cudagraph-capture-sizes "${graph_sizes[@]}" \
  --max-cudagraph-capture-size "${graph_sizes[-1]}" \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'

echo "started $container_name; follow startup with: sudo docker logs -f $container_name"
