#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash across node0 (rank 0, API) and node1 (rank 1). All config lives in config.env;
# override with environment variables, e.g.  MODE=low-latency CHUNKED_PREFILL_SIZE=8192 ./launch_cluster.sh
# The recipe directory must exist at the same absolute path on both nodes (it is rsync'ed to rank 1 below).
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
source ./config.env
wait_ready="${WAIT_READY_SECONDS:-3600}"

[[ "${SKIP_PREFLIGHT:-0}" == 1 ]] || ./preflight.sh

# Sync scripts (not caches/models/results/logs) to rank 1 so both ranks run identical launch logic.
rsync -a --exclude cache --exclude models --exclude hostrdma --exclude results --exclude logs --exclude '.*' "$script_dir/" "$RANK1_SSH:$script_dir/"

# Pass every overridable knob through to rank 1 explicitly.
pass=()
for v in ENGINE IMAGE VLLM_IMAGE CONTAINER_NAME SERVED_MODEL_NAME RANK0_IP RANK1_IP RANK1_RAIL1_IP MASTER_PORT API_PORT API_HOST FABRIC_IFACE NCCL_HCAS NCCL_TUNING \
         NCCL_DEBUG_LEVEL MODEL_DIR_RANK0 MODEL_DIR_RANK1 CACHE_ROOT TP_SIZE EP_SIZE MODE MEM_FRACTION_STATIC CONTEXT_LENGTH \
         CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS MAX_RUNNING_REQUESTS SWA_BOUNDED_REPLAY PREFILL_GRAPH_BACKEND \
         CUDA_GRAPH_MAX_BS_PREFILL CUDA_GRAPH_MAX_BS_DECODE KV_CACHE_DTYPE ENGRAM_HOST_TABLE ENGRAM_LAYOUT ENGRAM_PIN \
         LANGUAGE_MODEL_ONLY DISABLE_OVERLAP ALLREDUCE_FUSION HOST_RDMA HOST_RDMA_DIR NCCL_DEBUG_SUBSYS EXTRA_ARGS EXTRA_ENV \
         VLLM_PARALLEL VLLM_PP_LAYER_PARTITION VLLM_GPU_MEMORY_UTILIZATION VLLM_LANGUAGE_MODEL_ONLY VLLM_PREFIX_CACHING \
         VLLM_FLASHINFER_AUTOTUNE VLLM_ASYNC_SCHEDULING VLLM_KV_CACHE_DTYPE VLLM_COMPILATION_CONFIG VLLM_ENFORCE_EAGER VLLM_RUST_FRONTEND \
         VLLM_MASTER_PORT VLLM_DEEP_GEMM_WARMUP VLLM_PATCH_PP; do
  pass+=("$v=$(printf '%q' "${!v:-}")")
done

echo "== starting rank 1 on $RANK1_SSH"
case "$ENGINE" in sglang) serve=./serve_node.sh ;; vllm) serve=./serve_vllm_node.sh ;; *) echo "ENGINE must be sglang or vllm" >&2; exit 2 ;; esac
ssh -o BatchMode=yes "$RANK1_SSH" "cd $script_dir && env ${pass[*]} NODE_RANK=1 $serve"
echo "== starting rank 0 on $(hostname)"
NODE_RANK=0 $serve

# Stream both ranks' container logs to files from the start, so a crash that removes a container cannot lose them.
mkdir -p logs
live_ts="$(date +%Y%m%dT%H%M%S)"
nohup docker logs -f "$CONTAINER_NAME" > "logs/live-rank0-$live_ts.log" 2>&1 < /dev/null &
ssh -n -f -o BatchMode=yes "$RANK1_SSH" "cd $script_dir && mkdir -p logs && setsid nohup docker logs -f $CONTAINER_NAME > logs/live-rank1-$live_ts.log 2>&1 < /dev/null &" || true
echo "== live logs: logs/live-rank0-$live_ts.log (node1: logs/live-rank1-$live_ts.log)"

echo "== waiting for http://$API_HOST:$API_PORT/health (up to ${wait_ready}s); follow with ./logs.sh 0 | ./logs.sh 1"
start=$SECONDS
while (( SECONDS - start < wait_ready )); do
  if curl -fsS --max-time 3 "http://$API_HOST:$API_PORT/health" >/dev/null 2>&1; then
    echo "READY after $((SECONDS - start))s"
    docker logs "$CONTAINER_NAME" 2>&1 | grep -iE "attention.backend|moe.runner|fp8.gemm|kv cache|mem_fraction|max_total_num_tokens|engram|huge pages|capture cuda graph end|load weight end|allreduce fusion|Maximum concurrency|GPU KV cache size|Loading weights took|graph capturing finished|Data Direct|GDRDMA" | tail -24
    exit 0
  fi
  for r in 0 1; do
    if [[ $r == 0 ]]; then st="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo missing)"
    else st="$(ssh -o BatchMode=yes "$RANK1_SSH" docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo missing)"; fi
    if [[ "$st" != running ]]; then
      echo "rank $r container is '$st' - launch failed; last log lines:" >&2
      sleep 2
      if [[ $r == 0 ]]; then grep -vE "ncclIb|NCCL INFO" "logs/live-rank0-$live_ts.log" | tail -40 >&2
      else ssh -o BatchMode=yes "$RANK1_SSH" "grep -vE 'ncclIb|NCCL INFO' $script_dir/logs/live-rank1-$live_ts.log | tail -40" >&2; fi
      exit 1
    fi
  done
  sleep 10
done
echo "Timed out waiting for readiness" >&2
exit 1
