#!/usr/bin/env bash
set -euo pipefail

readonly container_name="${CONTAINER_NAME:?Set CONTAINER_NAME}"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly target_backend="${MOE_BACKEND:?Set MOE_BACKEND}"
readonly draft_backend="${MTP_DRAFT_MOE_BACKEND:?Set MTP_DRAFT_MOE_BACKEND}"
readonly minimum_kv_tokens="${MINIMUM_KV_TOKENS:?Set MINIMUM_KV_TOKENS}"

[[ "$container_name" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || {
  echo "Unsafe container name" >&2
  exit 2
}
[[ "$target_backend" == flashinfer_cutedsl ]] || {
  echo "Split-MTP gate requires a CuTeDSL target" >&2
  exit 2
}
[[ "$draft_backend" == flashinfer_cutlass ]] || {
  echo "Split-MTP gate requires a FlashInfer CUTLASS draft" >&2
  exit 2
}
[[ "$minimum_kv_tokens" =~ ^[0-9]+$ ]] || {
  echo "Invalid minimum KV-token gate" >&2
  exit 2
}

local_log=$(mktemp)
remote_log=$(mktemp)
local_env=$(mktemp)
remote_env=$(mktemp)
trap 'rm -f "$local_log" "$remote_log" "$local_env" "$remote_env"' EXIT

docker logs "$container_name" >"$local_log" 2>&1
# shellcheck disable=SC2029  # The validated name is expanded locally.
ssh "$remote" docker logs "$container_name" >"$remote_log" 2>&1
docker inspect "$container_name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  >"$local_env"
# shellcheck disable=SC2029  # The validated name is expanded locally.
ssh "$remote" docker inspect "$container_name" \
  --format '{{range .Config.Env}}{{println .}}{{end}}' >"$remote_env"

target_marker="'FLASHINFER_CUTEDSL' NvFp4 MoE backend"
draft_marker="Using FlashInfer CUTLASS Unquantized MoE backend"
fatal_pattern='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|illegal memory access|NV_ERR_INVALID_STATE|RmInitAdapter failed'

rank=0
for pair in "$local_log:$local_env" "$remote_log:$remote_env"; do
  log=${pair%%:*}
  environment=${pair#*:}
  rg -Fq "$target_marker" "$log" || {
    echo "rank${rank}: target CuTeDSL marker is absent" >&2
    exit 10
  }
  rg -Fq "$draft_marker" "$log" || {
    echo "rank${rank}: draft FlashInfer CUTLASS marker is absent" >&2
    exit 11
  }
  ! rg -qi "$fatal_pattern" "$log" || {
    echo "rank${rank}: fatal server marker appeared before requests" >&2
    exit 12
  }
  rg -Fxq 'VLLM_USE_V2_MODEL_RUNNER=0' "$environment" || {
    echo "rank${rank}: MRv1 environment pin is absent" >&2
    exit 13
  }
  capacity=$(rg -io 'GPU KV cache size: [0-9,]+ tokens' "$log" \
    | tail -n 1 | sed -E 's/[^0-9]//g')
  [[ "$capacity" =~ ^[0-9]+$ ]] || {
    echo "rank${rank}: GPU KV-token capacity is absent" >&2
    exit 14
  }
  (( capacity >= minimum_kv_tokens )) || {
    echo "rank${rank}: GPU KV-token capacity ${capacity} is below ${minimum_kv_tokens}" >&2
    exit 15
  }
  printf 'rank%s target=FLASHINFER_CUTEDSL draft=FLASHINFER_CUTLASS kv_tokens=%s mr_v1=yes\n' \
    "$rank" "$capacity"
  rank=$((rank + 1))
done

echo "Split-MTP backend and capacity gate passed before benchmark requests."
