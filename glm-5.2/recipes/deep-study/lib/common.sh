#!/usr/bin/env bash
set -euo pipefail

STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly STUDY_ROOT
readonly PLAN_RESOLVER="$STUDY_ROOT/resolve_plan.py"

load_site_config() {
  if [[ -n "${SITE_CONFIG:-}" ]]; then
    [[ -f "$SITE_CONFIG" ]] || {
      echo "SITE_CONFIG does not exist: $SITE_CONFIG" >&2
      return 2
    }
    # shellcheck source=/dev/null
    source "$SITE_CONFIG"
  fi
}

require_var() {
  local name="$1"
  [[ -n "${!name:-}" ]] || {
    echo "Required configuration is unset: $name" >&2
    return 2
  }
}

resolve_plan_env() {
  local profile="$1"
  local winner_backend="${2:-}"
  local args=("$profile" --format shell)
  if [[ -n "$winner_backend" ]]; then
    args+=(--winner-backend "$winner_backend")
  fi
  local assignments
  assignments="$(PYTHONDONTWRITEBYTECODE=1 python3 "$PLAN_RESOLVER" "${args[@]}")"
  eval "$assignments"
  export PROFILE_ID RUNTIME RUNTIME_IMAGE MODEL_REVISION SERVED_MODEL_NAME
  export CONTAINER_NAME TOPOLOGY TP_SIZE PP_SIZE EP_SIZE PP_LAYER_PARTITION
  export MOE_BACKEND FLASHINFER_AUTOTUNE MTP_TOKENS KV_CACHE_DTYPE
  export MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
  export CHUNKED_PREFILL_SIZE GPU_MEMORY_UTILIZATION BENCHMARK_MODE
  export CONCURRENCIES DECODE_CONTEXT MAX_OUTPUT_TOKENS DURATION_SECONDS
  export PREFILL_CONTEXTS MINIMUM_KV_TOKENS REPORTABLE
}

require_execution_opt_in() {
  [[ "${ALLOW_GPU_EXECUTION:-}" == "YES" ]] || {
    echo "Refusing execution: export ALLOW_GPU_EXECUTION=YES after confirming both GPUs are idle." >&2
    return 3
  }
}

require_safe_container_name() {
  local name="$1"
  [[ "$name" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || {
    echo "Unsafe or unexpected container name: $name" >&2
    return 3
  }
}

print_plan() {
  local profile="$1"
  local winner_backend="${2:-}"
  local args=("$profile")
  [[ -n "$winner_backend" ]] && args+=(--winner-backend "$winner_backend")
  PYTHONDONTWRITEBYTECODE=1 python3 "$PLAN_RESOLVER" "${args[@]}"
}
