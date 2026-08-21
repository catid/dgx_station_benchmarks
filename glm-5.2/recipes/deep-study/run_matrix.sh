#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here

usage() {
  echo "Usage: $0 STAGE [--winner-backend BACKEND] [--execute]" >&2
  echo "Stages: reproduction, backends, autotune, mtp, pp2, sglang, prefill-vllm, prefill-sglang" >&2
}
[[ $# -ge 1 ]] || { usage; exit 2; }
stage="$1"
shift
winner_backend="${WINNER_BACKEND:-}"
execute=no
while [[ $# -gt 0 ]]; do
  case "$1" in
    --winner-backend) winner_backend="${2:?missing backend}"; shift 2 ;;
    --execute) execute=yes; shift ;;
    *) usage; exit 2 ;;
  esac
done

case "$stage" in
  reproduction) profiles=(vllm-tp2-exact) ;;
  backends) profiles=(vllm-tp2-exact vllm-tp2-flashinfer-cutlass vllm-tp2-vllm-cutlass) ;;
  autotune) profiles=(vllm-tp2-winner-autotune-off vllm-tp2-winner-autotune-on) ;;
  mtp) profiles=(vllm-tp2-mtp0 vllm-tp2-mtp1 vllm-tp2-mtp2 vllm-tp2-mtp3 vllm-tp2-mtp5) ;;
  pp2) profiles=(vllm-pp2-balanced-bootstrap vllm-pp2-balanced) ;;
  sglang) profiles=(sglang-tp2-cutedsl sglang-pp2-balanced) ;;
  prefill-vllm) profiles=(vllm-tp2-prefill-4096 vllm-tp2-prefill-8192 vllm-tp2-prefill-16384 vllm-tp2-prefill-32768) ;;
  prefill-sglang) profiles=(sglang-pp2-prefill-4096 sglang-pp2-prefill-8192 sglang-pp2-prefill-16384 sglang-pp2-prefill-32768) ;;
  *) usage; exit 2 ;;
esac
if [[ "$stage" =~ ^(autotune|mtp)$ && -z "$winner_backend" ]]; then
  echo "$stage requires --winner-backend selected from the completed backend comparison." >&2
  exit 2
fi

# Resolve the whole stage before the first run so an invalid late profile cannot
# leave a partially executed matrix.
for profile in "${profiles[@]}"; do
  resolver_args=("$profile")
  [[ -n "$winner_backend" ]] && resolver_args+=(--winner-backend "$winner_backend")
  PYTHONDONTWRITEBYTECODE=1 python3 "$here/resolve_plan.py" "${resolver_args[@]}" >/dev/null
done

for profile in "${profiles[@]}"; do
  args=("$profile")
  [[ -n "$winner_backend" ]] && args+=(--winner-backend "$winner_backend")
  [[ "$execute" == yes ]] && args+=(--execute)
  bash "$here/run_headline.sh" "${args[@]}"
done
