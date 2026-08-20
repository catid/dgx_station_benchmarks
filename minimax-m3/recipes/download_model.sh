#!/usr/bin/env bash
set -euo pipefail

: "${MINIMAX_M3_LICENSE_ACCEPTED:?Read the MiniMax Community License, then set MINIMAX_M3_LICENSE_ACCEPTED=YES}"
[[ "$MINIMAX_M3_LICENSE_ACCEPTED" == YES ]] || {
  echo "MINIMAX_M3_LICENSE_ACCEPTED must equal YES" >&2
  exit 2
}
: "${MODEL_DIR:?Set MODEL_DIR to the target MiniMax-M3-NVFP4 directory}"

readonly repository="nvidia/MiniMax-M3-NVFP4"
readonly revision="901464083161bf8612a29ff7ad29914cd4ab4a85"
readonly expected_bytes=250137296832
readonly reserve_bytes=20000000000
parent_dir="$(dirname -- "$MODEL_DIR")"
readonly parent_dir

command -v hf >/dev/null || {
  echo "Install a current Hugging Face hf CLI" >&2
  exit 1
}
mkdir -p "$parent_dir" "$MODEL_DIR"
available_bytes="$(df -B1 --output=avail "$parent_dir" | tail -1 | tr -d ' ')"
current_bytes="$(find "$MODEL_DIR" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
remaining_bytes=$((expected_bytes - current_bytes))
(( remaining_bytes < 0 )) && remaining_bytes=0
required_free_bytes=$((remaining_bytes + reserve_bytes))
if [[ ! "$available_bytes" =~ ^[0-9]+$ ]] || (( available_bytes < required_free_bytes )); then
  echo "Need at least $required_free_bytes free bytes to finish with reserve; found $available_bytes" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
hf download "$repository" --revision "$revision" --local-dir "$MODEL_DIR"
"$(dirname -- "${BASH_SOURCE[0]}")/verify_checkpoint.sh" "$MODEL_DIR"
