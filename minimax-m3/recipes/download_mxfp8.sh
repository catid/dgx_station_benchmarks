#!/usr/bin/env bash
set -euo pipefail

: "${MINIMAX_M3_LICENSE_ACCEPTED:?Read the MiniMax Community License, then set MINIMAX_M3_LICENSE_ACCEPTED=YES}"
[[ "$MINIMAX_M3_LICENSE_ACCEPTED" == YES ]] || {
  echo "MINIMAX_M3_LICENSE_ACCEPTED must equal YES" >&2
  exit 2
}
: "${MXFP8_MODEL_DIR:?Set MXFP8_MODEL_DIR to the target MiniMax-M3-MXFP8 directory}"

readonly repository="MiniMaxAI/MiniMax-M3-MXFP8"
readonly revision="c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"
readonly expected_bytes=443776005285
readonly reserve_bytes=20000000000
parent_dir="$(dirname -- "$MXFP8_MODEL_DIR")"
readonly parent_dir

command -v hf >/dev/null || { echo "Install a current Hugging Face hf CLI" >&2; exit 1; }
mkdir -p "$parent_dir" "$MXFP8_MODEL_DIR"
available_bytes="$(df -B1 --output=avail "$parent_dir" | tail -1 | tr -d ' ')"
current_bytes="$(find "$MXFP8_MODEL_DIR" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
remaining_bytes=$((expected_bytes - current_bytes))
(( remaining_bytes < 0 )) && remaining_bytes=0
required_free_bytes=$((remaining_bytes + reserve_bytes))
if [[ ! "$available_bytes" =~ ^[0-9]+$ ]] || (( available_bytes < required_free_bytes )); then
  echo "Need at least $required_free_bytes free bytes to finish with reserve; found $available_bytes" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
hf download "$repository" --revision "$revision" --local-dir "$MXFP8_MODEL_DIR"
"$(dirname -- "${BASH_SOURCE[0]}")/verify_mxfp8.sh" "$MXFP8_MODEL_DIR"

