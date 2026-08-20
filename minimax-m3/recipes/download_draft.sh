#!/usr/bin/env bash
set -euo pipefail

: "${MINIMAX_M3_LICENSE_ACCEPTED:?Read the draft metadata and bundled MiniMax M3 license, then set MINIMAX_M3_LICENSE_ACCEPTED=YES}"
[[ "$MINIMAX_M3_LICENSE_ACCEPTED" == YES ]] || {
  echo "MINIMAX_M3_LICENSE_ACCEPTED must equal YES" >&2
  exit 2
}
: "${DRAFT_MODEL_DIR:?Set DRAFT_MODEL_DIR to the target MiniMax-M3-EAGLE3-GQA directory}"

readonly repository="Inferact/MiniMax-M3-EAGLE3-GQA"
readonly revision="96692486b5fd38ebf8fd2a5f6bb53427d30819a8"
command -v hf >/dev/null || { echo "Install a current Hugging Face hf CLI" >&2; exit 1; }
mkdir -p "$DRAFT_MODEL_DIR"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
hf download "$repository" --revision "$revision" --local-dir "$DRAFT_MODEL_DIR"
"$(dirname -- "${BASH_SOURCE[0]}")/verify_draft.sh" "$DRAFT_MODEL_DIR"

