#!/usr/bin/env bash
set -euo pipefail

readonly repo="tencent/Hy3-FP8"
readonly revision="ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b"
readonly model_dir="${MODEL_DIR:-$PWD/models/Hy3-FP8}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir

hf_bin="${HF_BIN:-}"
if [[ -z "$hf_bin" ]]; then
  hf_bin="$(command -v hf || true)"
fi
if [[ -z "$hf_bin" || ! -x "$hf_bin" ]]; then
  echo "Set HF_BIN to an executable Hugging Face CLI (hf)." >&2
  exit 1
fi

mkdir -p "$model_dir"
HF_XET_HIGH_PERFORMANCE=1 "$hf_bin" download "$repo" \
  --revision "$revision" \
  --local-dir "$model_dir"

"$hf_bin" cache verify "$repo" \
  --revision "$revision" \
  --local-dir "$model_dir" \
  --fail-on-missing-files

python3 "$script_dir/verify-checkpoint.py" "$model_dir"
