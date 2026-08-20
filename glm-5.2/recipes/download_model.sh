#!/usr/bin/env bash
set -euo pipefail

readonly repo_id="nvidia/GLM-5.2-NVFP4"
readonly revision="aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
readonly model_root="${MODEL_ROOT:-$PWD/models}"
readonly model_dir="${MODEL_DIR:-$model_root/GLM-5.2-NVFP4}"
readonly download_venv="${DOWNLOAD_VENV:-$PWD/.download-venv}"

python3 -m venv "$download_venv"
"$download_venv/bin/pip" install 'huggingface_hub==1.28.0'
mkdir -p "$model_root"
"$download_venv/bin/hf" download "$repo_id" \
  --revision "$revision" \
  --local-dir "$model_dir"
"$download_venv/bin/hf" cache verify "$repo_id" \
  --revision "$revision" \
  --local-dir "$model_dir" \
  --fail-on-missing-files

printf '%s\n' "$revision" >"$model_dir/PINNED_REVISION"
printf 'Verified checkpoint: %s at %s\n' "$repo_id" "$revision"
