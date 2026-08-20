#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
readonly tag="${MTP_IMAGE_TAG:-hy3-vllm:0.27.1-mtp-compile}"

docker build \
  --file "$script_dir/Dockerfile.mtp-compile" \
  --tag "$tag" \
  "$script_dir"

docker run --rm --entrypoint bash "$tag" -lc \
  'target=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/hy_v3_mtp.py; grep -n -B 1 "^class HYV3MTP" "$target"'
