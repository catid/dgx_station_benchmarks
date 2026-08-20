#!/usr/bin/env bash
set -Eeuo pipefail

readonly MODEL_ID="MiniMaxAI/MiniMax-H3"
readonly MODEL_REV="42ed227ee7df40d41602854ae760620d6eb651fe"
readonly EXPECTED_FL2VA_BYTES="144051182625"
readonly EXPECTED_SAFETENSORS="29"

: "${MODEL_DIR:?Set MODEL_DIR to an absolute destination path}"
[[ "${MODEL_DIR}" = /* ]] || {
  echo "MODEL_DIR must be absolute" >&2
  exit 2
}
command -v hf >/dev/null || {
  echo "Install a current huggingface_hub CLI first" >&2
  exit 2
}

mkdir -p "${MODEL_DIR}"
HF_HUB_DISABLE_TELEMETRY=1 hf download "${MODEL_ID}" \
  --revision "${MODEL_REV}" \
  --include 'model_index.json' \
  --include 'LICENSE' \
  --include 'README.md' \
  --include 'FL2VA/*' \
  --local-dir "${MODEL_DIR}" \
  --max-workers 8

actual_bytes="$({
  find "${MODEL_DIR}/FL2VA" -type f -printf '%s\n'
} | awk '{total += $1} END {printf "%.0f", total}')"
actual_safetensors="$({
  find "${MODEL_DIR}/FL2VA" -type f -name '*.safetensors' -print
} | wc -l)"

if [[ "${actual_bytes}" != "${EXPECTED_FL2VA_BYTES}" ]]; then
  echo "Unexpected FL2VA byte count: ${actual_bytes}" >&2
  exit 1
fi
if [[ "${actual_safetensors}" != "${EXPECTED_SAFETENSORS}" ]]; then
  echo "Unexpected safetensor count: ${actual_safetensors}" >&2
  exit 1
fi

printf 'Verified %s at %s: %s bytes, %s safetensors\n' \
  "${MODEL_ID}" "${MODEL_REV}" "${actual_bytes}" "${actual_safetensors}"

