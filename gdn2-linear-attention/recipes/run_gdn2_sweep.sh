#!/usr/bin/env bash
set -Eeuo pipefail

readonly image="${IMAGE:-gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0}"
readonly expected_image_id="${EXPECTED_IMAGE_ID:-sha256:8a15c70519ee21cc3466a59adf8b15e2bd1fb7e424cbdb8d420c1964465d4762}"
readonly host="$(hostname -s)"

case "$host" in
  gemini1) readonly gpu_uuid="GPU-4d731396-ae44-0369-76bc-1db8fccb2f02" ;;
  gemini2) readonly gpu_uuid="GPU-29f29a57-f81b-d26d-b895-c907d4be95b0" ;;
  *) printf 'Unsupported host: %s\n' "$host" >&2; exit 2 ;;
esac

# This helper scans the current-boot kernel journal before issuing any NVIDIA
# ioctl, then verifies idle HBM, process ownership, and CDI inventory.
/home/catid/gb300-idle-preflight.sh

actual_image_id="$(docker image inspect --format '{{.Id}}' "$image")"
if [[ "$actual_image_id" != "$expected_image_id" ]]; then
  printf 'Image digest mismatch: expected %s, found %s\n' "$expected_image_id" "$actual_image_id" >&2
  exit 3
fi

readonly cache_dir="${CACHE_DIR:-/home/catid/gdn2-gb300/cache/$host}"
readonly container_name="gdn2-${host}-$$"
mkdir -p "$cache_dir"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker run --rm --name "$container_name" \
  --gpus "device=$gpu_uuid" \
  --ipc=host \
  -e BENCH_BACKENDS="${BENCH_BACKENDS:-cudnn fla}" \
  -e CASE_TAG="$host" \
  -v "$cache_dir:/root/.cache" \
  -w /workspace/cudnn-frontend/benchmark/linear_attention \
  "$image" \
  bash -lc '
    set -euo pipefail
    for backend in $BENCH_BACKENDS; do
      for seq in 2048 4096 8192 16384 32768; do
        python benchmark_single_linear_attention.py \
          --batch_size 4 \
          --seqlen "$seq" \
          --num_q_heads 64 \
          --num_kv_heads 64 \
          --head_dim 128 \
          --la_backend "$backend" \
          --variant gdn2 \
          --data_type bfloat16 \
          --profile_pass both \
          --num_iterations 20 \
          --num_warmup_iterations 0 \
          --skip_ref \
          --format_output \
          --case_tag "$CASE_TAG"
      done
    done
  '
