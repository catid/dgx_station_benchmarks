#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE="lmsysorg/sglang@sha256:c3c427732dd726b6e1656dd3cb491bee3629a269c83c57496d26fe28b4d8c5ea"

: "${OUTPUT_DIR:?Set OUTPUT_DIR to an absolute result directory}"
[[ "${OUTPUT_DIR}" = /* ]] || {
  echo "OUTPUT_DIR must be absolute" >&2
  exit 2
}

server_host="${SERVER_HOST:-127.0.0.1}"
server_port="${SERVER_PORT:-30010}"
num_prompts="${NUM_PROMPTS:-3}"
max_concurrency="${MAX_CONCURRENCY:-1}"

mkdir -p "${OUTPUT_DIR}"
curl --fail "http://${server_host}:${server_port}/health" >/dev/null

docker run --rm \
  --network=host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env XDG_CACHE_HOME=/tmp/.cache \
  --tmpfs /tmp:rw,exec,nosuid,size=4g,mode=1777 \
  --volume "${OUTPUT_DIR}:/out" \
  "${IMAGE}" \
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --host "${server_host}" \
    --port "${server_port}" \
    --model MiniMaxAI/MiniMax-H3 \
    --dataset vbench \
    --task text-to-video \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${max_concurrency}" \
    --num-outputs-per-prompt 1 \
    --num-inference-steps 50 \
    --warmup-requests 1 \
    --warmup-inference-steps 50 \
    --extra-body '{"task":"t2va","conditions":[],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0},"seconds":5,"flow_shift":12.0,"audio_flow_shift":3.0}' \
    --output-file /out/benchmark.json
