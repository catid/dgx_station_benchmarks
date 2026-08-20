#!/usr/bin/env bash
set -euo pipefail

readonly topology="${1:?Usage: $0 {pp|tp} {0|1|2}}"
readonly mtp_tokens="${2:?Usage: $0 {pp|tp} {0|1|2}}"
readonly remote_host="${REMOTE_HOST:-node1}"
readonly container_name="${CONTAINER_NAME:-hy3-fp8-vllm}"
readonly result_root="${RESULT_ROOT:-$PWD/results}"
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly host="${BENCH_HOST:-127.0.0.1}"
readonly port="${BENCH_PORT:-30000}"
readonly model_revision="ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b"
readonly required_kv_tokens="${REQUIRED_KV_TOKENS:-1179904}"

case "$topology" in
  pp) readonly label=pp2 ;;
  tp) readonly label=tp2 ;;
  *) echo "Topology must be pp or tp." >&2; exit 2 ;;
esac
if [[ ! "$mtp_tokens" =~ ^[012]$ ]]; then
  echo "MTP depth must be 0, 1, or 2." >&2
  exit 2
fi
if [[ ! "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid container name: $container_name" >&2
  exit 2
fi
readonly run_name="${RESULT_RUN_NAME:-hy3-fp8-${label}-mtp${mtp_tokens}}"
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid RESULT_RUN_NAME: $run_name" >&2
  exit 2
fi

readonly run_dir="$result_root/$run_name"
mkdir -p "$run_dir/runtime"

git -C "$bench_dir" rev-parse HEAD > "$run_dir/runtime/llm-inference-bench-commit.txt"
docker inspect "$container_name" > "$run_dir/runtime/node0-inspect.json"
docker logs "$container_name" > "$run_dir/runtime/node0-server.log" 2>&1
nvidia-smi -q > "$run_dir/runtime/node0-nvidia-smi-q.txt"
curl --fail --silent --show-error "http://${host}:${port}/v1/models" \
  > "$run_dir/runtime/models.json"

# Preserve independently hashable remote artifacts in the exact format expected
# by the publication extractor. The base64 transport avoids shell quoting the JSON.
# shellcheck disable=SC2029
ssh "$remote_host" "docker inspect '$container_name' | base64 -w0" \
  | base64 -d > "$run_dir/runtime/node1-inspect.json"
# shellcheck disable=SC2029
ssh "$remote_host" "docker logs '$container_name' 2>&1 | base64 -w0" \
  | base64 -d > "$run_dir/runtime/node1-server.log"
# shellcheck disable=SC2029
ssh "$remote_host" \
  "nvidia-smi -q" > "$run_dir/runtime/node1-nvidia-smi-q.txt"

# Derive the publication settings from the launched container rather than from
# ambient shell defaults. This also catches a capture accidentally aimed at the
# wrong server profile.
python3 - "$run_dir/runtime/node0-inspect.json" "$topology" "$mtp_tokens" \
  "$required_kv_tokens" > "$run_dir/runtime/run-settings.txt" <<'PY'
import json
import sys

path, topology, mtp_tokens, required = sys.argv[1:]
record = json.load(open(path, encoding="utf-8"))[0]
cmd = record["Config"]["Cmd"]

def option(name):
    try:
        return cmd[cmd.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"launched container is missing {name}") from exc

expected_tp, expected_pp = ("1", "2") if topology == "pp" else ("2", "1")
if option("--tensor-parallel-size") != expected_tp or option("--pipeline-parallel-size") != expected_pp:
    raise SystemExit("launched topology does not match capture label")
if topology == "tp" and "--enable-expert-parallel" not in cmd:
    raise SystemExit("TP2 capture lacks --enable-expert-parallel")
if topology == "pp" and "--no-enable-flashinfer-autotune" not in cmd:
    raise SystemExit("PP2 capture lacks --no-enable-flashinfer-autotune")

print(f"topology={topology}")
print(f"mtp_tokens={mtp_tokens}")
print(f"gpu_memory_utilization={option('--gpu-memory-utilization')}")
print(f"kv_cache_dtype={option('--kv-cache-dtype')}")
print(f"max_model_len={option('--max-model-len')}")
print(f"max_num_seqs={option('--max-num-seqs')}")
print(f"max_num_batched_tokens={option('--max-num-batched-tokens')}")
print(f"required_kv_tokens={required}")
PY

runtime_image="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["Config"]["Image"])' "$run_dir/runtime/node0-inspect.json")"
readonly runtime_image
printf '%s\n' "$model_revision" > "$run_dir/runtime/model-revision.txt"
printf '%s\n' "$runtime_image" > "$run_dir/runtime/runtime-image.txt"

kv_tokens="$(sed -nE 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/p' \
  "$run_dir/runtime/node0-server.log" | tr -d ',' | sort -u)"
readonly kv_tokens
if [[ ! "$kv_tokens" =~ ^[0-9]+$ ]]; then
  echo "Could not resolve one global KV-cache capacity from the node0 log: $kv_tokens" >&2
  exit 2
fi
printf '%s\n' "$kv_tokens" > "$run_dir/runtime/kv-cache-tokens.txt"

echo "Captured runtime evidence in $run_dir/runtime"
