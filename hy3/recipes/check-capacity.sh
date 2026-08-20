#!/usr/bin/env bash
set -euo pipefail

readonly model_dir="${1:-${MODEL_DIR:-$PWD/models/Hy3-FP8}}"
hbm_mib="$(nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits \
  | awk -F ', ' '$2 ~ /GB300/ {print $1; exit}')"
readonly hbm_mib

if [[ -z "$hbm_mib" ]]; then
  echo "No GB300 reported by nvidia-smi." >&2
  exit 1
fi

weight_bytes=0
shard_count=0
shopt -s nullglob
for shard in "$model_dir"/model-*-of-*.safetensors; do
  weight_bytes=$((weight_bytes + $(stat -c %s "$shard")))
  shard_count=$((shard_count + 1))
done

python3 - "$hbm_mib" "$weight_bytes" "$shard_count" <<'PY'
import sys

expected_shards = 101
expected_weight_bytes = 299_889_838_946
hbm_mib, weight_bytes, shard_count = map(int, sys.argv[1:])
hbm_bytes = hbm_mib * 2**20
print(f"GB300 usable HBM: {hbm_mib:,} MiB = {hbm_bytes / 2**30:,.3f} GiB")
print(f"Downloaded weights: {weight_bytes:,} bytes = {weight_bytes / 2**30:,.3f} GiB")
print(f"Downloaded shards: {shard_count}")
if shard_count != expected_shards or weight_bytes != expected_weight_bytes:
    excess = expected_weight_bytes - hbm_bytes
    print(f"Checkpoint is incomplete; expected {expected_shards} shards and {expected_weight_bytes:,} weight bytes")
    print(f"Final result: NO FIT (complete weights exceed HBM by {excess / 2**30:,.3f} GiB before runtime allocations)")
    raise SystemExit(3)
if weight_bytes > hbm_bytes:
    excess = weight_bytes - hbm_bytes
    print(f"Result: NO FIT (weights exceed HBM by {excess / 2**30:,.3f} GiB before runtime allocations)")
    raise SystemExit(2)
print("Result: weight bytes fit, but runtime headroom must still be validated")
PY
