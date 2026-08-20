#!/usr/bin/env bash
set -euo pipefail

readonly model_dir="${1:-${MODEL_DIR:-$PWD/models/GLM-5.2-NVFP4}}"
hbm_mib="$(nvidia-smi --query-gpu=memory.total,name --format=csv,noheader,nounits \
  | awk -F ', ' '$2 ~ /GB300/ {print $1; exit}')"
readonly hbm_mib

if [[ -z "$hbm_mib" ]]; then
  echo "No NVIDIA GB300 found" >&2
  exit 1
fi

python3 - "$hbm_mib" "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

hbm_mib = int(sys.argv[1])
root = Path(sys.argv[2])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file()]
weight_bytes = sum((root / name).stat().st_size for name in shards if (root / name).is_file())
hbm_bytes = hbm_mib * 2**20
print(f"GB300 usable HBM: {hbm_mib:,} MiB = {hbm_bytes / 2**30:,.3f} GiB")
print(f"Checkpoint weights: {weight_bytes:,} bytes = {weight_bytes / 2**30:,.3f} GiB")
print(f"Indexed shards: {len(shards)}; missing: {len(missing)}")
if missing:
    raise SystemExit(3)
if weight_bytes > hbm_bytes:
    print(f"Result: NO FIT (weights exceed HBM by {(weight_bytes-hbm_bytes)/2**30:,.3f} GiB before runtime)")
    raise SystemExit(2)
print("Result: weight bytes fit; runtime headroom still requires validation")
PY
