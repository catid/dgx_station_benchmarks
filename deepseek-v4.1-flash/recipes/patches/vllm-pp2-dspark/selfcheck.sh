#!/usr/bin/env bash
# CPU-only: import the patched modules inside the image with the overlay bind-mounted, and
# exercise the relay helpers with fake tensors. Never requests a GPU.
# mounts.txt names the files relative to $RECIPE_DIR (the recipe directory two levels up).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
recipe_dir="$(cd ../.. && pwd)"
image="${VLLM_IMAGE:-vllm/vllm-openai:deepseekv41-flash-0909}"
mounts=()
while read -r flag spec; do
  [[ "$flag" == --volume && -n "$spec" ]] || continue
  spec="${spec//\$RECIPE_DIR/$recipe_dir}"
  [[ -s "${spec%%:*}" ]] || { echo "missing ${spec%%:*}" >&2; exit 1; }
  mounts+=(--volume "$spec")
done < mounts.txt
for f in *.py; do python3 -m py_compile "$f"; done
echo "py_compile ok"
docker run --rm "${mounts[@]}" --volume "$PWD/selfcheck.py:/tmp/selfcheck.py:ro" \
  --env VLLM_LOGGING_LEVEL=WARNING --env CUDA_VISIBLE_DEVICES= --entrypoint python3 "$image" /tmp/selfcheck.py
