#!/usr/bin/env bash
set -euo pipefail

readonly model_dir="${1:?Usage: $0 /absolute/path/to/MiniMax-M3-MXFP8}"
readonly expected_bytes=443776005285
readonly expected_shards=31

for file in config.json model.safetensors.index.json LICENSE tokenizer.json; do
  [[ -s "$model_dir/$file" ]] || { echo "Missing $model_dir/$file" >&2; exit 1; }
done
actual_shards="$(find -L "$model_dir" -maxdepth 1 -type f -name 'model-*-of-00031.safetensors' | wc -l)"
actual_bytes="$(find -L "$model_dir" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
[[ "$actual_shards" == "$expected_shards" ]] || { echo "Expected $expected_shards shards; found $actual_shards" >&2; exit 1; }
[[ "$actual_bytes" == "$expected_bytes" ]] || { echo "Expected $expected_bytes bytes; found $actual_bytes" >&2; exit 1; }

declare -A expected=(
  [config.json]=a3698fff231a6cd92f632dcbe50f0b1dbda716994f29ccde9f0fa9d2a87698bf
  [model.safetensors.index.json]=41126751cfdb44c2fb33522bcaaa6f6ca057462838b3493a347678e6e1ca24ee
  [LICENSE]=b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c
  [tokenizer.json]=bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e
)
for file in "${!expected[@]}"; do
  actual="$(sha256sum "$model_dir/$file" | awk '{print $1}')"
  [[ "$actual" == "${expected[$file]}" ]] || { echo "SHA-256 mismatch for $file" >&2; exit 1; }
done
[[ "$(jq '[.weight_map[]] | unique | length' "$model_dir/model.safetensors.index.json")" == "$expected_shards" ]] || {
  echo "Checkpoint index shard count mismatch" >&2; exit 1;
}
echo "PASS: MXFP8 checkpoint verified ($actual_shards shards, $actual_bytes bytes)"
