#!/usr/bin/env bash
set -euo pipefail

readonly model_dir="${1:?Usage: $0 /absolute/path/to/MiniMax-M3-NVFP4}"
readonly expected_bytes=250137296832
readonly expected_shards=88

for file in config.json model.safetensors.index.json hf_quant_config.json LICENSE tokenizer.json chat_template.jinja generation_config.json; do
  [[ -s "$model_dir/$file" ]] || { echo "Missing $model_dir/$file" >&2; exit 1; }
done

actual_shards="$(find "$model_dir" -maxdepth 1 -type f -name 'model-*-of-00088.safetensors' | wc -l)"
actual_bytes="$(find "$model_dir" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
[[ "$actual_shards" == "$expected_shards" ]] || {
  echo "Expected $expected_shards safetensor shards; found $actual_shards" >&2
  exit 1
}
[[ "$actual_bytes" == "$expected_bytes" ]] || {
  echo "Expected $expected_bytes repository bytes; found $actual_bytes" >&2
  exit 1
}

declare -A expected=(
  [config.json]=440d22038ffa0bcb93fa01e8d2d555e3840d939167c08eace81b9cce386fb117
  [model.safetensors.index.json]=1401838cb37be2f2242754dd593f3df27cf062a948bac29f70f5e694eeb268fc
  [hf_quant_config.json]=11581ca203bcd77dae53cc77b49392cb17545b097586249dfac12e093bfaa1ec
  [LICENSE]=b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c
  [tokenizer.json]=bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e
  [chat_template.jinja]=11421244f67553498e5c8112dae02802025bcc4305ec45ad380af95c96f9fe64
  [generation_config.json]=272d19efac5060ff8c9f76c7ed67909a5da1be466284ee4f1f1bd2a810c6fad2
)
for file in "${!expected[@]}"; do
  actual="$(sha256sum "$model_dir/$file" | awk '{print $1}')"
  [[ "$actual" == "${expected[$file]}" ]] || {
    echo "SHA-256 mismatch for $file" >&2
    exit 1
  }
done

index_shards="$(jq '[.weight_map[]] | unique | length' "$model_dir/model.safetensors.index.json")"
[[ "$index_shards" == "$expected_shards" ]] || {
  echo "Checkpoint index references $index_shards shards, expected $expected_shards" >&2
  exit 1
}
echo "PASS: $actual_shards shards and $actual_bytes bytes verified in $model_dir"
