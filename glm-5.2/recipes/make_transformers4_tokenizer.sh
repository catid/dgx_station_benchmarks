#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here
readonly model_dir="${1:?Usage: $0 MODEL_DIR [OUTPUT_DIR]}"
readonly output_dir="${2:-$PWD/tokenizer-transformers4}"

if [[ ! -s "$model_dir/tokenizer.json" ]]; then
  echo "Missing $model_dir/tokenizer.json" >&2
  exit 1
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir"
ln -s "$(realpath "$model_dir/tokenizer.json")" "$output_dir/tokenizer.json"
cp "$here/tokenizer-config-transformers4.json" "$output_dir/tokenizer_config.json"

printf 'Transformers-4 compatibility tokenizer: %s\n' "$output_dir"
