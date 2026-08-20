#!/usr/bin/env bash
set -euo pipefail

readonly draft_dir="${1:?Usage: $0 /absolute/path/to/MiniMax-M3-EAGLE3-GQA}"
readonly expected_bytes=6149993396

for file in config.json model.safetensors README.md 'MiniMax M3 LICENSE.txt'; do
  [[ -s "$draft_dir/$file" ]] || { echo "Missing $draft_dir/$file" >&2; exit 1; }
done
actual_bytes="$(find "$draft_dir" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
[[ "$actual_bytes" == "$expected_bytes" ]] || {
  echo "Expected $expected_bytes draft repository bytes; found $actual_bytes" >&2
  exit 1
}
[[ "$(sha256sum "$draft_dir/config.json" | awk '{print $1}')" == d26ea8053832f6df4a399d8abfac5ccbbfb2959f4786c9f069f0a21175d7d6aa ]] || {
  echo "Draft config SHA-256 mismatch" >&2; exit 1;
}
[[ "$(sha256sum "$draft_dir/model.safetensors" | awk '{print $1}')" == 0c6d6430519dd498342ccf7c94db8f7e90ba678c8ffbad3d198b35d4a0ffc33f ]] || {
  echo "Draft weights SHA-256 mismatch" >&2; exit 1;
}
[[ "$(sha256sum "$draft_dir/MiniMax M3 LICENSE.txt" | awk '{print $1}')" == b53f2fdda3049b0e9013207be51efc2d372cda1fcfdd8bb4bb8b22658ca5db9c ]] || {
  echo "Bundled MiniMax M3 license SHA-256 mismatch" >&2; exit 1;
}
echo "PASS: GQA EAGLE3 draft verified ($actual_bytes bytes)"

