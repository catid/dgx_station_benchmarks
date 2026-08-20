#!/usr/bin/env bash
set -euo pipefail

readonly phase="${1:?Usage: $0 before|after OUTPUT_DIR}"
readonly output_dir="${2:?Usage: $0 before|after OUTPUT_DIR}"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly iface="${FABRIC_IFACE:?Set FABRIC_IFACE}"
readonly hca="${FABRIC_HCA:?Set FABRIC_HCA}"
here="$(cd "$(dirname "$0")" && pwd)"
readonly here
[[ "$phase" == before || "$phase" == after ]] || { echo "phase must be before or after" >&2; exit 2; }
[[ "$iface" =~ ^[A-Za-z0-9_.:-]+$ && "$hca" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "Unsafe interface/HCA" >&2; exit 2; }
mkdir -p "$output_dir/$phase"

PYTHONDONTWRITEBYTECODE=1 python3 "$here/capture_roce_counters.py" \
  --iface "$iface" --hca "$hca" >"$output_dir/$phase/node0.json"
ssh "$remote" python3 - --iface "$iface" --hca "$hca" \
  <"$here/capture_roce_counters.py" >"$output_dir/$phase/node1.json"
