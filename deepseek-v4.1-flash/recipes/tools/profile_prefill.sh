#!/usr/bin/env bash
# Capture a torch-profiler trace of a single random-token prefill request on the running two-node SGLang server,
# then summarize GPU kernel time per rank. Usage: tools/profile_prefill.sh [isl=32768] [label=profile]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./config.env
isl="${1:-32768}"; label="${2:-profile}"
prof_dir_in="/root/.cache/sglang/profile-$label"                          # inside the container
prof_dir_r0="$CACHE_ROOT/sglang-dsv41/root-cache/sglang/profile-$label"    # rank 0 host path (the /root/.cache volume)
mkdir -p results/profile
curl -fsS "http://$API_HOST:$API_PORT/health" >/dev/null
echo "== start_profile -> $prof_dir_in"
curl -fsS -X POST "http://$API_HOST:$API_PORT/start_profile" -H 'Content-Type: application/json' \
  -d "{\"output_dir\": \"$prof_dir_in\", \"activities\": [\"CPU\", \"GPU\"], \"record_shapes\": false, \"with_stack\": false}"; echo
python3 - "$isl" "$API_HOST" "$API_PORT" <<'PY'
import json, random, sys, time, urllib.request
isl, host, port = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ids = [random.Random(f"profile/{isl}").randrange(1000, 120000) for _ in range(isl)]
body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "stream": False}).encode()
t = time.time()
r = urllib.request.urlopen(urllib.request.Request(f"http://{host}:{port}/generate", data=body, headers={"Content-Type": "application/json"}), timeout=600)
d = json.load(r); dt = time.time() - t
print(f"prefill {d['meta_info']['prompt_tokens']} tokens in {dt:.3f}s = {d['meta_info']['prompt_tokens']/dt:,.0f} tok/s")
PY
echo "== stop_profile"
curl -fsS -X POST "http://$API_HOST:$API_PORT/stop_profile"; echo
for i in $(seq 1 60); do ls "$prof_dir_r0"/*.trace.json* >/dev/null 2>&1 && break; sleep 2; done
ls -la "$prof_dir_r0"
for f in "$prof_dir_r0"/*.trace.json*; do
  echo "== rank0 trace: $f"; python3 tools/analyze_trace.py "$f" --top 30 | tee "results/profile/$label-rank0.txt"
done
if ssh -o BatchMode=yes "$RANK1_SSH" "ls $prof_dir_r0/*.trace.json* >/dev/null 2>&1"; then
  f="$(ssh -o BatchMode=yes "$RANK1_SSH" "ls $prof_dir_r0/*.trace.json* | head -1")"
  scp -q "$RANK1_SSH:$f" "results/profile/$label-rank1$(basename "$f" | sed 's/.*\(\.trace\.json.*\)/\1/')"
  echo "== rank1 trace: $f"; python3 tools/analyze_trace.py results/profile/$label-rank1* --top 15 | tee "results/profile/$label-rank1.txt"
fi
