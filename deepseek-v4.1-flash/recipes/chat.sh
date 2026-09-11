#!/usr/bin/env bash
# ./chat.sh [--thinking] "prompt"   -> one chat completion against the API (API_URL from config.env follows SWAP_RANKS)
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; source "$script_dir/config.env"
client_host="${API_URL#http://}"; client_host="${client_host%%:*}"; effort=none; if [[ "${1:-}" == --thinking ]]; then effort=high; shift; fi
prompt="${1:?prompt required}"
python3 - "$prompt" "$effort" "$client_host" "$API_PORT" "$SERVED_MODEL_NAME" <<'PY'
import json,sys,urllib.request,time
prompt,effort,host,port,model=sys.argv[1:6]
body={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":2048,"temperature":1.0,"top_p":0.95,"reasoning_effort":effort}
t=time.time()
r=urllib.request.urlopen(urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)
d=json.load(r); m=d["choices"][0]["message"]
if m.get("reasoning_content"): print("[reasoning]\n"+m["reasoning_content"]+"\n")
print(m.get("content")); u=d.get("usage",{}); print(f"\n[{u.get('prompt_tokens')} prompt / {u.get('completion_tokens')} completion tokens in {time.time()-t:.1f}s]")
PY
