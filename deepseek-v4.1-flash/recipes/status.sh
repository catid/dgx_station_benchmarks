#!/usr/bin/env bash
# Container + GB300 state on both nodes (node0 = this host, node1 = $RANK1_SSH) and API health (API_URL follows SWAP_RANKS).
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; source "$script_dir/config.env"
for h in node0 node1; do
  if [[ $h == node0 ]]; then pre=(); else pre=(ssh -o BatchMode=yes "$RANK1_SSH"); fi
  st="$("${pre[@]}" docker inspect -f '{{.State.Status}} since {{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null || echo absent)"
  gpu="$("${pre[@]}" nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw --format=csv,noheader | awk -F', ' '$1 ~ /GB300/ {print $2", util "$3", "$4}')"
  printf '%-8s container: %-45s GB300: %s\n' "$h" "$st" "$gpu"
done
curl -fsS --max-time 3 "$API_URL/health" >/dev/null && echo "API: healthy at $API_URL" || echo "API: not ready"
curl -fsS --max-time 3 "$API_URL/server_info" 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print("max_total_num_tokens:", d.get("max_total_num_tokens"), "| version:", d.get("version"))' 2>/dev/null || true
