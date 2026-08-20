#!/usr/bin/env bash
set -euo pipefail

readonly fabric_iface="${FABRIC_IFACE:-enP1p3s0f0np0}"
readonly fabric_hca="${FABRIC_HCA:-mlx5_0}"
readonly peer_ip="${1:?Usage: $0 PEER_IP}"

ip -brief address show dev "$fabric_iface"
ip -details link show dev "$fabric_iface" | sed -n '1,3p'
[[ "$(cat "/sys/class/net/$fabric_iface/mtu")" == 9000 ]] || {
  echo "$fabric_iface must use MTU 9000" >&2
  exit 1
}
[[ -e "/sys/class/infiniband/$fabric_hca" ]] || {
  echo "Missing HCA $fabric_hca" >&2
  exit 1
}
ping -I "$fabric_iface" -c 3 "$peer_ip"
printf 'Fabric checks passed for %s via %s (%s).\n' "$peer_ip" "$fabric_iface" "$fabric_hca"
