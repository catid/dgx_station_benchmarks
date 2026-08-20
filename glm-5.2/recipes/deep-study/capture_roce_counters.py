#!/usr/bin/env python3
"""Emit a read-only snapshot of RoCE and netdev counters as JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import subprocess
import time
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.:-]+$")


def read_integer_files(root: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    if not root.is_dir():
        return values
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        try:
            values[path.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return values


def ethtool_stats(interface: str) -> dict[str, int]:
    try:
        output = subprocess.run(
            ["ethtool", "-S", interface],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    values: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"\s*([^:]+):\s*(-?\d+)\s*$", line)
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", required=True)
    parser.add_argument("--hca", required=True)
    args = parser.parse_args()
    if not SAFE_NAME.fullmatch(args.iface) or not SAFE_NAME.fullmatch(args.hca):
        raise SystemExit("unsafe interface or HCA name")

    net_root = Path("/sys/class/net") / args.iface
    ib_root = Path("/sys/class/infiniband") / args.hca / "ports" / "1"
    if not net_root.is_dir() or not ib_root.is_dir():
        raise SystemExit("interface or HCA sysfs path is missing")
    snapshot = {
        "schema_version": 1,
        "host": platform.node(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "interface": args.iface,
        "hca": args.hca,
        "netdev": read_integer_files(net_root / "statistics"),
        "rdma": read_integer_files(ib_root / "counters"),
        "rdma_hw": read_integer_files(ib_root / "hw_counters"),
        "ethtool": ethtool_stats(args.iface),
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
