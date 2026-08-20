#!/usr/bin/env python3
"""Subtract paired RoCE snapshots and flag nonzero health-counter deltas."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


HEALTH_PATTERN = re.compile(
    r"error|discard|drop|retrans|timeout|rnr|pause|pfc|out.of.buffer|sequence",
    re.IGNORECASE,
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"invalid snapshot: {path}")
    return value


def seconds_between(before: dict, after: dict) -> float:
    start = dt.datetime.fromisoformat(before["timestamp_utc"])
    finish = dt.datetime.fromisoformat(after["timestamp_utc"])
    return (finish - start).total_seconds()


def delta_map(before: dict, after: dict, group: str) -> dict[str, int]:
    left = before.get(group, {})
    right = after.get(group, {})
    return {
        key: int(right[key]) - int(left[key])
        for key in sorted(left.keys() & right.keys())
        if int(right[key]) != int(left[key])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_root", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "nodes": {}, "health_counter_deltas": {}}
    for node in ("node0", "node1"):
        before = load(args.snapshot_root / "before" / f"{node}.json")
        after = load(args.snapshot_root / "after" / f"{node}.json")
        groups = {
            group: delta_map(before, after, group)
            for group in ("netdev", "rdma", "rdma_hw", "ethtool")
        }
        report["nodes"][node] = {
            "elapsed_seconds": seconds_between(before, after),
            "counter_deltas": groups,
        }
        report["health_counter_deltas"][node] = {
            f"{group}.{key}": value
            for group, values in groups.items()
            for key, value in values.items()
            if value and HEALTH_PATTERN.search(key)
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
