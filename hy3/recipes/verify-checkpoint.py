#!/usr/bin/env python3
"""Verify the structural invariants of the pinned Hy3-FP8 checkpoint."""

import json
import sys
from pathlib import Path

EXPECTED_SHARDS = 101
EXPECTED_WEIGHT_BYTES = 299_889_838_946


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} MODEL_DIR")
    root = Path(sys.argv[1]).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")

    index = json.loads(index_path.read_text())
    indexed = sorted(set(index["weight_map"].values()))
    shards = sorted(root.glob("model-*-of-*.safetensors"))
    missing = [name for name in indexed if not (root / name).is_file()]
    empty = [path.name for path in shards if path.stat().st_size == 0]
    weight_bytes = sum(path.stat().st_size for path in shards)

    print(f"checkpoint={root}")
    print(f"indexed_shards={len(indexed)}")
    print(f"present_shards={len(shards)}")
    print(f"weight_bytes={weight_bytes}")
    print(f"missing_shards={len(missing)}")
    print(f"empty_shards={len(empty)}")

    if len(indexed) != EXPECTED_SHARDS or len(shards) != EXPECTED_SHARDS:
        raise SystemExit("unexpected shard count")
    if weight_bytes != EXPECTED_WEIGHT_BYTES:
        raise SystemExit("unexpected aggregate safetensors byte count")
    if missing or empty:
        raise SystemExit(f"invalid checkpoint: missing={missing}, empty={empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
