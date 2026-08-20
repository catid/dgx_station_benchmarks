#!/usr/bin/env python3
"""Verify the pinned GLM-5.2 NVFP4 index and measured shard bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
EXPECTED_SHARDS = 47
EXPECTED_WEIGHT_FILE_BYTES = 464_823_042_096
EXPECTED_INDEX_TENSOR_BYTES = 464_795_267_072
EXPECTED_ARCHITECTURE = "GlmMoeDsaForCausalLM"
EXPECTED_QUANTIZATION = "NVFP4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--allow-missing-revision-marker",
        action="store_true",
        help="Allow checkpoints downloaded before download_model.sh wrote PINNED_REVISION",
    )
    args = parser.parse_args()
    root = args.model_dir.resolve()

    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (root / name).is_file()]
    file_bytes = sum((root / name).stat().st_size for name in shards if (root / name).is_file())
    metadata_bytes = int(index["metadata"]["total_size"])
    architectures = config.get("architectures", [])
    quant_algo = config.get("quantization_config", {}).get("quant_algo")
    marker = root / "PINNED_REVISION"

    checks = {
        "indexed_shards": len(shards) == EXPECTED_SHARDS,
        "missing_shards": not missing,
        "weight_file_bytes": file_bytes == EXPECTED_WEIGHT_FILE_BYTES,
        "index_tensor_bytes": metadata_bytes == EXPECTED_INDEX_TENSOR_BYTES,
        "architecture": EXPECTED_ARCHITECTURE in architectures,
        "quantization": quant_algo == EXPECTED_QUANTIZATION,
        "revision_marker": (
            args.allow_missing_revision_marker
            if not marker.exists()
            else marker.read_text().strip() == EXPECTED_REVISION
        ),
    }
    report = {
        "model_dir": str(root),
        "expected_revision": EXPECTED_REVISION,
        "indexed_shards": len(shards),
        "missing_shards": missing,
        "weight_file_bytes": file_bytes,
        "index_tensor_bytes": metadata_bytes,
        "architecture": architectures,
        "quantization": quant_algo,
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"Verification failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
