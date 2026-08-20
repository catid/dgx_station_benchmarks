#!/usr/bin/env python3
"""CPU-only validation of manifests and fail-closed plan rules."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_plan import MANIFEST_DIR, PlanError, load_json, resolve, validate  # noqa: E402


def must_reject(plan: dict, message: str) -> None:
    try:
        validate(plan)
    except PlanError:
        return
    raise AssertionError(f"unsafe plan was accepted: {message}")


def main() -> None:
    software = load_json(MANIFEST_DIR / "software.json")
    experiments = load_json(MANIFEST_DIR / "experiments.json")
    exclusions = load_json(MANIFEST_DIR / "exclusions.json")
    checkpoints = load_json(MANIFEST_DIR / "checkpoints.json")
    assert software["target"]["revision"] == "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
    assert all("@sha256:" in runtime["image"] for runtime in software["runtimes"].values())
    assert len(exclusions["exclusions"]) >= 10
    assert sum(item["disposition"] == "target" for item in checkpoints["checkpoints"]) == 1

    resolved = {}
    for profile_id, raw in experiments["profiles"].items():
        winner = "flashinfer_cutedsl" if raw.get("moe_backend") == "$WINNER_BACKEND" else None
        resolved[profile_id] = resolve(profile_id, winner)

    assert resolved["vllm-tp2-exact"]["moe_backend"] == "flashinfer_cutedsl"
    assert resolved["vllm-pp2-balanced"]["pp_layer_partition"] == "40,38"
    assert resolved["vllm-tp2-mtp5"]["speculation"]["num_speculative_tokens"] == 5
    assert resolved["sglang-pp2-balanced"]["speculation"]["method"] == "none"

    base = resolved["vllm-tp2-exact"]
    bad = copy.deepcopy(base)
    bad["cpu_offload_gb"] = 1
    must_reject(bad, "CPU offload")
    bad = copy.deepcopy(base)
    bad["topology"], bad["tp_size"], bad["pp_size"] = "pp2", 1, 2
    bad["expert_parallel_size"] = 1
    bad["pp_layer_partition"] = "39,39"
    must_reject(bad, "unbalanced PP")
    bad = copy.deepcopy(resolved["vllm-pp2-balanced"])
    bad["speculation"] = {"method": "mtp", "num_speculative_tokens": 1}
    must_reject(bad, "PP plus MTP")
    bad = copy.deepcopy(base)
    bad["moe_backend"] = "flashinfer_b12x"
    must_reject(bad, "SM120 backend on SM103")
    bad = copy.deepcopy(resolved["sglang-tp2-cutedsl"])
    bad["speculation"] = {"method": "mtp", "num_speculative_tokens": 1}
    must_reject(bad, "quarantined SGLang speculative TP2")
    for method in ("dspark", "dflash", "eagle3"):
        bad = copy.deepcopy(base)
        bad["speculation"] = {"method": method, "num_speculative_tokens": 1}
        must_reject(bad, f"excluded speculative method {method}")
    bad = copy.deepcopy(base)
    bad["moe_backend"] = "cutlass"
    bad["flashinfer_autotune"] = True
    must_reject(bad, "FlashInfer autotune on vLLM CUTLASS")

    summary = {
        "profiles_validated": len(resolved),
        "exclusions_validated": len(exclusions["exclusions"]),
        "negative_rules_validated": 9,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
