#!/usr/bin/env python3
"""Resolve and validate one pinned GLM-5.2 deep-study profile."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifests"
WINNER_BACKENDS = {"flashinfer_cutedsl", "flashinfer_cutlass", "cutlass"}
VLLM_BACKENDS = WINNER_BACKENDS | {"flashinfer_trtllm"}
SGLANG_BACKENDS = {
    "flashinfer_cutedsl",
    "flashinfer_cutlass",
    "flashinfer_trtllm",
    "flashinfer_trtllm_routed",
}
ALLOWED_MTP = {0, 1, 2, 3, 5}
ALLOWED_CHUNKS = {4096, 8192, 16384, 32768}
ALLOWED_CUDAGRAPH_SIZES = {1, 2, 4, 8, 16, 32, 64, 128}
ALLOWED_DRAFT_MOE_BACKENDS = {"flashinfer_cutlass"}


class PlanError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PlanError(f"{path}: expected schema_version 1 JSON object")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve(profile_id: str, winner_backend: str | None = None) -> dict[str, Any]:
    software = load_json(MANIFEST_DIR / "software.json")
    experiments = load_json(MANIFEST_DIR / "experiments.json")
    profiles = experiments.get("profiles", {})
    if profile_id not in profiles:
        raise PlanError(f"unknown profile: {profile_id}")

    plan = deep_merge(experiments["defaults"], profiles[profile_id])
    plan["profile_id"] = profile_id
    plan["container_name"] = f"glm52-deep-{profile_id}"
    if plan.get("moe_backend") == "$WINNER_BACKEND":
        if winner_backend not in WINNER_BACKENDS:
            choices = ", ".join(sorted(WINNER_BACKENDS))
            raise PlanError(
                f"{profile_id} requires --winner-backend; choose one of: {choices}"
            )
        plan["moe_backend"] = winner_backend
        plan["selected_winner_backend"] = winner_backend

    runtime = str(plan.get("runtime"))
    runtime_pin = software.get("runtimes", {}).get(runtime)
    if not isinstance(runtime_pin, dict):
        raise PlanError(f"unsupported runtime: {runtime}")
    plan["runtime_pin"] = runtime_pin
    plan["target"] = software["target"]
    validate(plan)
    return plan


def validate(plan: dict[str, Any]) -> None:
    profile_id = str(plan.get("profile_id", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,80}", profile_id):
        raise PlanError("profile ID is not safe for a named container")
    container = str(plan.get("container_name", ""))
    if container != f"glm52-deep-{profile_id}":
        raise PlanError("container name must be derived exactly from the profile ID")

    runtime = plan.get("runtime")
    topology = plan.get("topology")
    tp_size = int(plan.get("tp_size", 0))
    pp_size = int(plan.get("pp_size", 0))
    ep_size = int(plan.get("expert_parallel_size", 0))
    if topology == "tp2" and (tp_size, pp_size, ep_size) != (2, 1, 2):
        raise PlanError("tp2 requires tp=2, pp=1, ep=2")
    if topology == "pp2" and (tp_size, pp_size, ep_size) != (1, 2, 1):
        raise PlanError("pp2 requires tp=1, pp=2, ep=1")
    if topology not in {"tp2", "pp2"}:
        raise PlanError(f"unsupported topology: {topology}")

    partition = plan.get("pp_layer_partition")
    if topology == "pp2" and partition != "40,38":
        raise PlanError("every PP2 profile must use the audited 40,38 layer split")
    if topology == "tp2" and partition is not None:
        raise PlanError("TP2 must not set a pipeline layer partition")

    speculation = plan.get("speculation", {})
    method = speculation.get("method")
    mtp_tokens = int(speculation.get("num_speculative_tokens", -1))
    if mtp_tokens not in ALLOWED_MTP:
        raise PlanError(f"unsupported MTP token count: {mtp_tokens}")
    if mtp_tokens == 0 and method != "none":
        raise PlanError("MTP=0 must use method=none")
    if mtp_tokens > 0 and method != "mtp":
        raise PlanError("only native method=mtp is permitted when draft tokens are nonzero")
    if mtp_tokens > 0 and (runtime != "vllm" or topology != "tp2"):
        raise PlanError("the pinned matrix permits native MTP only on vLLM TP2")
    if topology == "pp2" and mtp_tokens != 0:
        raise PlanError("PP2 plus speculative decoding is forbidden")

    backend = plan.get("moe_backend")
    if runtime == "vllm" and backend not in VLLM_BACKENDS:
        raise PlanError(f"unsupported vLLM MoE backend: {backend}")
    if runtime == "sglang" and backend not in SGLANG_BACKENDS:
        raise PlanError(f"unsupported SGLang MoE runner: {backend}")
    if runtime == "sglang" and mtp_tokens:
        raise PlanError("SGLang speculative GLM TP2 is quarantined in this study")
    draft_moe_backend = speculation.get("moe_backend")
    if mtp_tokens == 0 and draft_moe_backend is not None:
        raise PlanError("MTP=0 must not set a draft MoE backend")
    if mtp_tokens > 0:
        if backend != "flashinfer_cutedsl":
            raise PlanError("the audited split-MTP path requires a CuTeDSL target")
        if draft_moe_backend not in ALLOWED_DRAFT_MOE_BACKENDS:
            raise PlanError(
                "CuTeDSL MTP requires the audited FlashInfer CUTLASS draft override"
            )
        if plan.get("vllm_use_v2_model_runner") != 0:
            raise PlanError("the audited split-MTP path requires VLLM_USE_V2_MODEL_RUNNER=0")
    if plan.get("phase") == "native-mtp" and plan.get("vllm_use_v2_model_runner") != 0:
        raise PlanError("the MTP control and candidates must use the same MRv1 path")
    if topology == "pp2" and bool(plan.get("flashinfer_autotune")):
        raise PlanError("FlashInfer autotune is forbidden for PP2")
    if bool(plan.get("flashinfer_autotune")) and backend == "cutlass":
        raise PlanError("FlashInfer autotune is not an A/B axis for vLLM's CUTLASS backend")

    if float(plan.get("cpu_offload_gb", -1)) != 0:
        raise PlanError("CPU offload is excluded")
    if plan.get("kv_cache_dtype") != "fp8_e4m3":
        raise PlanError("performance profiles must keep fp8_e4m3 KV fixed")
    if plan.get("model_revision") != plan["target"]["revision"]:
        raise PlanError("profile checkpoint revision does not match the pin")
    image = str(plan["runtime_pin"].get("image", ""))
    if "@sha256:" not in image:
        raise PlanError("runtime image must be digest-pinned")

    for field in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if int(plan.get(field, 0)) <= 0:
            raise PlanError(f"{field} must be positive")
    enforce_eager = plan.get("enforce_eager")
    if not isinstance(enforce_eager, bool):
        raise PlanError("enforce_eager must be boolean")
    block_size = plan.get("block_size")
    if block_size is not None and int(block_size) != 64:
        raise PlanError("only the source-audited 64-token block override is permitted")
    graph_sizes = [int(value) for value in plan.get("cudagraph_capture_sizes", [])]
    if enforce_eager:
        if runtime != "vllm" or graph_sizes:
            raise PlanError("eager correctness profiles must use vLLM with no CUDA graphs")
        if plan.get("phase") != "pp-correctness-smoke":
            raise PlanError("eager execution is restricted to the PP correctness smoke")
    elif (
        not graph_sizes
        or graph_sizes != sorted(set(graph_sizes))
        or any(value not in ALLOWED_CUDAGRAPH_SIZES for value in graph_sizes)
        or graph_sizes[-1] > int(plan["max_num_seqs"])
    ):
        raise PlanError("invalid or oversized CUDA graph capture grid")
    cache_profile_id = plan.get("cache_profile_id")
    if cache_profile_id is not None and not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{2,80}", str(cache_profile_id)
    ):
        raise PlanError("compiler-cache profile ID is unsafe")
    utilization = float(plan.get("gpu_memory_utilization", 0))
    if not 0.5 <= utilization <= 0.95:
        raise PlanError("GPU memory utilization must stay in [0.5, 0.95]")

    benchmark = plan.get("benchmark", {})
    if benchmark.get("mode") not in {"headline", "prefill-only", "bootstrap", "correctness-smoke"}:
        raise PlanError("unsupported benchmark mode")
    if (
        benchmark.get("mode") == "correctness-smoke"
        and plan.get("phase") != "pp-correctness-smoke"
    ):
        raise PlanError("correctness-smoke mode is restricted to its audited PP profile")
    if int(benchmark.get("minimum_kv_tokens", 0)) <= 0:
        raise PlanError("capacity gate requires a positive minimum_kv_tokens")
    prefill = [int(value) for value in benchmark.get("prefill_context_tokens", [])]
    if not prefill or max(prefill) >= int(plan["max_model_len"]):
        raise PlanError("prefill target must fit below max_model_len")
    if plan.get("phase") == "native-mtp-split-bootstrap":
        expected = {
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 16384,
        }
        if any(int(plan[key]) != value for key, value in expected.items()):
            raise PlanError("split-MTP bootstrap envelope must remain conservative")
        if graph_sizes != [1, 2, 4, 8, 16]:
            raise PlanError("split-MTP bootstrap CUDA graph grid must stop at 16")
        if [int(value) for value in benchmark.get("concurrencies", [])] != [1, 2, 4, 8, 16]:
            raise PlanError("split-MTP bootstrap concurrency grid must be C1-C16")
        if prefill != [8192]:
            raise PlanError("split-MTP bootstrap permits only the 8K prefill check")
        if int(benchmark.get("minimum_kv_tokens", 0)) != 24578:
            raise PlanError("split-MTP bootstrap capacity gate must cover 8,194 + 16x1,024")
    if plan.get("phase") == "native-mtp-split-short-context":
        expected = {
            "max_model_len": 9216,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 16384,
        }
        if any(int(plan[key]) != value for key, value in expected.items()):
            raise PlanError("split-MTP short-context envelope must remain fixed")
        if float(plan["gpu_memory_utilization"]) != 0.95:
            raise PlanError("split-MTP short-context utilization must remain 0.95")
        if graph_sizes != [1, 2, 4, 8]:
            raise PlanError("split-MTP short-context CUDA graph grid must stop at 8")
        if [int(value) for value in benchmark.get("concurrencies", [])] != [1, 2, 4, 8]:
            raise PlanError("split-MTP short-context grid must be C1-C8")
        if prefill != [8192]:
            raise PlanError("split-MTP short-context permits only the 8K prefill check")
        if int(benchmark.get("minimum_kv_tokens", 0)) != 16384:
            raise PlanError("split-MTP short-context gate must cover 8K + 8x1K")
        if cache_profile_id != "vllm-tp2-mtp1-split-bootstrap":
            raise PlanError("split-MTP short-context must reuse the measured P5 cache")
    if plan.get("phase") == "pp-correctness-smoke":
        expected = {
            "max_model_len": 16384,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 16384,
            "chunked_prefill_size": 16384,
        }
        if runtime != "vllm" or topology != "pp2":
            raise PlanError("PP correctness smoke must use vLLM PP2")
        if any(int(plan[key]) != value for key, value in expected.items()):
            raise PlanError("PP correctness-smoke envelope must remain fixed")
        if block_size != 64 or not enforce_eager or graph_sizes:
            raise PlanError("PP correctness smoke requires block64, eager mode, and no graphs")
        if plan["moe_backend"] != "flashinfer_cutedsl" or plan["flashinfer_autotune"]:
            raise PlanError("PP correctness smoke requires CuTeDSL with autotune off")
        if float(plan["gpu_memory_utilization"]) != 0.9:
            raise PlanError("PP correctness smoke utilization must remain 0.9")
        if benchmark.get("mode") != "correctness-smoke" or plan.get("reportable") is not False:
            raise PlanError("PP correctness smoke is nonreportable correctness evidence")
        if [int(value) for value in benchmark.get("concurrencies", [])] != [1]:
            raise PlanError("PP correctness smoke permits only C1")
        if int(benchmark.get("decode_context_tokens", 0)) != 8192:
            raise PlanError("PP correctness smoke requires the exact 8K prompt target")
        if int(benchmark.get("max_output_tokens", 0)) != 4608:
            raise PlanError("PP correctness smoke requires the retained 4,608-token audit")
        if (
            int(benchmark.get("duration_seconds", -1)) != 0
            or float(benchmark.get("temperature", -1)) != 0
            or benchmark.get("shared_prefix") is not True
        ):
            raise PlanError("PP correctness smoke workload controls must remain deterministic")
        if prefill != [8192] or int(benchmark.get("minimum_kv_tokens", 0)) != 12802:
            raise PlanError("PP correctness smoke capacity must cover 8,194 + 4,608 tokens")
    if plan.get("phase") == "prefill-chunk":
        chunk = int(plan.get("chunked_prefill_size", 0))
        if chunk not in ALLOWED_CHUNKS:
            raise PlanError(f"unsupported prefill chunk: {chunk}")
        if runtime == "vllm" and int(plan["max_num_batched_tokens"]) != chunk:
            raise PlanError("vLLM prefill chunk sweep must change only max_num_batched_tokens")


def shell_assignments(plan: dict[str, Any]) -> str:
    benchmark = plan["benchmark"]
    speculation = plan["speculation"]
    fields: dict[str, Any] = {
        "PROFILE_ID": plan["profile_id"],
        "RUNTIME": plan["runtime"],
        "RUNTIME_IMAGE": plan["runtime_pin"]["image"],
        "MODEL_REVISION": plan["target"]["revision"],
        "SERVED_MODEL_NAME": plan["target"]["served_model_name"],
        "CONTAINER_NAME": plan["container_name"],
        "TOPOLOGY": plan["topology"],
        "TP_SIZE": plan["tp_size"],
        "PP_SIZE": plan["pp_size"],
        "EP_SIZE": plan["expert_parallel_size"],
        "PP_LAYER_PARTITION": plan.get("pp_layer_partition") or "",
        "MOE_BACKEND": plan["moe_backend"],
        "FLASHINFER_AUTOTUNE": "on" if plan["flashinfer_autotune"] else "off",
        "MTP_TOKENS": speculation["num_speculative_tokens"],
        "MTP_DRAFT_MOE_BACKEND": speculation.get("moe_backend") or "",
        "VLLM_USE_V2_MODEL_RUNNER": (
            "" if plan.get("vllm_use_v2_model_runner") is None
            else plan["vllm_use_v2_model_runner"]
        ),
        "CUDAGRAPH_CAPTURE_SIZES": " ".join(
            map(str, plan["cudagraph_capture_sizes"])
        ),
        "MAX_CUDAGRAPH_CAPTURE_SIZE": (
            max(plan["cudagraph_capture_sizes"])
            if plan["cudagraph_capture_sizes"] else 0
        ),
        "BLOCK_SIZE": plan.get("block_size") or "",
        "ENFORCE_EAGER": "yes" if plan.get("enforce_eager") else "no",
        "CACHE_PROFILE_ID": plan.get("cache_profile_id") or "",
        "KV_CACHE_DTYPE": plan["kv_cache_dtype"],
        "MAX_MODEL_LEN": plan["max_model_len"],
        "MAX_NUM_SEQS": plan["max_num_seqs"],
        "MAX_NUM_BATCHED_TOKENS": plan["max_num_batched_tokens"],
        "CHUNKED_PREFILL_SIZE": plan["chunked_prefill_size"],
        "GPU_MEMORY_UTILIZATION": plan["gpu_memory_utilization"],
        "BENCHMARK_MODE": benchmark["mode"],
        "CONCURRENCIES": ",".join(map(str, benchmark["concurrencies"])),
        "DECODE_CONTEXT": benchmark["decode_context_tokens"],
        "MAX_OUTPUT_TOKENS": benchmark["max_output_tokens"],
        "DURATION_SECONDS": benchmark["duration_seconds"],
        "PREFILL_CONTEXTS": ",".join(map(str, benchmark["prefill_context_tokens"])),
        "MINIMUM_KV_TOKENS": benchmark["minimum_kv_tokens"],
        "REPORTABLE": "yes" if plan.get("reportable") else "no",
    }
    return "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in fields.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", nargs="?")
    parser.add_argument("--winner-backend", choices=sorted(WINNER_BACKENDS))
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = load_json(MANIFEST_DIR / "experiments.json")
    profiles = experiments.get("profiles", {})
    if args.list:
        for profile_id in profiles:
            print(profile_id)
        return 0
    if args.validate_all:
        failures: list[str] = []
        for profile_id, raw in profiles.items():
            winner = "flashinfer_cutedsl" if raw.get("moe_backend") == "$WINNER_BACKEND" else None
            try:
                resolve(profile_id, winner)
            except PlanError as exc:
                failures.append(f"{profile_id}: {exc}")
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print(f"validated {len(profiles)} profiles")
        return 0
    if not args.profile:
        raise PlanError("provide PROFILE, --list, or --validate-all")
    plan = resolve(args.profile, args.winner_backend)
    if args.format == "shell":
        print(shell_assignments(plan))
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanError as exc:
        print(f"plan rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
