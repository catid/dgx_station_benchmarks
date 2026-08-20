#!/usr/bin/env python3
"""Fail closed on incomplete, capacity-limited, or unpinned deep-study runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_plan import validate  # noqa: E402
from validate_quality import validate_natural  # noqa: E402


FATAL_LOG_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "illegal memory access",
    "traceback (most recent call last)",
    "nv_err_invalid_state",
    "rminitadapter failed",
)


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_inspect(path: Path) -> dict:
    value = json.loads(path.read_text())
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one container inspection")
    return value


def option(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def env_map(inspect: dict) -> dict[str, str]:
    result = {}
    for item in inspect.get("Config", {}).get("Env", []):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def validate_runtime(run: Path, plan: dict) -> tuple[list[str], list[str]]:
    runtime = run / "runtime"
    expected_image = plan["runtime_pin"]["image"]
    logs: list[str] = []
    for rank in (0, 1):
        item = load_inspect(runtime / f"node{rank}-inspect.json")
        config = item.get("Config", {})
        if config.get("Image") != expected_image:
            raise ValueError(f"node{rank}: runtime image is not the pinned digest")
        if item.get("Name", "").lstrip("/") != plan["container_name"]:
            raise ValueError(f"node{rank}: wrong named container")
        command = config.get("Cmd", [])
        if not isinstance(command, list):
            raise ValueError(f"node{rank}: invalid container command")
        environment = env_map(item)
        if plan["benchmark"]["mode"] != "bootstrap" and environment.get("NCCL_DEBUG") != "WARN":
            raise ValueError(f"node{rank}: reportable run used verbose NCCL logging")
        for forbidden_env in (
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_IB_QPS_PER_CONNECTION",
            "NCCL_IB_SPLIT_DATA_ON_QPS",
        ):
            if forbidden_env in environment:
                raise ValueError(f"node{rank}: headline run forced {forbidden_env}")
        if plan["topology"] == "pp2":
            partition_var = "VLLM_PP_LAYER_PARTITION" if plan["runtime"] == "vllm" else "SGLANG_PP_LAYER_PARTITION"
            if environment.get(partition_var) != "40,38":
                raise ValueError(f"node{rank}: PP layer split is not 40,38")
        if any(flag in command for flag in ("--cpu-offload-gb", "--cpu-offload")):
            raise ValueError(f"node{rank}: CPU offload appeared in the command")

        expected = {
            "--served-model-name": plan["target"]["served_model_name"],
            "--kv-cache-dtype": plan["kv_cache_dtype"],
        }
        if plan["runtime"] == "vllm":
            expected.update(
                {
                    "--tensor-parallel-size": str(plan["tp_size"]),
                    "--pipeline-parallel-size": str(plan["pp_size"]),
                    "--nnodes": "2",
                    "--node-rank": str(rank),
                    "--max-model-len": str(plan["max_model_len"]),
                    "--max-num-seqs": str(plan["max_num_seqs"]),
                    "--max-num-batched-tokens": str(plan["max_num_batched_tokens"]),
                    "--gpu-memory-utilization": str(plan["gpu_memory_utilization"]),
                    "--moe-backend": plan["moe_backend"],
                }
            )
            if plan["tp_size"] == 2 and "--enable-expert-parallel" not in command:
                raise ValueError(f"node{rank}: TP2 lacks expert parallelism")
            if "--enable-prefix-caching" not in command:
                raise ValueError(f"node{rank}: prefix caching is not enabled")
            autotune_flag = (
                "--enable-flashinfer-autotune"
                if plan["flashinfer_autotune"]
                else "--no-enable-flashinfer-autotune"
            )
            if autotune_flag not in command:
                raise ValueError(f"node{rank}: wrong FlashInfer autotune state")
            spec_value = option(command, "--speculative-config")
            expected_mtp = int(plan["speculation"]["num_speculative_tokens"])
            if expected_mtp:
                try:
                    spec = json.loads(spec_value or "")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"node{rank}: invalid speculative config") from exc
                if spec != {"method": "mtp", "num_speculative_tokens": expected_mtp}:
                    raise ValueError(f"node{rank}: wrong MTP configuration")
            elif spec_value is not None:
                raise ValueError(f"node{rank}: MTP0 must omit speculative config")
        else:
            expected.update(
                {
                    "--tp-size": str(plan["tp_size"]),
                    "--pp-size": str(plan["pp_size"]),
                    "--ep-size": str(plan["expert_parallel_size"]),
                    "--nnodes": "2",
                    "--node-rank": str(rank),
                    "--context-length": str(plan["max_model_len"]),
                    "--max-running-requests": str(plan["max_num_seqs"]),
                    "--chunked-prefill-size": str(plan["chunked_prefill_size"]),
                    "--mem-fraction-static": str(plan["gpu_memory_utilization"]),
                    "--moe-runner-backend": plan["moe_backend"],
                    "--quantization": "modelopt_fp4",
                }
            )
            if plan["pp_size"] == 2 and "--disable-overlap-schedule" not in command:
                raise ValueError(f"node{rank}: SGLang PP2 must disable overlap scheduling")
            if any(arg.startswith("--speculative-") for arg in command):
                raise ValueError(f"node{rank}: SGLang speculative flags are quarantined")
        for name, wanted in expected.items():
            if option(command, name) != wanted:
                raise ValueError(f"node{rank}: {name} does not prove {wanted}")

        log = (runtime / f"node{rank}-server.log").read_text(errors="replace")
        folded = log.casefold()
        for marker in FATAL_LOG_MARKERS:
            if marker in folded:
                raise ValueError(f"node{rank}: fatal marker in server log: {marker}")
        logs.append(log)

    joined_logs = "\n".join(logs)
    if plan["runtime"] == "vllm":
        backend_marker = {
            "flashinfer_cutedsl": "'FLASHINFER_CUTEDSL' NvFp4 MoE backend",
            "flashinfer_cutlass": "'FLASHINFER_CUTLASS' NvFp4 MoE backend",
            "cutlass": "'VLLM_CUTLASS' NvFp4 MoE backend",
            "flashinfer_trtllm": "'FLASHINFER_TRTLLM' NvFp4 MoE backend",
        }[plan["moe_backend"]]
        if backend_marker not in joined_logs:
            raise ValueError("server logs do not prove the requested NVFP4 MoE backend")
        for marker in ("FLASHINFER_MLA_SPARSE attention backend", "TRTLLM_RAGGED MLA prefill backend"):
            if marker not in joined_logs:
                raise ValueError(f"server logs do not prove fixed attention path: {marker}")
    elif plan["moe_backend"].casefold() not in joined_logs.casefold():
        raise ValueError("SGLang logs do not mention the requested MoE runner")

    capacities: list[str] = []
    patterns = (
        re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", re.I),
        re.compile(r"max_total_num_tokens[=:]\s*([0-9,]+)", re.I),
    )
    for log in logs:
        values = []
        for pattern in patterns:
            values.extend(int(match.replace(",", "")) for match in pattern.findall(log))
        if values:
            capacities.append(str(values[-1]))
    if not capacities:
        raise ValueError("server logs do not expose a KV/token capacity")
    minimum = int(plan["benchmark"]["minimum_kv_tokens"])
    if min(map(int, capacities)) < minimum:
        raise ValueError(f"KV/token capacity {capacities} is below required {minimum}")
    return capacities, logs


def validate_benchmark(run: Path, plan: dict) -> dict:
    data = load_object(run / "benchmark" / "llm-inference-bench.json")
    expected_prefill = {str(value) for value in plan["benchmark"]["prefill_context_tokens"]}
    prefill = data.get("prefill", {})
    if not expected_prefill.issubset(prefill):
        raise ValueError("benchmark is missing one or more prefill targets")
    for context in expected_prefill:
        cell = prefill[context]
        value = float(cell.get("tok_per_sec") or cell.get("client_tok_per_sec") or 0)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"prefill {context} lacks a finite positive measurement")

    results = data.get("results", [])
    if plan["benchmark"]["mode"] == "prefill-only":
        if results:
            raise ValueError("prefill-only profile unexpectedly contains decode rows")
        return {"decode_rows": 0, "prefill_rows": len(expected_prefill)}

    expected_c = [int(value) for value in plan["benchmark"]["concurrencies"]]
    by_c = {int(row.get("concurrency", -1)): row for row in results}
    if sorted(by_c) != expected_c or len(results) != len(expected_c):
        raise ValueError("decode concurrency grid is incomplete or duplicated")
    mtp = int(plan["speculation"]["num_speculative_tokens"])
    for concurrency in expected_c:
        row = by_c[concurrency]
        if int(row.get("num_errors", -1)) != 0:
            raise ValueError(f"C{concurrency}: request errors")
        if row.get("capacity_limited") is not False:
            raise ValueError(f"C{concurrency}: capacity limited")
        if int(row.get("max_running_reqs", 0)) < concurrency:
            raise ValueError(f"C{concurrency}: target concurrency was not resident")
        tps = float(row.get("aggregate_tps", 0))
        if not math.isfinite(tps) or tps <= 0:
            raise ValueError(f"C{concurrency}: invalid aggregate throughput")
        if mtp and concurrency == 1:
            if float(row.get("server_spec_accept_length", 0)) <= 0:
                raise ValueError("MTP run lacks a positive acceptance length")
            if int(row.get("server_spec_drafts", 0)) <= 0:
                raise ValueError("MTP run lacks draft-token telemetry")
    return {"decode_rows": len(results), "prefill_rows": len(expected_prefill)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    plan = load_object(run / "resolved-plan.json")
    validate(plan)
    for node in ("node0", "node1"):
        checkpoint = load_object(run / "runtime" / f"{node}-checkpoint-verification.json")
        checks = checkpoint.get("checks", {})
        if not checks or not all(checks.values()):
            raise ValueError(f"{node}: checkpoint verification did not pass every check")
        if checkpoint.get("expected_revision") != plan["target"]["revision"]:
            raise ValueError(f"{node}: checkpoint verifier used a different revision")
    if "Idle GB300 HBM:" not in (run / "runtime" / "launch.log").read_text(errors="replace"):
        raise ValueError("launch evidence does not prove the clean-idle HBM preflight passed")
    if (run / "runtime" / "model-revision.txt").read_text().strip() != plan["target"]["revision"]:
        raise ValueError("runtime model revision does not match the plan")
    expected_bench = "0b4185b5b435e948b199c9077a00b084864aa963"
    if (run / "runtime" / "llm-inference-bench-commit.txt").read_text().strip() != expected_bench:
        raise ValueError("runtime benchmark checkout does not match the pin")
    capacities, _ = validate_runtime(run, plan)
    benchmark = validate_benchmark(run, plan)
    quality = run / "quality" / "quality-audit.json"
    if plan["benchmark"]["mode"] != "prefill-only":
        validate_natural(quality)
    network_delta = load_object(run / "network" / "delta.json")
    report = {
        "profile_id": plan["profile_id"],
        "runtime": plan["runtime"],
        "topology": plan["topology"],
        "capacity_tokens_observed": [int(value) for value in capacities],
        "capacity_tokens_required": plan["benchmark"]["minimum_kv_tokens"],
        "benchmark": benchmark,
        "natural_quality": "not-run" if plan["benchmark"]["mode"] == "prefill-only" else "passed",
        "network_health_counter_deltas": network_delta.get("health_counter_deltas", {}),
        "status": "accepted-by-structural-validator",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
