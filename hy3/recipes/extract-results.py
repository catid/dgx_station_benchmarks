#!/usr/bin/env python3
"""Validate Hy3 result directories and emit compact publication artifacts.

Canonical result names are accepted only with the complete C1-C128 matrix,
``runtime/`` evidence, and a natural-quality audit.  The preserved 0.92-memory
PP2/MTP0 run has a separate name and is always labeled ``provisional_tuning``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EVIDENCE = DATA / "evidence"
RUNTIME = DATA / "runtime"
MODEL_REVISION = "ecc1d8e194e093f33177f2f0ef7ce8f397b2d68b"
RUNTIME_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
)
MTP_RUNTIME_IMAGE = "hy3-vllm:0.27.1-mtp-compile"
BENCHMARK_COMMIT = "0b4185b5b435e948b199c9077a00b084864aa963"
FINAL_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]
PROVISIONAL_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
CONFIGURATIONS = [
    (topology, mtp)
    for topology in ("pp2", "tp2")
    for mtp in (0, 1, 2)
]

THROUGHPUT_FIELDS = [
    "publication_status", "topology", "tensor_parallel_size",
    "pipeline_parallel_size", "expert_parallel", "mtp_tokens",
    "mtp_compile_fix", "gpu_memory_utilization", "flashinfer_autotune",
    "concurrency", "aggregate_output_tokens_per_second", "request_count",
    "completed_requests", "num_errors", "effective_concurrency",
    "max_running_requests", "capacity_limited", "prompt_tokens",
    "max_tokens", "temperature", "measurement_seconds",
    "server_spec_draft_tokens", "server_spec_accepted_tokens",
    "speculative_acceptance_rate", "headline_eligible",
    "headline_exclusion_reason", "model_revision", "runtime_image",
    "benchmark_commit", "result_file",
]
PREFILL_FIELDS = [
    "publication_status", "topology", "tensor_parallel_size",
    "pipeline_parallel_size", "expert_parallel", "mtp_tokens",
    "gpu_memory_utilization", "flashinfer_autotune", "context_tokens",
    "actual_prompt_tokens", "prompt_tokens_per_second", "ttft_seconds",
    "samples", "num_errors", "model_revision", "runtime_image",
    "benchmark_commit", "result_file",
]
QUALITY_FIELDS = [
    "topology", "tensor_parallel_size", "pipeline_parallel_size",
    "expert_parallel", "mtp_tokens", "mtp_compile_fix", "reasoning_effort",
    "outputs", "empty_outputs", "flagged_outputs",
    "max_repeated_8gram_fraction", "max_identical_character_run",
    "max_identical_word_run", "manual_review_status", "manual_review_notes",
    "source_file",
]
STABILITY_FIELDS = [
    "publication_status", "topology", "expert_parallel", "mtp_tokens",
    "gpu_memory_utilization", "concurrency", "context_tokens", "max_tokens",
    "duration_seconds", "aggregate_output_tokens_per_second",
    "server_output_tokens_per_second", "effective_concurrency",
    "max_running_requests", "num_errors", "server_spec_draft_tokens",
    "server_spec_accepted_tokens", "speculative_acceptance_rate",
    "observed_global_kv_tokens", "required_global_kv_tokens",
    "finding", "result_file",
]
KV_CAPACITY_FIELDS = [
    "publication_status", "topology", "expert_parallel", "mtp_tokens",
    "gpu_memory_utilization", "observed_global_kv_tokens",
    "required_global_kv_tokens", "margin_tokens", "capacity_status",
    "result_directory", "runtime_file",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def generic_node_name(name: str) -> str:
    """Map the measured hostnames to publication-safe rank names."""
    if re.match(r"^node[01]-", name):
        return name
    match = re.match(r"^.+?([12])-(.+)$", name)
    if match:
        return f"node{int(match.group(1)) - 1}-{match.group(2)}"
    return name


def generic_file_manifest(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        generic_node_name(path.name): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    }


def find_rank_file(runtime_dir: Path, rank: int, suffix: str) -> Path:
    generic = runtime_dir / f"node{rank}-{suffix}"
    if generic.is_file():
        return generic
    for path in sorted(runtime_dir.glob(f"*-{suffix}")):
        prefix = path.name[: -len(suffix) - 1]
        if prefix.endswith(str(rank + 1)):
            return path
    raise ValueError(
        f"{runtime_dir}: missing rank {rank} {suffix}; "
        f"accepted names are node{rank}-{suffix} or measured-host equivalent"
    )


def parse_settings(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing runtime settings {path}")
    settings = dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    )
    required = {
        "topology", "mtp_tokens", "gpu_memory_utilization", "kv_cache_dtype",
        "max_model_len", "max_num_seqs", "max_num_batched_tokens",
        "required_kv_tokens",
    }
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"{path}: missing settings {missing}")
    return settings


def topology_values(topology: str) -> tuple[int, int, bool]:
    return (1, 2, False) if topology == "pp2" else (2, 1, True)


def validate_benchmark(
    source: Path,
    expected_concurrencies: list[int],
    *,
    duration_seconds: float = 30.0,
    prefill_contexts: tuple[int, ...] = (8192, 65536, 131072),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(source)
    metadata = data.get("metadata", {})
    if metadata.get("decode_mode") != "duration":
        raise ValueError("decode_mode must be duration")
    if float(metadata.get("duration_per_test", 0)) != duration_seconds:
        raise ValueError(f"duration_per_test must be {duration_seconds:g} seconds")
    if int(metadata.get("max_tokens", 0)) != 1024:
        raise ValueError("max_tokens must be 1024")
    if float(metadata.get("temperature", -1)) != 0.0:
        raise ValueError("temperature must be 0")
    if metadata.get("ignore_eos") is not True:
        raise ValueError("ignore_eos must be true")

    results = sorted(data.get("results", []), key=lambda row: row["concurrency"])
    found = [int(row["concurrency"]) for row in results]
    if found != expected_concurrencies:
        raise ValueError(f"expected concurrencies {expected_concurrencies}, found {found}")
    for row in results:
        concurrency = int(row["concurrency"])
        if int(row.get("num_errors", -1)) != 0:
            raise ValueError(f"C{concurrency}: request errors present")
        if int(float(row.get("input_seq_len_avg", 0))) != 8192:
            raise ValueError(f"C{concurrency}: input length is not 8192")
        for sample in row.get("request_samples", []):
            if sample.get("completed") and (
                int(sample.get("input_tokens", 0)) != 8192
                or int(sample.get("output_tokens", 0)) != 1024
            ):
                raise ValueError(f"C{concurrency}: malformed completed request")

        if concurrency > 1 and int(row.get("max_running_reqs", 0)) != concurrency:
            raise ValueError(f"C{concurrency}: full offered concurrency was not resident")

    prefill = data.get("prefill", {})
    expected_prefill = [str(value) for value in prefill_contexts]
    if sorted(prefill, key=int) != expected_prefill:
        raise ValueError(f"prefill must contain exactly {expected_prefill}")
    for context, result in prefill.items():
        expected_actual = int(context) + 2
        if int(result.get("prompt_tokens", 0)) != expected_actual:
            raise ValueError(
                f"{context} prefill: expected API count {expected_actual}, "
                f"found {result.get('prompt_tokens')}"
            )
        if float(result.get("tok_per_sec", 0)) <= 0 or float(
            result.get("ttft_seconds", 0)
        ) <= 0:
            raise ValueError(f"{context} prefill: nonpositive measurement")
    return data, results


def compact_benchmark(
    source: Path,
    data: dict[str, Any],
    results: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    keep = [
        "concurrency", "context_tokens", "benchmark_mode",
        "measurement_seconds", "client_output_tokens", "server_output_tokens",
        "aggregate_source", "aggregate_tps", "per_request_avg_tps",
        "ttft_p50", "inter_token_latency_p50", "input_seq_len_avg",
        "output_seq_len_avg", "request_count", "completed_request_count",
        "num_errors", "avg_running_reqs", "max_running_reqs",
        "effective_concurrency", "avg_queue_reqs", "max_queue_reqs",
        "capacity_limited", "warmup_timed_out", "server_gen_throughput",
        "server_spec_draft_tokens", "server_spec_accepted_tokens",
        "spec_accept_len",
    ]
    metadata = data["metadata"]
    return {
        "source_file": f"{source.parent.name}/{source.name}",
        "source_sha256": sha256(source),
        "publication_status": status,
        "metadata": {
            key: metadata.get(key)
            for key in (
                "version", "engine", "model", "timestamp", "decode_mode",
                "duration_per_test", "max_tokens", "temperature", "ignore_eos",
                "concurrency_levels", "context_lengths",
            )
        },
        "results": [{key: row.get(key) for key in keep} for row in results],
        "prefill": data["prefill"],
    }


def runtime_settings(
    runtime_dir: Path, topology: str, mtp: int, *, provisional: bool = False
) -> tuple[float, bool, int, int, dict[str, Any]]:
    log_paths = [
        find_rank_file(runtime_dir, 0, "server.log"),
        find_rank_file(runtime_dir, 1, "server.log"),
    ]
    logs = [path.read_text(errors="replace") for path in log_paths]
    joined = "\n".join(logs)
    settings: dict[str, str] | None = None
    if provisional:
        utilization = 0.92
        if "Desired GPU memory utilization is (0.92," not in joined:
            raise ValueError(f"{runtime_dir}: logs do not prove utilization 0.92")
    else:
        settings = parse_settings(runtime_dir / "run-settings.txt")
        expected_mode = "pp" if topology == "pp2" else "tp"
        if settings["topology"] != expected_mode or int(settings["mtp_tokens"]) != mtp:
            raise ValueError(f"{runtime_dir}: topology/MTP settings do not match directory")
        if settings["kv_cache_dtype"] != "fp8_e4m3":
            raise ValueError(f"{runtime_dir}: expected fp8_e4m3 KV cache")
        if int(settings["max_model_len"]) != 262144:
            raise ValueError(f"{runtime_dir}: unexpected maximum model length")
        utilization = float(settings["gpu_memory_utilization"])
        if f"gpu_memory_utilization': {utilization}" not in joined:
            raise ValueError(f"{runtime_dir}: logs do not prove utilization {utilization}")
    autotune_disabled = all(
        "Skipping FlashInfer autotune because it is disabled" in log for log in logs
    )
    autotune_enabled = (
        "enable_flashinfer_autotune=True" in joined
        or "flashinfer.jit: [Autotuner]: Autotuning process starts" in joined
    )
    if topology == "pp2" and not autotune_disabled:
        raise ValueError("PP2 runtime does not prove FlashInfer autotune was disabled")
    if not autotune_disabled and not autotune_enabled:
        raise ValueError("runtime does not prove the FlashInfer autotune setting")
    flashinfer_autotune = not autotune_disabled
    if topology == "pp2":
        if "tensor_parallel_size=1, pipeline_parallel_size=2" not in joined:
            raise ValueError("runtime does not prove TP1/PP2 topology")
    else:
        if "tensor_parallel_size=2, pipeline_parallel_size=1" not in joined:
            raise ValueError("runtime does not prove TP2/PP1 topology")
        if not re.search(r"enable_expert_parallel['\"]?\s*[:=]\s*True", joined):
            raise ValueError("TP2 runtime does not prove expert parallel is enabled")
    if "Application startup complete" not in logs[0]:
        raise ValueError("node 0 runtime log does not show a healthy API")

    expected_image = RUNTIME_IMAGE if mtp == 0 else MTP_RUNTIME_IMAGE
    if provisional:
        inspect_path = find_rank_file(runtime_dir, 0, "inspect.json")
        inspect_data = json.loads(inspect_path.read_text())
        if not isinstance(inspect_data, list) or not inspect_data:
            raise ValueError(f"{inspect_path}: malformed Docker inspection")
        if inspect_data[0].get("Config", {}).get("Image") != expected_image:
            raise ValueError(f"{runtime_dir}: provisional runtime image mismatch")
        match = re.search(r"GPU KV cache size: ([0-9,]+) tokens", joined)
        if not match:
            raise ValueError(f"{runtime_dir}: missing provisional KV capacity")
        observed_kv = int(match.group(1).replace(",", ""))
        required_kv = 0
    else:
        model_revision_path = runtime_dir / "model-revision.txt"
        image_path = runtime_dir / "runtime-image.txt"
        benchmark_path = runtime_dir / "llm-inference-bench-commit.txt"
        for path in (model_revision_path, image_path, benchmark_path):
            if not path.is_file():
                raise ValueError(f"missing provenance file {path}")
        if model_revision_path.read_text().strip() != MODEL_REVISION:
            raise ValueError(f"{runtime_dir}: model revision mismatch")
        if image_path.read_text().strip() != expected_image:
            raise ValueError(f"{runtime_dir}: runtime image mismatch")
        if benchmark_path.read_text().strip() != BENCHMARK_COMMIT:
            raise ValueError(f"{runtime_dir}: benchmark commit mismatch")
        kv_path = runtime_dir / "kv-cache-tokens.txt"
        if not kv_path.is_file():
            raise ValueError(f"missing KV-capacity file {kv_path}")
        observed_kv = int(kv_path.read_text().strip())
        assert settings is not None
        required_kv = int(settings["required_kv_tokens"])
    if observed_kv < required_kv:
        raise ValueError(
            f"{runtime_dir}: KV capacity {observed_kv} is below {required_kv}"
        )
    manifest = {
        "files": generic_file_manifest(
            [path for path in runtime_dir.iterdir() if path.is_file()]
        ),
        "gpu_memory_utilization": utilization,
        "flashinfer_autotune": flashinfer_autotune,
        "observed_global_kv_tokens": observed_kv,
        "required_global_kv_tokens": required_kv,
        "kv_margin_tokens": observed_kv - required_kv if required_kv else None,
        "model_revision": MODEL_REVISION,
        "runtime_image": expected_image,
        "benchmark_commit": BENCHMARK_COMMIT,
        "provenance_limitations": (
            [
                "The legacy tuning capture proves image/topology/settings in "
                "Docker inspection and logs but predates per-run revision and "
                "benchmark-commit text files."
            ]
            if provisional else []
        ),
        "observations": (
            ["Both ranks logged that FlashInfer autotune was disabled."]
            if autotune_disabled
            else ["Runtime logs prove that FlashInfer autotune was enabled."]
        ) + ["Node 0 reached Application startup complete."],
    }
    return utilization, flashinfer_autotune, observed_kv, required_kv, manifest


def quality_rows(
    source: Path, topology: str, mtp: int, manual: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_json(source)
    outputs = data.get("outputs", [])
    if len(outputs) != 12:
        raise ValueError(f"quality audit must have 12 outputs, found {len(outputs)}")
    efforts = [str(output.get("reasoning_effort")) for output in outputs]
    if {effort: efforts.count(effort) for effort in set(efforts)} != {
        "no_think": 4, "low": 4, "high": 4,
    }:
        raise ValueError("quality audit must have four outputs per reasoning effort")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(output.get("sha256", "")))
        for output in outputs
    ):
        raise ValueError("quality audit contains a malformed response hash")
    compact = dict(data)
    compact["configuration"] = {"topology": topology, "mtp_tokens": mtp}
    compact["source_file"] = f"{source.parent.name}/{source.name}"
    compact["source_sha256"] = sha256(source)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(str(output.get("reasoning_effort", "unknown")), []).append(output)
    tp, pp, ep = topology_values(topology)
    review = manual.get(f"{topology}-mtp{mtp}", {})
    rows: list[dict[str, Any]] = []
    for effort, group in sorted(grouped.items()):
        analyses = [output.get("analysis", {}) for output in group]
        flags = [
            analysis.get("repeated_8gram_fraction", 0) >= 0.20
            or analysis.get("max_identical_word_run", 0) >= 4
            or analysis.get("max_identical_character_run", 0) >= 16
            for analysis in analyses
        ]
        rows.append({
            "topology": topology,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "expert_parallel": str(ep).lower(),
            "mtp_tokens": mtp,
            "mtp_compile_fix": str(mtp > 0).lower(),
            "reasoning_effort": effort,
            "outputs": len(group),
            "empty_outputs": sum(
                not (output.get("reasoning_content") or output.get("content"))
                for output in group
            ),
            "flagged_outputs": sum(flags),
            "max_repeated_8gram_fraction": max(
                analysis.get("repeated_8gram_fraction", 0) for analysis in analyses
            ),
            "max_identical_character_run": max(
                analysis.get("max_identical_character_run", 0) for analysis in analyses
            ),
            "max_identical_word_run": max(
                analysis.get("max_identical_word_run", 0) for analysis in analyses
            ),
            "manual_review_status": review.get("status", "pending"),
            "manual_review_notes": review.get("notes", ""),
            "source_file": f"data/evidence/{topology}-mtp{mtp}-quality.json",
        })
    return rows, compact


def failure_evidence(results_root: Path) -> None:
    specs = [
        {
            "directory": "hy3-fp8-pp2-mtp0-attempt2-hung",
            "status": "confirmed_distributed_autotune_deadlock",
            "required": [
                "flashinfer.jit: [Autotuner]: Autotuning process starts",
                "No available shared memory broadcast block found in 60 seconds",
            ],
            "finding": (
                "PP stages entered incompatible distributed FlashInfer autotune "
                "sequences; PP1 ended while PP0 remained blocked in collectives."
            ),
            "workaround": "Launch PP2 with --no-enable-flashinfer-autotune.",
        },
        {
            "directory": "hy3-fp8-tp2-mtp0-attempt1-no-ep-oom",
            "status": "confirmed_tp2_without_expert_parallel_oom",
            "required": ["tensor_parallel_size=2", "memory allocation failed with OOM"],
            "finding": (
                "Plain TP2 completed weight loading but exhausted each GB300 "
                "during initialization/warmup."
            ),
            "workaround": "Launch TP2 with --enable-expert-parallel.",
        },
    ]
    records: list[dict[str, Any]] = []
    for spec in specs:
        directory = results_root / spec["directory"]
        paths = sorted(directory.glob("*server.log"))
        if not paths:
            continue
        joined = "\n".join(path.read_text(errors="replace") for path in paths)
        missing = [marker for marker in spec["required"] if marker not in joined]
        if missing:
            raise ValueError(f"{directory}: missing failure markers {missing}")
        by_name = {
            generic_node_name(path.name): path.read_text(errors="replace")
            for path in paths
        }
        if spec["status"] == "confirmed_distributed_autotune_deadlock":
            node0 = by_name.get("node0-server.log", "")
            node1 = by_name.get("node1-server.log", "")
            if (
                "No available shared memory broadcast block found in 60 seconds"
                not in node0
                or "flashinfer.jit: [Autotuner]: Autotuning process ends" not in node1
            ):
                raise ValueError(
                    f"{directory}: logs do not prove PP0 blocked after PP1 ended"
                )
        elif not all("memory allocation failed with OOM" in text for text in by_name.values()):
            raise ValueError(f"{directory}: OOM is not present on both nodes")
        records.append({
            "result_directory": spec["directory"],
            "status": spec["status"],
            "finding": spec["finding"],
            "workaround": spec["workaround"],
            "source_files": {
                name: value for name, value in generic_file_manifest(paths).items()
            },
            "confirmed_markers": spec["required"],
        })

    attempt_name = "hy3-fp8-pp2-mtp0-attempt3-kv602176"
    attempt_runtime = results_root / attempt_name / "runtime"
    settings_path = attempt_runtime / "run-settings.txt"
    kv_path = attempt_runtime / "kv-cache-tokens.txt"
    try:
        node0_path = find_rank_file(attempt_runtime, 0, "server-failed.log")
        node1_path = find_rank_file(attempt_runtime, 1, "server-failed.log")
    except ValueError:
        node0_path = attempt_runtime / "node0-server-failed.log"
        node1_path = attempt_runtime / "node1-server-failed.log"
    required_paths = [settings_path, kv_path, node0_path, node1_path]
    if all(path.is_file() for path in required_paths):
        settings = dict(
            line.split("=", 1)
            for line in settings_path.read_text().splitlines()
            if "=" in line
        )
        observed_kv = int(kv_path.read_text().strip())
        required_kv = int(settings["required_kv_tokens"])
        if settings.get("gpu_memory_utilization") != "0.956":
            raise ValueError(f"{attempt_runtime}: expected utilization 0.956")
        if observed_kv != 602176 or required_kv != 1179904:
            raise ValueError(f"{attempt_runtime}: unexpected KV capacity record")
        node0 = node0_path.read_text(errors="replace")
        node1 = node1_path.read_text(errors="replace")
        joined = node0 + "\n" + node1
        required_markers = [
            "gpu_memory_utilization': 0.956",
            "GPU KV cache size: 602,176 tokens",
            "Skipping FlashInfer autotune because it is disabled",
            "Application startup complete",
        ]
        missing = [marker for marker in required_markers if marker not in joined]
        if missing:
            raise ValueError(f"{attempt_runtime}: missing guarded-run markers {missing}")
        if sum(
            "Skipping FlashInfer autotune because it is disabled" in log
            for log in (node0, node1)
        ) != 2:
            raise ValueError(f"{attempt_runtime}: both PP ranks did not skip autotune")

        def log_values(log: str) -> dict[str, float]:
            patterns = {
                "model_loading_gib": r"Model loading took ([0-9.]+) GiB memory",
                "available_kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
                "consumed_weights_plus_non_torch_gib": (
                    r"Actual usage is ([0-9.]+) GiB for consumed memory"
                ),
                "peak_activation_gib": r"([0-9.]+) GiB for peak activation",
                "cuda_graph_gib": r"([0-9.]+) GiB for CUDAGraph memory",
            }
            values: dict[str, float] = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, log)
                if not match:
                    raise ValueError(f"{attempt_runtime}: missing {key}")
                values[key] = float(match.group(1))
            return values

        source_files = generic_file_manifest(
            [path for path in attempt_runtime.iterdir() if path.is_file()]
        )
        runtime_record = {
            "result_directory": attempt_name,
            "status": "rejected_before_benchmark_insufficient_kv_capacity",
            "benchmark_started": False,
            "server_reached_health": True,
            "topology": "TP1/PP2",
            "mtp_tokens": 0,
            "gpu_memory_utilization": 0.956,
            "kv_cache_dtype": settings["kv_cache_dtype"],
            "max_model_len": int(settings["max_model_len"]),
            "observed_global_kv_tokens": observed_kv,
            "required_global_kv_tokens": required_kv,
            "kv_token_shortfall": required_kv - observed_kv,
            "fraction_of_required_kv": observed_kv / required_kv,
            "rank_runtime_memory": {
                "node0_pp0": log_values(node0),
                "node1_pp1": log_values(node1),
            },
            "source_files": source_files,
            "disposition": "rejected_capacity_profile",
            "notes": (
                "The API became healthy, but the capacity guard rejected the run "
                "before any decode, prefill, quality, or perplexity request. The "
                "retained server logs and KV-capacity record support this rejection; "
                "no unretained teardown summary is included as evidence."
            ),
        }
        (RUNTIME / "pp2-mtp0-attempt3-kv602176.json").write_text(
            json.dumps(runtime_record, indent=2) + "\n"
        )
        records.append(runtime_record)

    unsupported_name = "hy3-fp8-pp2-mtp1-attempt1-unsupported-mtp-pp"
    unsupported_runtime = results_root / unsupported_name / "runtime"
    if unsupported_runtime.is_dir():
        unsupported_logs = [
            find_rank_file(unsupported_runtime, 0, "server-failed.log"),
            find_rank_file(unsupported_runtime, 1, "server-failed.log"),
        ]
        unsupported_text = "\n".join(
            path.read_text(errors="replace") for path in unsupported_logs
        )
        marker = (
            "Pipeline parallelism is not supported for this model. Supported "
            "models implement the `SupportsPP` interface."
        )
        if marker not in unsupported_text or "Resolved architecture: HYV3MTPModel" not in unsupported_text:
            raise ValueError(f"{unsupported_runtime}: missing MTP/PP incompatibility proof")
        records.append({
            "result_directory": unsupported_name,
            "status": "confirmed_mtp_draft_model_lacks_pipeline_parallel_support",
            "affected_configurations": ["pp2-mtp1", "pp2-mtp2"],
            "benchmark_started": False,
            "finding": (
                "The pinned vLLM runtime resolves HYV3MTPModel, then rejects "
                "pipeline parallelism because the draft model lacks SupportsPP."
            ),
            "workaround": (
                "Use TP2 with expert parallel for MTP1/MTP2; do not patch or "
                "fabricate PP2 speculative results."
            ),
            "confirmed_markers": [
                "Resolved architecture: HYV3MTPModel",
                marker,
            ],
            "source_files": generic_file_manifest(unsupported_logs),
        })

    capacity_attempts = [
        {
            "name": "hy3-fp8-tp2-mtp1-c128-confirm60-attempt1-kv1179216",
            "mtp": 1,
            "utilization": 0.956,
            "observed": 1179216,
            "status": "rejected_capacity_profile_variance",
            "finding": (
                "The confirmation relaunch profiled 5.57 GiB peak activation "
                "instead of 5.08 GiB in the accepted full run, leaving 688 "
                "fewer KV tokens than the publication guard requires."
            ),
        },
        {
            "name": "hy3-fp8-tp2-mtp2-attempt1-kv1175760-util0956",
            "mtp": 2,
            "utilization": 0.956,
            "observed": 1175760,
            "status": "rejected_insufficient_kv_capacity",
            "finding": (
                "MTP2 at the 0.956 baseline missed the fixed KV guard by 4,144 "
                "tokens; one clean-baseline 0.958 profile supplied a 7,152-token reserve."
            ),
        },
    ]
    for spec in capacity_attempts:
        runtime_dir = results_root / spec["name"] / "runtime"
        if not runtime_dir.is_dir():
            continue
        settings = parse_settings(runtime_dir / "run-settings.txt")
        kv_path = runtime_dir / "kv-cache-tokens.txt"
        logs = [
            find_rank_file(runtime_dir, 0, "server-failed.log"),
            find_rank_file(runtime_dir, 1, "server-failed.log"),
        ]
        joined = "\n".join(path.read_text(errors="replace") for path in logs)
        observed = int(kv_path.read_text().strip())
        required = int(settings["required_kv_tokens"])
        if (
            observed != spec["observed"]
            or required != 1179904
            or float(settings["gpu_memory_utilization"]) != spec["utilization"]
            or int(settings["mtp_tokens"]) != spec["mtp"]
        ):
            raise ValueError(f"{runtime_dir}: capacity attempt settings mismatch")
        if (
            f"GPU KV cache size: {observed:,} tokens" not in joined
            or "Application startup complete" not in joined
        ):
            raise ValueError(f"{runtime_dir}: missing healthy-server/KV proof")
        source_paths = [
            path for path in runtime_dir.iterdir() if path.is_file()
        ]
        records.append({
            "result_directory": spec["name"],
            "status": spec["status"],
            "benchmark_started": False,
            "server_reached_health": True,
            "topology": "TP2/PP1 + expert parallel",
            "mtp_tokens": spec["mtp"],
            "gpu_memory_utilization": spec["utilization"],
            "observed_global_kv_tokens": observed,
            "required_global_kv_tokens": required,
            "kv_token_shortfall": required - observed,
            "finding": spec["finding"],
            "source_files": generic_file_manifest(source_paths),
        })
    if records:
        (DATA / "failure-evidence.json").write_text(
            json.dumps({"failures": records}, indent=2) + "\n"
        )


def headline_policy(
    status: str, topology: str, mtp: int, concurrency: int
) -> tuple[bool, str]:
    if status == "provisional_tuning":
        return False, "Provisional 0.92 tuning pass; final 0.956 run supersedes it."
    if topology == "tp2" and mtp == 1 and concurrency == 128:
        return False, (
            "Confirmed MTP1 C128 performance cliff; a separate 60-second "
            "0.958 run reproduced it at 957.2 tok/s."
        )
    if topology == "tp2" and mtp == 2 and concurrency >= 32:
        return False, (
            "MTP2 high-concurrency performance cliff begins at C32 despite "
            "full residency, zero errors, and normal draft acceptance."
        )
    return True, ""


def consume_run(
    run_dir: Path,
    topology: str,
    mtp: int,
    status: str,
    runtime_name: str,
    expected: list[int],
    manual: dict[str, Any],
    throughput: list[dict[str, Any]],
    prefill_rows: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    kv_capacity: list[dict[str, Any]],
) -> None:
    source = run_dir / "llm-inference-bench.json"
    runtime_dir = run_dir / runtime_name
    data, results = validate_benchmark(source, expected)
    utilization, flashinfer_autotune, observed_kv, required_kv, runtime_manifest = runtime_settings(
        runtime_dir, topology, mtp, provisional=status == "provisional_tuning"
    )
    suffix = "-092-provisional" if status == "provisional_tuning" else ""
    evidence_name = f"{topology}-mtp{mtp}{suffix}-benchmark.json"
    evidence_rel = f"data/evidence/{evidence_name}"
    compact = compact_benchmark(source, data, results, status)
    (EVIDENCE / evidence_name).write_text(json.dumps(compact, indent=2) + "\n")
    runtime_manifest.update({
        "source_directory": f"{run_dir.name}/{runtime_name}",
        "publication_status": status,
        "topology": topology,
        "mtp_tokens": mtp,
    })
    (RUNTIME / f"{topology}-mtp{mtp}{suffix}.json").write_text(
        json.dumps(runtime_manifest, indent=2) + "\n"
    )

    tp, pp, ep = topology_values(topology)
    common = {
        "publication_status": status,
        "topology": topology,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "expert_parallel": str(ep).lower(),
        "mtp_tokens": mtp,
        "gpu_memory_utilization": utilization,
        "flashinfer_autotune": str(flashinfer_autotune).lower(),
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE if mtp == 0 else MTP_RUNTIME_IMAGE,
        "benchmark_commit": BENCHMARK_COMMIT,
        "result_file": evidence_rel,
    }
    metadata = data["metadata"]
    for row in results:
        concurrency = int(row["concurrency"])
        headline_eligible, exclusion_reason = headline_policy(
            status, topology, mtp, concurrency
        )
        draft_tokens = int(row.get("server_spec_draft_tokens") or 0)
        accepted_tokens = int(row.get("server_spec_accepted_tokens") or 0)
        if mtp > 0 and draft_tokens <= 0:
            raise ValueError(f"C{concurrency}: MTP run lacks speculative counters")
        throughput.append(common | {
            "mtp_compile_fix": str(mtp > 0).lower(),
            "concurrency": concurrency,
            "aggregate_output_tokens_per_second": row["aggregate_tps"],
            "request_count": row["request_count"],
            "completed_requests": row["completed_request_count"],
            "num_errors": row["num_errors"],
            "effective_concurrency": row["effective_concurrency"],
            "max_running_requests": row["max_running_reqs"],
            "capacity_limited": str(row["capacity_limited"]).lower(),
            "prompt_tokens": int(row["input_seq_len_avg"]),
            "max_tokens": metadata["max_tokens"],
            "temperature": metadata["temperature"],
            "measurement_seconds": row["measurement_seconds"],
            "server_spec_draft_tokens": draft_tokens,
            "server_spec_accepted_tokens": accepted_tokens,
            "speculative_acceptance_rate": (
                accepted_tokens / draft_tokens if draft_tokens else ""
            ),
            "headline_eligible": str(headline_eligible).lower(),
            "headline_exclusion_reason": exclusion_reason,
        })
    for context, row in sorted(data["prefill"].items(), key=lambda item: int(item[0])):
        prefill_rows.append(common | {
            "context_tokens": int(context),
            "actual_prompt_tokens": row["prompt_tokens"],
            "prompt_tokens_per_second": row["tok_per_sec"],
            "ttft_seconds": row["ttft_seconds"],
            "samples": row["samples"],
            "num_errors": 0,
        })

    kv_capacity.append({
        "publication_status": status,
        "topology": topology,
        "expert_parallel": str(ep).lower(),
        "mtp_tokens": mtp,
        "gpu_memory_utilization": utilization,
        "observed_global_kv_tokens": observed_kv,
        "required_global_kv_tokens": required_kv if required_kv else "",
        "margin_tokens": observed_kv - required_kv if required_kv else "",
        "capacity_status": "accepted" if status == "accepted" else "provisional",
        "result_directory": run_dir.name,
        "runtime_file": f"data/runtime/{topology}-mtp{mtp}{suffix}.json",
    })

    if status == "accepted":
        quality_source = run_dir / "natural-quality-audit.json"
        rows, quality_compact = quality_rows(quality_source, topology, mtp, manual)
        quality.extend(rows)
        (EVIDENCE / f"{topology}-mtp{mtp}-quality.json").write_text(
            json.dumps(quality_compact, indent=2) + "\n"
        )


def consume_stability_confirmation(
    run_dir: Path,
    manual: dict[str, Any],
    stability: list[dict[str, Any]],
    kv_capacity: list[dict[str, Any]],
) -> None:
    source = run_dir / "llm-inference-bench.json"
    data, results = validate_benchmark(
        source,
        [128],
        duration_seconds=60.0,
        prefill_contexts=(8192,),
    )
    utilization, autotune, observed_kv, required_kv, runtime_manifest = runtime_settings(
        run_dir / "runtime", "tp2", 1
    )
    if utilization != 0.958 or not autotune:
        raise ValueError("stability confirmation must be TP2/MTP1 at util 0.958")
    result = results[0]
    draft = int(result.get("server_spec_draft_tokens") or 0)
    accepted = int(result.get("server_spec_accepted_tokens") or 0)
    if draft <= 0:
        raise ValueError("stability confirmation lacks MTP acceptance counters")
    evidence_name = "tp2-mtp1-c128-confirm60-util0958-benchmark.json"
    compact = compact_benchmark(
        source, data, results, "stability_confirmation"
    )
    compact["finding"] = (
        "The 60-second C128-only confirmation reproduced the high-concurrency "
        "MTP1 cliff with full residency and zero errors."
    )
    (EVIDENCE / evidence_name).write_text(json.dumps(compact, indent=2) + "\n")

    runtime_manifest.update({
        "source_directory": f"{run_dir.name}/runtime",
        "publication_status": "stability_confirmation",
        "topology": "tp2",
        "mtp_tokens": 1,
    })
    runtime_name = "tp2-mtp1-c128-confirm60-util0958.json"
    (RUNTIME / runtime_name).write_text(json.dumps(runtime_manifest, indent=2) + "\n")

    quality_source = run_dir / "natural-quality-audit.json"
    _, quality_compact = quality_rows(quality_source, "tp2", 1, manual)
    quality_compact["publication_status"] = "stability_confirmation"
    (EVIDENCE / "tp2-mtp1-c128-confirm60-util0958-quality.json").write_text(
        json.dumps(quality_compact, indent=2) + "\n"
    )

    stability.append({
        "publication_status": "stability_confirmation",
        "topology": "tp2",
        "expert_parallel": "true",
        "mtp_tokens": 1,
        "gpu_memory_utilization": utilization,
        "concurrency": 128,
        "context_tokens": int(result["input_seq_len_avg"]),
        "max_tokens": data["metadata"]["max_tokens"],
        "duration_seconds": result["measurement_seconds"],
        "aggregate_output_tokens_per_second": result["aggregate_tps"],
        "server_output_tokens_per_second": result["server_gen_throughput"],
        "effective_concurrency": result["effective_concurrency"],
        "max_running_requests": result["max_running_reqs"],
        "num_errors": result["num_errors"],
        "server_spec_draft_tokens": draft,
        "server_spec_accepted_tokens": accepted,
        "speculative_acceptance_rate": accepted / draft,
        "observed_global_kv_tokens": observed_kv,
        "required_global_kv_tokens": required_kv,
        "finding": "Confirmed high-concurrency performance cliff; exclude from headlines.",
        "result_file": f"data/evidence/{evidence_name}",
    })
    kv_capacity.append({
        "publication_status": "stability_confirmation",
        "topology": "tp2",
        "expert_parallel": "true",
        "mtp_tokens": 1,
        "gpu_memory_utilization": utilization,
        "observed_global_kv_tokens": observed_kv,
        "required_global_kv_tokens": required_kv,
        "margin_tokens": observed_kv - required_kv,
        "capacity_status": "accepted_confirmation",
        "result_directory": run_dir.name,
        "runtime_file": f"data/runtime/{runtime_name}",
    })


def rejected_capacity_rows(results_root: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "hy3-fp8-pp2-mtp0-attempt3-kv602176",
            "pp2", False, 0, "rejected_insufficient_kv_capacity",
        ),
        (
            "hy3-fp8-tp2-mtp1-c128-confirm60-attempt1-kv1179216",
            "tp2", True, 1, "rejected_below_guard",
        ),
        (
            "hy3-fp8-tp2-mtp2-attempt1-kv1175760-util0956",
            "tp2", True, 2, "rejected_below_guard",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, topology, ep, mtp, status in specs:
        runtime = results_root / name / "runtime"
        settings_path = runtime / "run-settings.txt"
        kv_path = runtime / "kv-cache-tokens.txt"
        if not settings_path.is_file() or not kv_path.is_file():
            continue
        settings = parse_settings(settings_path)
        observed = int(kv_path.read_text().strip())
        required = int(settings["required_kv_tokens"])
        rows.append({
            "publication_status": "rejected",
            "topology": topology,
            "expert_parallel": str(ep).lower(),
            "mtp_tokens": mtp,
            "gpu_memory_utilization": float(settings["gpu_memory_utilization"]),
            "observed_global_kv_tokens": observed,
            "required_global_kv_tokens": required,
            "margin_tokens": observed - required,
            "capacity_status": status,
            "result_directory": name,
            "runtime_file": (
                "data/runtime/pp2-mtp0-attempt3-kv602176.json"
                if name.endswith("attempt3-kv602176")
                else "data/failure-evidence.json"
            ),
        })
    return rows


def extract(results_root: Path) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manual_path = DATA / "manual-quality-review.json"
    manual = load_json(manual_path) if manual_path.is_file() else {}
    throughput: list[dict[str, Any]] = []
    prefill_rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    kv_capacity: list[dict[str, Any]] = []

    provisional = results_root / "hy3-fp8-pp2-mtp0-092-provisional"
    if provisional.is_dir():
        try:
            consume_run(
                provisional, "pp2", 0, "provisional_tuning", "runtime-092",
                PROVISIONAL_CONCURRENCIES, manual, throughput, prefill_rows,
                quality, kv_capacity,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"rejected {provisional.name}: {error}")
        else:
            print(f"consumed {provisional.name}: provisional_tuning (C128/quality absent)")
    else:
        print(f"pending {provisional.name}: preserved tuning directory not found")

    for topology, mtp in CONFIGURATIONS:
        name = f"hy3-fp8-{topology}-mtp{mtp}"
        run_dir = results_root / name
        missing = [
            path.name for path in (
                run_dir / "llm-inference-bench.json",
                run_dir / "natural-quality-audit.json",
                run_dir / "runtime",
            ) if not path.exists()
        ]
        if missing:
            if topology == "pp2" and mtp > 0:
                print(
                    f"unsupported {name}: pinned HYV3MTPModel lacks "
                    "pipeline-parallel SupportsPP"
                )
            else:
                print(f"pending {name}: missing {', '.join(missing)}")
            continue
        try:
            consume_run(
                run_dir, topology, mtp, "accepted", "runtime",
                FINAL_CONCURRENCIES, manual, throughput, prefill_rows, quality,
                kv_capacity,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"rejected {name}: {error}")
        else:
            print(f"consumed {name}: accepted")

    confirmation = results_root / "hy3-fp8-tp2-mtp1-c128-confirm60-util0958"
    if confirmation.is_dir():
        try:
            consume_stability_confirmation(
                confirmation, manual, stability, kv_capacity
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"rejected {confirmation.name}: {error}")
        else:
            print(f"consumed {confirmation.name}: stability_confirmation")
    else:
        print(f"pending {confirmation.name}: confirmation directory not found")

    kv_capacity.extend(rejected_capacity_rows(results_root))

    throughput.sort(key=lambda row: (
        row["publication_status"] != "provisional_tuning",
        row["topology"], row["mtp_tokens"], row["concurrency"],
    ))
    prefill_rows.sort(key=lambda row: (
        row["publication_status"] != "provisional_tuning",
        row["topology"], row["mtp_tokens"], row["context_tokens"],
    ))
    quality.sort(key=lambda row: (
        row["topology"], row["mtp_tokens"], row["reasoning_effort"],
    ))
    kv_capacity.sort(key=lambda row: (
        row["topology"], row["mtp_tokens"],
        float(row["gpu_memory_utilization"]), row["publication_status"],
    ))
    write_csv(DATA / "throughput.csv", THROUGHPUT_FIELDS, throughput)
    write_csv(DATA / "prefill.csv", PREFILL_FIELDS, prefill_rows)
    write_csv(DATA / "quality-summary.csv", QUALITY_FIELDS, quality)
    write_csv(DATA / "stability-confirmation.csv", STABILITY_FIELDS, stability)
    write_csv(DATA / "kv-capacity.csv", KV_CAPACITY_FIELDS, kv_capacity)
    failure_evidence(results_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, required=True,
        help="Directory containing Hy3 result directories",
    )
    args = parser.parse_args()
    extract(args.results_root.resolve())


if __name__ == "__main__":
    main()
