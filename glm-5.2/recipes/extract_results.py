#!/usr/bin/env python3
"""Validate GLM-5.2 runs and create publication-sized evidence.

Only complete, internally consistent PP2 and TP2 result directories are
accepted.  Missing or rejected runs contribute no rows: this script never
turns a missing measurement into zero or a placeholder value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
RUNTIME_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
)
RUNTIME_IMAGE_ID = "sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
BENCHMARK_COMMIT = "0b4185b5b435e948b199c9077a00b084864aa963"
LM_EVAL_COMMIT = "8a07e1110d060de48cfc7a9a7987b7659060b60b"
TOKENIZER_CONFIG_SHA256 = "24d1ad6cd9d0536c607358a6f7c87da99f369054e898242c46986baa5e2788ea"
TOKENIZER_JSON_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_CONTEXTS = [8192, 65536, 131072]
TOPOLOGIES = {
    "pp2": {"tensor_parallel_size": 1, "pipeline_parallel_size": 2,
            "expert_parallel": False, "flashinfer_autotune": False},
    "tp2": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
            "expert_parallel": True, "flashinfer_autotune": False},
}
QUALITY_MAX_TOKENS = 4096
QUALITY_KV_CACHE_DTYPE = "bfloat16"
STARTUP_FAILURES = {
    "pp2": (
        "Operator-observed PP2 startup failure at the configured 0.95 HBM "
        "utilization; the original startup logs were not retained. The operator "
        "observed pipeline-stage imbalance, one stage effectively full, and "
        "repeated small post-load allocation failures. No decode or prefill row "
        "was produced or accepted."
    ),
}

THROUGHPUT_FIELDS = [
    "topology", "tensor_parallel_size", "pipeline_parallel_size",
    "expert_parallel", "flashinfer_autotune", "model_revision",
    "runtime_image", "benchmark_commit", "kv_cache_dtype", "context_tokens",
    "output_tokens", "temperature", "concurrency", "request_count",
    "completed_requests", "aggregate_output_tok_s", "per_request_output_tok_s",
    "median_ttft_s", "median_itl_ms", "effective_concurrency",
    "max_running_requests", "capacity_limited", "num_errors", "source_json",
]
PREFILL_FIELDS = [
    "topology", "tensor_parallel_size", "pipeline_parallel_size",
    "expert_parallel", "flashinfer_autotune", "model_revision",
    "runtime_image", "benchmark_commit", "kv_cache_dtype", "context_tokens",
    "actual_prompt_tokens", "samples", "prompt_tok_s", "median_ttft_s",
    "num_errors", "source_json",
]
QUALITY_FIELDS = [
    "topology", "model_revision", "kv_cache_dtype", "audit_max_tokens",
    "prompt_index", "output_tokens",
    "finish_reason", "unique_word_ratio", "max_identical_character_run",
    "max_identical_word_run", "repeated_8gram_fraction", "flagged", "sha256",
    "manual_review_status", "manual_review_notes", "source_json",
]
PPL_FIELDS = [
    "topology", "model_revision", "runtime_image", "kv_cache_dtype", "dataset",
    "documents", "max_length", "batch_size", "word_ppl", "byte_ppl",
    "bits_per_byte", "lm_eval_commit", "source_json",
]

PUBLIC_HOST_REWRITES = {
    "".join(("gemini", "1")): "node0",
    "".join(("gemini", "2")): "node1",
    "".join(("dgx", "1")): "node0",
}
PUBLIC_ADDRESS_REWRITES = {
    # RFC 5737 documentation addresses keep the topology legible without
    # publishing an operator's actual fabric or management addressing.
    ".".join(map(str, (192, 168, 200, 1))): "192.0.2.1",
    ".".join(map(str, (192, 168, 200, 2))): "192.0.2.2",
}
GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F-]{16,}")


def portable_absolute_path(value: str) -> str:
    """Map known absolute evidence paths to stable public namespaces."""
    if not value.startswith("/"):
        return value
    parts = Path(value).parts
    for index, part in enumerate(parts):
        if part == "results" and index + 1 < len(parts) \
                and parts[index + 1].startswith("glm52-"):
            return "/" + "/".join(parts[index:])
    for marker in ("glm52-staging", "lm-evaluation-harness"):
        if marker in parts:
            index = parts.index(marker)
            return "/workspace/" + "/".join(parts[index:])
    return value


def scrub_public(value: Any) -> Any:
    """Recursively remove host-local identities from emitted evidence."""
    if isinstance(value, dict):
        return {key: scrub_public(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_public(item) for item in value]
    if not isinstance(value, str):
        return value
    value = portable_absolute_path(value)
    for private, public in PUBLIC_HOST_REWRITES.items():
        value = value.replace(private, public)
    for private, public in PUBLIC_ADDRESS_REWRITES.items():
        value = value.replace(private, public)
    return GPU_UUID_RE.sub("GPU-REDACTED", value)


def write_public_json(path: Path, value: Any, *, ensure_ascii: bool = True) -> None:
    path.write_text(
        json.dumps(scrub_public(value), indent=2, ensure_ascii=ensure_ascii) + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive, found {value!r}")
    return number


def exact_text(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {path}")
    actual = path.read_text().strip()
    if actual != expected:
        raise ValueError(f"{path}: expected {expected}, found {actual}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def option_value(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def inspect_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text())
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one Docker inspect object")
    return value


def validate_runtime(run_dir: Path, topology: str) -> dict[str, Any]:
    runtime = run_dir / "runtime"
    required = [
        "node0-inspect.json", "node1-inspect.json",
        "node0-server.log", "node1-server.log", "models.json",
        "metrics-before.txt", "metrics-after.txt", "kv-cache-tokens.txt",
        "node0-nvidia-smi.csv", "node1-nvidia-smi.csv",
        "llm-inference-bench-commit.txt", "lm-eval-commit.txt",
        "model-revision.txt",
    ]
    missing = [name for name in required if not (runtime / name).is_file()]
    if missing:
        raise ValueError(f"runtime evidence is incomplete: {', '.join(missing)}")
    exact_text(runtime / "model-revision.txt", MODEL_REVISION)
    exact_text(runtime / "llm-inference-bench-commit.txt", BENCHMARK_COMMIT)
    exact_text(runtime / "lm-eval-commit.txt", LM_EVAL_COMMIT)

    expected = TOPOLOGIES[topology]
    commands: list[list[str]] = []
    inspect_summary: dict[str, Any] = {}
    for rank, host in enumerate(("node0", "node1")):
        item = inspect_object(runtime / f"{host}-inspect.json")
        config = item.get("Config", {})
        command = config.get("Cmd")
        if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
            raise ValueError(f"{host} inspect lacks a string command array")
        commands.append(command)
        if config.get("Image") != RUNTIME_IMAGE or item.get("Image") != RUNTIME_IMAGE_ID:
            raise ValueError(f"{host} did not use the pinned runtime image")
        expected_options = {
            "--served-model-name": "GLM-5.2-NVFP4",
            "--tensor-parallel-size": str(expected["tensor_parallel_size"]),
            "--pipeline-parallel-size": str(expected["pipeline_parallel_size"]),
            "--nnodes": "2", "--node-rank": str(rank),
            "--kv-cache-dtype": "fp8_e4m3", "--max-model-len": "135168",
            "--reasoning-parser": "glm45", "--tool-call-parser": "glm47",
            "--moe-backend": "flashinfer_cutedsl",
        }
        for option, value in expected_options.items():
            if option_value(command, option) != value:
                raise ValueError(f"{host}: {option} does not prove {value}")
        for flag in ("--enable-prefix-caching", "--enable-auto-tool-choice"):
            if flag not in command:
                raise ValueError(f"{host}: missing {flag}")
        has_ep = "--enable-expert-parallel" in command
        has_no_autotune = "--no-enable-flashinfer-autotune" in command
        if has_ep != expected["expert_parallel"]:
            raise ValueError(f"{host}: expert-parallel setting disagrees with topology")
        if has_no_autotune == expected["flashinfer_autotune"]:
            raise ValueError(f"{host}: FlashInfer autotune setting disagrees with topology")
        inspect_summary[host] = {
            "config_image": config["Image"], "image_id": item["Image"],
            "command": command,
            "inspect_sha256": sha256(runtime / f"{host}-inspect.json"),
        }

    logs = [
        (runtime / f"{host}-server.log").read_text(errors="replace")
        for host in ("node0", "node1")
    ]
    if "Application startup complete" not in logs[0]:
        raise ValueError("node0 log does not prove API startup")
    if not all("NCCL" in log and "mlx5_0" in log for log in logs):
        raise ValueError("server logs do not prove NCCL selected mlx5_0 on both ranks")
    if not expected["flashinfer_autotune"] and not all(
        "Skipping FlashInfer autotune because it is disabled" in log for log in logs
    ):
        raise ValueError(f"{topology.upper()} logs do not prove FlashInfer autotune was disabled")
    if expected["flashinfer_autotune"] and not any(
        marker in "\n".join(logs)
        for marker in ("Autotuning process starts", "enable_flashinfer_autotune=True")
    ):
        raise ValueError(f"{topology.upper()} logs do not prove FlashInfer autotune was enabled")

    models = read_json(runtime / "models.json")
    model_rows = models.get("data", [])
    if len(model_rows) != 1 or model_rows[0].get("id") != "GLM-5.2-NVFP4":
        raise ValueError("/v1/models evidence does not expose GLM-5.2-NVFP4")
    if int(model_rows[0].get("max_model_len", 0)) != 135168:
        raise ValueError("/v1/models evidence does not prove max_model_len=135168")
    kv_tokens = int((runtime / "kv-cache-tokens.txt").read_text().strip())
    unshared_c128_tokens = (8192 + 1024 + 2) * 128
    shared_prefix_c128_tokens = (8192 + 2) + (1024 + 2) * 128
    if kv_tokens < shared_prefix_c128_tokens:
        raise ValueError(
            f"KV cache capacity {kv_tokens} cannot cover even the shared-prefix C128 cell"
        )
    capacity_basis = "unshared"
    if kv_tokens < unshared_c128_tokens:
        capacity_basis = "shared-prefix"
        c128_samples = re.findall(
            r"Running: 128 reqs, Waiting: 0 reqs.*?Prefix cache hit rate: ([0-9.]+)%",
            logs[0],
        )
        if not c128_samples or max(map(float, c128_samples)) < 80:
            raise ValueError(
                "C128 relies on shared-prefix KV reuse, but the server log does not prove "
                "128 resident requests with an >=80% prefix-cache hit rate"
            )

    # Publish only the generic, strictly required evidence names. Canonical raw
    # archives may also retain legacy host-named copies, but those names are not
    # part of the portable public manifest.
    source_files = {
        name: {
            "bytes": (runtime / name).stat().st_size,
            "sha256": sha256(runtime / name),
        }
        for name in sorted(required)
    }
    return {
        "source_directory": str(runtime), "topology": topology,
        "model_revision": MODEL_REVISION, "runtime_image": RUNTIME_IMAGE,
        "benchmark_commit": BENCHMARK_COMMIT, "lm_eval_commit": LM_EVAL_COMMIT,
        "kv_cache_tokens": kv_tokens,
        "c128_kv_capacity_basis": capacity_basis,
        "c128_shared_prefix_required_tokens": shared_prefix_c128_tokens,
        "c128_unshared_required_tokens": unshared_c128_tokens,
        "container_evidence": inspect_summary,
        "source_files": source_files,
    }


def validate_benchmark(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = read_json(path)
    metadata = data.get("metadata", {})
    expected_metadata = {
        "engine": "vllm", "model": "GLM-5.2-NVFP4",
        "decode_mode": "duration", "duration_per_test": 30.0,
        "max_tokens": 1024, "temperature": 0.0, "ignore_eos": True,
        "concurrency_levels": CONCURRENCIES, "context_lengths": [8192],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"benchmark metadata {key}: expected {expected!r}, found {metadata.get(key)!r}")
    results = sorted(data.get("results", []), key=lambda row: int(row["concurrency"]))
    if [int(row["concurrency"]) for row in results] != CONCURRENCIES:
        raise ValueError("benchmark does not contain exactly C1,C2,C4,C8,C16,C32,C64,C128")
    for row in results:
        concurrency = int(row["concurrency"])
        if int(row.get("num_errors", -1)) != 0:
            raise ValueError(f"C{concurrency}: num_errors is not zero")
        if int(float(row.get("input_seq_len_avg", 0))) != 8192:
            raise ValueError(f"C{concurrency}: average input is not 8,192 tokens")
        average_output = finite_positive(
            row.get("output_seq_len_avg"), f"C{concurrency} average sampled output"
        )
        if average_output > 1024:
            raise ValueError(f"C{concurrency}: average sampled output exceeds the 1,024-token cap")
        if int(row.get("request_count", 0)) < concurrency:
            raise ValueError(f"C{concurrency}: fewer than C requests were measured")
        if int(row.get("completed_request_count", -1)) < 0:
            raise ValueError(f"C{concurrency}: invalid completed-request count")
        if row.get("aggregate_source") != "openai_continuous_usage":
            raise ValueError(f"C{concurrency}: aggregate rate is not from the server usage counter")
        for key in ("aggregate_tps", "per_request_avg_tps", "ttft_p50"):
            finite_positive(row.get(key), f"C{concurrency} {key}")
        samples = row.get("request_samples", [])
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"C{concurrency}: missing request samples")
        for sample in samples:
            if int(sample.get("input_tokens", 0)) != 8192:
                raise ValueError(f"C{concurrency}: a sampled request is not exactly 8K input")
            sample_output = int(sample.get("output_tokens", 0))
            if sample_output <= 0 or sample_output > 1024:
                raise ValueError(f"C{concurrency}: sampled output is outside 1..1,024 tokens")
            if sample.get("completed") and sample_output != 1024:
                raise ValueError(f"C{concurrency}: a completed request did not reach 1,024 tokens")
    prefill = data.get("prefill", {})
    if sorted(map(int, prefill)) != PREFILL_CONTEXTS:
        raise ValueError("prefill does not contain exactly 8K, 64K, and 128K")
    for context in PREFILL_CONTEXTS:
        row = prefill[str(context)]
        if int(row.get("prompt_tokens", 0)) != context + 2:
            raise ValueError(f"{context} prefill API prompt count is not {context + 2}")
        if row.get("method") != "client" or int(row.get("samples", 0)) <= 0:
            raise ValueError(f"{context} prefill is not a sampled client measurement")
        finite_positive(row.get("tok_per_sec"), f"{context} prefill tok/s")
        finite_positive(row.get("ttft_seconds"), f"{context} prefill TTFT")
    return data, results


def analyze(text: str) -> dict[str, Any]:
    words = re.findall(r"\S+", text)
    folded = [word.casefold() for word in words]
    grams = [tuple(folded[index:index + 8]) for index in range(max(0, len(folded) - 7))]
    counts = Counter(grams)
    previous = None
    run = best_word_run = 0
    for word in folded:
        run = run + 1 if word == previous else 1
        previous = word
        best_word_run = max(best_word_run, run)
    character_run = max(
        (len(match.group(0)) for match in re.finditer(r"(\S)\1*", text)), default=0
    )
    repeated = sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))
    return {
        "unique_word_ratio": len(set(folded)) / max(1, len(folded)),
        "max_identical_character_run": character_run,
        "max_identical_word_run": best_word_run,
        "repeated_8gram_fraction": repeated,
        "flagged": not text.strip() or repeated >= 0.20 or best_word_run >= 4 or character_run >= 16,
    }


def validate_quality(path: Path, topology: str, manual: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = read_json(path)
    provenance_path = path.parent / "provenance.json"
    provenance = read_json(provenance_path)
    expected_provenance = {
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "topology": topology,
        "expert_parallel": topology == "tp2",
        "moe_backend": "flashinfer_cutedsl",
        "flashinfer_autotune": False,
        "kv_cache_dtype": QUALITY_KV_CACHE_DTYPE,
        "server_max_model_len": 8192,
        "audit_max_tokens": QUALITY_MAX_TOKENS,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"quality provenance {key}: expected {expected!r}, found {provenance.get(key)!r}"
            )
    quality_runtime = path.parent / "runtime"
    quality_logs = [quality_runtime / f"{host}-server.log" for host in ("node0", "node1")]
    if not all(log.is_file() for log in quality_logs):
        raise ValueError("quality audit is missing both BF16-KV server logs")
    rank0_quality_log = quality_logs[0].read_text(errors="replace")
    for marker in (
        "'max_model_len': 8192", "'kv_cache_dtype': 'bfloat16'",
        f"'tensor_parallel_size': {TOPOLOGIES[topology]['tensor_parallel_size']}",
        f"'enable_expert_parallel': {topology == 'tp2'}",
        "'enable_flashinfer_autotune': False", "'moe_backend': 'flashinfer_cutedsl'",
    ):
        if marker not in rank0_quality_log:
            raise ValueError(f"quality rank-0 log does not prove {marker}")
    settings = data.get("settings", {})
    if settings.get("temperature") != 0 or settings.get("max_tokens") != QUALITY_MAX_TOKENS:
        raise ValueError(
            f"quality audit settings must be temperature=0 and max_tokens={QUALITY_MAX_TOKENS}"
        )
    outputs = sorted(data.get("outputs", []), key=lambda row: int(row["prompt_index"]))
    if [int(row["prompt_index"]) for row in outputs] != [0, 1, 2, 3]:
        raise ValueError("quality audit must contain the four canonical prompts")
    review = manual.get(topology, {})
    status = review.get("status", "pending")
    if status not in {"pending", "clean", "flagged"}:
        raise ValueError(f"manual quality status {status!r} is invalid")
    rows: list[dict[str, Any]] = []
    for output in outputs:
        text = "\n".join(
            part for part in (output.get("reasoning_content"), output.get("content")) if part
        )
        if not text.strip():
            raise ValueError(f"quality prompt {output['prompt_index']} produced an empty output")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if output.get("sha256") != digest:
            raise ValueError(f"quality prompt {output['prompt_index']} has an invalid SHA-256")
        computed = analyze(text)
        recorded = output.get("analysis", {})
        for key in ("max_identical_character_run", "max_identical_word_run", "flagged"):
            if recorded.get(key) != computed[key]:
                raise ValueError(f"quality prompt {output['prompt_index']} has stale {key}")
        for key in ("unique_word_ratio", "repeated_8gram_fraction"):
            if not math.isclose(float(recorded.get(key, -1)), computed[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"quality prompt {output['prompt_index']} has stale {key}")
        usage = output.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens", 0))
        if completion_tokens <= 0 or completion_tokens > QUALITY_MAX_TOKENS:
            raise ValueError(f"quality prompt {output['prompt_index']} has invalid token usage")
        if output.get("finish_reason") != "stop":
            raise ValueError(
                f"quality prompt {output['prompt_index']} did not finish naturally"
            )
        rows.append({
            "topology": topology, "model_revision": MODEL_REVISION,
            "kv_cache_dtype": QUALITY_KV_CACHE_DTYPE,
            "audit_max_tokens": QUALITY_MAX_TOKENS,
            "prompt_index": output["prompt_index"], "output_tokens": completion_tokens,
            "finish_reason": output.get("finish_reason", ""),
            **computed, "sha256": digest, "manual_review_status": status,
            "manual_review_notes": review.get("notes", ""),
            "source_json": f"data/evidence/{topology}-quality.json",
        })
    compact = dict(data)
    compact.update({
        "source_file": str(path), "source_sha256": sha256(path),
        "model_revision": MODEL_REVISION, "topology": topology,
        "runtime_provenance": provenance,
        "runtime_log_sha256": {
            host: sha256(log) for host, log in zip(("node0", "node1"), quality_logs, strict=True)
        },
        "manual_review": {"status": status, "notes": review.get("notes", "")},
    })
    return rows, compact


def compact_benchmark(path: Path, data: dict[str, Any], results: list[dict[str, Any]], topology: str) -> dict[str, Any]:
    keep = [
        "concurrency", "context_tokens", "benchmark_mode", "measurement_seconds",
        "client_output_tokens", "server_output_tokens", "aggregate_source",
        "aggregate_tps", "per_request_avg_tps", "ttft_p50",
        "inter_token_latency_p50", "input_seq_len_avg", "output_seq_len_avg",
        "request_count", "completed_request_count", "num_errors", "avg_running_reqs",
        "max_running_reqs", "effective_concurrency", "avg_queue_reqs",
        "max_queue_reqs", "capacity_limited",
    ]
    metadata = data["metadata"]
    return {
        "source_file": str(path), "source_sha256": sha256(path), "topology": topology,
        "metadata": {key: metadata.get(key) for key in (
            "version", "engine", "model", "timestamp", "decode_mode",
            "duration_per_test", "max_tokens", "temperature", "ignore_eos",
            "concurrency_levels", "context_lengths",
        )},
        "results": [{key: row.get(key) for key in keep} for row in results],
        "prefill": data["prefill"],
    }


def find_wikitext(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in sorted((run_dir / "wikitext2").rglob("*.json")) if (run_dir / "wikitext2").is_dir() else []:
        try:
            data = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if "word_perplexity,none" in data.get("results", {}).get("wikitext", {}):
            candidates.append(path)
    if len(candidates) > 1:
        raise ValueError("multiple WikiText-2 result JSON files found; retain one canonical result")
    return candidates[0] if candidates else None


def validate_ppl_runtime(evidence_dir: Path) -> dict[str, Any]:
    exact_text(evidence_dir / "tokenizer-config-sha256.txt", TOKENIZER_CONFIG_SHA256)
    exact_text(evidence_dir / "tokenizer-json-sha256.txt", TOKENIZER_JSON_SHA256)
    runtime = evidence_dir / "runtime"
    required = [
        "node0-inspect.json", "node1-inspect.json",
        "node0-server.log", "node1-server.log", "models.json",
        "metrics-after.txt", "model-revision.txt", "wikitext-result-dir.txt",
    ]
    missing = [name for name in required if not (runtime / name).is_file()]
    if missing:
        raise ValueError(f"WikiText runtime evidence is incomplete: {', '.join(missing)}")
    exact_text(runtime / "model-revision.txt", MODEL_REVISION)
    expected_options = {
        "--served-model-name": "GLM-5.2-NVFP4",
        "--tensor-parallel-size": "2", "--pipeline-parallel-size": "1",
        "--nnodes": "2", "--kv-cache-dtype": "bfloat16",
        "--max-model-len": "4096", "--max-num-seqs": "4",
        "--max-num-batched-tokens": "4096", "--moe-backend": "flashinfer_cutedsl",
    }
    inspections: dict[str, Any] = {}
    for rank, host in enumerate(("node0", "node1")):
        item = inspect_object(runtime / f"{host}-inspect.json")
        config = item.get("Config", {})
        command = config.get("Cmd", [])
        if config.get("Image") != RUNTIME_IMAGE or item.get("Image") != RUNTIME_IMAGE_ID:
            raise ValueError(f"WikiText {host} did not use the pinned runtime image")
        for name, expected in expected_options.items():
            if option_value(command, name) != expected:
                raise ValueError(f"WikiText {host}: {name} does not prove {expected}")
        if option_value(command, "--node-rank") != str(rank):
            raise ValueError(f"WikiText {host}: wrong node rank")
        utilization = option_value(command, "--gpu-memory-utilization")
        if utilization is None or not math.isclose(float(utilization), 0.90):
            raise ValueError(f"WikiText {host}: GPU utilization is not 0.90")
        for flag in (
            "--enable-expert-parallel", "--no-enable-flashinfer-autotune",
            "--enable-prefix-caching",
        ):
            if flag not in command:
                raise ValueError(f"WikiText {host}: missing {flag}")
        inspections[host] = {
            "command": command,
            "inspect_sha256": sha256(runtime / f"{host}-inspect.json"),
        }
    logs = [
        (runtime / f"{host}-server.log").read_text(errors="replace")
        for host in ("node0", "node1")
    ]
    if "Application startup complete" not in logs[0]:
        raise ValueError("WikiText rank-0 log does not prove API startup")
    if not all("NCCL" in log and "mlx5_0" in log for log in logs):
        raise ValueError("WikiText logs do not prove NCCL used mlx5_0 on both ranks")
    for marker in (
        "'kv_cache_dtype': 'bfloat16'", "'max_model_len': 4096",
        "'enable_expert_parallel': True", "'enable_flashinfer_autotune': False",
        "'moe_backend': 'flashinfer_cutedsl'",
    ):
        if marker not in logs[0]:
            raise ValueError(f"WikiText rank-0 log does not prove {marker}")
    return {
        "container_evidence": inspections,
        "runtime_log_sha256": {
            host: sha256(runtime / f"{host}-server.log")
            for host in ("node0", "node1")
        },
        "models_sha256": sha256(runtime / "models.json"),
        "metrics_after_sha256": sha256(runtime / "metrics-after.txt"),
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
    }


def extract_ppl(path: Path, topology: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        evidence_dir = next(parent for parent in path.parents if parent.name == "wikitext2")
    except StopIteration as error:
        raise ValueError("WikiText-2 JSON must be beneath a wikitext2/ directory") from error
    exact_text(evidence_dir / "model-revision.txt", MODEL_REVISION)
    exact_text(evidence_dir / "runtime-image.txt", RUNTIME_IMAGE)
    exact_text(evidence_dir / "lm-eval-commit.txt", LM_EVAL_COMMIT)
    kv_cache_dtype = (evidence_dir / "kv-cache-dtype.txt").read_text().strip()
    if kv_cache_dtype != "bfloat16":
        raise ValueError("GLM-5.2 WikiText-2 must use the pinned BF16 KV profile")
    batch_size = int((evidence_dir / "batch-size.txt").read_text().strip())
    if batch_size <= 0:
        raise ValueError("WikiText-2 batch size must be positive")
    data = read_json(path)
    result = data.get("results", {}).get("wikitext", {})
    if int(result.get("sample_len", 0)) != 62:
        raise ValueError("WikiText-2 result must contain 62 documents")
    config = data.get("configs", {}).get("wikitext", {})
    if config.get("dataset_path") != "EleutherAI/wikitext_document_level":
        raise ValueError("WikiText-2 result is not document-level EleutherAI/wikitext")
    metrics = {
        "word_ppl": finite_positive(result.get("word_perplexity,none"), "word PPL"),
        "byte_ppl": finite_positive(result.get("byte_perplexity,none"), "byte PPL"),
        "bits_per_byte": finite_positive(result.get("bits_per_byte,none"), "bits/byte"),
    }
    max_length = int(data.get("max_length", 0))
    # lm-eval is configured with max_length=2048 and reserves one token for
    # generation in TemplateAPI, so the result JSON records 2047.
    if max_length != 2047:
        raise ValueError("WikiText-2 effective max_length must be 2047")
    if not str(data.get("git_hash", "")).endswith(f"-g{LM_EVAL_COMMIT[:8]}"):
        raise ValueError("WikiText-2 result JSON does not prove the pinned lm-eval commit")
    sample_paths = sorted(path.parent.glob("samples_wikitext_*.jsonl"))
    if len(sample_paths) != 1:
        raise ValueError("WikiText-2 must retain exactly one sample JSONL")
    document_ids: list[int] = []
    with sample_paths[0].open() as handle:
        for line in handle:
            sample = json.loads(line)
            document_ids.append(int(sample["doc_id"]))
    if document_ids != list(range(62)):
        raise ValueError("WikiText-2 sample JSONL must contain ordered document IDs 0..61")
    runtime_provenance = validate_ppl_runtime(evidence_dir)
    row = {
        "topology": topology, "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE, "kv_cache_dtype": kv_cache_dtype,
        "dataset": "EleutherAI/wikitext_document_level", "documents": 62,
        "max_length": 2047, "batch_size": batch_size, **metrics,
        "lm_eval_commit": LM_EVAL_COMMIT,
        "source_json": f"data/evidence/{topology}-wikitext2.json",
    }
    compact = {
        "source_file": str(path), "source_sha256": sha256(path),
        "topology": topology, "model_revision": MODEL_REVISION,
        "kv_cache_dtype": kv_cache_dtype, "batch_size": batch_size, "result": result,
        "config": config, "max_length": max_length,
        "git_hash": data.get("git_hash"), "lm_eval_version": data.get("lm_eval_version"),
        "samples_file": str(sample_paths[0]), "samples_sha256": sha256(sample_paths[0]),
        "runtime_provenance": runtime_provenance,
    }
    return row, compact


def consume_run(run_dir: Path, topology: str, output_root: Path, manual: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    benchmark_path = run_dir / "llm-inference-bench.json"
    quality_path = run_dir / "quality" / "quality-audit.json"
    if not benchmark_path.is_file() or not quality_path.is_file() or not (run_dir / "runtime").is_dir():
        raise ValueError("run requires llm-inference-bench.json, quality/quality-audit.json, and runtime/")
    runtime_manifest = validate_runtime(run_dir, topology)
    data, results = validate_benchmark(benchmark_path)
    quality_rows, quality_compact = validate_quality(quality_path, topology, manual)
    evidence = output_root / "data" / "evidence"
    runtime_out = output_root / "data" / "runtime"
    evidence.mkdir(parents=True, exist_ok=True)
    runtime_out.mkdir(parents=True, exist_ok=True)
    write_public_json(
        evidence / f"{topology}-benchmark.json",
        compact_benchmark(benchmark_path, data, results, topology),
    )
    write_public_json(
        evidence / f"{topology}-quality.json", quality_compact,
        ensure_ascii=False,
    )
    capped_audit_path = quality_path.parent / "quality-audit-1024-cap.json"
    if capped_audit_path.is_file():
        capped = read_json(capped_audit_path)
        if capped.get("settings") != {"temperature": 0, "max_tokens": 1024}:
            raise ValueError("preserved capped audit does not prove temperature=0/max_tokens=1024")
        if len(capped.get("outputs", [])) != 4:
            raise ValueError("preserved capped audit does not contain four outputs")
        capped.update({
            "source_file": str(capped_audit_path),
            "source_sha256": sha256(capped_audit_path),
            "model_revision": MODEL_REVISION,
            "topology": topology,
            "kv_cache_dtype": "fp8_e4m3",
            "publication_status": "preserved_noncanonical_token-cap audit",
        })
        write_public_json(
            evidence / f"{topology}-quality-1024-cap.json", capped,
            ensure_ascii=False,
        )
    write_public_json(runtime_out / f"{topology}.json", runtime_manifest)

    settings = TOPOLOGIES[topology]
    common = {
        "topology": topology, **settings, "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE, "benchmark_commit": BENCHMARK_COMMIT,
        "kv_cache_dtype": "fp8_e4m3",
    }
    throughput = []
    for row in results:
        throughput.append(common | {
            "context_tokens": int(row["input_seq_len_avg"]),
            "output_tokens": 1024, "temperature": 0,
            "concurrency": int(row["concurrency"]), "request_count": row["request_count"],
            "completed_requests": row["completed_request_count"],
            "aggregate_output_tok_s": row["aggregate_tps"],
            "per_request_output_tok_s": row["per_request_avg_tps"],
            "median_ttft_s": row["ttft_p50"],
            "median_itl_ms": float(row["inter_token_latency_p50"]) * 1000,
            "effective_concurrency": row["effective_concurrency"],
            "max_running_requests": row["max_running_reqs"],
            "capacity_limited": str(bool(row["capacity_limited"])).lower(),
            "num_errors": 0, "source_json": f"data/evidence/{topology}-benchmark.json",
        })
    prefill = []
    for context in PREFILL_CONTEXTS:
        row = data["prefill"][str(context)]
        prefill.append(common | {
            "context_tokens": context, "actual_prompt_tokens": row["prompt_tokens"],
            "samples": row["samples"], "prompt_tok_s": row["tok_per_sec"],
            "median_ttft_s": row["ttft_seconds"], "num_errors": 0,
            "source_json": f"data/evidence/{topology}-benchmark.json",
        })
    ppl_rows: list[dict[str, Any]] = []
    ppl_path = find_wikitext(run_dir)
    if ppl_path:
        ppl_row, ppl_compact = extract_ppl(ppl_path, topology)
        ppl_rows.append(ppl_row)
        write_public_json(evidence / f"{topology}-wikitext2.json", ppl_compact)
    return {"throughput": throughput, "prefill": prefill, "quality": quality_rows, "ppl": ppl_rows}


def extract(results_root: Path, output_root: Path) -> None:
    data_dir = output_root / "data"
    evidence = data_dir / "evidence"
    runtime = data_dir / "runtime"
    evidence.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    for path in list(evidence.glob("pp2-*.json")) + list(evidence.glob("tp2-*.json")):
        path.unlink()
    for path in (runtime / "pp2.json", runtime / "tp2.json"):
        path.unlink(missing_ok=True)
    manual_path = data_dir / "manual-quality-review.json"
    manual = read_json(manual_path) if manual_path.is_file() else {}
    collected = {key: [] for key in ("throughput", "prefill", "quality", "ppl")}
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    for topology in TOPOLOGIES:
        run_dir = results_root / f"glm52-nvfp4-{topology}"
        if not run_dir.is_dir():
            if topology in STARTUP_FAILURES:
                rejected[topology] = STARTUP_FAILURES[topology]
                print(f"rejected {run_dir.name}: {STARTUP_FAILURES[topology]}")
                continue
            print(f"pending {run_dir.name}: directory absent")
            continue
        try:
            rows = consume_run(run_dir, topology, output_root, manual)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            for path in evidence.glob(f"{topology}-*.json"):
                path.unlink()
            (runtime / f"{topology}.json").unlink(missing_ok=True)
            rejected[topology] = str(error)
            print(f"rejected {run_dir.name}: {error}")
            continue
        for key in collected:
            collected[key].extend(rows[key])
        accepted.append(topology)
        print(f"accepted {run_dir.name}")
    collected["throughput"].sort(key=lambda row: (row["topology"], row["concurrency"]))
    collected["prefill"].sort(key=lambda row: (row["topology"], row["context_tokens"]))
    collected["quality"].sort(key=lambda row: (row["topology"], row["prompt_index"]))
    collected["ppl"].sort(key=lambda row: row["topology"])
    write_csv(data_dir / "throughput.csv", THROUGHPUT_FIELDS, collected["throughput"])
    write_csv(data_dir / "prefill.csv", PREFILL_FIELDS, collected["prefill"])
    write_csv(data_dir / "quality-audit.csv", QUALITY_FIELDS, collected["quality"])
    write_csv(data_dir / "wikitext2-perplexity.csv", PPL_FIELDS, collected["ppl"])
    manifest = {
        "accepted_topologies": accepted, "rejected_topologies": rejected,
        "pending_topologies": sorted(set(TOPOLOGIES) - set(accepted) - set(rejected)),
        "required_concurrencies": CONCURRENCIES, "required_prefill_contexts": PREFILL_CONTEXTS,
        "model_revision": MODEL_REVISION, "runtime_image": RUNTIME_IMAGE,
        "benchmark_commit": BENCHMARK_COMMIT, "lm_eval_commit": LM_EVAL_COMMIT,
    }
    write_public_json(data_dir / "publication-manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path,
        default=PACKAGE_ROOT / "results",
        help="parent of glm52-nvfp4-pp2 and glm52-nvfp4-tp2",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_ROOT,
        help="GLM public-package root (primarily useful for tests)",
    )
    args = parser.parse_args()
    extract(args.results_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
