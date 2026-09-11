#!/usr/bin/env python3
"""Convert private DeepSeek-V4.1-Flash benchmark artifacts into this section's published tables.

Inputs are declared in ``data/sources.json`` with paths relative to ``--source-root`` (the
private benchmark working directory, which is never committed):

* ``bench_prefill.py`` JSONL files            -> ``prefill.csv`` (accepted lanes) and
                                                 ``diagnostic-prefill.csv`` (tuning ladder, spot checks)
* ``llm-inference-bench`` ``c*.json`` run dirs -> ``throughput.csv`` (decode)
* nccl-tests ``all_reduce_perf`` logs         -> ``fabric.csv``
* ``tools/analyze_trace.py`` summaries        -> ``profile.csv``
* lane dispositions                           -> ``qualification.csv``
* audited checkpoint facts                    -> ``checkpoint.json``

Hostnames are mapped to ``node0``/``node1`` (rank order) from the artifacts themselves or from
``--host-alias``; absolute private paths are never written. ``--check`` validates the committed
tables offline without reading any private artifact.

Every lane row carries the manifest's ``lane_label`` (stable, human-readable) and ``series_order``
(integer) so that charts and README tables share one label and one ordering per configuration.
``publication_status``/``rankable`` keep their meaning (accepted lanes rank, diagnostic lanes do
not); they never decide whether a lane is drawn.

Usage:
    python3 data/build_data.py --source-root /path/to/private/workdir [--copy-evidence]
    python3 data/build_data.py --check
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "sources.json"

PREFILL_FIELDS = (
    "lane", "lane_label", "series_order", "run_id", "profile", "engine", "mode", "publication_status", "rankable",
    "model_id", "model_revision", "runtime", "runtime_image_id", "runtime_commit",
    "topology", "tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size",
    "mtp_tokens", "kv_cache_dtype", "chunked_prefill_size", "max_prefill_tokens",
    "swa_bounded_replay", "nccl_hcas", "host_rdma", "ladder_step", "ladder_label",
    "isl_tokens", "actual_prompt_tokens_per_request", "concurrency", "requests",
    "warmup_requests", "flush_cache", "wall_seconds", "prompt_tokens_total",
    "aggregate_prompt_tokens_per_second", "per_request_prompt_tokens_per_second_mean",
    "ttft_mean_seconds", "ttft_p50_seconds", "ttft_p99_seconds", "server_prompt_tokens_delta",
    "token_count_match", "gpu_util_mean_pct", "gpu_power_mean_w", "gpu_power_max_w",
    "num_errors", "result_file", "source_artifact_sha256",
)
DIAGNOSTIC_PREFILL_FIELDS = PREFILL_FIELDS + ("diagnostic_status",)
THROUGHPUT_FIELDS = (
    "lane", "lane_label", "series_order", "run_id", "profile", "engine", "mode", "publication_status", "rankable",
    "model_id", "model_revision", "runtime", "runtime_image_id", "runtime_commit",
    "topology", "tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size",
    "mtp_tokens", "input_tokens", "target_output_tokens", "duration_seconds", "concurrency",
    "requests", "warmup_requests", "completed_requests", "effective_concurrency",
    "capacity_limited", "underfilled", "aggregate_output_tokens_per_second",
    "per_user_output_tokens_per_second_p50", "accept_length", "accept_rate",
    "mtp_accept_length", "engine_steps_per_second", "median_ttft_seconds", "median_itl_ms",
    "request_latency_p50_seconds", "num_errors", "benchmark_client", "benchmark_client_commit",
    "result_file", "source_artifact_sha256",
)
FABRIC_FIELDS = (
    "config_id", "label", "rails", "nccl_hcas", "nccl_tuning", "channels", "qps_per_connection",
    "message_bytes", "message_label", "elements", "time_us", "algbw_gbps", "busbw_gbps",
    "wrong_values", "in_place", "avg_busbw_gbps", "out_of_bounds_ok", "data_direct_detected",
    "nccl_tests_version", "source_file", "source_artifact_sha256",
)
PROFILE_FIELDS = (
    "profile_id", "label", "lane", "engine", "mode", "topology", "rank", "node", "isl_tokens",
    "kernel_wall_span_ms", "summed_kernel_ms", "kernel_count", "category", "category_ms",
    "category_pct", "source_file", "source_artifact_sha256",
)
QUALIFICATION_FIELDS = (
    "profile", "lane", "engine", "mode", "model_id", "model_revision", "runtime", "topology",
    "mtp_tokens", "status", "rankable", "evidence_run_id", "note",
)
TABLE_FIELDS = {
    "prefill.csv": PREFILL_FIELDS,
    "diagnostic-prefill.csv": DIAGNOSTIC_PREFILL_FIELDS,
    "throughput.csv": THROUGHPUT_FIELDS,
    "fabric.csv": FABRIC_FIELDS,
    "profile.csv": PROFILE_FIELDS,
    "qualification.csv": QUALIFICATION_FIELDS,
}
RATE_COLUMNS = {
    "prefill.csv": ("aggregate_prompt_tokens_per_second", "ttft_p50_seconds"),
    "diagnostic-prefill.csv": ("aggregate_prompt_tokens_per_second", "ttft_p50_seconds"),
    "throughput.csv": ("aggregate_output_tokens_per_second", "per_user_output_tokens_per_second_p50"),
    "fabric.csv": ("busbw_gbps", "algbw_gbps"),
    "profile.csv": ("category_ms",),
}
KEY_COLUMNS = {
    "prefill.csv": ("lane", "profile", "isl_tokens", "concurrency"),
    "diagnostic-prefill.csv": ("lane", "run_id", "isl_tokens", "concurrency"),
    "throughput.csv": ("lane", "profile", "concurrency"),
    "fabric.csv": ("config_id", "message_bytes", "in_place"),
    "profile.csv": ("profile_id", "rank", "category"),
    "qualification.csv": ("profile",),
}
STATUS_BY_DISPOSITION = {
    ("accepted", "prefill"): "PASS_RANKABLE_PREFILL",
    ("accepted", "decode"): "PASS_RANKABLE_DECODE",
    ("diagnostic", "prefill"): "DIAGNOSTIC_UNRANKED",
    ("diagnostic", "decode"): "DIAGNOSTIC_UNRANKED",
    ("smoke", "prefill"): "PASS_SMOKE_UNRANKED",
    ("smoke", "decode"): "PASS_SMOKE_UNRANKED",
    ("pending", "prefill"): "PENDING",
    ("pending", "decode"): "PENDING",
    ("not_measured", "prefill"): "NOT_MEASURED",
    ("not_measured", "decode"): "NOT_MEASURED",
    ("failed", "prefill"): "FAILED_BENCHMARK",
    ("failed", "decode"): "FAILED_BENCHMARK",
}
ROW_STATUSES = {"accepted", "diagnostic"}
NO_ROW_STATUSES = {"pending", "not_measured", "failed", "smoke"}

# Generic redaction signatures: private home/mount paths, management-style addresses and GPU UUIDs.
FORBIDDEN_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile(r"/mnt/[A-Za-z0-9_.-]+"),
    re.compile(r"/root/[A-Za-z0-9_.-]+"),
    re.compile(r"\bGPU" r"-[0-9a-fA-F]{8}-"),  # split so the pattern itself is not a leak-scan hit
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
)
ALLOWED_ADDRESS = re.compile(r"\b(127\.0\.0\.1|192\.168\.200\.\d{1,3}|192\.0\.2\.\d{1,3})\b")
ANY_ADDRESS = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


class BuildError(RuntimeError):
    pass


def boolean(value: object) -> str:
    return "true" if value else "false"


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return boolean(value)
    if isinstance(value, float):
        compact = f"{value:g}"  # 6 significant digits when that is lossless, else the exact repr
        return compact if float(compact) == value else repr(value)
    return str(value)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Redactor:
    """Maps rank hostnames to node0/node1 and refuses private identifiers in outputs."""

    RANK_LINE = re.compile(r"^#\s+Rank\s+(\d+)\s+Group\s+\d+\s+Pid\s+\d+\s+on\s+(\S+)\s+device", re.M)

    def __init__(self, aliases: dict[str, str]):
        self.aliases = dict(aliases)

    def learn_from_nccl_log(self, content: str) -> None:
        for rank, host in self.RANK_LINE.findall(content):
            self.aliases.setdefault(host, f"node{int(rank)}")

    def sanitize(self, content: str) -> str:
        for host, alias in sorted(self.aliases.items(), key=lambda item: -len(item[0])):
            content = re.sub(rf"(?<![A-Za-z0-9_.-]){re.escape(host)}(?![A-Za-z0-9_-])", alias, content)
        return content

    def problems(self, content: str, where: str) -> list[str]:
        found: list[str] = []
        for host in self.aliases:
            if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(host)}(?![A-Za-z0-9_-])", content):
                found.append(f"{where}: hostname {host!r} leaked")
        for pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(content)
            if match:
                found.append(f"{where}: forbidden identifier {match.group(0)!r}")
        for match in ANY_ADDRESS.finditer(content):
            if not ALLOWED_ADDRESS.match(match.group(0)):
                found.append(f"{where}: non-example address {match.group(0)!r}")
        return found


def resolve_inputs(root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            paths.extend(sorted(Path(match) for match in glob.glob(str(root / pattern))))
        else:
            paths.append(root / pattern)
    return paths


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise BuildError(f"{path.name}:{number}: invalid JSON ({error})") from error
    return records


def lane_columns(lane_id: str, lane: dict, sources: dict, metric: str) -> dict[str, str]:
    runtime = sources["runtimes"][lane["engine"]]
    server = lane.get("server", {})
    columns = {
        "lane": lane_id,
        "lane_label": lane["lane_label"],
        "series_order": text(int(lane["series_order"])),
        "profile": lane["profile"],
        "engine": lane["engine"],
        "mode": lane["mode"],
        "model_id": sources["model_id"],
        "model_revision": sources["model_revision"],
        "runtime": lane["engine"],
        "runtime_image_id": runtime["image_id"],
        "runtime_commit": runtime["commit"],
        "topology": lane["topology"],
        "tensor_parallel_size": text(lane["tensor_parallel_size"]),
        "pipeline_parallel_size": text(lane["pipeline_parallel_size"]),
        "expert_parallel_size": text(lane["expert_parallel_size"]),
        "mtp_tokens": "0",
    }
    if metric == "prefill":
        columns.update({
            "kv_cache_dtype": text(server.get("kv_cache_dtype")),
            "chunked_prefill_size": text(server.get("chunked_prefill_size")),
            "max_prefill_tokens": text(server.get("max_prefill_tokens")),
            "swa_bounded_replay": boolean(server.get("swa_bounded_replay")),
            "nccl_hcas": text(server.get("nccl_hcas")),
            "host_rdma": boolean(server.get("host_rdma")),
        })
    return columns


def prefill_rows_from_file(path: Path, base: dict[str, str], status: str, rankable: bool,
                           ladder: dict[str, str], strict: bool) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    notes: list[str] = []
    digest = sha256_of(path)
    for record in read_jsonl(path):
        errors = int(record.get("errors") or 0)
        rate = record.get("agg_prompt_tok_s")
        if errors or not rate or rate <= 0 or not record.get("ok", True):
            message = f"{path.name}: isl={record.get('isl')} C={record.get('concurrency')} has {errors} errors"
            if strict:
                raise BuildError(message + " (accepted lanes must be error-free)")
            notes.append(message + " -> dropped, never published as zero")
            continue
        metrics = record.get("metrics_delta") or {}
        server_delta = next((metrics[key] for key in metrics if key.endswith("prompt_tokens_total")), None)
        requests = int(record["requests"])
        prompt_total = int(record["prompt_tokens_total"])
        row = dict(base)
        row.update({
            "run_id": path.stem,
            "publication_status": status,
            "rankable": boolean(rankable),
            "ladder_step": ladder.get("step", ""),
            "ladder_label": ladder.get("label", ""),
            "isl_tokens": text(record["isl"]),
            "actual_prompt_tokens_per_request": text(prompt_total // requests if requests else ""),
            "concurrency": text(record["concurrency"]),
            "requests": text(requests),
            "warmup_requests": text(record.get("warmup")),
            "flush_cache": boolean(record.get("flush_cache")),
            "wall_seconds": text(record["wall_s"]),
            "prompt_tokens_total": text(prompt_total),
            "aggregate_prompt_tokens_per_second": text(rate),
            "per_request_prompt_tokens_per_second_mean": text(record.get("req_tok_s_mean")),
            "ttft_mean_seconds": text(record.get("ttft_mean_s")),
            "ttft_p50_seconds": text(record.get("ttft_p50_s")),
            "ttft_p99_seconds": text(record.get("ttft_p99_s")),
            "server_prompt_tokens_delta": text(int(server_delta)) if server_delta is not None else "",
            "token_count_match": boolean(record.get("token_count_match")),
            "gpu_util_mean_pct": text(record.get("gpu_util_mean_pct")),
            "gpu_power_mean_w": text(record.get("gpu_power_mean_w")),
            "gpu_power_max_w": text(record.get("gpu_power_max_w")),
            "num_errors": text(errors),
            "result_file": path.name,
            "source_artifact_sha256": digest,
        })
        server = record.get("server") or {}
        for key in ("chunked_prefill_size", "max_prefill_tokens"):
            if key in server and base.get(key) and text(server[key]) != base[key]:
                raise BuildError(f"{path.name}: server {key}={server[key]} disagrees with sources.json {base[key]}")
        rows.append(row)
    return rows, notes


def check_prefill_grid(rows: list[dict[str, str]], contract: dict, lane_id: str) -> None:
    expected = {(isl, c) for isl in contract["isl_tokens"] for c in contract["concurrency"]}
    seen = {(int(row["isl_tokens"]), int(row["concurrency"])) for row in rows}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise BuildError(f"lane {lane_id}: accepted prefill grid incomplete; missing={missing} extra={extra}")
    if len(seen) != len(rows):
        raise BuildError(f"lane {lane_id}: duplicate prefill points")
    for row in rows:
        c = int(row["concurrency"])
        if int(row["requests"]) != max(4, 4 * c):
            raise BuildError(f"lane {lane_id}: C{c} ran {row['requests']} requests, contract is max(4, 4C)")
        if row["token_count_match"] != "true":
            raise BuildError(f"lane {lane_id}: token count mismatch at isl={row['isl_tokens']} C{c}")
        if row["server_prompt_tokens_delta"] and row["server_prompt_tokens_delta"] != row["prompt_tokens_total"]:
            raise BuildError(f"lane {lane_id}: server counter delta != client prompt tokens at isl={row['isl_tokens']} C{c}")


def decode_rows_from_dir(run_dir: Path, base: dict[str, str], status: str, rankable: bool,
                         client: dict, strict: bool) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    notes: list[str] = []
    files = sorted(run_dir.glob("c*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not files:
        raise BuildError(f"{run_dir.name}: no c*.json files")
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            message = f"{run_dir.name}/{path.name}: unreadable JSON ({error})"
            if strict:
                raise BuildError(message + " (accepted lanes need complete cell files)") from error
            notes.append(message + " -> skipped, cell treated as not measured")
            continue
        metadata = payload.get("metadata") or {}
        digest = sha256_of(path)
        for cell in payload.get("results") or []:
            mode = cell.get("benchmark_mode", "")
            if mode != "request-count":  # duration / burst-e2e cells are a different contract; never published here
                notes.append(f"{run_dir.name}/{path.name}: C{cell.get('concurrency')} is a {mode or 'unknown'} cell, "
                             "not the finite-request layer -> skipped")
                continue
            errors = int(cell.get("num_errors") or 0)
            rate = cell.get("aggregate_tps")
            if errors or not rate or rate <= 0:
                message = f"{run_dir.name}/{path.name}: C{cell.get('concurrency')} has {errors} errors"
                if strict:
                    raise BuildError(message + " (accepted lanes must be error-free)")
                notes.append(message + " -> dropped, never published as zero")
                continue
            dspark = base["mode"] == "dspark"
            accept_length = cell.get("server_spec_accept_length") if dspark else None
            itl_p50 = cell.get("inter_token_latency_p50")
            row = dict(base)
            row.update({
                "run_id": run_dir.name,
                "publication_status": status,
                "rankable": boolean(rankable),
                "input_tokens": text(cell.get("context_tokens")),
                "target_output_tokens": text(metadata.get("max_tokens")),
                "duration_seconds": text(cell.get("measurement_wall_seconds") or cell.get("wall_time")),
                "concurrency": text(cell["concurrency"]),
                "requests": text(cell.get("request_count")),
                "warmup_requests": text(cell.get("warmup_request_count")),
                "completed_requests": text(cell.get("completed_request_count")),
                "effective_concurrency": text(cell.get("effective_concurrency")),
                "capacity_limited": boolean(cell.get("capacity_limited")),
                "underfilled": boolean(cell.get("underfilled")),
                "aggregate_output_tokens_per_second": text(round(float(rate), 1)),
                "per_user_output_tokens_per_second_p50": text(cell.get("output_tps_per_user_p50")),
                "accept_length": text(accept_length) if accept_length else "",
                "accept_rate": text(cell.get("server_spec_accept_rate")) if accept_length else "",
                "mtp_accept_length": "",
                # SGLang AR and vLLM report no engine-step counter (0.0): leave the cell empty, never publish a zero.
                "engine_steps_per_second": text(cell.get("server_steps_per_s")) if (cell.get("server_steps_per_s") or 0) > 0 else "",
                "median_ttft_seconds": text(cell.get("ttft_p50")),
                "median_itl_ms": text(round(float(itl_p50) * 1000, 3)) if itl_p50 is not None else "",
                "request_latency_p50_seconds": text(cell.get("request_latency_p50")),
                "num_errors": text(errors),
                "benchmark_client": f"{client['name']} {client['version']}",
                "benchmark_client_commit": client["commit"],
                "result_file": f"{run_dir.name}/{path.name}",
                "source_artifact_sha256": digest,
            })
            rows.append(row)
    return rows, notes


def decode_run_dirs(block: dict) -> list[str]:
    """A decode block names one ``run_dir`` or a ``run_dirs`` list (a later run that adds cells, such as a C64
    point measured after the C1-C32 sweep, is merged into the same lane; duplicate concurrencies are refused)."""
    if block.get("run_dirs"):
        if block.get("run_dir"):
            raise BuildError("decode block lists both run_dir and run_dirs")
        return list(block["run_dirs"])
    return [block["run_dir"]] if block.get("run_dir") else []


def check_decode_rows(rows: list[dict[str, str]], lane_id: str) -> None:
    seen = set()
    for row in rows:
        c = int(row["concurrency"])
        if c in seen:
            raise BuildError(f"lane {lane_id}: duplicate decode concurrency C{c}")
        seen.add(c)
        if row["requests"] and int(row["requests"]) != 5 * c:
            raise BuildError(f"lane {lane_id}: C{c} ran {row['requests']} requests, contract is 5C")
        if row["underfilled"] == "true":
            raise BuildError(f"lane {lane_id}: C{c} underfilled; accepted rows must hold offered concurrency")


FABRIC_ROW = re.compile(
    r"^\s+(\d+)\s+(\d+)\s+float\s+sum\s+-1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
    re.M,
)


def size_label(num_bytes: int) -> str:
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):g} GiB"
    return f"{num_bytes / (1 << 20):g} MiB"


def fabric_rows(entry: dict, path: Path, redactor: Redactor) -> list[dict[str, str]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    redactor.learn_from_nccl_log(content)
    digest = sha256_of(path)
    avg = re.search(r"# Avg bus bandwidth\s*:\s*([\d.]+)", content)
    oob = re.search(r"# Out of bounds values\s*:\s*(\d+)\s*(OK|FAILED)?", content)
    version = re.search(r"# nccl-tests version (\S+)", content)
    data_direct = len(re.findall(r"Data Direct DMA Interface is detected", content))
    rows = []
    for match in FABRIC_ROW.finditer(content):
        size, count, t_out, alg_out, bus_out, wrong_out, t_in, alg_in, bus_in, wrong_in = match.groups()
        for in_place, values in ((False, (t_out, alg_out, bus_out, wrong_out)), (True, (t_in, alg_in, bus_in, wrong_in))):
            rows.append({
                "config_id": entry["config_id"],
                "label": entry["label"],
                "rails": text(entry["rails"]),
                "nccl_hcas": entry["nccl_hcas"],
                "nccl_tuning": entry["nccl_tuning"],
                "channels": text(entry.get("channels")),
                "qps_per_connection": text(entry.get("qps_per_connection")),
                "message_bytes": size,
                "message_label": size_label(int(size)),
                "elements": count,
                "time_us": values[0],
                "algbw_gbps": values[1],
                "busbw_gbps": values[2],
                "wrong_values": values[3],
                "in_place": boolean(in_place),
                "avg_busbw_gbps": avg.group(1) if avg else "",
                "out_of_bounds_ok": boolean(oob and oob.group(1) == "0"),
                "data_direct_detected": boolean(data_direct > 0),
                "nccl_tests_version": version.group(1) if version else "",
                "source_file": path.name,
                "source_artifact_sha256": digest,
            })
    if not rows:
        raise BuildError(f"{path.name}: no all_reduce_perf result rows found")
    return rows


PROFILE_HEADER = re.compile(
    r"GPU kernel wall span:\s*([\d.]+)\s*ms;\s*summed kernel time:\s*([\d.]+)\s*ms;\s*kernels:\s*(\d+)"
)
PROFILE_CATEGORY = re.compile(r"^\s{2}(\S.*?)\s+([\d.]+)\s+([\d.]+)%\s*$", re.M)


def profile_rows(entry: dict, rank: str, path: Path, lane: dict) -> list[dict[str, str]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    header = PROFILE_HEADER.search(content)
    if not header:
        raise BuildError(f"{path.name}: missing 'GPU kernel wall span' header")
    section = content.split("By category", 1)[1].split("\n\n", 1)[0] if "By category" in content else ""
    rows = []
    for category, ms, pct in PROFILE_CATEGORY.findall(section):
        rows.append({
            "profile_id": entry["profile_id"],
            "label": entry["label"],
            "lane": entry["lane"],
            "engine": lane["engine"],
            "mode": lane["mode"],
            "topology": lane["topology"],
            "rank": rank,
            "node": f"node{int(rank)}",
            "isl_tokens": text(entry.get("isl_tokens")),
            "kernel_wall_span_ms": header.group(1),
            "summed_kernel_ms": header.group(2),
            "kernel_count": header.group(3),
            "category": category.strip(),
            "category_ms": ms,
            "category_pct": pct,
            "source_file": path.name,
            "source_artifact_sha256": sha256_of(path),
        })
    if not rows:
        raise BuildError(f"{path.name}: no category rows found")
    return rows


def qualification_rows(sources: dict, evidence: dict[tuple[str, str], str]) -> list[dict[str, str]]:
    rows = []
    for lane_id, lane in sources["lanes"].items():
        for metric in ("prefill", "decode"):
            block = lane[metric]
            status = block["publication_status"]
            if status not in ROW_STATUSES | NO_ROW_STATUSES:
                raise BuildError(f"lane {lane_id}: unknown {metric} publication_status {status!r}")
            rankable = status == "accepted" and bool(block.get("rankable"))
            label = STATUS_BY_DISPOSITION[(status, metric)]
            if status == "diagnostic" and block.get("diagnostic_status"):
                label = f"DIAGNOSTIC_UNRANKED_{block['diagnostic_status']}"
            rows.append({
                "profile": f"{lane_id}_{metric}",
                "lane": lane_id,
                "engine": lane["engine"],
                "mode": lane["mode"],
                "model_id": sources["model_id"],
                "model_revision": sources["model_revision"],
                "runtime": lane["engine"],
                "topology": lane["topology"],
                "mtp_tokens": "0",
                "status": label,
                "rankable": boolean(rankable),
                "evidence_run_id": evidence.get((lane_id, metric), ""),
                "note": block.get("note", ""),
            })
    for extra in sources.get("extra_qualification", []):
        rows.append({
            "profile": extra["profile"],
            "lane": extra.get("lane", ""),
            "engine": extra.get("engine", ""),
            "mode": extra.get("mode", ""),
            "model_id": sources["model_id"],
            "model_revision": sources["model_revision"],
            "runtime": extra.get("runtime", ""),
            "topology": extra.get("topology", ""),
            "mtp_tokens": "",
            "status": extra["status"],
            "rankable": boolean(extra.get("rankable")),
            "evidence_run_id": extra.get("evidence_run_id", ""),
            "note": extra.get("note", ""),
        })
    return rows


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def verify_checkpoint(checkpoint: dict, checkpoint_dir: Path) -> None:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    facts = {
        "architecture": config["architectures"][0],
        "index_total_size_bytes": index["metadata"]["total_size"],
        "index_tensor_entries": len(index["weight_map"]),
        "safetensors_files": len(shards),
        "safetensors_bytes": sum((checkpoint_dir / shard).stat().st_size for shard in shards),
    }
    for key, value in facts.items():
        if checkpoint.get(key) != value:
            raise BuildError(f"checkpoint.json {key}={checkpoint.get(key)!r} but checkpoint dir says {value!r}")
    quant = config.get("quantization_config", {})
    if quant.get("expert_dtype") != checkpoint["precision"]["expert_dtype"]:
        raise BuildError("checkpoint.json precision.expert_dtype disagrees with config.json")


def build(args: argparse.Namespace) -> int:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    root = Path(args.source_root).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    redactor = Redactor(dict(alias.split("=", 1) for alias in args.host_alias))
    notes: list[str] = []
    evidence: dict[tuple[str, str], str] = {}
    evidence_files: list[tuple[Path, str]] = []  # (source, name under data/evidence/)
    ladder = {entry["lane"]: {"step": text(entry["step"]), "label": entry["label"]}
              for entry in sources.get("tuning_ladder", [])}

    prefill: list[dict[str, str]] = []
    diagnostic: list[dict[str, str]] = []
    decode: list[dict[str, str]] = []
    for lane_id, lane in sources["lanes"].items():
        block = lane["prefill"]
        status = block["publication_status"]
        base = lane_columns(lane_id, lane, sources, "prefill")
        files = resolve_inputs(root, block.get("jsonl", []))
        if status in ROW_STATUSES:
            if not files or not all(path.is_file() for path in files):
                raise BuildError(f"lane {lane_id}: prefill status {status!r} but inputs are missing: {block.get('jsonl')}")
            rows: list[dict[str, str]] = []
            for path in files:
                found, dropped = prefill_rows_from_file(
                    path, base, status, status == "accepted" and bool(block.get("rankable")),
                    ladder.get(lane_id, {}), strict=(status == "accepted"))
                rows.extend(found)
                notes.extend(dropped)
                evidence_files.append((path, path.name))
            if status == "accepted":
                check_prefill_grid(rows, sources["prefill_contract"], lane_id)
                prefill.extend(rows)
            else:
                for row in rows:
                    row["diagnostic_status"] = block.get("diagnostic_status", "DIAGNOSTIC")
                diagnostic.extend(rows)
            evidence[(lane_id, "prefill")] = ";".join(sorted({row["run_id"] for row in rows}))
        elif files and any(path.is_file() for path in files):
            notes.append(f"lane {lane_id}: prefill inputs exist but status is {status!r}; no rows written")
        for path in resolve_inputs(root, block.get("diagnostic_jsonl", [])):
            if not path.is_file():
                raise BuildError(f"lane {lane_id}: diagnostic input missing: {path.name}")
            found, dropped = prefill_rows_from_file(path, base, "diagnostic", False, ladder.get(lane_id, {}), strict=False)
            for row in found:
                row["diagnostic_status"] = block.get("diagnostic_status", "DIAGNOSTIC")
            diagnostic.extend(found)
            notes.extend(dropped)
            evidence_files.append((path, path.name))

        block = lane["decode"]
        status = block["publication_status"]
        run_dirs = [root / entry for entry in decode_run_dirs(block)]
        if status in ROW_STATUSES:
            missing = [entry.name for entry in run_dirs if not entry.is_dir()]
            if not run_dirs or missing:
                raise BuildError(f"lane {lane_id}: decode status {status!r} but run_dir is missing {missing or ''}")
            base = lane_columns(lane_id, lane, sources, "decode")
            rows: list[dict[str, str]] = []
            for run_dir in run_dirs:
                found, dropped = decode_rows_from_dir(
                    run_dir, base, status, status == "accepted" and bool(block.get("rankable")),
                    sources["decode_client"], strict=(status == "accepted"))
                rows.extend(found)
                notes.extend(dropped)
                evidence_files.extend((path, f"{run_dir.name}/{path.name}") for path in sorted(run_dir.glob("c*.json")))
            if status == "accepted":
                check_decode_rows(rows, lane_id)
            decode.extend(rows)
            evidence[(lane_id, "decode")] = ";".join(run_dir.name for run_dir in run_dirs)
        elif any(entry.is_dir() for entry in run_dirs):
            notes.append(f"lane {lane_id}: decode run_dir exists but status is {status!r}; no rows written")

    fabric: list[dict[str, str]] = []
    for entry in sources.get("fabric", []):
        path = root / entry["log"]
        if not path.is_file():
            notes.append(f"fabric {entry['config_id']}: log missing ({entry['log']}); skipped")
            continue
        fabric.extend(fabric_rows(entry, path, redactor))
        evidence_files.append((path, path.name))

    profile: list[dict[str, str]] = []
    for entry in sources.get("profiles", []):
        lane = sources["lanes"][entry["lane"]]
        for rank, relative in sorted(entry["ranks"].items()):
            path = root / relative
            if not path.is_file():
                notes.append(f"profile {entry['profile_id']} rank {rank}: missing ({relative}); skipped")
                continue
            profile.extend(profile_rows(entry, rank, path, lane))
            evidence_files.append((path, path.name))

    qualification = qualification_rows(sources, evidence)

    write_csv(out / "prefill.csv", PREFILL_FIELDS, prefill)
    write_csv(out / "diagnostic-prefill.csv", DIAGNOSTIC_PREFILL_FIELDS, diagnostic)
    write_csv(out / "throughput.csv", THROUGHPUT_FIELDS, decode)
    write_csv(out / "fabric.csv", FABRIC_FIELDS, fabric)
    write_csv(out / "profile.csv", PROFILE_FIELDS, profile)
    write_csv(out / "qualification.csv", QUALIFICATION_FIELDS, qualification)
    checkpoint = sources["checkpoint"]
    if args.checkpoint_dir:
        verify_checkpoint(checkpoint, Path(args.checkpoint_dir))
    (out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    evidence_problems: list[str] = []
    if args.copy_evidence:
        evidence_dir = out / "evidence"
        evidence_dir.mkdir(exist_ok=True)
        sums = []
        for path, name in evidence_files:
            target = evidence_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in {".log", ".txt", ".jsonl", ".json"}:
                sanitized = redactor.sanitize(path.read_text(encoding="utf-8", errors="replace"))
                target.write_text(sanitized, encoding="utf-8")
                evidence_problems.extend(redactor.problems(sanitized, f"evidence/{name}"))
            else:
                shutil.copyfile(path, target)
            sums.append(f"{sha256_of(target)}  {name}")
        (evidence_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    problems = check_tables(out, redactor) + evidence_problems
    for note in notes:
        print(f"note: {note}")
    print(f"prefill.csv: {len(prefill)} accepted rows; diagnostic-prefill.csv: {len(diagnostic)}; "
          f"throughput.csv: {len(decode)}; fabric.csv: {len(fabric)}; profile.csv: {len(profile)}; "
          f"qualification.csv: {len(qualification)}")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("PASS: tables written and validated")
    return 0


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


LANE_TABLES = ("prefill.csv", "diagnostic-prefill.csv", "throughput.csv")


def check_lane_labels(data_dir: Path, tables: dict[str, list[dict[str, str]]]) -> list[str]:
    """Every lane row names its lane_label/series_order, identically across tables and matching the manifest."""
    problems: list[str] = []
    seen: dict[str, tuple[str, str]] = {}
    for name in LANE_TABLES:
        for row in tables.get(name, []):
            label, order = row.get("lane_label", ""), row.get("series_order", "")
            if not label:
                problems.append(f"{name}: lane {row.get('lane')!r} has an empty lane_label")
            if not order.isdigit() or int(order) <= 0:
                problems.append(f"{name}: lane {row.get('lane')!r} series_order {order!r} is not a positive integer")
            previous = seen.setdefault(row.get("lane", ""), (label, order))
            if previous != (label, order):
                problems.append(f"{name}: lane {row.get('lane')!r} label/order {label!r}/{order!r} differ from {previous!r}")
    by_order: dict[str, str] = {}
    for lane, (_, order) in seen.items():
        if by_order.setdefault(order, lane) != lane:
            problems.append(f"series_order {order} is shared by lanes {by_order[order]!r} and {lane!r}")
    manifest = data_dir / "sources.json"
    if manifest.is_file():
        lanes = json.loads(manifest.read_text(encoding="utf-8")).get("lanes", {})
        for lane, (label, order) in seen.items():
            expected = lanes.get(lane)
            if expected is None:
                problems.append(f"lane {lane!r} has rows but no manifest entry")
            elif (expected.get("lane_label"), text(int(expected.get("series_order", 0)))) != (label, order):
                problems.append(f"lane {lane!r} label/order {label!r}/{order!r} disagree with sources.json")
    return problems


def check_tables(data_dir: Path, redactor: Redactor | None = None) -> list[str]:
    """Offline validation of committed tables: schema, unique keys, positive rates, dispositions, redaction."""
    problems: list[str] = []
    redactor = redactor or Redactor({})
    tables: dict[str, list[dict[str, str]]] = {}
    for name, fields in TABLE_FIELDS.items():
        path = data_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        header, rows = read_csv(path)
        tables[name] = rows
        if tuple(header) != fields:
            problems.append(f"{name}: header differs from build_data.py schema")
        keys = [tuple(row.get(column, "") for column in KEY_COLUMNS[name]) for row in rows]
        if len(keys) != len(set(keys)):
            problems.append(f"{name}: duplicate key rows")
        for row in rows:
            for column in RATE_COLUMNS.get(name, ()):
                value = row.get(column, "")
                try:
                    number = float(value)
                except ValueError:
                    problems.append(f"{name}: {column}={value!r} is not numeric")
                    continue
                if not math.isfinite(number) or number <= 0:
                    problems.append(f"{name}: {column}={value!r} must be finite and > 0 (failed cells are never zero)")
            if "num_errors" in row and row["num_errors"] not in ("", "0"):
                problems.append(f"{name}: row with num_errors={row['num_errors']} must not be published")
        problems.extend(redactor.problems(path.read_text(encoding="utf-8"), name))

    problems.extend(check_lane_labels(data_dir, tables))
    accepted = [row for row in tables.get("prefill.csv", []) if row["publication_status"] == "accepted"]
    if len(accepted) != len(tables.get("prefill.csv", [])):
        problems.append("prefill.csv: only accepted rows belong here")
    if any(row["rankable"] != "true" for row in accepted):
        problems.append("prefill.csv: accepted rows must be rankable")
    for row in tables.get("diagnostic-prefill.csv", []):
        if row["rankable"] != "false" or row["publication_status"] != "diagnostic":
            problems.append("diagnostic-prefill.csv: rows must be diagnostic and non-rankable")
    for row in tables.get("throughput.csv", []):
        if row["publication_status"] == "accepted" and row["rankable"] != "true":
            problems.append("throughput.csv: accepted rows must be rankable")
        if row["publication_status"] not in ROW_STATUSES:
            problems.append(f"throughput.csv: unexpected publication_status {row['publication_status']!r}")

    qualification = tables.get("qualification.csv", [])
    lanes_with_prefill = {row["lane"] for row in accepted}
    lanes_with_decode = {row["lane"] for row in tables.get("throughput.csv", []) if row["publication_status"] == "accepted"}
    for row in qualification:
        rankable = row["rankable"] == "true"
        if rankable != row["status"].startswith("PASS_RANKABLE"):
            problems.append(f"qualification.csv: {row['profile']} rankable={row['rankable']} disagrees with status {row['status']}")
        if row["status"] == "PASS_RANKABLE_PREFILL" and row["lane"] not in lanes_with_prefill:
            problems.append(f"qualification.csv: {row['profile']} claims accepted prefill but prefill.csv has no rows")
        if row["status"] == "PASS_RANKABLE_DECODE" and row["lane"] not in lanes_with_decode:
            problems.append(f"qualification.csv: {row['profile']} claims accepted decode but throughput.csv has no rows")
    for lane in lanes_with_prefill:
        if not any(row["lane"] == lane and row["status"] == "PASS_RANKABLE_PREFILL" for row in qualification):
            problems.append(f"prefill.csv: lane {lane} has accepted rows without a PASS_RANKABLE_PREFILL ledger row")
    for lane in lanes_with_decode:
        if not any(row["lane"] == lane and row["status"] == "PASS_RANKABLE_DECODE" for row in qualification):
            problems.append(f"throughput.csv: lane {lane} has accepted rows without a PASS_RANKABLE_DECODE ledger row")

    checkpoint_path = data_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        problems.append("checkpoint.json: missing")
    else:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key in ("model_id", "revision", "architecture", "safetensors_files", "index_total_size_bytes", "precision"):
            if key not in checkpoint:
                problems.append(f"checkpoint.json: missing {key}")
        for name in ("prefill.csv", "diagnostic-prefill.csv", "throughput.csv", "qualification.csv"):
            for row in tables.get(name, []):
                if row.get("model_id") and row["model_id"] != checkpoint.get("model_id"):
                    problems.append(f"{name}: model_id {row['model_id']!r} differs from checkpoint.json")
                if row.get("model_revision") and row["model_revision"] != checkpoint.get("revision"):
                    problems.append(f"{name}: model_revision differs from checkpoint.json")
        problems.extend(redactor.problems(checkpoint_path.read_text(encoding="utf-8"), "checkpoint.json"))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES), help="manifest (default: data/sources.json)")
    parser.add_argument("--source-root", help="private working directory that the manifest paths are relative to")
    parser.add_argument("--output-dir", default=str(HERE), help="where the CSV/JSON tables are written (default: data/)")
    parser.add_argument("--checkpoint-dir", help="local checkpoint directory; verifies checkpoint.json facts when given")
    parser.add_argument("--host-alias", action="append", default=[], metavar="HOST=nodeN",
                        help="extra hostname -> node alias (rank hosts are learned from nccl-tests logs automatically)")
    parser.add_argument("--copy-evidence", action="store_true", help="copy sanitized source artifacts into data/evidence/")
    parser.add_argument("--check", action="store_true", help="validate the committed tables offline and exit")
    args = parser.parse_args(argv)
    try:
        if args.check:
            problems = check_tables(Path(args.output_dir))
            for problem in problems:
                print(f"FAIL: {problem}")
            if problems:
                return 1
            print("PASS: DeepSeek-V4.1-Flash tables are schema-complete, unique, zero-free, and redacted")
            return 0
        if not args.source_root:
            parser.error("--source-root is required unless --check is given")
        return build(args)
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
