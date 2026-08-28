#!/usr/bin/env python3
"""Import completed GLM-5.3 NVFP4 TP2 raw result bundles.

The importer validates pinned manifests, completion records, exact request-count
decode cells, and the standalone TP2/AR cold-prefill cells before replacing the
matching publication profiles in the section's compact and diagnostic CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_THROUGHPUT = HERE / "throughput.csv"
DEFAULT_PREFILL = HERE / "prefill.csv"
DEFAULT_DIAGNOSTIC_THROUGHPUT = HERE / "diagnostic-throughput.csv"
DEFAULT_DIAGNOSTIC_PREFILL = HERE / "diagnostic-prefill.csv"

MODEL_ID = "LibertAIDAI/GLM-5.3-Flash-NVFP4"
MODEL_REVISION = "aa28e1f54130286c95fee10d0705c74ce8743734"
WEIGHT_REVISION = "11d73216cd636238e82e1d77fe1042ffab36e7fa"
RUNTIME_CONFIG_SHA256 = "5db46f44956e4a8a0cc8ed54b6d77bf99dd7c1ec90c58975d1952560768513d5"
BENCH_VERSION = "0.4.29"
BENCH_COMMIT = "0b4185b5b435e948b199c9077a00b084864aa963"
TOPOLOGY = "cross_node_tp2"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_CONTEXTS = (8192, 65536, 131072)

AR_RUNTIME_IMAGE = "glm53-nvfp4-sglang:pr36507-c3b7482"
AR_RUN_PROFILE = "tp2-mtp0"
AR_PUBLICATION_PROFILE = "NVFP4 TP2/AR"
AR_PREFILL_SHA256 = "d624a68252542bd06e7422d204decac5fec25d33b7856e9630af82c152b17b90"

DFLASH_RUN_PROFILE = "tp2-dflash2"
DFLASH_PUBLICATION_PROFILE = "NVFP4 TP2/DFlash2"
DFLASH_RUNTIME_IMAGE = "glm53-dflash2-sglang:5277926"
DFLASH_RUNTIME_COMMIT = "52779266e668039bed838fe25ef84ffb014d22f2"
DFLASH_DRAFT_ID = "incoai/GLM-5.3-Flash-DFlash2"
DFLASH_DRAFT_REVISION = "7d74cdd881ed7e32c31175984a67823127b66cfe"
DFLASH_PROPOSED_TOKENS = 7

# Backwards-compatible names used by the importer tests and downstream callers.
RUNTIME_IMAGE = AR_RUNTIME_IMAGE
RUN_PROFILE = AR_RUN_PROFILE
PUBLICATION_PROFILE = AR_PUBLICATION_PROFILE

THROUGHPUT_FIELDS = (
    "run_id",
    "profile",
    "publication_status",
    "model_id",
    "model_revision",
    "topology",
    "mtp_tokens",
    "input_tokens",
    "target_output_tokens",
    "duration_seconds",
    "concurrency",
    "aggregate_output_tokens_per_second",
    "effective_concurrency",
    "max_running_requests",
    "saturation_observed",
    "mtp_accept_length",
    "engine_steps_per_second",
    "median_ttft_seconds",
    "median_itl_ms",
    "num_errors",
    "source_artifact_sha256",
    "result_file",
)

PREFILL_FIELDS = (
    "run_id",
    "profile",
    "publication_status",
    "model_id",
    "model_revision",
    "topology",
    "mtp_tokens",
    "nominal_context_tokens",
    "actual_prompt_tokens",
    "samples",
    "prompt_tokens_per_second",
    "server_prompt_tokens_per_second",
    "median_ttft_seconds",
    "server_cached_tokens",
    "num_errors",
    "source_artifact_sha256",
    "result_file",
)

DIAGNOSTIC_THROUGHPUT_FIELDS = (
    "model_id",
    "model_revision",
    "runtime",
    "topology",
    "mtp_tokens",
    "run_id",
    "measurement_status",
    "rankable",
    "concurrency",
    "input_tokens",
    "target_output_tokens",
    "measurement_seconds",
    "aggregate_output_tokens_per_second",
    "completed_requests_in_window",
    "effective_concurrency",
    "max_running_requests",
    "underfilled",
    "capacity_limited",
    "num_errors",
    "ttft_p50_seconds",
    "itl_p50_seconds",
    "mtp_accept_length",
    "engine_steps_per_second",
    "source_artifact_sha256",
)

DIAGNOSTIC_PREFILL_FIELDS = (
    "model_id",
    "model_revision",
    "runtime",
    "topology",
    "mtp_tokens",
    "run_id",
    "measurement_status",
    "rankable",
    "nominal_context_tokens",
    "actual_prompt_tokens",
    "samples",
    "client_prompt_tokens_per_second",
    "client_ttft_seconds",
    "server_prompt_tokens_per_second",
    "server_cached_tokens",
    "source_artifact_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def regular_file(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    return path


def json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(regular_file(path, label).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool), f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    require(math.isfinite(result), f"{label} is not finite")
    return result


def exact_int(value: Any, label: str) -> int:
    number = finite_number(value, label)
    require(number.is_integer(), f"{label} is not integral")
    return int(number)


def first_line(path: Path, label: str) -> str:
    lines = regular_file(path, label).read_text(encoding="utf-8").splitlines()
    require(lines, f"{label} is empty")
    return lines[0]


def key_values(path: Path, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in regular_file(path, label).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        require(key not in result, f"duplicate {label} field: {key}")
        result[key] = value
    return result


def validate_result_root(
    root_arg: Path,
) -> tuple[Path, dict[str, Any], str, int, bool]:
    require(root_arg.is_dir() and not root_arg.is_symlink(), f"unsafe result root: {root_arg}")
    root = root_arg.resolve(strict=True)
    manifest = json_object(root / "run-manifest.json", "run manifest")
    require(manifest.get("schema_version") == 1, "run-manifest schema differs")
    require(manifest.get("run_id") == root.name, "run-manifest ID differs")
    profile = manifest.get("profile")
    require(profile in (AR_RUN_PROFILE, DFLASH_RUN_PROFILE),
            "run-manifest profile differs")

    topology = manifest.get("topology")
    require(isinstance(topology, dict), "run-manifest topology is absent")
    require(topology == {"nodes": 2, "tp": 2, "ep": 1},
            "run-manifest is not the two-Station TP2/EP1 topology")

    benchmark = manifest.get("benchmark")
    require(isinstance(benchmark, dict), "run-manifest benchmark contract is absent")
    require(benchmark.get("commit") == BENCH_COMMIT, "benchmark client commit differs")
    require(benchmark.get("decode_concurrency") == list(CONCURRENCIES),
            "decode concurrency contract differs")
    require(benchmark.get("input_tokens") == 8192, "decode input contract differs")
    require(benchmark.get("output_tokens") == 1024, "decode output contract differs")
    require(benchmark.get("measured_requests") == "5*C", "decode request contract differs")
    require(benchmark.get("warmups") == "C", "decode warmup contract differs")

    raw = root / "benchmark/raw"
    require(raw.is_dir() and not raw.is_symlink(), "raw benchmark directory is absent")
    require(first_line(raw / "STATUS.txt", "benchmark status") == "COMPLETE",
            "benchmark client has not completed")

    if profile == AR_RUN_PROFILE:
        model = manifest.get("model")
        require(isinstance(model, dict), "run-manifest model is absent")
        require(model.get("repository") == MODEL_ID, "model repository differs")
        require(model.get("revision") == MODEL_REVISION, "model revision differs")
        require(model.get("weight_revision") == WEIGHT_REVISION, "weight revision differs")
        require(model.get("runtime_config_sha256") == RUNTIME_CONFIG_SHA256,
                "runtime configuration hash differs")
        require(model.get("quantization") == "modelopt_fp4", "quantization differs")
        require(manifest.get("runtime_image") == AR_RUNTIME_IMAGE, "runtime image differs")
        mtp = manifest.get("mtp")
        require(isinstance(mtp, dict), "run-manifest MTP configuration is absent")
        require(mtp.get("enabled") is False and mtp.get("steps") == 0,
                "run-manifest is not autoregressive MTP0")
        require(benchmark.get("cold_prefill") == ["8K", "64K", "128K"],
                "cold-prefill contract differs")
        require(first_line(root / "STATUS.retry.txt", "launcher retry status")
                == "COMPLETE_MEASURED_RAW", "retained launcher retry is not complete")
        require(key_values(
            root / "runtime/cleanup-verdict.txt", "cleanup verdict"
        ).get("outcome") == "PASS_FORCED_EXACT_NAMES", "exact-name cleanup did not pass")
        require(key_values(
            root / "postflight/verdict.retry.txt", "postflight retry verdict"
        ).get("outcome") == "PASS", "retained postflight retry did not pass")
        return root, manifest, AR_PUBLICATION_PROFILE, 0, True

    target = manifest.get("target")
    require(isinstance(target, dict), "run-manifest target is absent")
    require(target.get("repository") == MODEL_ID, "target repository differs")
    require(target.get("revision") == MODEL_REVISION, "target revision differs")
    require(target.get("weight_revision") == WEIGHT_REVISION, "weight revision differs")
    require(target.get("quantization") == "modelopt_fp4", "target quantization differs")
    require(target.get("kv_cache_dtype") == "fp8_e4m3", "target KV-cache dtype differs")

    draft = manifest.get("draft")
    require(isinstance(draft, dict), "run-manifest DFlash2 draft is absent")
    require(draft.get("repository") == DFLASH_DRAFT_ID, "DFlash2 draft repository differs")
    require(draft.get("revision") == DFLASH_DRAFT_REVISION, "DFlash2 draft revision differs")
    require(draft.get("proposed_tokens") == DFLASH_PROPOSED_TOKENS,
            "DFlash2 proposal width differs")
    require(draft.get("quantization") == "unquant", "DFlash2 draft quantization differs")
    require(draft.get("dtype") == "bfloat16", "DFlash2 draft dtype differs")

    runtime = manifest.get("runtime")
    require(isinstance(runtime, dict), "run-manifest runtime is absent")
    require(runtime.get("image") == DFLASH_RUNTIME_IMAGE, "DFlash2 runtime image differs")
    require(runtime.get("source_commit") == DFLASH_RUNTIME_COMMIT,
            "DFlash2 runtime commit differs")

    cold_prefill = benchmark.get("cold_prefill")
    require(isinstance(cold_prefill, dict), "DFlash2 prefill reference is absent")
    require(cold_prefill.get("rerun") is False, "DFlash2 unexpectedly reran prefill")
    require(cold_prefill.get("sha256") == AR_PREFILL_SHA256,
            "DFlash2 prefill reference hash differs")
    prefill_reference = json_object(raw / "prefill/reused-base.json", "prefill reference")
    require(prefill_reference.get("schema") == "benchmark-prefill-reference/v1",
            "DFlash2 prefill reference schema differs")
    require(prefill_reference.get("rerun") is False,
            "DFlash2 prefill reference claims a rerun")
    require(prefill_reference.get("source") == cold_prefill.get("source"),
            "DFlash2 prefill reference source differs")
    require(prefill_reference.get("sha256") == AR_PREFILL_SHA256,
            "DFlash2 prefill reference artifact differs")
    require(not (raw / "prefill/cold.json").exists(),
            "DFlash2 bundle unexpectedly contains a fresh prefill result")

    require(first_line(root / "STATUS.txt", "launcher status") == "COMPLETE_MEASURED_RAW",
            "DFlash2 launcher status is not complete")
    require(key_values(
        root / "runtime/cleanup-verdict.txt", "cleanup verdict"
    ).get("outcome") == "PASS_FORCED_EXACT_NAMES", "exact-name cleanup did not pass")
    require(key_values(
        root / "postflight/verdict.txt", "postflight verdict"
    ).get("outcome") == "PASS", "DFlash2 postflight did not pass")
    return root, manifest, DFLASH_PUBLICATION_PROFILE, 0, False


def validate_metadata(metadata: Any, *, max_tokens: int, label: str) -> dict[str, Any]:
    require(isinstance(metadata, dict), f"{label} metadata is absent")
    require(metadata.get("version") == BENCH_VERSION, f"{label} client version differs")
    require(metadata.get("engine") == "sglang", f"{label} engine differs")
    require(metadata.get("model") == MODEL_ID, f"{label} served model differs")
    require(metadata.get("max_tokens") == max_tokens, f"{label} output target differs")
    require(finite_number(metadata.get("temperature"), f"{label} temperature") == 0,
            f"{label} temperature differs")
    require(metadata.get("ignore_eos") is True, f"{label} did not ignore EOS")
    return metadata


def decode_rows(
    root: Path,
    concurrency: int,
    path: Path,
    publication_profile: str,
    mtp_tokens: int,
) -> tuple[dict[str, str], dict[str, str]]:
    data = json_object(path, f"C{concurrency} decode result")
    metadata = validate_metadata(data.get("metadata"), max_tokens=1024,
                                 label=f"C{concurrency} decode")
    target = 5 * concurrency
    require(metadata.get("concurrency_levels") == [concurrency],
            "decode offered concurrency differs")
    require(metadata.get("context_lengths") == [8192], "decode input target differs")
    require(metadata.get("request_count") == target, "decode request target differs")
    require(metadata.get("warmup_request_count") == concurrency,
            "decode warmup target differs")

    results = data.get("results")
    require(isinstance(results, list) and len(results) == 1 and isinstance(results[0], dict),
            f"C{concurrency} does not contain exactly one result")
    cell = results[0]
    require(exact_int(cell.get("concurrency"), "decode concurrency") == concurrency,
            "decode result concurrency differs")
    require(exact_int(cell.get("context_tokens"), "decode input tokens") == 8192,
            "decode input tokens differ")
    for field in ("request_count_target", "request_count", "completed_request_count"):
        require(exact_int(cell.get(field), f"decode {field}") == target,
                f"decode {field} differs")
    require(exact_int(cell.get("num_completed"), "decode num_completed") == target,
            "decode num_completed differs")
    require(exact_int(cell.get("num_errors"), "decode errors") == 0,
            "decode cell has request errors")
    require(cell.get("aggregate_source") == "openai_completed_usage",
            "decode aggregate source differs")
    output_tokens = exact_int(cell.get("client_output_tokens"), "decode output tokens")
    require(output_tokens == target * 1024, "decode output-token total differs")
    seconds = finite_number(cell.get("measurement_seconds"), "decode measurement seconds")
    throughput = finite_number(cell.get("aggregate_tps"), "decode throughput")
    require(seconds > 0 and throughput > 0, "decode timing is not positive")
    require(math.isclose(throughput, output_tokens / seconds, rel_tol=1e-5, abs_tol=1e-5),
            "decode throughput does not recompute")
    ttft = finite_number(cell.get("ttft_p50"), "decode TTFT")
    itl = finite_number(cell.get("inter_token_latency_p50"), "decode ITL")
    effective = finite_number(cell.get("effective_concurrency"), "effective concurrency")
    max_running = exact_int(cell.get("max_running_reqs"), "maximum running requests")
    underfilled = cell.get("underfilled")
    capacity_limited = cell.get("capacity_limited")
    require(ttft > 0 and itl > 0 and effective > 0 and max_running > 0,
            "decode latency/concurrency is invalid")
    require(isinstance(underfilled, bool) and isinstance(capacity_limited, bool),
            "decode capacity flags are absent")
    source_hash = sha256(path)

    diagnostic = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": "sglang",
        "topology": TOPOLOGY,
        "mtp_tokens": str(mtp_tokens),
        "run_id": root.name,
        "measurement_status": "MEASURED",
        "rankable": "true",
        "concurrency": str(concurrency),
        "input_tokens": "8192",
        "target_output_tokens": "1024",
        "measurement_seconds": str(cell["measurement_seconds"]),
        "aggregate_output_tokens_per_second": str(cell["aggregate_tps"]),
        "completed_requests_in_window": str(target),
        "effective_concurrency": str(cell["effective_concurrency"]),
        "max_running_requests": str(max_running),
        "underfilled": str(underfilled).lower(),
        "capacity_limited": str(capacity_limited).lower(),
        "num_errors": "0",
        "ttft_p50_seconds": str(cell["ttft_p50"]),
        "itl_p50_seconds": str(cell["inter_token_latency_p50"]),
        "mtp_accept_length": str(cell.get("server_accept_len_effective", 0.0)),
        "engine_steps_per_second": str(cell.get("server_steps_per_s", 0.0)),
        "source_artifact_sha256": source_hash,
    }
    compact = {
        "run_id": root.name,
        "profile": publication_profile,
        "publication_status": "measured",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": TOPOLOGY,
        "mtp_tokens": str(mtp_tokens),
        "input_tokens": "8192",
        "target_output_tokens": "1024",
        "duration_seconds": str(cell["measurement_seconds"]),
        "concurrency": str(concurrency),
        "aggregate_output_tokens_per_second": str(cell["aggregate_tps"]),
        "effective_concurrency": str(cell["effective_concurrency"]),
        "max_running_requests": str(max_running),
        "saturation_observed": str(underfilled).lower(),
        "mtp_accept_length": str(cell.get("server_accept_len_effective", 0.0)),
        "engine_steps_per_second": str(cell.get("server_steps_per_s", 0.0)),
        "median_ttft_seconds": str(cell["ttft_p50"]),
        "median_itl_ms": str(itl * 1000),
        "num_errors": "0",
        "source_artifact_sha256": source_hash,
        "result_file": "data/diagnostic-throughput.csv",
    }
    return compact, diagnostic


def prefill_rows(root: Path, path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = json_object(path, "cold-prefill result")
    metadata = validate_metadata(data.get("metadata"), max_tokens=1, label="cold prefill")
    require(metadata.get("standalone_prefill") is True, "prefill is not standalone")
    require(metadata.get("prefill_only") is True, "prefill result includes decode")
    cells = data.get("prefill")
    require(isinstance(cells, dict), "prefill cells are absent")
    require(set(cells) == {str(value) for value in PREFILL_CONTEXTS},
            "prefill contexts differ")
    source_hash = sha256(path)
    compact_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []
    for context in PREFILL_CONTEXTS:
        cell = cells[str(context)]
        require(isinstance(cell, dict), f"prefill {context} cell is not an object")
        prompt_tokens = exact_int(cell.get("prompt_tokens"), f"prefill {context} prompt tokens")
        require(context <= prompt_tokens <= context + 8,
                f"prefill {context} prompt length is outside the targeting allowance")
        samples = exact_int(cell.get("samples"), f"prefill {context} samples")
        throughput = finite_number(cell.get("tok_per_sec"), f"prefill {context} throughput")
        ttft = finite_number(cell.get("ttft_seconds"), f"prefill {context} TTFT")
        require(samples > 0 and throughput > 0 and ttft > 0,
                f"prefill {context} measurement is invalid")
        require(abs((prompt_tokens / ttft) / throughput - 1) < 0.01,
                f"prefill {context} throughput does not match prompt tokens / TTFT")
        require(cell.get("method") == "client", f"prefill {context} method differs")
        validation = cell.get("server_validation")
        require(isinstance(validation, dict), f"prefill {context} validation is absent")
        require(validation.get("method") == "" and finite_number(
            validation.get("tok_per_sec"), f"prefill {context} server throughput") == 0,
            f"prefill {context} unexpectedly reports a server rate")

        diagnostic_rows.append({
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime": "sglang",
            "topology": TOPOLOGY,
            "mtp_tokens": "0",
            "run_id": root.name,
            "measurement_status": "MEASURED",
            "rankable": "true",
            "nominal_context_tokens": str(context),
            "actual_prompt_tokens": str(prompt_tokens),
            "samples": str(samples),
            "client_prompt_tokens_per_second": str(cell["tok_per_sec"]),
            "client_ttft_seconds": str(cell["ttft_seconds"]),
            "server_prompt_tokens_per_second": "",
            "server_cached_tokens": "",
            "source_artifact_sha256": source_hash,
        })
        compact_rows.append({
            "run_id": root.name,
            "profile": PUBLICATION_PROFILE,
            "publication_status": "measured",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "topology": TOPOLOGY,
            "mtp_tokens": "0",
            "nominal_context_tokens": str(context),
            "actual_prompt_tokens": str(prompt_tokens),
            "samples": str(samples),
            "prompt_tokens_per_second": str(cell["tok_per_sec"]),
            "server_prompt_tokens_per_second": "",
            "median_ttft_seconds": str(cell["ttft_seconds"]),
            "server_cached_tokens": "",
            "num_errors": "0",
            "source_artifact_sha256": source_hash,
            "result_file": "data/diagnostic-prefill.csv",
        })
    return compact_rows, diagnostic_rows


def import_result(root_arg: Path) -> dict[str, list[dict[str, str]]]:
    root, _manifest, publication_profile, mtp_tokens, has_prefill = validate_result_root(
        root_arg
    )
    raw = root / "benchmark/raw"
    throughput: list[dict[str, str]] = []
    diagnostic_throughput: list[dict[str, str]] = []
    for concurrency in CONCURRENCIES:
        compact, diagnostic = decode_rows(
            root, concurrency, regular_file(raw / f"fixed/c{concurrency}.json",
                                            f"C{concurrency} decode result"),
            publication_profile,
            mtp_tokens,
        )
        throughput.append(compact)
        diagnostic_throughput.append(diagnostic)
    if has_prefill:
        prefill, diagnostic_prefill = prefill_rows(
            root, regular_file(raw / "prefill/cold.json", "cold-prefill result")
        )
    else:
        prefill, diagnostic_prefill = [], []
    return {
        "throughput": throughput,
        "prefill": prefill,
        "diagnostic_throughput": diagnostic_throughput,
        "diagnostic_prefill": diagnostic_prefill,
    }


def read_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == fields, f"CSV header differs: {path}")
        return list(reader)


def replace_profile_rows(
    existing: list[dict[str, str]], imported: list[dict[str, str]]
) -> list[dict[str, str]]:
    if not imported:
        return existing
    profiles = {row["profile"] for row in imported}
    require(len(profiles) == len({(row["profile"], row["run_id"]) for row in imported}),
            "one imported profile maps to multiple run IDs")
    rows = [
        row for row in existing
        if row.get("profile") not in profiles
    ]
    rows.extend(imported)
    return rows


def replace_run_rows(
    existing: list[dict[str, str]],
    imported: list[dict[str, str]],
    replaced_run_ids: set[str],
) -> list[dict[str, str]]:
    if not imported:
        return existing
    imported_run_ids = {row["run_id"] for row in imported}
    rows = [
        row for row in existing
        if row.get("run_id") not in replaced_run_ids | imported_run_ids
    ]
    rows.extend(imported)
    return rows


def render_csv(rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def atomic_write(path: Path, text: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"refusing to replace unsafe CSV: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_roots", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    imported_sets = [import_result(root) for root in args.result_roots]
    imported = {
        key: [row for result in imported_sets for row in result[key]]
        for key in (
            "throughput", "prefill", "diagnostic_throughput", "diagnostic_prefill"
        )
    }
    imported_profiles = {row["profile"] for row in imported["throughput"]}
    require(len(imported_profiles) == len(args.result_roots),
            "result roots contain duplicate publication profiles")

    existing_throughput = read_rows(DEFAULT_THROUGHPUT, THROUGHPUT_FIELDS)
    replaced_decode_run_ids = {
        row["run_id"] for row in existing_throughput
        if row.get("profile") in imported_profiles
    }
    merged_throughput = replace_profile_rows(existing_throughput, imported["throughput"])
    merged_diagnostic_throughput = replace_run_rows(
        read_rows(DEFAULT_DIAGNOSTIC_THROUGHPUT, DIAGNOSTIC_THROUGHPUT_FIELDS),
        imported["diagnostic_throughput"],
        replaced_decode_run_ids,
    )

    existing_prefill = read_rows(DEFAULT_PREFILL, PREFILL_FIELDS)
    imported_prefill_profiles = {row["profile"] for row in imported["prefill"]}
    replaced_prefill_run_ids = {
        row["run_id"] for row in existing_prefill
        if row.get("profile") in imported_prefill_profiles
    }
    merged_prefill = replace_profile_rows(existing_prefill, imported["prefill"])
    merged_diagnostic_prefill = replace_run_rows(
        read_rows(DEFAULT_DIAGNOSTIC_PREFILL, DIAGNOSTIC_PREFILL_FIELDS),
        imported["diagnostic_prefill"],
        replaced_prefill_run_ids,
    )

    rendered = (
        (DEFAULT_THROUGHPUT, render_csv(merged_throughput, THROUGHPUT_FIELDS)),
        (DEFAULT_PREFILL, render_csv(merged_prefill, PREFILL_FIELDS)),
        (DEFAULT_DIAGNOSTIC_THROUGHPUT,
         render_csv(merged_diagnostic_throughput, DIAGNOSTIC_THROUGHPUT_FIELDS)),
        (DEFAULT_DIAGNOSTIC_PREFILL,
         render_csv(merged_diagnostic_prefill, DIAGNOSTIC_PREFILL_FIELDS)),
    )
    if args.check:
        changed = [
            str(path) for path, text in rendered
            if path.read_text(encoding="utf-8") != text
        ]
        require(not changed, f"imported CSVs differ: {', '.join(changed)}")
    else:
        for path, text in rendered:
            atomic_write(path, text)
    print(json.dumps({
        "mode": "check" if args.check else "write",
        "run_ids": [root.name for root in args.result_roots],
        "decode_rows": len(imported["throughput"]),
        "prefill_rows": len(imported["prefill"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
