#!/usr/bin/env python3
"""Import the completed GLM-5.3 NVFP4 TP2/AR raw result bundle.

The importer validates the pinned manifest, completion/retry seals, exact
request-count decode cells, and standalone cold-prefill cells before replacing
the matching model rows in the section's compact and diagnostic CSVs.
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
RUNTIME_IMAGE = "glm53-nvfp4-sglang:pr36507-c3b7482"
BENCH_VERSION = "0.4.29"
BENCH_COMMIT = "0b4185b5b435e948b199c9077a00b084864aa963"
RUN_PROFILE = "tp2-mtp0"
PUBLICATION_PROFILE = "NVFP4 TP2/AR"
TOPOLOGY = "cross_node_tp2"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_CONTEXTS = (8192, 65536, 131072)

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


def validate_result_root(root_arg: Path) -> tuple[Path, dict[str, Any]]:
    require(root_arg.is_dir() and not root_arg.is_symlink(), f"unsafe result root: {root_arg}")
    root = root_arg.resolve(strict=True)
    manifest = json_object(root / "run-manifest.json", "run manifest")
    require(manifest.get("schema_version") == 1, "run-manifest schema differs")
    require(manifest.get("run_id") == root.name, "run-manifest ID differs")
    require(manifest.get("profile") == RUN_PROFILE, "run-manifest profile differs")

    model = manifest.get("model")
    require(isinstance(model, dict), "run-manifest model is absent")
    require(model.get("repository") == MODEL_ID, "model repository differs")
    require(model.get("revision") == MODEL_REVISION, "model revision differs")
    require(model.get("weight_revision") == WEIGHT_REVISION, "weight revision differs")
    require(model.get("runtime_config_sha256") == RUNTIME_CONFIG_SHA256,
            "runtime configuration hash differs")
    require(model.get("quantization") == "modelopt_fp4", "quantization differs")
    require(manifest.get("runtime_image") == RUNTIME_IMAGE, "runtime image differs")

    topology = manifest.get("topology")
    require(isinstance(topology, dict), "run-manifest topology is absent")
    require(topology == {"nodes": 2, "tp": 2, "ep": 1},
            "run-manifest is not the two-Station TP2/EP1 topology")

    mtp = manifest.get("mtp")
    require(isinstance(mtp, dict), "run-manifest MTP configuration is absent")
    require(mtp.get("enabled") is False and mtp.get("steps") == 0,
            "run-manifest is not autoregressive MTP0")

    benchmark = manifest.get("benchmark")
    require(isinstance(benchmark, dict), "run-manifest benchmark contract is absent")
    require(benchmark.get("commit") == BENCH_COMMIT, "benchmark client commit differs")
    require(benchmark.get("decode_concurrency") == list(CONCURRENCIES),
            "decode concurrency contract differs")
    require(benchmark.get("cold_prefill") == ["8K", "64K", "128K"],
            "cold-prefill contract differs")
    require(benchmark.get("input_tokens") == 8192, "decode input contract differs")
    require(benchmark.get("output_tokens") == 1024, "decode output contract differs")

    raw = root / "benchmark/raw"
    require(raw.is_dir() and not raw.is_symlink(), "raw benchmark directory is absent")
    require(first_line(raw / "STATUS.txt", "benchmark status") == "COMPLETE",
            "benchmark client has not completed")
    require(first_line(root / "STATUS.retry.txt", "launcher retry status")
            == "COMPLETE_MEASURED_RAW", "retained launcher retry is not complete")
    require(key_values(root / "runtime/cleanup-verdict.txt", "cleanup verdict").get("outcome")
            == "PASS_FORCED_EXACT_NAMES", "exact-name cleanup did not pass")
    require(key_values(root / "postflight/verdict.retry.txt", "postflight retry verdict").get("outcome")
            == "PASS", "retained postflight retry did not pass")
    return root, manifest


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


def decode_rows(root: Path, concurrency: int, path: Path) -> tuple[dict[str, str], dict[str, str]]:
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
    require(exact_int(cell.get("num_errors"), "decode errors") == 0,
            "decode cell has request errors")
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
        "mtp_tokens": "0",
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
        "profile": PUBLICATION_PROFILE,
        "publication_status": "measured",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": TOPOLOGY,
        "mtp_tokens": "0",
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
    root, _manifest = validate_result_root(root_arg)
    raw = root / "benchmark/raw"
    throughput: list[dict[str, str]] = []
    diagnostic_throughput: list[dict[str, str]] = []
    for concurrency in CONCURRENCIES:
        compact, diagnostic = decode_rows(
            root, concurrency, regular_file(raw / f"fixed/c{concurrency}.json",
                                            f"C{concurrency} decode result")
        )
        throughput.append(compact)
        diagnostic_throughput.append(diagnostic)
    prefill, diagnostic_prefill = prefill_rows(
        root, regular_file(raw / "prefill/cold.json", "cold-prefill result")
    )
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


def replace_model_rows(existing: list[dict[str, str]], imported: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        row for row in existing
        if not (row.get("model_id") == MODEL_ID and row.get("model_revision") == MODEL_REVISION)
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
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    imported = import_result(args.result_root)
    outputs = (
        (DEFAULT_THROUGHPUT, THROUGHPUT_FIELDS, imported["throughput"]),
        (DEFAULT_PREFILL, PREFILL_FIELDS, imported["prefill"]),
        (DEFAULT_DIAGNOSTIC_THROUGHPUT, DIAGNOSTIC_THROUGHPUT_FIELDS,
         imported["diagnostic_throughput"]),
        (DEFAULT_DIAGNOSTIC_PREFILL, DIAGNOSTIC_PREFILL_FIELDS,
         imported["diagnostic_prefill"]),
    )
    rendered: list[tuple[Path, str]] = []
    for path, fields, new_rows in outputs:
        merged = replace_model_rows(read_rows(path, fields), new_rows)
        rendered.append((path, render_csv(merged, fields)))
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
        "run_id": args.result_root.name,
        "decode_rows": len(imported["throughput"]),
        "prefill_rows": len(imported["prefill"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
