#!/usr/bin/env python3
"""Import raw Qwen NVFP4-4p89 DGX benchmark cells into ``dgx-overlays.csv``.

Each input is a result root produced by ``qwen38-4p89-bench/run_pair.sh`` or
``run_single.sh``. Only measured JSON files are imported. Missing cells stay
missing, which lets a running profile be published without inventing the rest
of its curve.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "dgx-overlays.csv"

MODEL_ID = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89"
MODEL_REVISION = "ee0cea634a371acd1caeaed8e95b90e4344c16b4"
RUNTIME_IMAGE = "qwen38-4p89-sglang:runtime-v1"
RUNTIME = f"SGLang ({RUNTIME_IMAGE})"
BENCH_VERSION = "0.4.29"
BENCH_COMMIT = "0b4185b5b435e948b199c9077a00b084864aa963"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_CONTEXTS = (8192, 32768, 65536, 131072)

FIELDS = (
    "source_id",
    "source_kind",
    "publication_status",
    "platform_label",
    "model_id",
    "model_revision",
    "runtime",
    "profile",
    "topology",
    "mtp_tokens",
    "metric",
    "concurrency",
    "nominal_context_tokens",
    "input_tokens",
    "target_output_tokens",
    "target_requests",
    "completed_requests",
    "num_errors",
    "measurement_seconds",
    "throughput",
    "ttft_p50_seconds",
    "itl_p50_seconds",
    "effective_concurrency",
    "queue_fraction",
    "capacity_limited",
    "source_path",
    "source_sha256",
    "notes",
)


@dataclass(frozen=True)
class ProfileSpec:
    publication_profile: str
    topology: str
    mtp_tokens: int
    nodes: int
    tp_size: int
    ep_size: int
    platform_label: str
    notes: str


PROFILE_SPECS = {
    "tp1-mtp0": ProfileSpec(
        "NVFP4 TP1/MTP0",
        "single_node_tp1_ep_disabled",
        0,
        1,
        1,
        1,
        "DGX Station",
        "one_engine_on_one_station",
    ),
    "tp2-mtp0": ProfileSpec(
        "NVFP4 TP2/MTP0",
        "cross_node_tp2_ep_disabled",
        0,
        2,
        2,
        1,
        "DGX Station pair",
        "one_distributed_engine_across_two_stations",
    ),
    "tp2-mtp3": ProfileSpec(
        "NVFP4 TP2/MTP3",
        "cross_node_tp2_ep_disabled",
        3,
        2,
        2,
        1,
        "DGX Station pair",
        "one_distributed_engine_across_two_stations",
    ),
    "tep2-mtp0": ProfileSpec(
        "NVFP4 TEP2/MTP0",
        "cross_node_tep2_ep2",
        0,
        2,
        2,
        2,
        "DGX Station pair",
        "one_distributed_engine_across_two_stations",
    ),
    "tep2-mtp3": ProfileSpec(
        "NVFP4 TEP2/MTP3",
        "cross_node_tep2_ep2",
        3,
        2,
        2,
        2,
        "DGX Station pair",
        "one_distributed_engine_across_two_stations",
    ),
}
PROFILE_ORDER = {name: index for index, name in enumerate(PROFILE_SPECS)}
PUBLICATION_TO_PROFILE = {
    spec.publication_profile: name for name, spec in PROFILE_SPECS.items()
}


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


def validate_manifest(root: Path) -> tuple[str, ProfileSpec]:
    manifest = json_object(root / "run-manifest.json", "run manifest")
    require(manifest.get("schema_version") == 1, "run-manifest schema differs")
    require(manifest.get("run_id") == root.name, "run-manifest ID differs from directory")
    profile = manifest.get("profile")
    require(isinstance(profile, str) and profile in PROFILE_SPECS,
            f"unsupported benchmark profile: {profile!r}")
    spec = PROFILE_SPECS[profile]

    model = manifest.get("model")
    require(isinstance(model, dict), "run-manifest model is absent")
    require(model.get("repository") == MODEL_ID, "model repository differs")
    require(model.get("revision") == MODEL_REVISION, "model revision differs")
    require(model.get("quantization") == "modelopt_mixed", "quantization differs")
    require(manifest.get("runtime_image") == RUNTIME_IMAGE, "runtime image differs")

    topology = manifest.get("topology")
    require(isinstance(topology, dict), "run-manifest topology is absent")
    require(
        topology.get("nodes") == spec.nodes and topology.get("tp") == spec.tp_size,
        "run-manifest node/tensor-parallel topology differs",
    )
    require(topology.get("ep") == spec.ep_size, "expert-parallel size differs")

    mtp = manifest.get("mtp")
    require(isinstance(mtp, dict), "run-manifest MTP configuration is absent")
    require(mtp.get("enabled") is (spec.mtp_tokens > 0), "MTP enablement differs")
    require(mtp.get("steps") == spec.mtp_tokens, "MTP step count differs")

    benchmark = manifest.get("benchmark")
    require(isinstance(benchmark, dict), "run-manifest benchmark contract is absent")
    require(benchmark.get("commit") == BENCH_COMMIT, "benchmark client commit differs")
    require(benchmark.get("decode_concurrency") == list(CONCURRENCIES),
            "decode concurrency contract differs")
    require(benchmark.get("input_tokens") == 8192, "decode input contract differs")
    require(benchmark.get("output_tokens") == 1024, "decode output contract differs")
    return profile, spec


def validate_metadata(
    metadata: Any,
    *,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    require(isinstance(metadata, dict), f"{label} metadata is absent")
    require(metadata.get("version") == BENCH_VERSION, f"{label} client version differs")
    require(metadata.get("engine") == "sglang", f"{label} engine differs")
    require(metadata.get("model") == MODEL_ID, f"{label} served model differs")
    require(metadata.get("max_tokens") == max_tokens, f"{label} output target differs")
    require(finite_number(metadata.get("temperature"), f"{label} temperature") == 0,
            f"{label} temperature differs")
    require(metadata.get("ignore_eos") is True, f"{label} did not ignore EOS")
    return metadata


def common_row(root: Path, spec: ProfileSpec, path: Path) -> dict[str, str]:
    return {
        "source_id": root.name,
        "source_kind": "raw_benchmark_measurement",
        "publication_status": "MEASURED_CURRENT",
        "platform_label": spec.platform_label,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": RUNTIME,
        "profile": spec.publication_profile,
        "topology": spec.topology,
        "mtp_tokens": str(spec.mtp_tokens),
        "source_path": str(path),
        "source_sha256": sha256(path),
        "notes": spec.notes,
    }


def decode_row(
    root: Path,
    spec: ProfileSpec,
    concurrency: int,
    path: Path,
) -> dict[str, str]:
    data = json_object(path, f"C{concurrency} decode result")
    metadata = validate_metadata(
        data.get("metadata"), max_tokens=1024, label=f"C{concurrency} decode"
    )
    target = 5 * concurrency
    require(metadata.get("concurrency_levels") == [concurrency],
            "decode offered concurrency differs")
    require(metadata.get("context_lengths") == [8192], "decode input target differs")
    require(metadata.get("request_count") == target, "decode request target differs")
    require(metadata.get("warmup_request_count") == concurrency,
            "decode warmup count differs")

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
    queue = finite_number(cell.get("queue_fraction"), "queue fraction")
    capacity_limited = cell.get("capacity_limited")
    require(ttft > 0 and itl > 0 and effective > 0, "decode latency/concurrency is invalid")
    require(0 <= queue <= 1, "decode queue fraction is invalid")
    require(isinstance(capacity_limited, bool), "decode capacity flag is absent")

    row = common_row(root, spec, path)
    row.update({
        "metric": "decode",
        "concurrency": str(concurrency),
        "nominal_context_tokens": "",
        "input_tokens": "8192",
        "target_output_tokens": "1024",
        "target_requests": str(target),
        "completed_requests": str(target),
        "num_errors": "0",
        "measurement_seconds": str(cell["measurement_seconds"]),
        "throughput": str(cell["aggregate_tps"]),
        "ttft_p50_seconds": str(cell["ttft_p50"]),
        "itl_p50_seconds": str(cell["inter_token_latency_p50"]),
        "effective_concurrency": str(cell["effective_concurrency"]),
        "queue_fraction": str(cell["queue_fraction"]),
        "capacity_limited": str(capacity_limited).lower(),
    })
    return row


def prefill_rows(root: Path, spec: ProfileSpec, path: Path) -> list[dict[str, str]]:
    data = json_object(path, "cold-prefill result")
    metadata = validate_metadata(data.get("metadata"), max_tokens=1, label="cold prefill")
    require(metadata.get("standalone_prefill") is True, "prefill is not standalone")
    require(metadata.get("prefill_only") is True, "prefill result includes decode")
    cells = data.get("prefill")
    require(isinstance(cells, dict), "prefill cells are absent")
    require(set(cells) <= {str(value) for value in PREFILL_CONTEXTS},
            "prefill contains an unexpected context")

    rows: list[dict[str, str]] = []
    for context in PREFILL_CONTEXTS:
        cell = cells.get(str(context))
        if cell is None:
            continue
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

        row = common_row(root, spec, path)
        row.update({
            "metric": "prefill",
            "concurrency": "1",
            "nominal_context_tokens": str(context),
            "input_tokens": str(prompt_tokens),
            "target_output_tokens": "1",
            "target_requests": str(samples),
            "completed_requests": str(samples),
            "num_errors": "0",
            "measurement_seconds": "",
            "throughput": str(cell["tok_per_sec"]),
            "ttft_p50_seconds": str(cell["ttft_seconds"]),
            "itl_p50_seconds": "",
            "effective_concurrency": "1.0",
            "queue_fraction": "0.0",
            "capacity_limited": "false",
        })
        rows.append(row)
    return rows


def import_result(root_arg: Path, *, require_complete: bool = False) -> tuple[str, list[dict[str, str]]]:
    require(root_arg.is_dir() and not root_arg.is_symlink(),
            f"missing or unsafe result root: {root_arg}")
    root = root_arg.resolve(strict=True)
    profile, spec = validate_manifest(root)
    raw = root / "benchmark/raw"
    require(raw.is_dir() and not raw.is_symlink(), f"raw benchmark directory is absent: {raw}")

    rows: list[dict[str, str]] = []
    observed_decode: set[int] = set()
    for concurrency in CONCURRENCIES:
        path = raw / f"fixed/c{concurrency}.json"
        if not path.exists():
            continue
        rows.append(decode_row(root, spec, concurrency, path))
        observed_decode.add(concurrency)

    observed_prefill: set[int] = set()
    prefill_path = raw / "prefill/cold.json"
    if prefill_path.exists():
        imported_prefill = prefill_rows(root, spec, prefill_path)
        rows.extend(imported_prefill)
        observed_prefill = {int(row["nominal_context_tokens"]) for row in imported_prefill}

    require(rows, f"result root has no completed measurement JSON: {root}")
    if require_complete:
        require(
            regular_file(root / "STATUS.txt", "result status")
            .read_text(encoding="utf-8")
            .strip()
            == "COMPLETE_MEASURED_RAW",
            "result launcher has not completed cleanup and postflight",
        )
        require(
            regular_file(raw / "STATUS.txt", "benchmark status")
            .read_text(encoding="utf-8")
            .strip()
            == "COMPLETE",
            "benchmark client has not completed",
        )
        require(observed_decode == set(CONCURRENCIES),
                f"profile is missing decode cells: {sorted(set(CONCURRENCIES) - observed_decode)}")
        require(observed_prefill == set(PREFILL_CONTEXTS),
                f"profile is missing prefill cells: {sorted(set(PREFILL_CONTEXTS) - observed_prefill)}")
    return profile, rows


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    require(path.is_file() and not path.is_symlink(), f"unsafe output CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == FIELDS, "existing dgx-overlays.csv header differs")
        return list(reader)


def row_sort_key(row: dict[str, str]) -> tuple[int, int, int]:
    profile = PUBLICATION_TO_PROFILE.get(row.get("profile", ""))
    metric_order = 0 if row.get("metric") == "decode" else 1
    axis = int(row.get("concurrency") or row.get("nominal_context_tokens") or 0)
    return PROFILE_ORDER.get(profile, len(PROFILE_ORDER)), metric_order, axis


def merge_rows(
    existing: list[dict[str, str]],
    imported: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Replace supplied profiles and drop every row from the old checkpoint."""
    replaced = {PROFILE_SPECS[profile].publication_profile for profile in imported}
    rows = [
        row for row in existing
        if row.get("model_id") == MODEL_ID and row.get("profile") not in replaced
    ]
    for profile in PROFILE_SPECS:
        rows.extend(imported.get(profile, []))

    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        require(set(row) == set(FIELDS), "DGX overlay row fields differ")
        axis = row["concurrency"] if row["metric"] == "decode" else row["nominal_context_tokens"]
        key = (row["profile"], row["metric"], axis)
        require(key not in keys, f"duplicate DGX overlay row after merge: {key}")
        keys.add(key)
    return sorted(rows, key=row_sort_key)


def render_csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def atomic_write(path: Path, text: str) -> None:
    require(not path.exists() or (path.is_file() and not path.is_symlink()),
            f"refusing to replace unsafe output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require C1-C64 and all four cold-prefill contexts in every input root",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="require one input root for every supported DGX profile",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already equals the requested import",
    )
    args = parser.parse_args()

    imported: dict[str, list[dict[str, str]]] = {}
    for root in args.result_roots:
        profile, rows = import_result(root, require_complete=args.require_complete)
        require(profile not in imported, f"duplicate result profile: {profile}")
        imported[profile] = rows
    if args.require_all:
        require(set(imported) == set(PROFILE_SPECS),
                f"missing profiles: {sorted(set(PROFILE_SPECS) - set(imported))}")

    rendered = render_csv(merge_rows(read_existing(args.output), imported))
    if args.check:
        require(args.output.exists() and args.output.read_text(encoding="utf-8") == rendered,
                f"{args.output} does not match the raw result roots")
    else:
        atomic_write(args.output, rendered)
    print(json.dumps({
        "output": str(args.output),
        "profiles_imported": sorted(imported, key=PROFILE_ORDER.get),
        "rows_imported": sum(len(rows) for rows in imported.values()),
        "mode": "check" if args.check else "write",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
