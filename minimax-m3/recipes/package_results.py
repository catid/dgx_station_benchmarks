#!/usr/bin/env python3
"""Normalize and sanitize measured MiniMax M3 evidence for publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MODES = ("disabled", "adaptive", "enabled")
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
CHECKPOINT = "nvidia/MiniMax-M3-NVFP4@901464083161bf8612a29ff7ad29914cd4ab4a85"
MXFP8_CHECKPOINT = "MiniMaxAI/MiniMax-M3-MXFP8@c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {key: sanitize(item) for key, item in value.items()}
        if "hostname" in sanitized:
            sanitized["hostname"] = "benchmark-host"
        if "uname" in sanitized and isinstance(sanitized["uname"], str):
            sanitized["uname"] = re.sub(r"Linux\s+\S+", "Linux benchmark-host", sanitized["uname"], count=1)
        if isinstance(sanitized.get("tokenizer"), str) and sanitized["tokenizer"].startswith("/"):
            sanitized["tokenizer"] = "/model"
        if isinstance(sanitized.get("config_source"), str) and "/lm_eval/" in sanitized["config_source"]:
            sanitized["config_source"] = "/tools/lm-evaluation-harness/lm_eval/" + sanitized[
                "config_source"
            ].split("/lm_eval/", 1)[1]
        return sanitized
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"\b(?:gemini|dgx)\d+\b", "benchmark-host", value, flags=re.IGNORECASE)
        value = re.sub(r" at 0x[0-9a-fA-F]+>", " at 0xSANITIZED>", value)
        return value.replace("127.0.0.1", "loopback")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cell(document: dict[str, Any], concurrency: int) -> dict[str, Any]:
    matches = [row for row in document["results"] if int(row["concurrency"]) == concurrency]
    if len(matches) != 1:
        raise ValueError(f"Expected one C={concurrency} row, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--ppl-root", type=Path)
    parser.add_argument("--mxfp8-root", type=Path)
    parser.add_argument("--mxfp8-legacy-json", type=Path)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    data_dir = args.package_root / "data"
    evidence_dir = data_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    canonical = {mode: load(args.raw_root / mode / "llm-inference-bench.json") for mode in MODES}
    selected_reruns = {
        ("disabled", 2): load(args.raw_root / "reruns" / "disabled-c2-30.json"),
        ("adaptive", 8): load(args.raw_root / "reruns" / "adaptive-c8-60.json"),
        ("enabled", 8): load(args.raw_root / "reruns" / "enabled-c8-60.json"),
    }

    for mode, document in canonical.items():
        if document["metadata"]["version"] != "0.4.29":
            raise ValueError(f"Unexpected llm-inference-bench version for {mode}")
        write_json(evidence_dir / f"nvfp4-{mode}-llm-inference-bench.json", sanitize(document))
    for (mode, concurrency), document in selected_reruns.items():
        write_json(evidence_dir / f"nvfp4-{mode}-c{concurrency}-selected-rerun.json", sanitize(document))

    throughput_rows: list[dict[str, Any]] = []
    for mode in MODES:
        for concurrency in CONCURRENCIES:
            if concurrency > 16:
                throughput_rows.append(
                    {
                        "topology": "1x-gb300",
                        "stations": 1,
                        "checkpoint": CHECKPOINT,
                        "speculative_mode": "none",
                        "thinking_mode": mode,
                        "context_tokens": 8192,
                        "output_token_cap": 1024,
                        "concurrency": concurrency,
                        "duration_seconds": "",
                        "aggregate_output_tokens_per_second": "",
                        "per_stream_output_tokens_per_second": "",
                        "effective_concurrency": "",
                        "completed_requests": 0,
                        "errors": 0,
                        "underfilled": "",
                        "capacity_limited": "true",
                        "selection_note": "skipped: requested KV exceeds 156160-token measured budget",
                        "source": f"evidence/nvfp4-{mode}-llm-inference-bench.json",
                    }
                )
                continue
            document = selected_reruns.get((mode, concurrency), canonical[mode])
            measured = cell(document, concurrency)
            if int(measured["num_errors"]) != 0 or bool(measured["capacity_limited"]):
                raise ValueError(f"Selected {mode} C={concurrency} cell did not pass validation")
            duration = float(document["metadata"]["duration_per_test"])
            note = "canonical 30-second matrix"
            if (mode, concurrency) in selected_reruns:
                note = "selected clean rerun after canonical boundary-underfill flag"
            throughput_rows.append(
                {
                    "topology": "1x-gb300",
                    "stations": 1,
                    "checkpoint": CHECKPOINT,
                    "speculative_mode": "none",
                    "thinking_mode": mode,
                    "context_tokens": int(measured["context_tokens"]),
                    "output_token_cap": 1024,
                    "concurrency": concurrency,
                    "duration_seconds": f"{duration:g}",
                    "aggregate_output_tokens_per_second": f"{float(measured['aggregate_tps']):.6f}",
                    "per_stream_output_tokens_per_second": f"{float(measured['per_request_avg_tps']):.6f}",
                    "effective_concurrency": f"{float(measured['effective_concurrency']):g}",
                    "completed_requests": int(measured["completed_request_count"]),
                    "errors": int(measured["num_errors"]),
                    "underfilled": str(bool(measured["underfilled"])).lower(),
                    "capacity_limited": str(bool(measured["capacity_limited"])).lower(),
                    "selection_note": note,
                    "source": (
                        f"evidence/nvfp4-{mode}-c{concurrency}-selected-rerun.json"
                        if (mode, concurrency) in selected_reruns
                        else f"evidence/nvfp4-{mode}-llm-inference-bench.json"
                    ),
                }
            )

    mxfp8_canonical: dict[str, dict[str, Any]] = {}
    if args.mxfp8_root:
        mxfp8_canonical = {
            mode: load(args.mxfp8_root / f"{mode}-canonical" / "llm-inference-bench.json")
            for mode in MODES
        }
        for mode, document in mxfp8_canonical.items():
            metadata = document["metadata"]
            if metadata["version"] != "0.4.29" or metadata["model"] != "MiniMax-M3-MXFP8":
                raise ValueError(f"Unexpected MXFP8 benchmark identity for {mode}")
            if int(metadata["max_total_tokens"]) != 743168:
                raise ValueError(f"Unexpected MXFP8 KV budget for {mode}")
            write_json(
                evidence_dir / f"mxfp8-{mode}-llm-inference-bench.json",
                sanitize(document),
            )

        confirmation = load(args.mxfp8_root / "disabled-c32" / "llm-inference-bench.json")
        confirmation_cell = cell(confirmation, 32)
        if int(confirmation_cell["num_errors"]) or bool(confirmation_cell["capacity_limited"]):
            raise ValueError("MXFP8 isolated graph-C32 confirmation did not pass")
        write_json(
            evidence_dir / "mxfp8-disabled-c32-confirmation.json",
            sanitize(confirmation),
        )

        if args.mxfp8_legacy_json:
            legacy = load(args.mxfp8_legacy_json)
            write_json(
                evidence_dir / "mxfp8-disabled-superseded-eager.json",
                sanitize(
                    {
                        "status": "superseded",
                        "reason": "C32 ran outside the captured C1-C16 CUDA-graph set",
                        "replacement": "mxfp8-disabled-llm-inference-bench.json",
                        "benchmark": legacy,
                    }
                ),
            )

        for mode, document in mxfp8_canonical.items():
            measured_cells = {int(row["concurrency"]): row for row in document["results"]}
            for concurrency in CONCURRENCIES:
                source = f"evidence/mxfp8-{mode}-llm-inference-bench.json"
                if concurrency not in measured_cells:
                    required_tokens = concurrency * (8192 + 1024)
                    throughput_rows.append(
                        {
                            "topology": "2x-gb300-pp2",
                            "stations": 2,
                            "checkpoint": MXFP8_CHECKPOINT,
                            "speculative_mode": "none",
                            "thinking_mode": mode,
                            "context_tokens": 8192,
                            "output_token_cap": 1024,
                            "concurrency": concurrency,
                            "duration_seconds": "",
                            "aggregate_output_tokens_per_second": "",
                            "per_stream_output_tokens_per_second": "",
                            "effective_concurrency": "",
                            "completed_requests": 0,
                            "errors": 0,
                            "underfilled": "",
                            "capacity_limited": "true",
                            "selection_note": (
                                f"skipped: {required_tokens}-token request set exceeds "
                                "743168-token measured PP2 KV budget"
                            ),
                            "source": source,
                        }
                    )
                    continue
                measured = measured_cells[concurrency]
                if int(measured["num_errors"]) != 0 or bool(measured["capacity_limited"]):
                    raise ValueError(f"MXFP8 {mode} C={concurrency} cell did not pass validation")
                note = "canonical graph-C32 no-async PP2 profile"
                if concurrency == 32:
                    note += "; isolated C32 confirmation also passed"
                elif concurrency == 64:
                    note += "; measured eager path beyond the C32 CUDA-graph tier"
                throughput_rows.append(
                    {
                        "topology": "2x-gb300-pp2",
                        "stations": 2,
                        "checkpoint": MXFP8_CHECKPOINT,
                        "speculative_mode": "none",
                        "thinking_mode": mode,
                        "context_tokens": int(measured["context_tokens"]),
                        "output_token_cap": 1024,
                        "concurrency": concurrency,
                        "duration_seconds": f"{float(document['metadata']['duration_per_test']):g}",
                        "aggregate_output_tokens_per_second": f"{float(measured['aggregate_tps']):.6f}",
                        "per_stream_output_tokens_per_second": f"{float(measured['per_request_avg_tps']):.6f}",
                        "effective_concurrency": "",
                        "completed_requests": int(measured["completed_request_count"]),
                        "errors": int(measured["num_errors"]),
                        "underfilled": "",
                        "capacity_limited": "false",
                        "selection_note": note,
                        "source": source,
                    }
                )

    write_csv(
        data_dir / "throughput.csv",
        [
            "topology",
            "stations",
            "checkpoint",
            "speculative_mode",
            "thinking_mode",
            "context_tokens",
            "output_token_cap",
            "concurrency",
            "duration_seconds",
            "aggregate_output_tokens_per_second",
            "per_stream_output_tokens_per_second",
            "effective_concurrency",
            "completed_requests",
            "errors",
            "underfilled",
            "capacity_limited",
            "selection_note",
            "source",
        ],
        throughput_rows,
    )

    disabled_prefill = canonical["disabled"]["prefill"]
    prefill_rows = []
    for context in (8192, 65536, 131072):
        result = disabled_prefill[str(context)]
        prefill_rows.append(
            {
                "topology": "1x-gb300",
                "stations": 1,
                "checkpoint": CHECKPOINT,
                "thinking_mode": "disabled",
                "context_tokens": context,
                "cached_tokens": int(result["server_validation"]["cached_tokens"]),
                "client_prompt_tokens": int(result["prompt_tokens"]),
                "client_prompt_tokens_per_second": int(result["tok_per_sec"]),
                "time_to_first_token_seconds": f"{float(result['ttft_seconds']):.3f}",
                "samples": int(result["samples"]),
                "errors": 0,
                "source": "evidence/nvfp4-disabled-llm-inference-bench.json",
            }
        )
    if mxfp8_canonical:
        mxfp8 = mxfp8_canonical["disabled"]
        for context in (8192, 65536, 131072):
            result = mxfp8["prefill"][str(context)]
            prefill_rows.append(
                {
                    "topology": "2x-gb300-pp2",
                    "stations": 2,
                    "checkpoint": MXFP8_CHECKPOINT,
                    "thinking_mode": "disabled",
                    "context_tokens": context,
                    "cached_tokens": 0,
                    "client_prompt_tokens": int(result["prompt_tokens"]),
                    "client_prompt_tokens_per_second": int(result["tok_per_sec"]),
                    "time_to_first_token_seconds": f"{float(result['ttft_seconds']):.3f}",
                    "samples": int(result["samples"]),
                    "errors": 0,
                    "source": "evidence/mxfp8-disabled-llm-inference-bench.json",
                }
            )
    write_csv(
        data_dir / "prefill.csv",
        [
            "topology",
            "stations",
            "checkpoint",
            "thinking_mode",
            "context_tokens",
            "cached_tokens",
            "client_prompt_tokens",
            "client_prompt_tokens_per_second",
            "time_to_first_token_seconds",
            "samples",
            "errors",
            "source",
        ],
        prefill_rows,
    )

    quality_rows = []
    quality_tracks: list[tuple[str, int, str, str]] = [
        ("1x-gb300", 1, CHECKPOINT, "nvfp4"),
    ]
    if args.mxfp8_root:
        quality_tracks.append(("2x-gb300-pp2", 2, MXFP8_CHECKPOINT, "mxfp8"))

    for topology, stations, checkpoint, prefix in quality_tracks:
        quality_evidence: dict[str, Any] = {
            "checkpoint": checkpoint,
            "topology": topology,
            "method": "Natural prompts, model-card sampling, full output retained during audit",
            "manual_review_scope": "Degeneration/repetition only; not a factual-accuracy benchmark",
            "cells": [],
        }
        evidence_name = f"{prefix}-natural-output-audit.json"
        for mode in MODES:
            for concurrency in (1, 64, 128):
                report_path = args.raw_root / mode / f"natural-c{concurrency}.json"
                if prefix == "mxfp8":
                    report_path = args.mxfp8_root / f"{mode}-natural" / f"natural-c{concurrency}.json"
                report = load(report_path)
                outputs = report["outputs"]
                repeated_max = max(float(item["analysis"]["repeated_8gram_fraction"]) for item in outputs)
                word_run_max = max(int(item["analysis"]["max_identical_word_run"]) for item in outputs)
                character_run_max = max(int(item["analysis"]["max_identical_character_run"]) for item in outputs)
                length_stops = sum(item["finish_reason"] == "length" for item in outputs)
                answer_empty = sum(not str(item.get("content") or "").strip() for item in outputs)
                reasoning_nonempty = sum(bool(str(item.get("reasoning_content") or "").strip()) for item in outputs)
                unique_hashes = len({item["sha256"] for item in outputs})
                total_completion_tokens = sum(int(item["usage"]["completion_tokens"]) for item in outputs)
                automatic_flags = sum(bool(item["analysis"]["flagged"]) for item in outputs)
                quality_rows.append(
                    {
                        "topology": topology,
                        "stations": stations,
                        "checkpoint": checkpoint,
                        "speculative_mode": "none",
                        "thinking_mode": mode,
                        "concurrency": concurrency,
                        "outputs": len(outputs),
                        "unique_output_hashes": unique_hashes,
                        "errors": 0,
                        "empty_outputs": int(report["summary"]["empty"]),
                        "answer_empty_outputs": answer_empty,
                        "reasoning_nonempty_outputs": reasoning_nonempty,
                        "automatic_flags": automatic_flags,
                        "manual_degenerate_outputs": 0,
                        "manual_pass": "true",
                        "max_repeated_8gram_fraction": f"{repeated_max:.9f}",
                        "max_identical_word_run": word_run_max,
                        "max_identical_character_run": character_run_max,
                        "length_cap_stops": length_stops,
                        "total_completion_tokens": total_completion_tokens,
                        "source": f"evidence/{evidence_name}",
                    }
                )

                retained = list(outputs[:5])
                retained_hashes = {item["sha256"] for item in retained}
                for item in outputs:
                    if item["finish_reason"] == "length" or not str(item.get("content") or "").strip():
                        if item["sha256"] not in retained_hashes:
                            retained.append(item)
                            retained_hashes.add(item["sha256"])
                quality_evidence["cells"].append(
                    {
                        "thinking_mode": mode,
                        "concurrency": concurrency,
                        "settings": report["settings"],
                        "summary": report["summary"],
                        "unique_output_hashes": unique_hashes,
                        "answer_empty_outputs": answer_empty,
                        "reasoning_nonempty_outputs": reasoning_nonempty,
                        "finish_reasons": {
                            "stop": sum(item["finish_reason"] == "stop" for item in outputs),
                            "length": length_stops,
                        },
                        "max_repeated_8gram_fraction": repeated_max,
                        "max_identical_word_run": word_run_max,
                        "max_identical_character_run": character_run_max,
                        "manual_degenerate_outputs": 0,
                        "manual_pass": True,
                        "all_output_sha256": [item["sha256"] for item in outputs],
                        "retained_samples": sanitize(retained),
                    }
                )
        write_json(evidence_dir / evidence_name, quality_evidence)
    write_csv(
        data_dir / "natural-quality.csv",
        [
            "topology",
            "stations",
            "checkpoint",
            "speculative_mode",
            "thinking_mode",
            "concurrency",
            "outputs",
            "unique_output_hashes",
            "errors",
            "empty_outputs",
            "answer_empty_outputs",
            "reasoning_nonempty_outputs",
            "automatic_flags",
            "manual_degenerate_outputs",
            "manual_pass",
            "max_repeated_8gram_fraction",
            "max_identical_word_run",
            "max_identical_character_run",
            "length_cap_stops",
            "total_completion_tokens",
            "source",
        ],
        quality_rows,
    )

    ppl_rows: list[dict[str, Any]] = []
    if args.ppl_root:
        result_paths = sorted((args.ppl_root / "lm-eval" / "MiniMax-M3-NVFP4").glob("results_*.json"))
        if len(result_paths) != 1:
            raise ValueError(f"Expected one lm-eval result, found {len(result_paths)}")
        ppl = load(result_paths[0])
        profile = load(args.ppl_root / "bf16-kv-server-profile.json")
        decode = load(args.ppl_root / "bf16-kv-decode.json")
        wikitext = ppl["results"]["wikitext"]
        decode_cell = cell(decode, 1)
        if int(decode_cell["num_errors"]) or bool(decode_cell["capacity_limited"]):
            raise ValueError("BF16-KV C1 decode validation failed")
        write_json(
            evidence_dir / "nvfp4-wikitext2.json",
            sanitize(
                {
                    "checkpoint": CHECKPOINT,
                    "server_profile": profile,
                    "results": ppl["results"],
                    "configs": ppl["configs"],
                    "versions": ppl["versions"],
                    "n-samples": ppl["n-samples"],
                }
            ),
        )
        write_json(evidence_dir / "nvfp4-bf16-kv-decode.json", sanitize(decode))
        ppl_rows.append(
            {
                "topology": "1x-gb300",
                "stations": 1,
                "checkpoint": CHECKPOINT,
                "kv_cache_dtype": profile["kv_cache_dtype"],
                "effective_max_length": 2047,
                "batch_size": 4,
                "documents": int(wikitext["sample_len"]),
                "word_perplexity": f"{float(wikitext['word_perplexity,none']):.9f}",
                "byte_perplexity": f"{float(wikitext['byte_perplexity,none']):.9f}",
                "bits_per_byte": f"{float(wikitext['bits_per_byte,none']):.9f}",
                "decode_tokens_per_second": f"{float(decode_cell['aggregate_tps']):.6f}",
                "source": "evidence/nvfp4-wikitext2.json; evidence/nvfp4-bf16-kv-decode.json",
            }
        )
    if args.mxfp8_root:
        mxfp8_ppl_paths = sorted(
            (args.mxfp8_root / "ppl" / "lm-eval-u095" / "MiniMax-M3-MXFP8").glob("results_*.json")
        )
        if len(mxfp8_ppl_paths) != 1:
            raise ValueError(f"Expected one MXFP8 lm-eval result, found {len(mxfp8_ppl_paths)}")
        mxfp8_ppl = load(mxfp8_ppl_paths[0])
        mxfp8_decode = load(args.mxfp8_root / "ppl" / "bf16-kv-c1" / "llm-inference-bench.json")
        mxfp8_wikitext = mxfp8_ppl["results"]["wikitext"]
        mxfp8_decode_cell = cell(mxfp8_decode, 1)
        if int(mxfp8_decode_cell["num_errors"]) or bool(mxfp8_decode_cell["capacity_limited"]):
            raise ValueError("MXFP8 BF16-KV C1 decode validation failed")
        mxfp8_profile = {
            "served_model": "MiniMax-M3-MXFP8",
            "thinking_mode": "disabled",
            "kv_cache_dtype": "bfloat16",
            "max_model_len": "9216",
            "max_num_seqs": "4",
            "max_num_batched_tokens": "8192",
            "gpu_memory_utilization": "0.95",
            "tensor_parallel_size": "1",
            "pipeline_parallel_size": "2",
            "cuda_graph_capture_sizes": [1, 2, 4],
            "max_cudagraph_capture_size": 4,
            "async_scheduling": False,
            "automatic_tool_parser": False,
            "language_model_only": True,
            "cpu_offload_requested": False,
            "measured_global_kv_tokens": 313984,
        }
        write_json(
            evidence_dir / "mxfp8-wikitext2.json",
            sanitize(
                {
                    "checkpoint": MXFP8_CHECKPOINT,
                    "server_profile": mxfp8_profile,
                    "results": mxfp8_ppl["results"],
                    "configs": mxfp8_ppl["configs"],
                    "versions": mxfp8_ppl["versions"],
                    "n-samples": mxfp8_ppl["n-samples"],
                }
            ),
        )
        write_json(evidence_dir / "mxfp8-bf16-kv-decode.json", sanitize(mxfp8_decode))
        ppl_rows.append(
            {
                "topology": "2x-gb300-pp2",
                "stations": 2,
                "checkpoint": MXFP8_CHECKPOINT,
                "kv_cache_dtype": "bfloat16",
                "effective_max_length": 2047,
                "batch_size": 4,
                "documents": int(mxfp8_wikitext["sample_len"]),
                "word_perplexity": f"{float(mxfp8_wikitext['word_perplexity,none']):.9f}",
                "byte_perplexity": f"{float(mxfp8_wikitext['byte_perplexity,none']):.9f}",
                "bits_per_byte": f"{float(mxfp8_wikitext['bits_per_byte,none']):.9f}",
                "decode_tokens_per_second": f"{float(mxfp8_decode_cell['aggregate_tps']):.6f}",
                "source": "evidence/mxfp8-wikitext2.json; evidence/mxfp8-bf16-kv-decode.json",
            }
        )
    write_csv(
        data_dir / "wikitext2-perplexity.csv",
        [
            "topology",
            "stations",
            "checkpoint",
            "kv_cache_dtype",
            "effective_max_length",
            "batch_size",
            "documents",
            "word_perplexity",
            "byte_perplexity",
            "bits_per_byte",
            "decode_tokens_per_second",
            "source",
        ],
        ppl_rows,
    )

    manifest = []
    for path in sorted(evidence_dir.glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"{digest}  {path.name}")
    (evidence_dir / "SHA256SUMS").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
