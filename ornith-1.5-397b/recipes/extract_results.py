#!/usr/bin/env python3
"""Regenerate compact 1x and 2x summaries from retained raw artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "1x" / "raw"
OUT = ROOT / "data" / "1x"
RAW2 = ROOT / "data" / "2x" / "raw"
OUT2 = ROOT / "data" / "2x"
MODEL = "Ornith-1.5-397B-NVFP4"


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text())


def load2(topology: str, name: str) -> dict:
    return json.loads((RAW2 / topology.lower() / name).read_text())


def write_csv(
    name: str, fieldnames: list[str], rows: list[dict], directory: Path = OUT
) -> None:
    path = directory / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_decode() -> None:
    data = load("decode-8k-1024.json")
    results = sorted(data["results"], key=lambda row: row["concurrency"])
    expected = [1, 2, 4, 8, 16]
    assert [row["concurrency"] for row in results] == expected
    assert all(row["num_errors"] == 0 for row in results)

    rows = []
    for result in results:
        rows.append(
            {
                "stations": 1,
                "topology": "TP1",
                "model": MODEL,
                "weight_quantization": "ModelOpt NVFP4 W4A4 group16",
                "kv_cache_dtype": "fp8_e4m3",
                "input_tokens": int(result["input_seq_len_avg"]),
                "output_tokens": int(result["output_seq_len_avg"]),
                "concurrency": result["concurrency"],
                "aggregate_tps": result["aggregate_tps"],
                "per_request_tps": result["per_request_avg_tps"],
                "ttft_seconds": result["ttft_p50"],
                "itl_seconds": result["inter_token_latency_p50"],
                "avg_running": result["avg_running_reqs"],
                "max_running": result["max_running_reqs"],
                "capacity_limited": str(result["capacity_limited"]).lower(),
                "num_errors": result["num_errors"],
                "aggregate_source": result["aggregate_source"],
                "measurement_seconds": result["measurement_seconds"],
            }
        )
    write_csv("decode-throughput.csv", list(rows[0]), rows)


def extract_prefill() -> None:
    specs = [
        ("prefill-8k-64k.json", "8192", "8K", ""),
        ("prefill-8k-64k.json", "65536", "64K", ""),
        (
            "prefill-128k-canonical.json",
            "131072",
            "128K canonical CLI",
            "The chat completion API reports two more prompt tokens than /tokenize.",
        ),
        (
            "prefill-128k-exact-api.json",
            "131070",
            "128K exact API",
            "Calibrated to 131070 via /tokenize so API usage is exactly 131072.",
        ),
    ]
    rows = []
    for filename, key, label, notes in specs:
        result = load(filename)["prefill"][key]
        rows.append(
            {
                "stations": 1,
                "topology": "TP1",
                "model": MODEL,
                "kv_cache_dtype": "fp8_e4m3",
                "server_max_model_len": 135168,
                "target_label": label,
                "tokenized_target": int(key),
                "actual_prompt_tokens": result["prompt_tokens"],
                "ttft_seconds": result["ttft_seconds"],
                "prompt_tps": result["tok_per_sec"],
                "samples": result["samples"],
                "method": result["method"],
                "notes": notes,
            }
        )
    write_csv("prefill.csv", list(rows[0]), rows)


def extract_perplexity() -> None:
    data = load("wikitext2-bf16-kv.json")
    result = data["results"]["wikitext"]
    row = {
        "stations": 1,
        "topology": "TP1",
        "model": MODEL,
        "kv_cache_dtype": "bfloat16",
        "task": "EleutherAI/wikitext_document_level",
        "documents": result["sample_len"],
        "max_length": 2048,
        "api_batch_size": 4,
        "word_perplexity": result["word_perplexity,none"],
        "byte_perplexity": result["byte_perplexity,none"],
        "bits_per_byte": result["bits_per_byte,none"],
    }
    assert row["documents"] == 62
    write_csv("wikitext2-perplexity.csv", list(row), [row])


def extract_quality() -> None:
    records = []
    for profile in ("natural", "forced"):
        data = load(f"audit-{profile}.json")
        assert data["prompt_tokens"] == 8192
        for output in data["outputs"]:
            analysis = output["analysis"]
            usage = output["usage"]
            flagged = (
                analysis["repeated_8gram_fraction"] >= 0.20
                or analysis["max_identical_word_run"] >= 4
                or analysis["max_identical_character_run"] >= 16
            )
            records.append(
                {
                    "profile": profile,
                    "index": output["index"],
                    "sha256": output["sha256"],
                    "finish_reason": output["finish_reason"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "characters": analysis["characters"],
                    "words": analysis["words"],
                    "unique_word_ratio": analysis["unique_word_ratio"],
                    "max_identical_character_run": analysis[
                        "max_identical_character_run"
                    ],
                    "max_identical_word_run": analysis["max_identical_word_run"],
                    "repeated_2gram_fraction": analysis["repeated_2gram_fraction"],
                    "repeated_4gram_fraction": analysis["repeated_4gram_fraction"],
                    "repeated_8gram_fraction": analysis["repeated_8gram_fraction"],
                    "automatic_degeneration_flag": flagged,
                }
            )

    assert len(records) == 8
    assert all(record["completion_tokens"] == 1024 for record in records)
    summary = {
        "method": {
            "profiles": ["natural_eos_policy", "forced_ignore_eos"],
            "requests_per_profile": 4,
            "temperature": 0,
            "prompt_tokens": 8192,
            "max_completion_tokens": 1024,
            "flag_rule": (
                "repeated_8gram_fraction >= 0.20, identical-word run >= 4, "
                "or identical-character run >= 16"
            ),
            "manual_review": "All eight full outputs were coherent and nondegenerate.",
        },
        "summary": {
            "outputs": len(records),
            "flagged": sum(record["automatic_degeneration_flag"] for record in records),
            "max_repeated_8gram_fraction": max(
                record["repeated_8gram_fraction"] for record in records
            ),
            "max_identical_word_run": max(
                record["max_identical_word_run"] for record in records
            ),
            "max_identical_character_run": max(
                record["max_identical_character_run"] for record in records
            ),
        },
        "outputs": records,
    }
    (OUT / "quality-audit-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def decode_row(result: dict, topology: str) -> dict:
    return {
        "stations": 2,
        "topology": topology,
        "model": MODEL,
        "weight_quantization": "ModelOpt NVFP4 W4A4 group16",
        "kv_cache_dtype": "fp8_e4m3",
        "input_tokens": int(result["input_seq_len_avg"]),
        "output_tokens": int(result["output_seq_len_avg"]),
        "concurrency": result["concurrency"],
        "aggregate_tps": result["aggregate_tps"],
        "per_request_tps": result["per_request_avg_tps"],
        "ttft_seconds": result["ttft_p50"],
        "itl_seconds": result["inter_token_latency_p50"],
        "avg_running": result["avg_running_reqs"],
        "max_running": result["max_running_reqs"],
        "capacity_limited": str(result["capacity_limited"]).lower(),
        "num_errors": result["num_errors"],
        "aggregate_source": result["aggregate_source"],
        "measurement_seconds": result["measurement_seconds"],
    }


def extract_2x_decode() -> None:
    expected = [1, 2, 4, 8, 16, 32, 64, 128]
    rows = []
    stability_rows = []
    for topology in ("PP2", "TP2"):
        short = load2(topology, "decode-prefill-30s.json")
        results = sorted(short["results"], key=lambda row: row["concurrency"])
        assert short["metadata"]["duration_per_test"] == 30.0
        assert [row["concurrency"] for row in results] == expected
        assert all(row["num_errors"] == 0 for row in results)
        rows.extend(decode_row(result, topology) for result in results)

        stable_data = load2(topology, "c128-60s.json")
        stable = stable_data["results"]
        assert stable_data["metadata"]["duration_per_test"] == 60.0
        assert len(stable) == 1 and stable[0]["concurrency"] == 128
        assert stable[0]["num_errors"] == 0
        for role, result, duration in (
            ("diagnostic_short", results[-1], 30),
            ("headline_stable", stable[0], 60),
        ):
            stability_rows.append(
                {
                    "stations": 2,
                    "topology": topology,
                    "model": MODEL,
                    "kv_cache_dtype": "fp8_e4m3",
                    "input_tokens": int(result["input_seq_len_avg"]),
                    "output_tokens": int(result["output_seq_len_avg"]),
                    "concurrency": 128,
                    "target_duration_seconds": duration,
                    "measurement_seconds": result["measurement_seconds"],
                    "aggregate_tps": result["aggregate_tps"],
                    "per_request_tps": result["per_request_avg_tps"],
                    "ttft_seconds": result["ttft_p50"],
                    "itl_seconds": result["inter_token_latency_p50"],
                    "avg_running": result["avg_running_reqs"],
                    "max_running": result["max_running_reqs"],
                    "capacity_limited": str(result["capacity_limited"]).lower(),
                    "num_errors": result["num_errors"],
                    "role": role,
                    "notes": (
                        "Retained for transparency; transient short-window value."
                        if role == "diagnostic_short"
                        else "Stable 60-second C128 topology comparison and headline."
                    ),
                }
            )

    write_csv("decode-throughput.csv", list(rows[0]), rows, OUT2)
    write_csv("c128-stability.csv", list(stability_rows[0]), stability_rows, OUT2)


def extract_2x_prefill() -> None:
    rows = []
    for topology in ("PP2", "TP2"):
        data = load2(topology, "decode-prefill-30s.json")
        for target, label in (("8192", "8K"), ("65536", "64K"), ("131072", "128K canonical CLI")):
            result = data["prefill"][target]
            rows.append(
                {
                    "stations": 2,
                    "topology": topology,
                    "model": MODEL,
                    "kv_cache_dtype": "fp8_e4m3",
                    "server_max_model_len": 135168,
                    "target_label": label,
                    "tokenized_target": int(target),
                    "actual_prompt_tokens": result["prompt_tokens"],
                    "ttft_seconds": result["ttft_seconds"],
                    "prompt_tps": result["tok_per_sec"],
                    "samples": result["samples"],
                    "method": result["method"],
                    "notes": "Canonical /tokenize target; retain API-observed token count.",
                }
            )
    write_csv("prefill.csv", list(rows[0]), rows, OUT2)


def extract_2x_quality() -> None:
    records = []
    topology_summary = {}
    for topology in ("PP2", "TP2"):
        data = load2(topology, "natural-quality-audit.json")
        assert data["prompt_tokens"] == 8192
        topology_records = []
        for output in data["outputs"]:
            analysis = output["analysis"]
            usage = output["usage"]
            flagged = (
                analysis["repeated_8gram_fraction"] >= 0.20
                or analysis["max_identical_word_run"] >= 4
                or analysis["max_identical_character_run"] >= 16
            )
            record = {
                "topology": topology,
                "index": output["index"],
                "sha256": output["sha256"],
                "finish_reason": output["finish_reason"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                **analysis,
                "automatic_degeneration_flag": flagged,
            }
            topology_records.append(record)
            records.append(record)
        assert len(topology_records) == 4
        assert all(record["completion_tokens"] == 1024 for record in topology_records)
        topology_summary[topology] = {
            "outputs": len(topology_records),
            "unique_output_hashes": len({record["sha256"] for record in topology_records}),
            "flagged": sum(record["automatic_degeneration_flag"] for record in topology_records),
            "max_repeated_8gram_fraction": max(record["repeated_8gram_fraction"] for record in topology_records),
            "max_identical_word_run": max(record["max_identical_word_run"] for record in topology_records),
            "max_identical_character_run": max(record["max_identical_character_run"] for record in topology_records),
        }

    summary = {
        "method": {
            "profile": "natural_eos_policy",
            "requests_per_topology": 4,
            "temperature": 0,
            "prompt_tokens": 8192,
            "max_completion_tokens": 1024,
            "flag_rule": (
                "repeated_8gram_fraction >= 0.20, identical-word run >= 4, "
                "or identical-character run >= 16"
            ),
            "manual_review": "All eight full outputs were coherent and nondegenerate.",
            "duplicate_note": (
                "Temperature-zero requests may be byte-identical; duplicate hashes "
                "are determinism, not within-output degeneration."
            ),
        },
        "summary_by_topology": topology_summary,
        "outputs": records,
    }
    (OUT2 / "quality-audit-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT2.mkdir(parents=True, exist_ok=True)
    extract_decode()
    extract_prefill()
    extract_perplexity()
    extract_quality()
    extract_2x_decode()
    extract_2x_prefill()
    extract_2x_quality()


if __name__ == "__main__":
    main()
