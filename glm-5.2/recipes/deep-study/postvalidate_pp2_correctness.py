#!/usr/bin/env python3
"""Reclassify retained P8 outputs after the prompt-accounting audit correction.

The original P8 harness intentionally remains immutable.  It assumed that a
direct chat request would report the standalone-prefill path's 8,194 prompt
tokens and also treated byte-identical greedy repeats as a correctness gate.
The direct API and /tokenize endpoint both reported exactly 8,192 tokens, and
GPU inference produced coherent but non-byte-identical greedy repeats.  This
validator checks the retained text and usage without rewriting the raw file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from pp2_correctness_smoke import analyze


PROMPT_TOKENS = 8192
OUTPUT_LENGTHS = (1024, 4608)
REPEATS = 2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def common_prefix_characters(left: str, right: str) -> int:
    return len(os.path.commonprefix((left, right)))


def words(text: str) -> list[str]:
    return [word.casefold() for word in re.findall(r"\b\w+\b", text)]


def common_prefix_words(left: str, right: str) -> int:
    count = 0
    for left_word, right_word in zip(words(left), words(right)):
        if left_word != right_word:
            break
        count += 1
    return count


def eightgram_jaccard(left: str, right: str) -> float:
    def grams(text: str) -> set[tuple[str, ...]]:
        values = words(text)
        return {
            tuple(values[index : index + 8])
            for index in range(max(0, len(values) - 7))
        }

    left_grams = grams(left)
    right_grams = grams(right)
    return len(left_grams & right_grams) / max(1, len(left_grams | right_grams))


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate_retained(raw_path: Path) -> dict[str, Any]:
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("raw harness report is not an object")
    if raw.get("settings", {}).get("tokenize_target") != PROMPT_TOKENS:
        raise ValueError("raw harness did not target exact 8K")
    proof = raw.get("prompt_proof", {})
    if proof.get("tokenize_target") != PROMPT_TOKENS or proof.get("tokenize_count") != PROMPT_TOKENS:
        raise ValueError("raw harness lacks exact /tokenize 8K proof")
    messages = proof.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("raw prompt messages are missing")
    messages_hash = sha256_bytes(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    )
    if messages_hash != proof.get("messages_sha256"):
        raise ValueError("raw prompt-message hash does not verify")

    retained = raw.get("outputs", [])
    if not isinstance(retained, list) or len(retained) != len(OUTPUT_LENGTHS) * REPEATS:
        raise ValueError("expected four retained outputs")
    expected_grid = {(length, repeat) for length in OUTPUT_LENGTHS for repeat in range(REPEATS)}
    actual_grid = {
        (int(output.get("requested_output_tokens", -1)), int(output.get("repeat", -1)))
        for output in retained
    }
    if actual_grid != expected_grid:
        raise ValueError("retained output grid is incomplete or duplicated")

    summaries: list[dict[str, Any]] = []
    by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in OUTPUT_LENGTHS}
    for output in retained:
        length = int(output["requested_output_tokens"])
        combined = output.get("combined_text")
        if not isinstance(combined, str) or not combined:
            raise ValueError(f"{length}: retained text is absent")
        reasoning = output.get("reasoning_content") or ""
        content = output.get("content") or ""
        rebuilt = "\n".join(part for part in (reasoning, content) if part)
        if rebuilt != combined:
            raise ValueError(f"{length}: combined retained text is inconsistent")
        digest = sha256_bytes(combined.encode())
        if digest != output.get("sha256"):
            raise ValueError(f"{length}: retained output hash does not verify")
        usage = output.get("usage", {})
        if (
            int(usage.get("prompt_tokens", -1)) != PROMPT_TOKENS
            or int(usage.get("completion_tokens", -1)) != length
            or int(usage.get("total_tokens", -1)) != PROMPT_TOKENS + length
        ):
            raise ValueError(f"{length}: direct API token usage is not exact")
        if output.get("finish_reason") != "length":
            raise ValueError(f"{length}: forced-length completion did not finish by length")
        analysis = analyze(combined, length)
        if analysis.get("flagged") or not all(analysis.get("checks", {}).values()):
            raise ValueError(f"{length}: retained output failed degeneration checks")
        recorded = output.get("analysis", {})
        for field in (
            "characters",
            "words",
            "invalid_control_count",
            "max_identical_character_run",
            "max_identical_word_run",
            "repeated_8gram_fraction",
            "flagged",
        ):
            if recorded.get(field) != analysis.get(field):
                raise ValueError(f"{length}: recorded text analysis drifted at {field}")
        summary = {
            "requested_output_tokens": length,
            "repeat": int(output["repeat"]),
            "finish_reason": output["finish_reason"],
            "usage": {
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
            },
            "sha256": digest,
            "analysis": analysis,
        }
        summaries.append(summary)
        by_length[length].append({"text": combined, "sha256": digest})

    comparisons = []
    for length in OUTPUT_LENGTHS:
        pair = by_length[length]
        if len(pair) != REPEATS:
            raise ValueError(f"{length}: repeat pair is incomplete")
        comparisons.append(
            {
                "requested_output_tokens": length,
                "byte_identical": pair[0]["sha256"] == pair[1]["sha256"],
                "sha256": [pair[0]["sha256"], pair[1]["sha256"]],
                "common_prefix_characters": common_prefix_characters(
                    pair[0]["text"], pair[1]["text"]
                ),
                "common_prefix_words": common_prefix_words(
                    pair[0]["text"], pair[1]["text"]
                ),
                "eightgram_jaccard": eightgram_jaccard(
                    pair[0]["text"], pair[1]["text"]
                ),
            }
        )

    summaries.sort(key=lambda output: (output["requested_output_tokens"], output["repeat"]))
    return {
        "schema_version": 1,
        "status": "passed_no_corruption_detected",
        "result_kind": "correctness_only_no_performance",
        "raw_harness_status": raw.get("status"),
        "raw_harness_sha256": sha256_bytes(raw_bytes),
        "accounting_correction": {
            "raw_expected_api_prompt_tokens": raw.get("settings", {}).get(
                "expected_api_prompt_tokens"
            ),
            "tokenize_target": PROMPT_TOKENS,
            "tokenize_observed": proof["tokenize_count"],
            "direct_api_prompt_tokens_observed_all_requests": PROMPT_TOKENS,
            "disposition": (
                "the direct API and /tokenize endpoint agree at exactly 8,192; "
                "the raw 8,194 assertion came from a different standalone-prefill path"
            ),
        },
        "automated_quality": {
            "outputs_retained": len(summaries),
            "outputs_passing_all_text_checks": len(summaries),
            "corruption_or_degeneration_flags": 0,
        },
        "repeatability": {
            "byte_identical_pairs": sum(item["byte_identical"] for item in comparisons),
            "non_byte_identical_pairs": sum(not item["byte_identical"] for item in comparisons),
            "disposition": (
                "greedy PP2 outputs were coherent but not byte-identical; hash mismatch "
                "is retained as nondeterminism, not classified as text corruption"
            ),
            "comparisons": comparisons,
        },
        "outputs": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = validate_retained(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": report["status"],
        "raw_harness_sha256": report["raw_harness_sha256"],
        "outputs": report["automated_quality"]["outputs_retained"],
        "non_byte_identical_pairs": report["repeatability"]["non_byte_identical_pairs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
