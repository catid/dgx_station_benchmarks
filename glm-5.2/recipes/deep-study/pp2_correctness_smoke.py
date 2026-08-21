#!/usr/bin/env python3
"""Deterministic long-generation correctness smoke for vLLM PP2.

This is deliberately a correctness test, not a throughput benchmark.  It uses
the pinned llm-inference-bench prompt builder, targets its 8K token context
exactly through vLLM's /tokenize endpoint, and retains two 1K and two >4K
greedy generations for byte-for-byte repeatability and degeneration checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_BENCH_COMMIT = "0b4185b5b435e948b199c9077a00b084864aa963"
TOKENIZE_TARGET = 8192
# P8 measured that direct chat usage and /tokenize agree at exactly 8,192.
# The 8,194 value in prior standalone-prefill JSON is path-specific accounting.
API_PROMPT_TOKENS = TOKENIZE_TARGET
OUTPUT_LENGTHS = (1024, 4608)
REPEATS = 2


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_benchmark_helpers(bench_dir: Path):
    actual = subprocess.run(
        ["git", "-C", str(bench_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != EXPECTED_BENCH_COMMIT:
        raise ValueError(f"llm-inference-bench checkout is not pinned: {actual}")
    sys.path.insert(0, str(bench_dir))
    from llm_decode_bench import build_messages, generate_padding_text

    return actual, build_messages, generate_padding_text


def tokenize_count(
    client: httpx.Client, model: str, messages: list[dict[str, str]]
) -> int:
    response = client.post("/tokenize", json={"model": model, "messages": messages})
    response.raise_for_status()
    value = int(response.json().get("count", 0))
    if value <= 0:
        raise ValueError(f"/tokenize returned an invalid count: {response.text[:500]}")
    return value


def exact_messages(
    client: httpx.Client,
    model: str,
    build_messages,
    generate_padding_text,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    prefix = "[GLM52_P8_PP2_CORRECTNESS] "
    base = generate_padding_text(TOKENIZE_TARGET * 4)

    def text_for(characters: int) -> str:
        if characters <= 0:
            return ""
        return (prefix + base)[:characters]

    low = 0
    high = TOKENIZE_TARGET * 8
    low_count = tokenize_count(client, model, build_messages(TOKENIZE_TARGET, text_for(low)))
    high_count = tokenize_count(client, model, build_messages(TOKENIZE_TARGET, text_for(high)))
    if low_count >= TOKENIZE_TARGET or high_count < TOKENIZE_TARGET:
        raise ValueError(
            "could not bracket exact 8K prompt: "
            f"low={low_count}, high={high_count}, high_chars={high}"
        )

    while low + 1 < high:
        middle = (low + high) // 2
        count = tokenize_count(
            client, model, build_messages(TOKENIZE_TARGET, text_for(middle))
        )
        if count < TOKENIZE_TARGET:
            low, low_count = middle, count
        else:
            high, high_count = middle, count

    # Token counts can occasionally jump by more than one at a character
    # boundary.  Search a small deterministic neighborhood, but never accept
    # an approximate context.
    exact_characters: int | None = None
    exact_count = 0
    start = max(0, low - 32)
    stop = min(len(prefix + base), high + 33)
    for candidate in range(start, stop + 1):
        count = tokenize_count(
            client, model, build_messages(TOKENIZE_TARGET, text_for(candidate))
        )
        if count == TOKENIZE_TARGET:
            exact_characters, exact_count = candidate, count
            break
    if exact_characters is None:
        raise ValueError(
            f"exact /tokenize target {TOKENIZE_TARGET} is unreachable near "
            f"{low_count}@{low} and {high_count}@{high} characters"
        )

    context_text = text_for(exact_characters)
    messages = build_messages(TOKENIZE_TARGET, context_text)
    proof = {
        "tokenize_target": TOKENIZE_TARGET,
        "tokenize_count": exact_count,
        "context_characters": len(context_text),
        "context_sha256": hashlib.sha256(context_text.encode()).hexdigest(),
        "messages_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "messages": messages,
    }
    return messages, proof


def analyze(text: str, expected_tokens: int) -> dict[str, Any]:
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    folded = [word.casefold() for word in words]
    grams = [tuple(folded[index : index + 8]) for index in range(max(0, len(folded) - 7))]
    gram_counts = Counter(grams)
    repeated_8gram_fraction = sum(
        count - 1 for count in gram_counts.values() if count > 1
    ) / max(1, len(grams))

    previous = None
    current_word_run = 0
    maximum_word_run = 0
    for word in folded:
        current_word_run = current_word_run + 1 if word == previous else 1
        previous = word
        maximum_word_run = max(maximum_word_run, current_word_run)
    maximum_character_run = max(
        (len(match.group(0)) for match in re.finditer(r"(\S)\1*", text)),
        default=0,
    )

    nonspace = [character for character in text if not character.isspace()]
    ascii_fraction = sum(character.isascii() for character in nonspace) / max(1, len(nonspace))
    alphanumeric_fraction = sum(character.isalnum() for character in nonspace) / max(1, len(nonspace))
    control_count = sum(
        unicodedata.category(character) in {"Cc", "Cs"}
        and character not in "\n\r\t"
        for character in text
    )
    printable_fraction = sum(
        character.isprintable() or character in "\n\r\t" for character in text
    ) / max(1, len(text))
    minimum_words = 100 if expected_tokens == 1024 else 400
    checks = {
        "nonempty": bool(text.strip()),
        "minimum_words": len(words) >= minimum_words,
        "printable_fraction": printable_fraction >= 0.995,
        "ascii_fraction": ascii_fraction >= 0.80,
        "alphanumeric_fraction": alphanumeric_fraction >= 0.35,
        "no_replacement_character": "\ufffd" not in text,
        "no_invalid_controls": control_count == 0,
        "max_identical_character_run": maximum_character_run < 16,
        "max_identical_word_run": maximum_word_run < 8,
        "repeated_8gram_fraction": repeated_8gram_fraction < 0.20,
    }
    return {
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(folded)) / max(1, len(folded)),
        "printable_fraction": printable_fraction,
        "ascii_fraction": ascii_fraction,
        "alphanumeric_fraction": alphanumeric_fraction,
        "invalid_control_count": control_count,
        "max_identical_character_run": maximum_character_run,
        "max_identical_word_run": maximum_word_run,
        "repeated_8gram_fraction": repeated_8gram_fraction,
        "checks": checks,
        "flagged": not all(checks.values()),
    }


def generate(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    output_tokens: int,
    repeat: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": False,
    }
    started = time.perf_counter()
    response = client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    data = response.json()
    choices = data.get("choices", [])
    if len(choices) != 1:
        raise ValueError(f"expected one completion choice, got {len(choices)}")
    choice = choices[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content") or ""
    combined = "\n".join(part for part in (reasoning, content) if part)
    usage = data.get("usage") or {}
    actual_prompt = int(usage.get("prompt_tokens", -1))
    actual_completion = int(usage.get("completion_tokens", -1))
    exact_usage = {
        "prompt_tokens": actual_prompt == API_PROMPT_TOKENS,
        "completion_tokens": actual_completion == output_tokens,
        "total_tokens": int(usage.get("total_tokens", -1))
        == API_PROMPT_TOKENS + output_tokens,
    }
    analysis = analyze(combined, output_tokens)
    return {
        "requested_output_tokens": output_tokens,
        "repeat": repeat,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "exact_usage_checks": exact_usage,
        "reasoning_content": reasoning,
        "content": content,
        "combined_text": combined,
        "sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "analysis": analysis,
        "passed": all(exact_usage.values()) and not analysis["flagged"],
    }


def evaluate(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    outputs = report.get("outputs", [])
    if len(outputs) != len(OUTPUT_LENGTHS) * REPEATS:
        failures.append(f"expected four retained outputs, got {len(outputs)}")
    for output in outputs:
        if not output.get("passed"):
            failures.append(
                f"{output.get('requested_output_tokens')} token repeat "
                f"{output.get('repeat')} failed usage or degeneration checks"
            )
    for length in OUTPUT_LENGTHS:
        pair = [output for output in outputs if output.get("requested_output_tokens") == length]
        if len(pair) != REPEATS:
            failures.append(f"{length}: expected {REPEATS} repeats")
    return failures


def repeatability(report: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    for length in OUTPUT_LENGTHS:
        pair = [
            output
            for output in report.get("outputs", [])
            if output.get("requested_output_tokens") == length
        ]
        pair.sort(key=lambda output: int(output.get("repeat", -1)))
        if len(pair) != REPEATS:
            continue
        comparisons.append(
            {
                "requested_output_tokens": length,
                "byte_identical": (
                    pair[0].get("sha256") == pair[1].get("sha256")
                    and pair[0].get("combined_text") == pair[1].get("combined_text")
                ),
                "sha256": [pair[0].get("sha256"), pair[1].get("sha256")],
            }
        )
    return {
        "comparisons": comparisons,
        "byte_identical_pairs": sum(item["byte_identical"] for item in comparisons),
        "non_byte_identical_pairs": sum(not item["byte_identical"] for item in comparisons),
        "note": "hash mismatches are recorded as nondeterminism; text-quality checks gate correctness",
    }


def run(args: argparse.Namespace) -> int:
    # Keep the pure text analyzer importable by the CPU-only manifest tests on
    # hosts whose system Python does not carry the benchmark's HTTP extras.
    global httpx
    import httpx

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "model": args.model,
        "server": args.base_url,
        "settings": {
            "tokenize_target": TOKENIZE_TARGET,
            "expected_api_prompt_tokens": API_PROMPT_TOKENS,
            "output_lengths": list(OUTPUT_LENGTHS),
            "repeats_per_length": REPEATS,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "sequential": True,
        },
        "outputs": [],
        "failures": [],
    }
    atomic_json(args.output, report)
    try:
        commit, build_messages, generate_padding_text = load_benchmark_helpers(
            args.bench_dir
        )
        report["llm_inference_bench_commit"] = commit
        timeout = httpx.Timeout(1800.0, connect=30.0)
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
        with httpx.Client(base_url=args.base_url, timeout=timeout, limits=limits) as client:
            health = client.get("/health")
            health.raise_for_status()
            messages, prompt_proof = exact_messages(
                client, args.model, build_messages, generate_padding_text
            )
            report["prompt_proof"] = prompt_proof
            atomic_json(args.output, report)
            for output_tokens in OUTPUT_LENGTHS:
                for repeat in range(REPEATS):
                    result = generate(
                        client, args.model, messages, output_tokens, repeat
                    )
                    report["outputs"].append(result)
                    atomic_json(args.output, report)
        report["failures"] = evaluate(report)
        report["repeatability"] = repeatability(report)
        report["status"] = "passed" if not report["failures"] else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    atomic_json(args.output, report)
    summary = {
        "status": report["status"],
        "outputs_retained": len(report["outputs"]),
        "failures": report["failures"],
        "hashes": [output.get("sha256") for output in report["outputs"]],
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
