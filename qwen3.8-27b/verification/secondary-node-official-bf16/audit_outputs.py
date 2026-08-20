#!/usr/bin/env python3
"""Audit saved natural-decode outputs with the report's degeneration rule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def analyze_text(text: str) -> dict[str, float | int | str | bool]:
    words = re.findall(r"\S+", text)
    lowered = [word.casefold() for word in words]
    grams = [tuple(lowered[i : i + 8]) for i in range(max(0, len(lowered) - 7))]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)

    best_word_run = 0
    current = 0
    previous = None
    for word in lowered:
        current = current + 1 if word == previous else 1
        previous = word
        best_word_run = max(best_word_run, current)

    best_phrase_repeats = 1
    for width in range(2, min(65, len(lowered) // 2 + 1)):
        for offset in range(width):
            previous_block = None
            repeats = 0
            for start in range(offset, len(lowered) - width + 1, width):
                block = lowered[start : start + width]
                repeats = repeats + 1 if block == previous_block else 1
                best_phrase_repeats = max(best_phrase_repeats, repeats)
                previous_block = block

    repeated_fraction = repeated / max(1, len(grams))
    flagged = best_phrase_repeats >= 4 or repeated_fraction >= 0.20
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(lowered)) / max(1, len(lowered)),
        "max_identical_word_run": best_word_run,
        "max_identical_character_run": max(
            (len(match.group(0)) for match in re.finditer(r"(.)\1*", text)),
            default=0,
        ),
        "repeated_8gram_fraction": repeated_fraction,
        "max_consecutive_phrase_repeats": best_phrase_repeats,
        "flagged_degenerate": flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    outputs = []
    for run in data["runs"]:
        text = run.get("output_text") or run.get("content_text") or ""
        row = {"run_index": run["run_index"], **analyze_text(text)}
        outputs.append(row)

    result = {
        "flag_rule": (
            "max_consecutive_phrase_repeats >= 4 or "
            "repeated_8gram_fraction >= 0.20"
        ),
        "outputs": len(outputs),
        "flagged_degenerate": sum(row["flagged_degenerate"] for row in outputs),
        "unique_outputs": len({row["sha256"] for row in outputs}),
        "all_match_node0_reference_sha256": all(
            row["sha256"]
            == "09bfd7b6b9a1a33c0e7685ebcabe22b887fd4f3f3bf154446d3a8f738d0cf2db"
            for row in outputs
        ),
        "runs": outputs,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
