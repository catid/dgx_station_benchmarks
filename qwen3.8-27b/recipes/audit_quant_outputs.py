#!/usr/bin/env python3
"""Score retained completion-stats text for intra-output degeneration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPEATED_8GRAM_THRESHOLD = 0.20
CONSECUTIVE_PHRASE_THRESHOLD = 4


def analyze_text(text: str) -> dict[str, int | float | str]:
    words = re.findall(r"\S+", text)
    lowered = [word.casefold() for word in words]
    word_ids: dict[str, int] = {}
    tokens = [word_ids.setdefault(word, len(word_ids) + 1) for word in lowered]

    best_word_run = 0
    current_word_run = 0
    previous = None
    for token in tokens:
        current_word_run = current_word_run + 1 if token == previous else 1
        previous = token
        best_word_run = max(best_word_run, current_word_run)

    grams = [tuple(tokens[index : index + 8]) for index in range(max(0, len(tokens) - 7))]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)

    mask = (1 << 64) - 1
    base = 1_000_003
    prefixes = [0]
    powers = [1]
    for token in tokens:
        prefixes.append((prefixes[-1] * base + token) & mask)
        powers.append((powers[-1] * base) & mask)

    def block_hash(start: int, width: int) -> int:
        return (prefixes[start + width] - prefixes[start] * powers[width]) & mask

    best_phrase_repeats = 1 if tokens else 0
    best_phrase_width = 0
    max_width = min(64, len(tokens) // 2)
    for width in range(2, max_width + 1):
        for offset in range(width):
            previous_start = None
            previous_hash = None
            repeats = 0
            for start in range(offset, len(tokens) - width + 1, width):
                current_hash = block_hash(start, width)
                same = (
                    previous_start is not None
                    and current_hash == previous_hash
                    and tokens[start : start + width]
                    == tokens[previous_start : previous_start + width]
                )
                repeats = repeats + 1 if same else 1
                if repeats > best_phrase_repeats:
                    best_phrase_repeats = repeats
                    best_phrase_width = width
                previous_start = start
                previous_hash = current_hash

    repeated_8gram_fraction = repeated / max(1, len(grams))
    return {
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(lowered)) / max(1, len(lowered)),
        "max_identical_word_run": best_word_run,
        "max_identical_character_run": max(
            (len(match.group(0)) for match in re.finditer(r"(.)\1*", text)),
            default=0,
        ),
        "repeated_8gram_fraction": repeated_8gram_fraction,
        "max_consecutive_phrase_repeats": best_phrase_repeats,
        "repeated_phrase_width_words": best_phrase_width,
        "flagged": (
            repeated_8gram_fraction >= REPEATED_8GRAM_THRESHOLD
            or best_phrase_repeats >= CONSECUTIVE_PHRASE_THRESHOLD
        ),
    }


def scan(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{path}: expected a top-level runs list")
    outputs = []
    missing_text = 0
    for array_index, run in enumerate(runs):
        text = run.get("output_text")
        if not isinstance(text, str) or not text:
            missing_text += 1
            continue
        analysis = analyze_text(text)
        outputs.append(
            {
                "array_index": array_index,
                "run_index": run.get("run_index"),
                "completion_tokens": run.get("completion_tokens"),
                "finish_reason": run.get("finish_reason"),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                **analysis,
            }
        )
    flagged = [output for output in outputs if output["flagged"]]
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runs": len(runs),
        "outputs_scored": len(outputs),
        "missing_output_text": missing_text,
        "flagged_outputs": len(flagged),
        "flagged": flagged,
        "per_output": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-flagged",
        action="store_true",
        help="Exit successfully even if an output is flagged. Missing text still fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [scan(path) for path in args.inputs]
    summary = {
        "rule": {
            "repeated_8gram_fraction_gte": REPEATED_8GRAM_THRESHOLD,
            "max_consecutive_phrase_repeats_gte": CONSECUTIVE_PHRASE_THRESHOLD,
        },
        "files": files,
        "totals": {
            "runs": sum(item["runs"] for item in files),
            "outputs_scored": sum(item["outputs_scored"] for item in files),
            "missing_output_text": sum(item["missing_output_text"] for item in files),
            "flagged_outputs": sum(item["flagged_outputs"] for item in files),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["totals"], indent=2))
    if summary["totals"]["missing_output_text"]:
        raise SystemExit("Audit failed: one or more runs did not retain output_text")
    if summary["totals"]["flagged_outputs"] and not args.allow_flagged:
        print("Audit failed: degenerate outputs were flagged", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
