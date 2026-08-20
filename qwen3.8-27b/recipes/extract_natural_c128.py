#!/usr/bin/env python3
"""Recover the published Qwen natural C128 rows from retained raw results.

The raw benchmark JSON contains generated text and machine diagnostics, so it is
not copied into the public package. This extractor verifies the five exact raw
files, writes compact per-run provenance without text, and deterministically
merges the C128 measurements into the public CSV and repetition audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
MODEL = "Qwen3.8-27B"
FLAG_RULE = "phrase_repeats >= 4 or repeated_8gram_fraction >= 0.20"
MODE_ORDER = (
    "autoregressive",
    "DFlash1 (community)",
    "DFlash2",
    "DSpark",
    "MTP",
)
SOURCES = {
    "autoregressive": {
        "display_mode": "autoregressive",
        "sha256": "eabebc3557fa4b06766720a30f540c15f8a322da841109ba05995b923a28c44d",
        "bytes": 5_742_504,
    },
    "dflash1-community": {
        "display_mode": "DFlash1 (community)",
        "sha256": "f527190a24fe85ff515b18a79d99fe66bfa242aa9b1fe1090cd761e847e39653",
        "bytes": 5_908_430,
    },
    "dflash2": {
        "display_mode": "DFlash2",
        "sha256": "ee7054779a6eb2d818dd2b7543b26dab0838565fb8f6bab0406649bd6bbc46b7",
        "bytes": 5_636_140,
    },
    "dspark": {
        "display_mode": "DSpark",
        "sha256": "21838ccee3a460eb3938be2d0d68f334d14c7db954f979a1516548b6e14f4d5e",
        "bytes": 5_766_292,
    },
    "mtp": {
        "display_mode": "MTP",
        "sha256": "4cefb22bdc7e7d1126c98c84e7df6a1df9e9ba96f01c32239663767908b7683c",
        "bytes": 5_936_580,
    },
}


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )

    def validate_finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"non-finite JSON number in {path}")
        if isinstance(item, dict):
            for child in item.values():
                validate_finite(child)
        elif isinstance(item, list):
            for child in item:
                validate_finite(child)

    validate_finite(value)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def analyze_text(text: str) -> dict[str, int | float]:
    words = re.findall(r"\S+", text)
    lowered = [word.casefold() for word in words]
    grams = [tuple(lowered[index : index + 8]) for index in range(max(0, len(lowered) - 7))]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)

    best_word_run = 0
    current = 0
    previous = None
    for word in lowered:
        current = current + 1 if word == previous else 1
        previous = word
        best_word_run = max(best_word_run, current)

    token_vocabulary: dict[str, int] = {}
    token_ids = []
    for word in lowered:
        if word not in token_vocabulary:
            token_vocabulary[word] = len(token_vocabulary) + 1
        token_ids.append(token_vocabulary[word])
    mask = (1 << 64) - 1
    base = 1_000_003
    prefix = [0]
    powers = [1]
    for token_id in token_ids:
        prefix.append((prefix[-1] * base + token_id + 1) & mask)
        powers.append((powers[-1] * base) & mask)

    def block_hash(start: int, width: int) -> int:
        return (prefix[start + width] - prefix[start] * powers[width]) & mask

    best_phrase_repeats = 1
    for width in range(2, min(65, len(lowered) // 2 + 1)):
        for offset in range(width):
            previous_start = None
            previous_hash = None
            repeats = 0
            for start in range(offset, len(lowered) - width + 1, width):
                current_hash = block_hash(start, width)
                same = (
                    previous_start is not None
                    and current_hash == previous_hash
                    and lowered[start : start + width]
                    == lowered[previous_start : previous_start + width]
                )
                repeats = repeats + 1 if same else 1
                best_phrase_repeats = max(best_phrase_repeats, repeats)
                previous_start = start
                previous_hash = current_hash

    return {
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(lowered)) / max(1, len(lowered)),
        "max_identical_word_run": best_word_run,
        "max_identical_character_run": max(
            (len(match.group(0)) for match in re.finditer(r"(.)\1*", text)),
            default=0,
        ),
        "repeated_8gram_fraction": repeated / max(1, len(grams)),
        "max_consecutive_phrase_repeats": best_phrase_repeats,
    }


def validate_source(path: Path, source: dict[str, object]) -> dict:
    payload = path.read_bytes()
    if len(payload) != source["bytes"]:
        raise ValueError(f"unexpected source size for {path}")
    if hashlib.sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError(f"unexpected source SHA-256 for {path}")
    data = strict_json(path)
    metadata = data["metadata"]
    runs = data["runs"]
    summary = data["selected_summary"]
    prefill = data["prefill_scout"]
    if (
        metadata["version"] != "0.4.29"
        or metadata["engine"] != "sglang"
        or metadata["model"] != MODEL
        or int(metadata["fixed_concurrency"]) != 128
        or int(metadata["requested_runs"]) != 640
        or metadata["concurrency_levels_requested"] != [128]
        or int(metadata["max_tokens"]) != 1024
        or float(metadata["temperature"]) != 0.0
        or metadata["disable_thinking"] is not True
        or metadata["reasoning_effort"] is not None
        or int(data["selected_concurrency"]) != 128
        or len(runs) != 640
    ):
        raise ValueError(f"unexpected workload metadata in {path}")
    if any(
        int(run["run_index"]) != index
        or int(run["concurrency"]) != 128
        or run["ok"] is not True
        or run["error"]
        or int(run["prompt_tokens"]) != 8011
        or int(run["completion_tokens"]) <= 0
        or float(run["gen_elapsed"]) <= 0
        or float(run["ttft"]) <= 0
        or not (run.get("output_text") or run.get("content_text"))
        for index, run in enumerate(runs, start=1)
    ):
        raise ValueError(f"invalid or incomplete run set in {path}")

    completion_tokens = [int(run["completion_tokens"]) for run in runs]
    generation_seconds = [float(run["gen_elapsed"]) for run in runs]
    ttft_seconds = [float(run["ttft"]) for run in runs]
    if (
        int(summary["attempted"]) != 640
        or int(summary["completed"]) != 640
        or int(summary["errors"]) != 0
        or int(summary["hit_max_tokens"]) != sum(tokens == 1024 for tokens in completion_tokens)
        or not close(float(summary["completion_tokens"]["avg"]), sum(completion_tokens) / 640)
        or not close(float(summary["gen_elapsed"]["avg"]), sum(generation_seconds) / 640)
        or not close(float(summary["ttft"]["avg"]), sum(ttft_seconds) / 640)
        or not close(
            float(summary["aggregate_gen_tok_s"]),
            sum(completion_tokens) / sum(generation_seconds),
        )
        or prefill["ok"] is not True
        or int(prefill["prompt_tokens"]) != 8011
        or float(prefill["ttft"]) <= 0
    ):
        raise ValueError(f"selected summary does not match retained runs in {path}")
    return data


def audit_output(mode: str, run: dict, analysis_cache: dict[str, dict]) -> dict:
    text = run.get("output_text") or run.get("content_text") or ""
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest not in analysis_cache:
        analysis_cache[digest] = analyze_text(text)
    return {
        "model": MODEL,
        "mode": mode,
        "concurrency": 128,
        "run_index": int(run["run_index"]),
        "finish_reason": run["finish_reason"],
        "sha256": digest,
        **analysis_cache[digest],
    }


def summary_for(outputs: list[dict]) -> dict:
    flagged = [
        row
        for row in outputs
        if int(row["max_consecutive_phrase_repeats"]) >= 4
        or float(row["repeated_8gram_fraction"]) >= 0.2
    ]
    return {
        "model": outputs[0]["model"],
        "mode": outputs[0]["mode"],
        "concurrency": int(outputs[0]["concurrency"]),
        "outputs": len(outputs),
        "flagged_degenerate": len(flagged),
        "flagged_fraction": len(flagged) / len(outputs),
        "unique_outputs": len({row["sha256"] for row in outputs}),
        "min_unique_word_ratio": min(float(row["unique_word_ratio"]) for row in outputs),
        "max_repeated_8gram_fraction": max(
            float(row["repeated_8gram_fraction"]) for row in outputs
        ),
        "max_identical_word_run": max(int(row["max_identical_word_run"]) for row in outputs),
        "max_identical_character_run": max(
            int(row["max_identical_character_run"]) for row in outputs
        ),
        "max_consecutive_phrase_repeats": max(
            int(row["max_consecutive_phrase_repeats"]) for row in outputs
        ),
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    path = DATA_ROOT / "wikitext2-natural-decode.csv"
    fieldnames = [
        "model",
        "mode",
        "concurrency",
        "completed",
        "completion_tokens_avg",
        "hit_max_tokens",
        "gen_tok_s_per_stream",
        "ttft_seconds_avg",
        "prefill_scout_tok_s",
        "prefill_scout_ttft_seconds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_root",
        type=Path,
        help="directory containing {autoregressive,dflash1-community,dflash2,dspark,mtp}/c128.json",
    )
    args = parser.parse_args()

    csv_path = DATA_ROOT / "wikitext2-natural-decode.csv"
    with csv_path.open(newline="") as handle:
        existing_rows = list(csv.DictReader(handle))
    existing_rows = [row for row in existing_rows if int(row["concurrency"]) != 128]
    expected_existing = {(mode, concurrency) for mode in MODE_ORDER for concurrency in (1, 64)}
    existing_keys = {(row["mode"], int(row["concurrency"])) for row in existing_rows}
    if len(existing_rows) != len(expected_existing) or existing_keys != expected_existing:
        raise ValueError("existing natural-decode CSV must contain exactly Qwen C1/C64")

    audit_path = DATA_ROOT / "wikitext2-quality-audit.json"
    audit = strict_json(audit_path)
    if audit["flag_rule"] != FLAG_RULE:
        raise ValueError("unexpected existing repetition-audit rule")
    retained_outputs = [
        row for row in audit["outputs"] if int(row["concurrency"]) != 128
    ]
    retained_keys = {(row["mode"], int(row["concurrency"])) for row in retained_outputs}
    if retained_keys != expected_existing:
        raise ValueError("existing audit must contain exactly Qwen C1/C64 outputs")

    analysis_cache: dict[str, dict] = {}
    new_rows = []
    new_outputs = []
    compact_sources = []
    for source_mode, source in SOURCES.items():
        path = args.source_root / source_mode / "c128.json"
        data = validate_source(path, source)
        mode = str(source["display_mode"])
        summary = data["selected_summary"]
        prefill = data["prefill_scout"]
        outputs = [audit_output(mode, run, analysis_cache) for run in data["runs"]]
        if summary_for(outputs)["flagged_degenerate"] != 0:
            raise ValueError(f"published zero-flag C128 claim changed for {mode}")
        new_outputs.extend(outputs)
        new_rows.append(
            {
                "model": MODEL,
                "mode": mode,
                "concurrency": 128,
                "completed": int(summary["completed"]),
                "completion_tokens_avg": summary["completion_tokens"]["avg"],
                "hit_max_tokens": int(summary["hit_max_tokens"]),
                "gen_tok_s_per_stream": summary["aggregate_gen_tok_s"],
                "ttft_seconds_avg": summary["ttft"]["avg"],
                "prefill_scout_tok_s": int(prefill["prompt_tokens"]) / float(prefill["ttft"]),
                "prefill_scout_ttft_seconds": prefill["ttft"],
            }
        )
        compact_sources.append(
            {
                "mode": mode,
                "raw_source": f"qwen3.8-27b/{source_mode}/c128.json",
                "raw_source_sha256": source["sha256"],
                "raw_source_bytes": source["bytes"],
                "concurrency": 128,
                "selected_summary": {
                    "attempted": int(summary["attempted"]),
                    "completed": int(summary["completed"]),
                    "errors": int(summary["errors"]),
                    "hit_max_tokens": int(summary["hit_max_tokens"]),
                    "completion_tokens_avg": summary["completion_tokens"]["avg"],
                    "ttft_seconds_avg": summary["ttft"]["avg"],
                    "generation_seconds_avg": summary["gen_elapsed"]["avg"],
                    "generation_tok_s_per_stream": summary["aggregate_gen_tok_s"],
                    "prefill_scout_tok_s": int(prefill["prompt_tokens"]) / float(prefill["ttft"]),
                    "prefill_scout_ttft_seconds": prefill["ttft"],
                },
                "runs": [
                    {
                        "run_index": int(run["run_index"]),
                        "prompt_tokens": int(run["prompt_tokens"]),
                        "completion_tokens": int(run["completion_tokens"]),
                        "ttft_seconds": run["ttft"],
                        "generation_seconds": run["gen_elapsed"],
                        "finish_reason": run["finish_reason"],
                        "output_sha256": output["sha256"],
                    }
                    for run, output in zip(data["runs"], outputs, strict=True)
                ],
            }
        )

    mode_index = {mode: index for index, mode in enumerate(MODE_ORDER)}
    all_rows = [*existing_rows, *new_rows]
    all_rows.sort(key=lambda row: (mode_index[str(row["mode"])], int(row["concurrency"])))
    write_csv(all_rows)

    all_outputs = [*retained_outputs, *new_outputs]
    all_outputs.sort(
        key=lambda row: (
            mode_index[str(row["mode"])],
            int(row["concurrency"]),
            int(row["run_index"]),
        )
    )
    summaries = []
    for mode in MODE_ORDER:
        for concurrency in (1, 64, 128):
            subset = [
                row
                for row in all_outputs
                if row["mode"] == mode and int(row["concurrency"]) == concurrency
            ]
            expected = 5 * concurrency
            if len(subset) != expected:
                raise ValueError(f"expected {expected} audit outputs for {mode} C{concurrency}")
            summaries.append(summary_for(subset))
    audit_path.write_text(
        json.dumps(
            {
                "flag_rule": FLAG_RULE,
                "summary": summaries,
                "diagnostics": audit["diagnostics"],
                "outputs": all_outputs,
            },
            indent=2,
        )
        + "\n"
    )

    compact = {
        "schema_version": 1,
        "model": MODEL,
        "method": {
            "benchmark": "llm-inference-bench 0.4.29",
            "engine": "SGLang",
            "prompt_tokens": 8011,
            "output_tokens_max": 1024,
            "temperature": 0.0,
            "thinking_disabled": True,
            "respect_eos": True,
            "measured_requests": 640,
            "concurrency": 128,
        },
        "audit_derivation": {
            "flag_rule": FLAG_RULE,
            "generated_text_retained_in_public_file": False,
            "identity": "run_index + finish_reason + SHA-256(output_text)",
        },
        "sources": compact_sources,
    }
    (DATA_ROOT / "wikitext2-natural-c128-source.json").write_text(
        json.dumps(compact, indent=2) + "\n"
    )

    print("recovered 5 C128 rows and 3,200 audited output identities")


if __name__ == "__main__":
    main()
