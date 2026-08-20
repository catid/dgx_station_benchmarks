#!/usr/bin/env python3
"""Build the deterministic natural WikiText-2 continuation prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from lm_eval.tasks.wikitext.preprocess_wikitext import wikitext_detokenizer
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=8_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    dataset = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="test",
    )
    instruction = (
        "The following is reference text from WikiText-2. Continue it naturally in the "
        "same encyclopedic style. Do not discuss this instruction and do not repeat the "
        "existing text verbatim.\n\n"
    )
    pieces: list[str] = []
    selected_rows: list[int] = []
    for row_index, document in enumerate(dataset):
        detokenized = wikitext_detokenizer(document)
        candidate = instruction + "\n\n".join(pieces + [detokenized])
        candidate_tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if len(candidate_tokens) > args.target_tokens:
            current = instruction + "\n\n".join(pieces)
            remaining = args.target_tokens - len(tokenizer.encode(current, add_special_tokens=False))
            if remaining > 0:
                separator = "\n\n" if pieces else ""
                doc_tokens = tokenizer.encode(separator + detokenized, add_special_tokens=False)
                pieces.append(tokenizer.decode(doc_tokens[:remaining], skip_special_tokens=True))
                selected_rows.append(row_index)
            break
        pieces.append(detokenized)
        selected_rows.append(row_index)

    prompt = instruction + "\n\n".join(pieces)
    token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    encoded = prompt.encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    args.metadata_output.write_text(
        json.dumps(
            {
                "dataset": "EleutherAI/wikitext_document_level",
                "dataset_config": "wikitext-2-raw-v1",
                "split": "test",
                "tokenizer": args.tokenizer,
                "target_tokens": args.target_tokens,
                "actual_raw_prompt_tokens": token_count,
                "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
                "selected_row_indices": selected_rows,
                "instruction": instruction,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} with {token_count:,} raw prompt tokens")


if __name__ == "__main__":
    main()
