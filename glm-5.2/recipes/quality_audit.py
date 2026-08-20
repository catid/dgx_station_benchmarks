#!/usr/bin/env python3
"""Generate natural GLM answers and flag obvious repetition or empty output."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import httpx

PROMPTS = [
    "Explain why the daytime sky appears blue and sunsets often appear red. Be precise but concise.",
    "A box contains 4 red, 5 blue, and 6 green balls. Two are drawn without replacement. What is the probability they have different colors? Show your reasoning.",
    "Write a correct Python function that merges overlapping closed integer intervals, then briefly explain its time complexity.",
    "Compare tensor parallelism and pipeline parallelism for serving a large language model across two networked GPUs. Discuss latency and throughput tradeoffs.",
]


def analyze(text: str) -> dict[str, float | int | bool]:
    words = re.findall(r"\S+", text)
    folded = [word.casefold() for word in words]
    grams = [tuple(folded[i : i + 8]) for i in range(max(0, len(folded) - 7))]
    counts = Counter(grams)
    previous = None
    run = best_word_run = 0
    for word in folded:
        run = run + 1 if word == previous else 1
        previous = word
        best_word_run = max(best_word_run, run)
    character_run = max(
        (len(match.group(0)) for match in re.finditer(r"(\S)\1*", text)),
        default=0,
    )
    repeated_fraction = sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))
    flagged = not text.strip() or repeated_fraction >= 0.20 or best_word_run >= 4 or character_run >= 16
    return {
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(folded)) / max(1, len(folded)),
        "max_identical_character_run": character_run,
        "max_identical_word_run": best_word_run,
        "repeated_8gram_fraction": repeated_fraction,
        "flagged": flagged,
    }


async def run(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"
    timeout = httpx.Timeout(1800.0, connect=30.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        async def request(index: int, prompt: str) -> dict:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            content = message.get("content") or ""
            combined = "\n".join(part for part in (reasoning, content) if part)
            return {
                "prompt_index": index,
                "prompt": prompt,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "reasoning_content": reasoning,
                "content": content,
                "sha256": hashlib.sha256(combined.encode()).hexdigest(),
                "analysis": analyze(combined),
            }

        outputs = await asyncio.gather(
            *(request(index, prompt) for index, prompt in enumerate(PROMPTS))
        )

    report = {
        "model": args.model,
        "server": base_url,
        "settings": {"temperature": 0, "max_tokens": args.max_tokens},
        "outputs": outputs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "quality-audit.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if any(output["analysis"]["flagged"] for output in outputs):
        print(f"WARNING: one or more outputs need review in {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GLM-5.2-NVFP4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("results/quality"))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
