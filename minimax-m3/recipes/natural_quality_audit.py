#!/usr/bin/env python3
"""Retain MiniMax M3 natural outputs and flag mechanical degeneration."""

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
    "Describe how sparse attention can reduce long-context inference cost, including one limitation or tradeoff.",
]


def analyze(text: str) -> dict[str, float | int | bool]:
    words = re.findall(r"\S+", text)
    folded = [word.casefold() for word in words]
    grams = [tuple(folded[index : index + 8]) for index in range(max(0, len(folded) - 7))]
    counts = Counter(grams)
    repeated_fraction = sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))
    previous = None
    run = best_word_run = 0
    for word in folded:
        run = run + 1 if word == previous else 1
        previous = word
        best_word_run = max(best_word_run, run)
    character_run = max((len(match.group(0)) for match in re.finditer(r"(.)\1*", text)), default=0)
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
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    base_url = f"http://{args.host}:{args.port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        async def request(index: int) -> dict:
            prompt = PROMPTS[index % len(PROMPTS)]
            async with semaphore:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_tokens": args.max_tokens,
                        "chat_template_kwargs": {"thinking_mode": args.thinking_mode},
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
                "request_index": index,
                "prompt": prompt,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "reasoning_content": reasoning,
                "content": content,
                "sha256": hashlib.sha256(combined.encode()).hexdigest(),
                "analysis": analyze(combined),
            }

        outputs = await asyncio.gather(*(request(index) for index in range(args.requests)))

    report = {
        "model": args.model,
        "server": base_url,
        "settings": {
            "thinking_mode": args.thinking_mode,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "outputs": len(outputs),
            "empty": sum(not (item["content"] or item["reasoning_content"]) for item in outputs),
            "flagged": sum(bool(item["analysis"]["flagged"]) for item in outputs),
        },
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MiniMax-M3-NVFP4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--thinking-mode", choices=("disabled", "adaptive", "enabled"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests is None:
        args.requests = max(args.concurrency, len(PROMPTS))
    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
