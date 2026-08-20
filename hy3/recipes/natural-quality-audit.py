#!/usr/bin/env python3
"""Run natural Hy3 prompts and record repetition/degeneration evidence."""

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


def analyze(text: str) -> dict:
    words = re.findall(r"\S+", text)
    folded = [word.casefold() for word in words]
    grams = [tuple(folded[index : index + 8]) for index in range(max(0, len(folded) - 7))]
    counts = Counter(grams)
    repeated_8gram_fraction = (
        sum(count - 1 for count in counts.values() if count > 1)
        / max(1, len(grams))
    )
    run = best_word_run = 0
    previous = None
    for word in folded:
        run = run + 1 if word == previous else 1
        previous = word
        best_word_run = max(best_word_run, run)
    max_character_run = max(
        (len(match.group(0)) for match in re.finditer(r"(.)\1*", text)),
        default=0,
    )
    return {
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(folded)) / max(1, len(folded)),
        "max_identical_character_run": max_character_run,
        "max_identical_word_run": best_word_run,
        "repeated_8gram_fraction": repeated_8gram_fraction,
        "repetition_flagged": (
            repeated_8gram_fraction >= 0.20
            or best_word_run >= 4
            or max_character_run >= 16
        ),
    }


async def main(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(1800.0, connect=30.0)
    base_url = f"http://{args.host}:{args.port}"
    efforts = [item.strip() for item in args.reasoning_efforts.split(",")]
    unknown = sorted(set(efforts) - {"no_think", "low", "high"})
    if unknown:
        raise SystemExit(f"unsupported reasoning efforts: {unknown}")
    jobs = [(effort, index, prompt) for effort in efforts for index, prompt in enumerate(PROMPTS)]

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        async def request(effort: str, index: int, prompt: str) -> dict:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "chat_template_kwargs": {"reasoning_effort": effort},
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
                "reasoning_effort": effort,
                "prompt_index": index,
                "prompt": prompt,
                "finish_reason": choice.get("finish_reason"),
                "usage": data.get("usage"),
                "reasoning_content": reasoning,
                "content": content,
                "sha256": hashlib.sha256(combined.encode()).hexdigest(),
                "analysis": analyze(combined),
            }

        outputs = await asyncio.gather(*(request(*job) for job in jobs))

    report = {
        "model": args.model,
        "server": base_url,
        "configuration": {
            "topology": args.topology,
            "mtp_tokens": args.mtp_tokens,
        },
        "settings": {
            "reasoning_efforts": efforts,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Hy3-FP8")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--topology", choices=("pp2", "tp2"), required=True)
    parser.add_argument("--mtp-tokens", choices=(0, 1, 2), type=int, required=True)
    parser.add_argument("--reasoning-efforts", default="no_think,low,high")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
