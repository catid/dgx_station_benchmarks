#!/usr/bin/env python3
"""Capture full outputs and report simple loop/repetition indicators."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx


def analyze(text: str) -> dict:
    words = re.findall(r"\S+", text)
    folded = [word.casefold() for word in words]
    result = {
        "characters": len(text),
        "words": len(words),
        "unique_word_ratio": len(set(folded)) / max(1, len(folded)),
        "max_identical_character_run": max(
            (len(match.group(0)) for match in re.finditer(r"(.)\1*", text)),
            default=0,
        ),
    }
    previous = None
    run = best = 0
    for word in folded:
        run = run + 1 if word == previous else 1
        previous = word
        best = max(best, run)
    result["max_identical_word_run"] = best
    for width in (2, 4, 8):
        grams = [tuple(folded[index : index + width]) for index in range(len(folded) - width + 1)]
        counts = Counter(grams)
        result[f"repeated_{width}gram_fraction"] = (
            sum(count - 1 for count in counts.values() if count > 1)
            / max(1, len(grams))
        )
        if width == 8 and counts:
            gram, count = counts.most_common(1)[0]
            result["most_common_8gram"] = " ".join(gram)
            result["most_common_8gram_count"] = count
    return result


def degeneration_flag(analysis: dict) -> bool:
    return (
        analysis["repeated_8gram_fraction"] >= 0.20
        or analysis["max_identical_word_run"] >= 4
        or analysis["max_identical_character_run"] >= 16
    )


async def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.bench_dir))
    from llm_decode_bench import build_messages, generate_padding_text

    timeout = httpx.Timeout(1800.0, connect=30.0)
    base_url = f"http://{args.host}:{args.port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        padding = generate_padding_text(args.context_tokens * 4)
        low, high = 0, args.context_tokens * 10
        best_text, best_count = "", 0
        while low <= high:
            middle = (low + high) // 2
            candidate = padding[:middle]
            response = await client.post(
                "/tokenize",
                json={
                    "model": args.model,
                    "messages": build_messages(args.context_tokens, candidate),
                },
            )
            response.raise_for_status()
            count = int(response.json()["count"])
            if abs(count - args.context_tokens) < abs(best_count - args.context_tokens):
                best_text, best_count = candidate, count
            if count < args.context_tokens:
                low = middle + 1
            elif count > args.context_tokens:
                high = middle - 1
            else:
                best_text, best_count = candidate, count
                break

        if best_count != args.context_tokens:
            raise RuntimeError(f"prompt calibration got {best_count}, wanted {args.context_tokens}")
        payload = {
            "model": args.model,
            "messages": build_messages(args.context_tokens, best_text),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if args.ignore_eos:
            payload["ignore_eos"] = True

        async def request(index: int) -> dict:
            started = time.perf_counter()
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            content = message.get("content") or ""
            text = "\n".join(part for part in (reasoning, content) if part)
            analysis = analyze(text)
            return {
                "index": index,
                "elapsed_seconds": time.perf_counter() - started,
                "finish_reason": data["choices"][0].get("finish_reason"),
                "usage": data.get("usage"),
                "reasoning_content": reasoning,
                "content": content,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "analysis": analysis,
                "automatic_degeneration_flag": degeneration_flag(analysis),
            }

        outputs = await asyncio.gather(*(request(index) for index in range(args.requests)))

    report = {
        "model": args.model,
        "server": base_url,
        "prompt_tokens": best_count,
        "settings": {
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
        },
        "summary": {
            "flag_rule": (
                "repeated_8gram_fraction >= 0.20, identical-word run >= 4, "
                "or identical-character run >= 16"
            ),
            "flagged": sum(output["automatic_degeneration_flag"] for output in outputs),
            "manual_review_required": True,
        },
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["summary"]["flagged"]:
        raise SystemExit("automatic degeneration rule flagged one or more outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    asyncio.run(run(parser.parse_args()))
