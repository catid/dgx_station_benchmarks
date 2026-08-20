#!/usr/bin/env python3
"""Validate the publication data and render all DeepSeek-V4-Flash charts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
CHART_ROOT = PACKAGE_ROOT / "charts"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
NATURAL_CONCURRENCIES = (1, 64, 128)
MODES = ("autoregressive", "dspark")
NATURAL_MODES = ("autoregressive", "DSpark")
THINKING_LEVELS = ("low", "high", "max")
LABELS = {"autoregressive": "AR", "dspark": "DSpark", "DSpark": "DSpark"}
COLORS = {"autoregressive": "#9aa4b2", "dspark": "#a78bfa", "DSpark": "#a78bfa"}
RED = "#f87171"
EXPECTED_C128_TPS = {
    ("autoregressive", "low"): 5808.4595783182485,
    ("autoregressive", "high"): 5807.953896180938,
    ("autoregressive", "max"): 5596.63020297993,
    ("dspark", "low"): 6511.123362317942,
    ("dspark", "high"): 6458.590824943702,
    ("dspark", "max"): 6185.39244982964,
}
EXPECTED_RAW_SOURCES = {
    "autoregressive": (
        "bc09799b824cb50f8252d754f5d5a4a3a9d9335a8baa95c7005149c8153cc6f1",
        6_784_743,
    ),
    "DSpark": (
        "d8ad82b39ac77f4ecd7b9ac212c59725e2ebbc5dbb11d4f06d78e2b383c920dc",
        6_795_178,
    ),
}
EXPECTED_PROMPT_SHA256 = (
    "3e9f7399cf5e1bdfd4fa48a0676dce2a809a67b243731461cb18c97f250b34bb"
)


def csv_rows(filename: str) -> list[dict[str, str]]:
    with (DATA_ROOT / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate_throughput() -> list[dict[str, str]]:
    rows = csv_rows("throughput.csv")
    expected = {
        (mode, thinking, concurrency)
        for mode in MODES
        for thinking in THINKING_LEVELS
        for concurrency in CONCURRENCIES
    }
    keys = [
        (row["mode"], row["thinking"], int(row["concurrency"])) for row in rows
    ]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("throughput.csv must contain exactly the 48 C1-C128 cells")

    for row in rows:
        key = (row["mode"], row["thinking"], int(row["concurrency"]))
        concurrency = key[2]
        if (
            int(row["context_tokens"]) != 8192
            or int(row["max_tokens"]) != 1024
            or float(row["temperature"]) != 0.0
            or row["ignore_eos"].lower() != "true"
            or row["disable_thinking"].lower() != "false"
            or row["reasoning_effort"] != row["thinking"]
            or int(row["request_count"]) != 5 * concurrency
            or int(row["completed_requests"]) != int(row["request_count"])
            or int(row["errors"]) != 0
            or float(row["aggregate_tps"]) <= 0
        ):
            raise ValueError(f"unexpected workload, result, or errors in cell {key}")
        if concurrency == 128 and not close(
            float(row["aggregate_tps"]), EXPECTED_C128_TPS[key[:2]]
        ):
            raise ValueError(f"published C128 throughput changed in cell {key}")
    return rows


def validate_prefill() -> list[dict[str, str]]:
    rows = csv_rows("prefill.csv")
    if [int(row["context_tokens"]) for row in rows] != [8192, 32768, 65536, 131072]:
        raise ValueError("prefill.csv must contain the ordered 8K/32K/64K/128K cells")
    for row in rows:
        if (
            row["configuration"] != "DeepSeek 8K/FP8"
            or int(row["prompt_tokens"]) != int(row["context_tokens"]) + 2
            or float(row["tok_per_sec"]) <= 0
            or float(row["ttft_seconds"]) <= 0
            or int(row["samples"]) <= 0
        ):
            raise ValueError("invalid prefill row")
    return rows


def validate_perplexity() -> dict[str, str]:
    rows = csv_rows("wikitext2-perplexity.csv")
    if len(rows) != 1:
        raise ValueError("wikitext2-perplexity.csv must contain exactly one result")
    row = rows[0]
    if (
        row["model"] != "DeepSeek-V4-Flash-0731"
        or row["cache_dtype"] != "FP8 KV"
        or int(row["documents"]) != 62
        or not close(float(row["word_perplexity"]), 6.01452934894402)
    ):
        raise ValueError("unexpected WikiText-2 perplexity result")
    return row


def validate_c128_source() -> dict[str, dict]:
    source = json.loads((DATA_ROOT / "wikitext2-natural-c128-source.json").read_text())
    if (
        source.get("schema_version") != 1
        or source["prompt"]["prompt_text_sha256"] != EXPECTED_PROMPT_SHA256
        or source["audit_derivation"]["flag_rule"]
        != "phrase_repeats >= 4 or repeated_8gram_fraction >= 0.20"
    ):
        raise ValueError("unexpected C128 provenance header")

    sources = {item["mode"]: item for item in source["sources"]}
    if set(sources) != set(NATURAL_MODES):
        raise ValueError("C128 provenance must contain AR and DSpark")
    for mode, item in sources.items():
        expected_sha256, expected_bytes = EXPECTED_RAW_SOURCES[mode]
        runs = item["runs"]
        if (
            item["raw_source_sha256"] != expected_sha256
            or int(item["raw_source_bytes"]) != expected_bytes
            or int(item["concurrency"]) != 128
            or len(runs) != 640
            or [int(run["run_index"]) for run in runs] != list(range(1, 641))
        ):
            raise ValueError(f"invalid C128 source identity or run set for {mode}")
        if any(
            int(run["prompt_tokens"]) != 8004
            or int(run["completion_tokens"]) <= 0
            or float(run["ttft_seconds"]) <= 0
            or float(run["generation_seconds"]) <= 0
            or len(run["output_sha256"]) != 64
            for run in runs
        ):
            raise ValueError(f"invalid compact C128 run for {mode}")

        summary = item["selected_summary"]
        completion_tokens = [int(run["completion_tokens"]) for run in runs]
        ttfts = [float(run["ttft_seconds"]) for run in runs]
        generation_times = [float(run["generation_seconds"]) for run in runs]
        if (
            int(summary["attempted"]) != 640
            or int(summary["completed"]) != 640
            or int(summary["errors"]) != 0
            or int(summary["hit_max_tokens"])
            != sum(tokens == 1024 for tokens in completion_tokens)
            or not close(
                float(summary["completion_tokens_avg"]),
                sum(completion_tokens) / len(completion_tokens),
            )
            or not close(float(summary["ttft_seconds_avg"]), sum(ttfts) / len(ttfts))
            or not close(
                float(summary["generation_seconds_avg"]),
                sum(generation_times) / len(generation_times),
            )
            or not close(
                float(summary["generation_tok_s_per_stream"]),
                sum(completion_tokens) / sum(generation_times),
            )
        ):
            raise ValueError(f"C128 compact summary does not match its runs for {mode}")
    return sources


def validate_natural_decode(sources: dict[str, dict]) -> list[dict[str, str]]:
    rows = csv_rows("wikitext2-natural-decode.csv")
    expected = {
        (mode, concurrency)
        for mode in NATURAL_MODES
        for concurrency in NATURAL_CONCURRENCIES
    }
    keys = [(row["mode"], int(row["concurrency"])) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("natural-decode CSV must contain exactly AR/DSpark C1/C64/C128")
    for row in rows:
        mode = row["mode"]
        concurrency = int(row["concurrency"])
        if (
            row["model"] != "DeepSeek-V4-Flash-0731"
            or int(row["completed"]) != 5 * concurrency
            or float(row["gen_tok_s_per_stream"]) <= 0
            or float(row["ttft_seconds_avg"]) <= 0
        ):
            raise ValueError(f"invalid natural-decode row {(mode, concurrency)}")
        if concurrency == 128:
            source_summary = sources[mode]["selected_summary"]
            if (
                int(row["completed"]) != int(source_summary["completed"])
                or not close(
                    float(row["completion_tokens_avg"]),
                    float(source_summary["completion_tokens_avg"]),
                )
                or not close(
                    float(row["gen_tok_s_per_stream"]),
                    float(source_summary["generation_tok_s_per_stream"]),
                )
                or not close(
                    float(row["ttft_seconds_avg"]),
                    float(source_summary["ttft_seconds_avg"]),
                )
            ):
                raise ValueError(f"natural C128 CSV does not match provenance for {mode}")
    return rows


def validate_quality_audit(
    natural_rows: list[dict[str, str]], sources: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    audit = json.loads((DATA_ROOT / "wikitext2-quality-audit.json").read_text())
    if audit["flag_rule"] != "phrase_repeats >= 4 or repeated_8gram_fraction >= 0.20":
        raise ValueError("unexpected quality-audit rule")
    expected_counts = {
        (row["mode"], int(row["concurrency"])): int(row["completed"])
        for row in natural_rows
    }
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    seen = set()
    for row in audit["outputs"]:
        key = (row["mode"], int(row["concurrency"]))
        identity = (*key, int(row["run_index"]))
        if identity in seen:
            raise ValueError(f"duplicate audit output {identity}")
        seen.add(identity)
        groups[key].append(row)
    if set(groups) != set(expected_counts):
        raise ValueError("quality audit does not cover every natural-decode cell")

    summaries = {(row["mode"], int(row["concurrency"])): row for row in audit["summary"]}
    if set(summaries) != set(expected_counts):
        raise ValueError("quality summaries do not cover every natural-decode cell")
    for key, expected_count in expected_counts.items():
        outputs = groups[key]
        if len(outputs) != expected_count:
            raise ValueError(f"quality output count mismatch for {key}")
        flagged = [
            row
            for row in outputs
            if int(row["max_consecutive_phrase_repeats"]) >= 4
            or float(row["repeated_8gram_fraction"]) >= 0.2
        ]
        summary = summaries[key]
        if (
            int(summary["outputs"]) != expected_count
            or int(summary["flagged_degenerate"]) != len(flagged)
            or not close(float(summary["flagged_fraction"]), len(flagged) / len(outputs))
            or int(summary["unique_outputs"])
            != len({row["sha256"] for row in outputs})
        ):
            raise ValueError(f"quality summary mismatch for {key}")
        if key[1] == 128:
            source_identities = {
                (int(run["run_index"]), run["finish_reason"], run["output_sha256"])
                for run in sources[key[0]]["runs"]
            }
            audit_identities = {
                (int(row["run_index"]), row["finish_reason"], row["sha256"])
                for row in outputs
            }
            if source_identities != audit_identities:
                raise ValueError(f"C128 audit identities do not match provenance for {key[0]}")
    return audit["summary"], audit["outputs"]


def read_and_validate() -> dict:
    throughput = validate_throughput()
    prefill = validate_prefill()
    perplexity = validate_perplexity()
    sources = validate_c128_source()
    natural = validate_natural_decode(sources)
    quality_summaries, quality_outputs = validate_quality_audit(natural, sources)
    return {
        "throughput": throughput,
        "prefill": prefill,
        "perplexity": perplexity,
        "natural": natural,
        "quality_summaries": quality_summaries,
        "quality_outputs": quality_outputs,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#07111f",
            "savefig.facecolor": "#07111f",
            "axes.facecolor": "#0b1728",
            "axes.edgecolor": "#435067",
            "axes.labelcolor": "#e6edf7",
            "xtick.color": "#b9c5d8",
            "ytick.color": "#b9c5d8",
            "text.color": "#f6f8fb",
            "grid.color": "#314056",
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "path.simplify": False,
        }
    )


def save(figure: plt.Figure, filename: str, rect: tuple[float, float, float, float]) -> None:
    figure.tight_layout(rect=rect)
    CHART_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        CHART_ROOT / filename,
        dpi=150,
        metadata={"Software": "dgx_station_benchmarks/render_charts.py"},
    )
    plt.close(figure)


def render_throughput(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for axis, thinking in zip(axes, THINKING_LEVELS, strict=True):
        for mode in MODES:
            cells = sorted(
                (row for row in rows if row["mode"] == mode and row["thinking"] == thinking),
                key=lambda row: int(row["concurrency"]),
            )
            x_values = [int(row["concurrency"]) for row in cells]
            y_values = [float(row["aggregate_tps"]) for row in cells]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                markersize=5,
                linewidth=2.4,
                color=COLORS[mode],
                label=LABELS[mode],
            )
            limited = [
                (x_value, y_value)
                for x_value, y_value, row in zip(x_values, y_values, cells, strict=True)
                if row["capacity_limited_flag"].lower() == "true"
            ]
            if limited:
                axis.scatter(
                    [point[0] for point in limited],
                    [point[1] for point in limited],
                    marker="x",
                    s=80,
                    linewidths=2.2,
                    color=RED,
                    zorder=5,
                )
            axis.annotate(
                f"{y_values[-1]:,.1f}",
                (x_values[-1], y_values[-1]),
                xytext=(-8, 9),
                textcoords="offset points",
                ha="right",
                color=COLORS[mode],
                fontsize=9.5,
                fontweight="bold",
            )
        axis.set_title(f"Thinking: {thinking}", loc="left", fontsize=16, pad=12)
        axis.set_xscale("log", base=2)
        axis.set_xlim(0.8, 155)
        axis.set_xticks(CONCURRENCIES)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"C{int(value)}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.set_ylim(0, 7300)
        axis.grid(True, alpha=0.42, linewidth=0.8)
        axis.set_xlabel("Offered request concurrency")
        axis.set_ylabel("Aggregate output tokens/s")

    legend = [
        Line2D(
            [0], [0], color=COLORS[mode], marker="o", linewidth=2.4,
            markersize=5, label=LABELS[mode]
        )
        for mode in MODES
    ]
    legend.append(
        Line2D(
            [0], [0], color=RED, marker="x", linestyle="None",
            markersize=8, markeredgewidth=2, label="Capacity-limited"
        )
    )
    figure.legend(handles=legend, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.895))
    figure.suptitle(
        "DeepSeek-V4-Flash-0731 on one NVIDIA GB300",
        x=0.055,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.056,
        0.90,
        "304B total / 13B active • native FP4 experts + FP8 dense • "
        "8K input / 1K output • SGLang",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.055,
        0.025,
        "Raw throughput. Natural audit: C64 flagged 8.4% AR / 11.9% DSpark; "
        "C128 flagged 10.0% in both modes.",
        color="#f59e8b",
        fontsize=10,
    )
    save(figure, "deepseek-throughput.png", (0.035, 0.07, 0.99, 0.85))


def render_prefill(rows: list[dict[str, str]]) -> None:
    contexts = [int(row["context_tokens"]) // 1024 for row in rows]
    throughput = [float(row["tok_per_sec"]) for row in rows]
    ttft = [float(row["ttft_seconds"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    specs = (
        (axes[0], throughput, "Client-observed prompt throughput", "Prompt tokens/s", "{value:,.0f}"),
        (axes[1], ttft, "Time to first token", "TTFT (seconds)", "{value:.3f}s"),
    )
    for axis, values, title, ylabel, value_format in specs:
        axis.plot(contexts, values, marker="o", markersize=7, linewidth=3, color="#a78bfa")
        for x_value, y_value in zip(contexts, values, strict=True):
            axis.annotate(
                value_format.format(value=y_value),
                (x_value, y_value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="#c4b5fd",
            )
        axis.set_title(title, loc="left", fontsize=17, pad=12)
        axis.set_xticks(contexts, [f"{value}K" for value in contexts])
        axis.set_xlabel("Prompt length")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.42, linewidth=0.8)
        axis.margins(x=0.07, y=0.16)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))

    figure.suptitle(
        "DeepSeek-V4-Flash cold prefill through 128K",
        x=0.055,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.056,
        0.90,
        "One NVIDIA GB300 • standalone cold-prefill medians • 8K chunks • FP8 hybrid cache",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.055,
        0.025,
        "SGLang warned that explicit FP8 cache scales were unavailable and used unit fallback scales.",
        color="#9fb0c8",
        fontsize=10,
    )
    save(figure, "prefill-128k.png", (0.045, 0.08, 0.99, 0.84))


def render_quality(
    perplexity: dict[str, str],
    natural_rows: list[dict[str, str]],
    quality_summaries: list[dict],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 8))

    ppl = float(perplexity["word_perplexity"])
    axes[0].bar(["FP8 hybrid KV"], [ppl], width=0.58, color="#a78bfa")
    axes[0].bar_label(axes[0].containers[0], labels=[f"{ppl:.4f}"], padding=8, fontsize=12)
    axes[0].set_title("Canonical WikiText-2 PPL", loc="left", fontsize=16, pad=12)
    axes[0].set_ylabel("Word perplexity (lower is better)")
    axes[0].set_ylim(0, 7.3)
    axes[0].grid(True, axis="y", alpha=0.42, linewidth=0.8)

    x_values = list(range(len(NATURAL_CONCURRENCIES)))
    width = 0.34
    natural_by_key = {
        (row["mode"], int(row["concurrency"])): row for row in natural_rows
    }
    summary_by_key = {
        (row["mode"], int(row["concurrency"])): row for row in quality_summaries
    }
    for index, mode in enumerate(NATURAL_MODES):
        positions = [value + (index - 0.5) * width for value in x_values]
        rates = [
            float(natural_by_key[(mode, concurrency)]["gen_tok_s_per_stream"])
            for concurrency in NATURAL_CONCURRENCIES
        ]
        bars = axes[1].bar(
            positions,
            rates,
            width=width,
            color=COLORS[mode],
            label=LABELS[mode],
        )
        axes[1].bar_label(bars, labels=[f"{value:.1f}" for value in rates], padding=4, fontsize=9)
        flagged = [
            100 * float(summary_by_key[(mode, concurrency)]["flagged_fraction"])
            for concurrency in NATURAL_CONCURRENCIES
        ]
        bars = axes[2].bar(
            positions,
            flagged,
            width=width,
            color=COLORS[mode],
            label=LABELS[mode],
        )
        axes[2].bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in flagged],
            padding=4,
            fontsize=9,
        )

    axes[1].set_title("Natural 8K continuation", loc="left", fontsize=16, pad=12)
    axes[1].set_ylabel("Generation tokens/s per stream")
    axes[1].set_xticks(x_values, [f"C{value}" for value in NATURAL_CONCURRENCIES])
    axes[1].set_ylim(0, 360)
    axes[1].grid(True, axis="y", alpha=0.42, linewidth=0.8)
    axes[1].legend(loc="upper right")

    axes[2].set_title("Repetition audit", loc="left", fontsize=16, pad=12)
    axes[2].set_ylabel("Flagged outputs")
    axes[2].set_xticks(x_values, [f"C{value}" for value in NATURAL_CONCURRENCIES])
    axes[2].set_ylim(0, 14)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    axes[2].grid(True, axis="y", alpha=0.42, linewidth=0.8)
    axes[2].legend(loc="upper left")

    figure.suptitle(
        "DeepSeek-V4-Flash quality + natural-text decode",
        x=0.055,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.056,
        0.91,
        "EleutherAI WikiText-2 • one NVIDIA GB300 • EOS respected • up to 1,024 new tokens",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.055,
        0.025,
        "Natural rate = sum(tokens) / sum(per-request generation time). Audit flags ≥4 phrase repeats "
        "or repeated 8-gram fraction ≥0.20.",
        color="#9fb0c8",
        fontsize=10,
    )
    save(figure, "wikitext2-quality-decode.png", (0.04, 0.075, 0.99, 0.85))


def main() -> None:
    configure_style()
    data = read_and_validate()
    render_throughput(data["throughput"])
    render_prefill(data["prefill"])
    render_quality(data["perplexity"], data["natural"], data["quality_summaries"])


if __name__ == "__main__":
    main()
