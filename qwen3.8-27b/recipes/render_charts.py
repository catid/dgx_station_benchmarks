#!/usr/bin/env python3
"""Validate package data and render the four main Qwen publication charts."""

from __future__ import annotations

import csv
import json
import math
import re
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
MODES = ("autoregressive", "dflash1-community", "dflash2", "dspark", "mtp")
NATURAL_MODES = ("autoregressive", "DFlash1 (community)", "DFlash2", "DSpark", "MTP")
THINKING_LEVELS = ("none", "low", "medium", "xhigh")
LABELS = {
    "autoregressive": "AR",
    "dflash1-community": "DFlash1 community",
    "dflash2": "DFlash2",
    "dspark": "DSpark",
    "mtp": "MTP",
    "DFlash1 (community)": "DFlash1 community",
    "DFlash2": "DFlash2",
    "DSpark": "DSpark",
    "MTP": "MTP",
}
COLORS = {
    "autoregressive": "#9aa4b2",
    "dflash1-community": "#f59e0b",
    "dflash2": "#22d3ee",
    "dspark": "#a78bfa",
    "mtp": "#34d399",
    "DFlash1 (community)": "#f59e0b",
    "DFlash2": "#22d3ee",
    "DSpark": "#a78bfa",
    "MTP": "#34d399",
}
PREFILL_CONFIGURATIONS = {
    "Qwen official 2K/FP8": ("Official 2K / FP8", "#64748b"),
    "Qwen 8K/BF16": ("8K / BF16", "#f59e0b"),
    "Qwen 8K/FP8": ("8K / FP8", "#22d3ee"),
    "Qwen 16K/FP8 eager": ("16K / FP8 eager", "#34d399"),
    "Qwen 32K/FP8 eager": ("32K / FP8 eager", "#a78bfa"),
}
RED = "#f87171"
FLAG_RULE = "phrase_repeats >= 4 or repeated_8gram_fraction >= 0.20"
EXPECTED_FIXED_C128_SHA256 = {
    ("autoregressive", "low"): "511ccbe6d2004231f48bad1e07658993e6f338dfa63597f7041e6d011634f2f8",
    ("autoregressive", "medium"): "6db57f16ecc5d1ec0451dc669c3ea0b2a2ccf8beab13c2d1e699eeb037ba2022",
    ("autoregressive", "none"): "ac0c3bb917dd1888a7031becf0eb9dd5e214796bbd2648daf229dbd24fd254b9",
    ("autoregressive", "xhigh"): "7179cc6a9720896d6dcdef7bb2e5a5359de62282b7db3601718ce3eea97f23e6",
    ("dflash1-community", "low"): "adf92339d64b1a7bb8503c48374a029e2c78a2f5c09c179785a8bb5b012673a3",
    ("dflash1-community", "medium"): "4c87a82b4abc599177c4417242898b13e400cdf16bc633b2dffbaa89096e7077",
    ("dflash1-community", "none"): "2335905549249e1b727011020047d70edb28c0b0865fc745cb00670a7eda9da2",
    ("dflash1-community", "xhigh"): "a27763bfa14b91f3ab5ddc83cc3553d2e63bf60eab57b33cbe31e2596b654314",
    ("dflash2", "low"): "2fb7be00653d74e9994b188260d5428c14cbb99c9c439903cf1629a4e5ceca54",
    ("dflash2", "medium"): "1ed984d4ed130a9e5404a7a917a957a21305a79b649c4decba825acc6ab55f16",
    ("dflash2", "none"): "ef3176e61743bfc398e026b4c85627ad707d8e54db86506ff737a9df0d74af79",
    ("dflash2", "xhigh"): "cc65c5453b516ff1bdf8de4a3d6c2043afcd0deb0e63f5b6e8538a6971e14fdf",
    ("dspark", "low"): "6ae8caddf2c46ce73e60da703db4161cd26f342044efca48c306d363a6fb3140",
    ("dspark", "medium"): "e3919822988ea8d44b74c866461aa01fd131a9658b3cef7a59a94298e43f9748",
    ("dspark", "none"): "ec79593c7375be42606b89ceacaabf440e8d5629a9bca838dc366f5945c492e6",
    ("dspark", "xhigh"): "81fce6d653cb89f17181ed1e26b01597f0d80acee2fef10922a842266745b27d",
    ("mtp", "low"): "d06d5d5751f2e58764a37cdaa8b8dae94ba293c42e2e7b69b849580c955ada6c",
    ("mtp", "medium"): "76a80664a2d89a01d37d349a5505d7e334cf268bc572373ce25ccf6437145ee0",
    ("mtp", "none"): "72d91e97d915e63df03eaf757ab3a235506a5cd2b36844c695766904e808f5c0",
    ("mtp", "xhigh"): "a4d93de64f60ad4ac76b1bfeaa18578586dfa84c72d319708385403900335b3e",
}
EXPECTED_NATURAL_C128_SOURCES = {
    "autoregressive": ("eabebc3557fa4b06766720a30f540c15f8a322da841109ba05995b923a28c44d", 5_742_504),
    "DFlash1 (community)": ("f527190a24fe85ff515b18a79d99fe66bfa242aa9b1fe1090cd761e847e39653", 5_908_430),
    "DFlash2": ("ee7054779a6eb2d818dd2b7543b26dab0838565fb8f6bab0406649bd6bbc46b7", 5_636_140),
    "DSpark": ("21838ccee3a460eb3938be2d0d68f334d14c7db954f979a1516548b6e14f4d5e", 5_766_292),
    "MTP": ("4cefb22bdc7e7d1126c98c84e7df6a1df9e9ba96f01c32239663767908b7683c", 5_936_580),
}


def csv_rows(filename: str) -> list[dict[str, str]]:
    with (DATA_ROOT / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate_fixed_c128_provenance(rows: list[dict[str, str]]) -> None:
    provenance = json.loads((DATA_ROOT / "official-bf16-c128-provenance.json").read_text())
    if provenance.get("schema_version") != 1:
        raise ValueError("unexpected fixed-length C128 provenance schema")
    sources = provenance["sources"]
    for key, expected_sha256 in EXPECTED_FIXED_C128_SHA256.items():
        mode, thinking = key
        source = sources[mode][thinking]
        if source["sha256"] != expected_sha256 or not re.fullmatch(
            r"[0-9a-f]{64}", source["sha256"]
        ):
            raise ValueError(f"unexpected fixed C128 source identity for {key}")
        matches = [
            row
            for row in rows
            if row["mode"] == mode
            and row["thinking"] == thinking
            and int(row["concurrency"]) == 128
        ]
        if len(matches) != 1 or not close(
            float(matches[0]["aggregate_tps"]), float(source["aggregate_tps"])
        ):
            raise ValueError(f"fixed C128 CSV/provenance mismatch for {key}")


def validate_throughput() -> list[dict[str, str]]:
    all_rows = csv_rows("throughput.csv")
    rows = [row for row in all_rows if row["mode"] in MODES]
    expected = {
        (mode, thinking, concurrency)
        for mode in MODES
        for thinking in THINKING_LEVELS
        for concurrency in CONCURRENCIES
    }
    keys = [(row["mode"], row["thinking"], int(row["concurrency"])) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("throughput.csv must contain exactly 160 official BF16 cells")
    for row in rows:
        mode, thinking, concurrency = row["mode"], row["thinking"], int(row["concurrency"])
        thinking_disabled = thinking == "none"
        expected_effort = "" if thinking_disabled else thinking
        if (
            int(row["context_tokens"]) != 8192
            or int(row["max_tokens"]) != 1024
            or float(row["temperature"]) != 0.0
            or row["ignore_eos"].lower() != "true"
            or (row["disable_thinking"].lower() == "true") != thinking_disabled
            or row["reasoning_effort"] != expected_effort
            or int(row["request_count"]) != 5 * concurrency
            or int(row["completed_requests"]) != int(row["request_count"])
            or int(row["errors"]) != 0
            or float(row["aggregate_tps"]) <= 0
        ):
            raise ValueError(f"unexpected workload or result in {(mode, thinking, concurrency)}")
    validate_fixed_c128_provenance(rows)
    return rows


def validate_prefill() -> list[dict[str, str]]:
    all_rows = csv_rows("prefill.csv")
    rows = [row for row in all_rows if row["configuration"] in PREFILL_CONFIGURATIONS]
    expected = {
        (configuration, context)
        for configuration in PREFILL_CONFIGURATIONS
        for context in (8192, 32768, 65536, 131072)
    }
    keys = [(row["configuration"], int(row["context_tokens"])) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("prefill.csv must contain all 20 official Qwen cells")
    for row in rows:
        if (
            int(row["prompt_tokens"]) != int(row["context_tokens"]) + 2
            or float(row["tok_per_sec"]) <= 0
            or float(row["ttft_seconds"]) <= 0
            or int(row["samples"]) <= 0
        ):
            raise ValueError(f"invalid prefill row {row['configuration']}")
    return rows


def validate_perplexity() -> dict[str, str]:
    rows = [row for row in csv_rows("wikitext2-perplexity.csv") if row["model"] == "Qwen3.8-27B"]
    if len(rows) != 1:
        raise ValueError("expected one official Qwen WikiText-2 result")
    row = rows[0]
    if (
        row["cache_dtype"] != "BF16 KV/Mamba"
        or int(row["documents"]) != 62
        or not close(float(row["word_perplexity"]), 9.294200611359534)
    ):
        raise ValueError("unexpected official Qwen WikiText-2 result")
    return row


def validate_natural_source() -> dict[str, dict]:
    compact = json.loads((DATA_ROOT / "wikitext2-natural-c128-source.json").read_text())
    if (
        compact.get("schema_version") != 1
        or compact.get("model") != "Qwen3.8-27B"
        or compact["audit_derivation"]["flag_rule"] != FLAG_RULE
        or int(compact["method"]["prompt_tokens"]) != 8011
        or int(compact["method"]["measured_requests"]) != 640
        or int(compact["method"]["concurrency"]) != 128
    ):
        raise ValueError("unexpected natural C128 compact provenance header")
    sources = {source["mode"]: source for source in compact["sources"]}
    if set(sources) != set(NATURAL_MODES):
        raise ValueError("natural C128 provenance must contain all five modes")
    for mode, source in sources.items():
        expected_sha256, expected_bytes = EXPECTED_NATURAL_C128_SOURCES[mode]
        runs = source["runs"]
        if (
            source["raw_source_sha256"] != expected_sha256
            or int(source["raw_source_bytes"]) != expected_bytes
            or int(source["concurrency"]) != 128
            or len(runs) != 640
            or [int(run["run_index"]) for run in runs] != list(range(1, 641))
        ):
            raise ValueError(f"invalid natural C128 source identity or run set for {mode}")
        if any(
            int(run["prompt_tokens"]) != 8011
            or int(run["completion_tokens"]) <= 0
            or float(run["ttft_seconds"]) <= 0
            or float(run["generation_seconds"]) <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", run["output_sha256"])
            for run in runs
        ):
            raise ValueError(f"invalid compact natural C128 run for {mode}")
        summary = source["selected_summary"]
        tokens = [int(run["completion_tokens"]) for run in runs]
        generation = [float(run["generation_seconds"]) for run in runs]
        ttft = [float(run["ttft_seconds"]) for run in runs]
        if (
            int(summary["attempted"]) != 640
            or int(summary["completed"]) != 640
            or int(summary["errors"]) != 0
            or int(summary["hit_max_tokens"]) != sum(value == 1024 for value in tokens)
            or not close(float(summary["completion_tokens_avg"]), sum(tokens) / 640)
            or not close(float(summary["generation_seconds_avg"]), sum(generation) / 640)
            or not close(float(summary["ttft_seconds_avg"]), sum(ttft) / 640)
            or not close(float(summary["generation_tok_s_per_stream"]), sum(tokens) / sum(generation))
        ):
            raise ValueError(f"natural C128 summary does not match runs for {mode}")
    return sources


def validate_natural_decode(sources: dict[str, dict]) -> list[dict[str, str]]:
    rows = csv_rows("wikitext2-natural-decode.csv")
    expected = {(mode, concurrency) for mode in NATURAL_MODES for concurrency in NATURAL_CONCURRENCIES}
    keys = [(row["mode"], int(row["concurrency"])) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("natural decode CSV must contain five modes at C1/C64/C128")
    for row in rows:
        mode, concurrency = row["mode"], int(row["concurrency"])
        if (
            row["model"] != "Qwen3.8-27B"
            or int(row["completed"]) != 5 * concurrency
            or float(row["gen_tok_s_per_stream"]) <= 0
            or float(row["ttft_seconds_avg"]) <= 0
        ):
            raise ValueError(f"invalid natural decode row {(mode, concurrency)}")
        if concurrency == 128:
            summary = sources[mode]["selected_summary"]
            if (
                int(row["completed"]) != int(summary["completed"])
                or not close(float(row["completion_tokens_avg"]), float(summary["completion_tokens_avg"]))
                or not close(float(row["gen_tok_s_per_stream"]), float(summary["generation_tok_s_per_stream"]))
                or not close(float(row["ttft_seconds_avg"]), float(summary["ttft_seconds_avg"]))
                or not close(float(row["prefill_scout_tok_s"]), float(summary["prefill_scout_tok_s"]))
            ):
                raise ValueError(f"natural C128 CSV/provenance mismatch for {mode}")
    return rows


def validate_quality(
    natural_rows: list[dict[str, str]], sources: dict[str, dict]
) -> list[dict]:
    audit = json.loads((DATA_ROOT / "wikitext2-quality-audit.json").read_text())
    if audit["flag_rule"] != FLAG_RULE:
        raise ValueError("unexpected repetition-audit rule")
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
            raise ValueError(f"duplicate quality output {identity}")
        seen.add(identity)
        groups[key].append(row)
    summaries = {(row["mode"], int(row["concurrency"])): row for row in audit["summary"]}
    if set(groups) != set(expected_counts) or set(summaries) != set(expected_counts):
        raise ValueError("quality audit does not cover every natural decode cell")
    for key, expected_count in expected_counts.items():
        outputs = groups[key]
        flagged = [
            row
            for row in outputs
            if int(row["max_consecutive_phrase_repeats"]) >= 4
            or float(row["repeated_8gram_fraction"]) >= 0.2
        ]
        summary = summaries[key]
        if (
            len(outputs) != expected_count
            or int(summary["outputs"]) != expected_count
            or int(summary["flagged_degenerate"]) != len(flagged)
            or not close(float(summary["flagged_fraction"]), len(flagged) / len(outputs))
            or int(summary["unique_outputs"]) != len({row["sha256"] for row in outputs})
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
                raise ValueError(f"natural C128 audit identities changed for {key[0]}")
    return audit["summary"]


def read_and_validate() -> dict:
    throughput = validate_throughput()
    prefill = validate_prefill()
    perplexity = validate_perplexity()
    sources = validate_natural_source()
    natural = validate_natural_decode(sources)
    quality = validate_quality(natural, sources)
    return {
        "throughput": throughput,
        "prefill": prefill,
        "perplexity": perplexity,
        "natural": natural,
        "quality": quality,
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
        metadata={"Software": "dgx_station_benchmarks/qwen3.8-27b/render_charts.py"},
    )
    plt.close(figure)


def plot_mode(axis: plt.Axes, rows: list[dict[str, str]], mode: str, thinking: str) -> None:
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
        markersize=4.5,
        linewidth=2.2,
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
            s=58,
            linewidths=1.8,
            color=RED,
            zorder=5,
        )


def configure_throughput_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.8, 155)
    axis.set_xticks(CONCURRENCIES)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"C{int(value)}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.set_ylim(0, 6900)
    axis.set_xlabel("Offered request concurrency")
    axis.set_ylabel("Aggregate output tokens/s")
    axis.grid(True, alpha=0.42, linewidth=0.8)


def mode_legend() -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=COLORS[mode], marker="o", linewidth=2.2, markersize=5, label=LABELS[mode])
        for mode in MODES
    ]
    handles.append(
        Line2D([0], [0], color=RED, marker="x", linestyle="None", markersize=7, markeredgewidth=2, label="Capacity-limited")
    )
    return handles


def render_thinking_grid(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    for axis, thinking in zip(axes.flat, THINKING_LEVELS, strict=True):
        for mode in MODES:
            plot_mode(axis, rows, mode, thinking)
        configure_throughput_axis(axis)
        axis.set_title(f"Thinking: {thinking}", loc="left", fontsize=16, pad=10)
    figure.legend(handles=mode_legend(), ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.89))
    figure.suptitle(
        "Qwen3.8-27B speculative decoding × thinking level",
        x=0.055,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.056,
        0.91,
        "One NVIDIA GB300 • BF16 target/KV/Mamba • 8K input / 1K forced output • SGLang",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.055,
        0.018,
        "Five measured requests per offered stream. C128 values use the separately retained high-capacity profile; red × marks capacity-limited cells.",
        color="#9fb0c8",
        fontsize=10,
    )
    save(figure, "qwen-thinking-grid.png", (0.035, 0.055, 0.99, 0.84))


def render_low_throughput(rows: list[dict[str, str]]) -> None:
    figure, axis = plt.subplots(figsize=(16, 8))
    for mode in MODES:
        plot_mode(axis, rows, mode, "low")
    configure_throughput_axis(axis)
    axis.set_title("Thinking: low", loc="left", fontsize=17, pad=12)
    axis.legend(handles=mode_legend(), ncol=3, loc="upper left")
    figure.suptitle(
        "Qwen3.8-27B on one NVIDIA GB300",
        x=0.06,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.061,
        0.90,
        "8K input + 1K forced output • low thinking • BF16 target/KV/Mamba • SGLang",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.06,
        0.025,
        "Aggregate output tokens / wall time; temperature 0; ignore EOS; 5×C measured requests after C warmups.",
        color="#9fb0c8",
        fontsize=10,
    )
    save(figure, "qwen-throughput-low.png", (0.04, 0.07, 0.99, 0.85))


def render_prefill(rows: list[dict[str, str]]) -> None:
    figure, axis = plt.subplots(figsize=(16, 8))
    for configuration, (label, color) in PREFILL_CONFIGURATIONS.items():
        cells = sorted(
            (row for row in rows if row["configuration"] == configuration),
            key=lambda row: int(row["context_tokens"]),
        )
        axis.plot(
            [int(row["context_tokens"]) // 1024 for row in cells],
            [float(row["tok_per_sec"]) for row in cells],
            marker="o",
            markersize=6,
            linewidth=2.5,
            label=label,
            color=color,
        )
    axis.set_xticks([8, 32, 64, 128], ["8K", "32K", "64K", "128K"])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    axis.set_xlabel("Prompt length")
    axis.set_ylabel("Client-observed prompt tokens/s")
    axis.grid(True, alpha=0.42, linewidth=0.8)
    axis.legend(ncol=2, loc="lower center")
    figure.suptitle(
        "Qwen3.8-27B cold prefill through 128K",
        x=0.06,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.061,
        0.90,
        "One NVIDIA GB300 • standalone cold-prefill medians • SGLang",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.06,
        0.025,
        "FP8-KV curves are speed-oriented; explicit cache scales were unavailable and SGLang used unit fallback scales.",
        color="#9fb0c8",
        fontsize=10,
    )
    save(figure, "prefill-128k.png", (0.04, 0.07, 0.99, 0.85))


def render_quality(
    perplexity: dict[str, str],
    natural_rows: list[dict[str, str]],
    quality_summaries: list[dict],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 8))
    ppl = float(perplexity["word_perplexity"])
    bars = axes[0].bar(["BF16 KV/Mamba"], [ppl], width=0.58, color="#22d3ee")
    axes[0].bar_label(bars, labels=[f"{ppl:.4f}"], padding=8, fontsize=12)
    axes[0].set_title("Canonical WikiText-2 PPL", loc="left", fontsize=16, pad=12)
    axes[0].set_ylabel("Word perplexity (lower is better)")
    axes[0].set_ylim(0, 11)
    axes[0].grid(True, axis="y", alpha=0.42, linewidth=0.8)

    natural_by_key = {(row["mode"], int(row["concurrency"])): row for row in natural_rows}
    summary_by_key = {(row["mode"], int(row["concurrency"])): row for row in quality_summaries}
    for mode in NATURAL_MODES:
        rates = [
            float(natural_by_key[(mode, concurrency)]["gen_tok_s_per_stream"])
            for concurrency in NATURAL_CONCURRENCIES
        ]
        axes[1].plot(
            range(len(NATURAL_CONCURRENCIES)),
            rates,
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=COLORS[mode],
            label=LABELS[mode],
        )
        repeats = [
            int(summary_by_key[(mode, concurrency)]["max_consecutive_phrase_repeats"])
            for concurrency in NATURAL_CONCURRENCIES
        ]
        axes[2].plot(
            range(len(NATURAL_CONCURRENCIES)),
            repeats,
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=COLORS[mode],
            label=LABELS[mode],
        )

    for axis in axes[1:]:
        axis.set_xticks(range(len(NATURAL_CONCURRENCIES)), [f"C{value}" for value in NATURAL_CONCURRENCIES])
        axis.grid(True, alpha=0.42, linewidth=0.8)
    axes[1].set_title("Natural 8K continuation", loc="left", fontsize=16, pad=12)
    axes[1].set_ylabel("Generation tokens/s per stream")
    axes[1].set_ylim(0, 340)
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].set_title("Worst phrase-repeat run", loc="left", fontsize=16, pad=12)
    axes[2].set_ylabel("Consecutive phrase repeats")
    axes[2].set_ylim(0, 4.5)
    axes[2].axhline(4, color=RED, linestyle="--", linewidth=1.5, label="Flag threshold")
    axes[2].legend(loc="upper left", fontsize=9)

    figure.suptitle(
        "Qwen3.8-27B quality + natural-text decode",
        x=0.055,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    figure.text(
        0.056,
        0.91,
        "EleutherAI WikiText-2 • one NVIDIA GB300 • temperature 0 • EOS respected • up to 1,024 new tokens",
        color="#aab8cc",
        fontsize=13,
    )
    figure.text(
        0.055,
        0.025,
        "Natural rate = sum(tokens) / sum(per-request generation time). Audit also flags repeated 8-gram fraction ≥0.20; all 4,825 outputs passed.",
        color="#9fb0c8",
        fontsize=10,
    )
    save(figure, "wikitext2-quality-decode.png", (0.04, 0.075, 0.99, 0.85))


def main() -> None:
    configure_style()
    data = read_and_validate()
    render_thinking_grid(data["throughput"])
    render_low_throughput(data["throughput"])
    render_prefill(data["prefill"])
    render_quality(data["perplexity"], data["natural"], data["quality"])


if __name__ == "__main__":
    main()
