#!/usr/bin/env python3
"""Render Qwen3.8-Flash-Next charts from section-owned machine data."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"
DATA = ROOT / "data"
OUTPUT_DIR = CHARTS
CHART_NAMES = (
    "dgx-nvfp4-decode.png",
    "dgx-nvfp4-prefill.png",
    "decode-throughput.png",
    "cold-prefill-throughput.png",
    "tep4-ar-decode-comparison.png",
    "tep4-ar-prefill-comparison.png",
    "tep4-mtp3-decode-comparison.png",
    "tep4-mtp3-prefill-comparison.png",
)
PUBLISHED_EXTERNAL_STATUSES = {"SEALED_PRIMARY_EXTERNAL"}
ACCEPTED_DGX_STATUSES = {
    "MEASURED_CURRENT",
    "SEALED_RANKABLE",
    "VALIDATED_RANKABLE",
}
DGX_MODEL_ID = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89"
DGX_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_CONTEXTS = (8192, 32768, 65536, 131072)
DGX_HEADLINE_SERIES = {
    "NVFP4 TP1/MTP0": (
        "1× DGX Station GB300 · TP1/AR",
        "#E6EDF3",
        "-",
        "P",
    ),
    "NVFP4 TP1/MTP3": (
        "1× DGX Station GB300 · TP1/MTP3",
        "#FF7B72",
        "-",
        "X",
    ),
    "NVFP4 TP2/MTP0": (
        "2× DGX Station GB300 · TP2/AR",
        "#58A6FF",
        "-",
        "o",
    ),
    "NVFP4 TP2/MTP3": (
        "2× DGX Station GB300 · TP2/MTP3",
        "#D2A8FF",
        "-",
        "o",
    ),
    "NVFP4 TEP2/MTP0": (
        "2× DGX Station GB300 · TEP2/AR",
        "#61DDAA",
        "-",
        "s",
    ),
    "NVFP4 TEP2/MTP3": (
        "2× DGX Station GB300 · TEP2/MTP3",
        "#F6BD16",
        "-",
        "^",
    ),
}
DGX_HEADLINE_PLATFORMS = {
    "NVFP4 TP1/MTP0": "DGX Station",
    "NVFP4 TP1/MTP3": "DGX Station",
    "NVFP4 TP2/MTP0": "DGX Station pair",
    "NVFP4 TP2/MTP3": "DGX Station pair",
    "NVFP4 TEP2/MTP0": "DGX Station pair",
    "NVFP4 TEP2/MTP3": "DGX Station pair",
}
RTX_COMPARISON_SERIES = {
    "nvfp4_tep4_ar": (
        "4× RTX PRO 6000 · RadixArk NVFP4@7b719225 · TEP4/AR",
        "#FFA657",
        "D",
    ),
    "nvfp4_tep4_mtp3": (
        "4× RTX PRO 6000 · RadixArk NVFP4@7b719225 · TEP4/MTP3",
        "#FF7B72",
        "X",
    ),
}

BACKGROUND = "#0E1117"
PANEL = "#151A23"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
GRID = "#30363D"

SERIES = {
    "fp8_tp4_ar": ("Official FP8/vLLM · TP4/AR", "#58A6FF", "-", "o"),
    "fp8_tp4_mtp3": ("Official FP8/vLLM · TP4/MTP3", "#D2A8FF", "-", "o"),
    "fp8_tep4_ar": ("Official FP8/vLLM · TEP4/AR", "#61DDAA", "-", "o"),
    "fp8_tep4_mtp3": ("Official FP8/vLLM · TEP4/MTP3", "#F6BD16", "-", "o"),
    "nvfp4_tep4_ar": ("Radix NVFP4/SGLang · TEP4/AR", "#FFA657", "--", "D"),
    "nvfp4_tep4_mtp3": ("Radix NVFP4/SGLang · TEP4/MTP3", "#FF7B72", "--", "D"),
}

def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#697386",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": "#C9D1D9",
            "ytick.color": "#C9D1D9",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "legend.facecolor": PANEL,
            "legend.edgecolor": "#484F58",
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_by_profile(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["profile"]].append(row)
    return grouped


def published_external_rows(filename: str) -> list[dict[str, str]]:
    return [
        row for row in read_rows(DATA / filename)
        if row.get("publication_status") in PUBLISHED_EXTERNAL_STATUSES
        and row.get("profile") in SERIES
    ]


def accepted_overlay_rows(metric: str) -> list[dict[str, str]]:
    rows = read_rows(DATA / "dgx-overlays.csv")
    return [
        row
        for row in rows
        if row.get("metric") == metric
        and row.get("throughput")
        and row.get("publication_status") in ACCEPTED_DGX_STATUSES
        and row.get("model_id") == DGX_MODEL_ID
    ]


def headline_dgx_rows(metric: str) -> list[dict[str, str]]:
    """Select measured C1-C64 rows from the one- and two-Station lanes."""
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    axis_value = "concurrency" if metric == "decode" else "nominal_context_tokens"
    for row in accepted_overlay_rows(metric):
        profile = row.get("profile", "")
        series = DGX_HEADLINE_SERIES.get(profile)
        if series is None:
            continue
        if row.get("platform_label") != DGX_HEADLINE_PLATFORMS[profile]:
            continue
        value = int(row[axis_value])
        if metric == "decode" and value not in DGX_CONCURRENCIES:
            continue
        if metric == "prefill" and value not in PREFILL_CONTEXTS:
            continue
        key = (profile, value)
        if key in seen:
            raise ValueError(f"duplicate DGX headline row: {profile} {metric} {value}")
        seen.add(key)
        selected.append(row)
    return selected


def headline_dgx_series(metric: str) -> dict[str, list[dict[str, str]]]:
    grouped = group_by_profile(headline_dgx_rows(metric))
    axis_value = "concurrency" if metric == "decode" else "nominal_context_tokens"
    return {
        profile: sorted(grouped.get(profile, []), key=lambda row: int(row[axis_value]))
        for profile in DGX_HEADLINE_SERIES
        if grouped.get(profile)
    }


def dynamic_y_upper(values: list[float]) -> float:
    """Give the tallest published series 12% headroom without a fixed ceiling."""
    return max(1.0, max(values, default=0.0) * 1.12)


def rtx_nvfp4_decode(profile: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in published_external_rows("throughput.csv"):
        concurrency = int(row["concurrency"])
        if row["profile"] != profile or concurrency > 64:
            continue
        values[concurrency] = float(row["aggregate_output_tokens_per_second"])
    return values


def rtx_nvfp4_prefill(profile: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in published_external_rows("prefill.csv"):
        if row["profile"] != profile:
            continue
        context = int(row["nominal_context_tokens"])
        values[context] = float(row["client_prompt_tokens_per_second"])
    return values


def render_dgx_decode() -> None:
    grouped = headline_dgx_series("decode")
    figure, axis = plt.subplots(figsize=(10.8, 6.6))
    plotted_values: list[float] = []
    for profile, (label, color, linestyle, marker) in DGX_HEADLINE_SERIES.items():
        rows = grouped.get(profile, [])
        if not rows:
            continue
        values = [float(row["throughput"]) for row in rows]
        plotted_values.extend(values)
        axis.plot(
            [int(row["concurrency"]) for row in rows],
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.8,
            marker=marker,
            markersize=6.5,
            label=label,
        )

    for profile, (label, color, marker) in RTX_COMPARISON_SERIES.items():
        rtx_comparison = rtx_nvfp4_decode(profile)
        rtx_concurrencies = [
            value for value in DGX_CONCURRENCIES if value in rtx_comparison
        ]
        if not rtx_concurrencies:
            continue
        rtx_values = [rtx_comparison[value] for value in rtx_concurrencies]
        plotted_values.extend(rtx_values)
        axis.plot(
            rtx_concurrencies,
            rtx_values,
            color=color,
            linestyle="--",
            linewidth=2.8,
            marker=marker,
            markersize=6.5,
            markerfacecolor=BACKGROUND,
            markeredgewidth=1.5,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(DGX_CONCURRENCIES, [str(value) for value in DGX_CONCURRENCIES])
    axis.set_xlim(0.85, 72)
    axis.set_ylim(0, dynamic_y_upper(plotted_values))
    axis.set_xlabel("Offered concurrency")
    axis.set_ylabel("Output tokens/s")
    axis.set_title("Qwen3.8-Flash-Next · DGX Station NVFP4 fixed decode")
    axis.legend(loc="upper left", framealpha=0.92)
    figure.text(
        0.5,
        0.018,
        "RTX comparisons: patched SGLang · one 4-GPU server · 8,192 input + 1,024 output tokens · temperature 0",
        ha="center",
        color=MUTED,
        fontsize=8.8,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(
        OUTPUT_DIR / "dgx-nvfp4-decode.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)


def render_dgx_prefill() -> None:
    grouped = headline_dgx_series("prefill")
    figure, axis = plt.subplots(figsize=(10.8, 6.6))
    positions = list(range(len(PREFILL_CONTEXTS)))
    context_labels = ("8K", "32K", "64K", "128K")
    position_by_context = {
        context: position for position, context in enumerate(PREFILL_CONTEXTS)
    }
    plotted_values: list[float] = []
    for profile, (label, color, linestyle, marker) in DGX_HEADLINE_SERIES.items():
        rows = grouped.get(profile, [])
        if not rows:
            continue
        values = [float(row["throughput"]) for row in rows]
        plotted_values.extend(values)
        axis.plot(
            [position_by_context[int(row["nominal_context_tokens"])] for row in rows],
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.8,
            marker=marker,
            markersize=6.5,
            label=label,
        )

    for profile, (label, color, marker) in RTX_COMPARISON_SERIES.items():
        rtx_comparison = rtx_nvfp4_prefill(profile)
        rtx_contexts = [
            context for context in PREFILL_CONTEXTS if context in rtx_comparison
        ]
        if not rtx_contexts:
            continue
        rtx_values = [rtx_comparison[context] for context in rtx_contexts]
        plotted_values.extend(rtx_values)
        axis.plot(
            [position_by_context[context] for context in rtx_contexts],
            rtx_values,
            color=color,
            linestyle="--",
            linewidth=2.8,
            marker=marker,
            markersize=6.5,
            markerfacecolor=BACKGROUND,
            markeredgewidth=1.5,
            label=label,
        )

    axis.set_xlim(-0.15, 3.15)
    axis.set_xticks(positions)
    axis.set_xticklabels(context_labels, ha="center")
    axis.set_ylim(0, dynamic_y_upper(plotted_values))
    axis.set_xlabel("Nominal cold-prefill target")
    axis.set_ylabel("Client prompt tokens/s")
    axis.set_title("Qwen3.8-Flash-Next · DGX Station NVFP4 cold prefill")
    axis.legend(loc="lower center", framealpha=0.92)
    figure.text(
        0.5,
        0.018,
        "RTX comparisons: patched SGLang · one 4-GPU server · C1 cold prefill · one output token",
        ha="center",
        color=MUTED,
        fontsize=8.8,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(
        OUTPUT_DIR / "dgx-nvfp4-prefill.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)


def render_decode() -> None:
    grouped = group_by_profile(published_external_rows("throughput.csv"))
    figure, axis = plt.subplots(figsize=(12.2, 7.2))

    for profile, (label, color, linestyle, marker) in SERIES.items():
        rows = sorted(grouped.get(profile, []), key=lambda row: int(row["concurrency"]))
        if not rows:
            continue
        axis.plot(
            [int(row["concurrency"]) for row in rows],
            [float(row["aggregate_output_tokens_per_second"]) for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.7,
            marker=marker,
            markersize=6.5,
            markerfacecolor=BACKGROUND if profile.startswith("nvfp4") else color,
            markeredgewidth=1.5,
        )

    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128]
    axis.set_xscale("log", base=2)
    axis.set_xticks(concurrencies, [str(value) for value in concurrencies])
    axis.set_xlim(0.85, 150)
    axis.set_ylim(0, 4050)
    axis.set_xlabel("Offered concurrency")
    axis.set_ylabel("Aggregate output tokens/s")
    axis.set_title("Qwen3.8-Flash-Next · 4× RTX PRO 6000 · fixed decode")
    axis.legend(loc="upper left", framealpha=0.92, ncol=2, fontsize=8.8)
    figure.text(
        0.5,
        0.018,
        "One 4-GPU server · RTX PRO 6000 Blackwell Max-Q · PCIe Gen5, no NVLink",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(
        OUTPUT_DIR / "decode-throughput.png", dpi=180, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)


def render_prefill() -> None:
    grouped = group_by_profile(published_external_rows("prefill.csv"))
    figure, axis = plt.subplots(figsize=(12.2, 7.2))

    for profile, (label, color, linestyle, marker) in SERIES.items():
        rows = sorted(grouped.get(profile, []), key=lambda row: int(row["nominal_context_tokens"]))
        if not rows:
            continue
        axis.plot(
            [int(row["nominal_context_tokens"]) for row in rows],
            [float(row["client_prompt_tokens_per_second"]) for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.7,
            marker=marker,
            markersize=6.5,
            markerfacecolor=BACKGROUND if profile.startswith("nvfp4") else color,
            markeredgewidth=1.5,
        )

    contexts = [8192, 32768, 65536, 131072]
    axis.set_xticks(contexts, ["8K", "32K", "64K", "128K"])
    axis.set_xlim(4000, 136000)
    axis.set_ylim(9000, 16500)
    axis.set_xlabel("Nominal cold-prefill target")
    axis.set_ylabel("Client prompt tokens/s")
    axis.set_title("Qwen3.8-Flash-Next · 4× RTX PRO 6000 · C1 cold prefill")
    axis.legend(loc="lower left", framealpha=0.92, ncol=2, fontsize=8.8)
    figure.text(
        0.5,
        0.018,
        "One 4-GPU server · RTX PRO 6000 Blackwell Max-Q · PCIe Gen5, no NVLink",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(
        OUTPUT_DIR / "cold-prefill-throughput.png", dpi=180, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)


def render_same_topology_decode(
    *,
    mode: str,
    output_name: str,
    footer: str,
) -> None:
    grouped = group_by_profile(published_external_rows("throughput.csv"))
    fp8_profile = f"fp8_tep4_{mode}"
    nvfp4_profile = f"nvfp4_tep4_{mode}"
    fp8 = {
        int(row["concurrency"]): float(row["aggregate_output_tokens_per_second"])
        for row in grouped[fp8_profile]
    }
    nvfp4 = {
        int(row["concurrency"]): float(row["aggregate_output_tokens_per_second"])
        for row in grouped[nvfp4_profile]
    }
    common = sorted(fp8.keys() & nvfp4.keys())
    delta = [(nvfp4[value] / fp8[value] - 1.0) * 100.0 for value in common]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    left.plot(common, [fp8[value] for value in common], color="#F6BD16", lw=2.7, marker="o", label="Official FP8/vLLM")
    left.plot(
        common,
        [nvfp4[value] for value in common],
        color="#FF7B72",
        lw=2.7,
        ls="--",
        marker="D",
        markerfacecolor=BACKGROUND,
        label="Radix NVFP4/SGLang",
    )
    left.set_xscale("log", base=2)
    left.set_xticks(common, [str(value) for value in common])
    left.set_xlabel("Offered concurrency")
    left.set_ylabel("Aggregate output tokens/s")
    mode_label = "AR" if mode == "ar" else "MTP3"
    left.set_title(f"TEP4/{mode_label} absolute rate")
    left.legend(loc="upper left", fontsize=9)

    colors = ["#61DDAA" if value >= 0 else "#FF7B72" for value in delta]
    right.bar(range(len(common)), delta, color=colors, width=0.72)
    right.axhline(0, color="#C9D1D9", lw=1)
    right.set_xticks(range(len(common)), [str(value) for value in common])
    right.set_xlabel("Offered concurrency")
    right.set_ylabel("NVFP4/SGLang change vs FP8/vLLM (%)")
    right.set_title("End-to-end system delta")
    lower = min(-8.0, min(delta) - 4.0)
    upper = max(10.0, max(delta) + 4.0)
    right.set_ylim(lower, upper)
    for x_value, y_value in enumerate(delta):
        right.text(x_value, y_value + (0.8 if y_value >= 0 else -1.0), f"{y_value:+.1f}%", ha="center", va="bottom" if y_value >= 0 else "top", fontsize=8.5)

    figure.suptitle(
        f"Qwen3.8-Flash-Next · 4× RTX PRO 6000 TEP4/{mode_label} · decode",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.018,
        footer,
        ha="center",
        color=MUTED,
        fontsize=9.4,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_same_topology_prefill(*, mode: str, output_name: str, footer: str) -> None:
    grouped = group_by_profile(published_external_rows("prefill.csv"))
    fp8_profile = f"fp8_tep4_{mode}"
    nvfp4_profile = f"nvfp4_tep4_{mode}"
    fp8 = {
        int(row["nominal_context_tokens"]): float(row["client_prompt_tokens_per_second"])
        for row in grouped[fp8_profile]
    }
    nvfp4 = {
        int(row["nominal_context_tokens"]): float(row["client_prompt_tokens_per_second"])
        for row in grouped[nvfp4_profile]
    }
    common = sorted(fp8.keys() & nvfp4.keys())
    delta = [(nvfp4[value] / fp8[value] - 1.0) * 100.0 for value in common]
    labels = ["8K", "32K", "64K", "128K"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    left.plot(range(len(common)), [fp8[value] for value in common], color="#F6BD16", lw=2.7, marker="o", label="Official FP8/vLLM")
    left.plot(
        range(len(common)),
        [nvfp4[value] for value in common],
        color="#FF7B72",
        lw=2.7,
        ls="--",
        marker="D",
        markerfacecolor=BACKGROUND,
        label="Radix NVFP4/SGLang",
    )
    left.set_xticks(range(len(common)), labels)
    left.set_xlabel("Nominal cold-prefill target")
    left.set_ylabel("Client prompt tokens/s")
    mode_label = "AR" if mode == "ar" else "MTP3"
    left.set_title(f"TEP4/{mode_label} absolute rate")
    left.legend(loc="lower left", fontsize=9)

    right.bar(range(len(common)), delta, color="#61DDAA", width=0.68)
    right.axhline(0, color="#C9D1D9", lw=1)
    right.set_xticks(range(len(common)), labels)
    right.set_xlabel("Nominal cold-prefill target")
    right.set_ylabel("NVFP4/SGLang change vs FP8/vLLM (%)")
    right.set_title("End-to-end system delta")
    right.set_ylim(min(0.0, min(delta) - 1.0), max(15.0, max(delta) + 1.0))
    for x_value, y_value in enumerate(delta):
        right.text(x_value, y_value + 0.35, f"{y_value:+.1f}%", ha="center", fontsize=8.8)

    figure.suptitle(
        f"Qwen3.8-Flash-Next · 4× RTX PRO 6000 TEP4/{mode_label} · cold prefill",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.018,
        footer,
        ha="center",
        color=MUTED,
        fontsize=9.4,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_all(output_dir: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    render_dgx_decode()
    render_dgx_prefill()
    render_decode()
    render_prefill()
    render_same_topology_decode(
        mode="ar",
        output_name="tep4-ar-decode-comparison.png",
        footer="One 4-GPU TEP4 server · RTX PRO 6000 Blackwell Max-Q",
    )
    render_same_topology_prefill(
        mode="ar",
        output_name="tep4-ar-prefill-comparison.png",
        footer="One 4-GPU TEP4 server · RTX PRO 6000 Blackwell Max-Q",
    )
    render_same_topology_decode(
        mode="mtp3",
        output_name="tep4-mtp3-decode-comparison.png",
        footer="One 4-GPU TEP4 server · RTX PRO 6000 Blackwell Max-Q",
    )
    render_same_topology_prefill(
        mode="mtp3",
        output_name="tep4-mtp3-prefill-comparison.png",
        footer="One 4-GPU TEP4 server · RTX PRO 6000 Blackwell Max-Q",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output_dir:
        parser.error("--check and --output-dir are mutually exclusive")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="qwen38-flash-next-charts-") as directory:
            generated = Path(directory)
            render_all(generated)
            changed = [
                name for name in CHART_NAMES
                if not (CHARTS / name).is_file()
                or (CHARTS / name).read_bytes() != (generated / name).read_bytes()
            ]
        if changed:
            raise SystemExit(f"chart render differs: {', '.join(changed)}")
        print("PASS: committed Qwen charts match the pinned renderer")
        return
    render_all(args.output_dir or CHARTS)


if __name__ == "__main__":
    main()
