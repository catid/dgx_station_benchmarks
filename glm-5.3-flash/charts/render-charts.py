#!/usr/bin/env python3
"""Render GLM-5.3-Flash throughput charts."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
OUTPUT_DIR = CHARTS
CHART_NAMES = ("decode-throughput.png", "prefill-throughput.png")
EXTERNAL_COLOR = "#F6903D"
GB300_COLORS = ("#5B8FF9", "#61DDAA", "#F6BD16", "#5D7092", "#E8684A")
PUBLISHED_EXTERNAL_STATUS = "EXTERNAL_USER_SUPPLIED"
PROFILE_ORDER = {
    "TP2/MTP0": 0,
    "TP2/MTP5": 1,
    "TEP2/MTP5": 2,
    "NVFP4 TP2/AR": 3,
    "NVFP4 TP2/DFlash2": 4,
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0E1117",
            "axes.facecolor": "#151A23",
            "axes.edgecolor": "#697386",
            "axes.labelcolor": "#E6EDF3",
            "text.color": "#E6EDF3",
            "xtick.color": "#C9D1D9",
            "ytick.color": "#C9D1D9",
            "grid.color": "#30363D",
            "font.size": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def measured_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_rows(path) if row.get("publication_status") == "measured"]


def external_rows(metric: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_rows(DATA / "external-rtx-pro-6000.csv")
        if row.get("metric") == metric
        and row.get("source_status") == PUBLISHED_EXTERNAL_STATUS
    ]


def group_profiles(
    rows: list[dict[str, str]], order_key: str
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["profile"]].append(row)
    return {
        profile: sorted(cells, key=lambda row: int(row[order_key]))
        for profile, cells in sorted(
            grouped.items(), key=lambda item: PROFILE_ORDER.get(item[0], len(PROFILE_ORDER))
        )
    }


def render_decode() -> None:
    external = external_rows("decode")
    external.sort(key=lambda row: int(row["concurrency"]))
    gb300 = measured_rows(DATA / "throughput.csv")
    gb300_profiles = group_profiles(gb300, "concurrency")

    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    for index, (profile, cells) in enumerate(gb300_profiles.items()):
        axis.plot(
            [int(row["concurrency"]) for row in cells],
            [float(row["aggregate_output_tokens_per_second"]) for row in cells],
            marker="o",
            linewidth=2.7,
            color=GB300_COLORS[index % len(GB300_COLORS)],
            label=f"2× DGX Station GB300 — {profile}",
        )
    axis.plot(
        [int(row["concurrency"]) for row in external],
        [float(row["tokens_per_second"]) for row in external],
        marker="o",
        linewidth=2.7,
        linestyle="--",
        color=EXTERNAL_COLOR,
        label="4× RTX PRO 6000 — MTP5",
    )
    axis.set_title("GLM-5.3-Flash — decode throughput")
    axis.set_xlabel("Request concurrency")
    axis.set_ylabel("Aggregate output tokens/second")
    ticks = sorted({1, 2, 4, 8, 10, *[int(row["concurrency"]) for row in gb300]})
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.65)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "decode-throughput.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_prefill() -> None:
    external = external_rows("prefill")
    external.sort(key=lambda row: int(row["context_tokens"]))
    gb300 = measured_rows(DATA / "prefill.csv")
    gb300_profiles = group_profiles(gb300, "nominal_context_tokens")

    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    for index, (profile, cells) in enumerate(gb300_profiles.items()):
        axis.plot(
            [int(row["nominal_context_tokens"]) // 1024 for row in cells],
            [float(row["prompt_tokens_per_second"]) for row in cells],
            marker="o",
            linewidth=2.7,
            color=GB300_COLORS[index % len(GB300_COLORS)],
            label=f"2× DGX Station GB300 — {profile}",
        )
    x_external = [int(row["context_tokens"]) // 1024 for row in external]
    axis.plot(
        x_external,
        [float(row["tokens_per_second"]) for row in external],
        marker="o",
        linewidth=2.7,
        linestyle="--",
        color=EXTERNAL_COLOR,
        label="4× RTX PRO 6000 — MTP5",
    )
    axis.set_title("GLM-5.3-Flash — prefill throughput")
    axis.set_xlabel("Prompt context (Ki tokens)")
    axis.set_ylabel("Prompt tokens/second")
    axis.set_xscale("log", base=2)
    axis.set_xticks((8, 64, 128))
    axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}K"))
    rates = [float(row["tokens_per_second"]) for row in external]
    rates.extend(float(row["prompt_tokens_per_second"]) for row in gb300)
    axis.set_ylim(0, max(rates) * 1.05)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, alpha=0.65)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "prefill-throughput.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_all(output_dir: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    render_decode()
    render_prefill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output_dir:
        parser.error("--check and --output-dir are mutually exclusive")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="glm53-charts-") as directory:
            generated = Path(directory)
            render_all(generated)
            changed = [
                name for name in CHART_NAMES
                if not (CHARTS / name).is_file()
                or (CHARTS / name).read_bytes() != (generated / name).read_bytes()
            ]
        if changed:
            raise SystemExit(f"chart render differs: {', '.join(changed)}")
        print("PASS: committed GLM charts match the pinned renderer")
        return
    render_all(args.output_dir or CHARTS)


if __name__ == "__main__":
    main()
