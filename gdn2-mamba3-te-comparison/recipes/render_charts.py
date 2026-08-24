#!/usr/bin/env python3
"""Render the comparison README charts from data/comparison.csv.

This intentionally uses Pillow rather than a plotting framework so the chart
build has one small dependency and remains deterministic in the benchmark
environment.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "comparison.csv"
CHART_DIR = ROOT / "charts"

WIDTH, HEIGHT = 1600, 900
BACKGROUND = "#F8FAFC"
PANEL = "#FFFFFF"
INK = "#0F172A"
MUTED = "#475569"
GRID = "#CBD5E1"
BLUE = "#2563EB"
GREEN = "#65A30D"
PURPLE = "#7C3AED"
CYAN = "#0891B2"
ORANGE = "#EA580C"

FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
REGULAR = FONT_DIR / "LiberationSans-Regular.ttf"
BOLD = FONT_DIR / "LiberationSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else REGULAR), size=size)


def read_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_chart(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((38, 32, WIDTH - 38, HEIGHT - 34), 24, fill=PANEL, outline=GRID, width=2)
    draw.text((82, 67), title, fill=INK, font=font(43, bold=True))
    draw.text((84, 124), subtitle, fill=MUTED, font=font(22))
    return image, draw


def footer(draw: ImageDraw.ImageDraw, note: str) -> None:
    draw.line((84, 822, WIDTH - 84, 822), fill=GRID, width=2)
    draw.text((84, 835), note, fill=MUTED, font=font(17))


def text_center(draw: ImageDraw.ImageDraw, x: float, y: float, value: str, size: int, color: str = INK, bold: bool = False) -> None:
    selected = font(size, bold=bold)
    box = draw.textbbox((0, 0), value, font=selected)
    draw.text((x - (box[2] - box[0]) / 2, y), value, fill=color, font=selected)


def value_label(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000:.0f}k"


def draw_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    maximum: float,
    tick_count: int,
    formatter,
) -> None:
    left, top, right, bottom = bounds
    for index in range(tick_count + 1):
        value = maximum * index / tick_count
        y = bottom - (bottom - top) * index / tick_count
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = formatter(value)
        label_font = font(18)
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - 16 - (box[2] - box[0]), y - 10), label, fill=MUTED, font=label_font)
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)


def legend_item(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, label: str) -> None:
    draw.rounded_rectangle((x, y + 3, x + 34, y + 20), 7, fill=color)
    draw.text((x + 46, y), label, fill=INK, font=font(20, bold=True))


def render_gdn2(rows: Iterable[dict[str, str]]) -> None:
    selected = [row for row in rows if row["family"] == "gdn2"]
    by_variant = {
        variant: sorted(
            (row for row in selected if row["variant"] == variant),
            key=lambda row: int(row["sequence_length"]),
        )
        for variant in ("cudnn", "fla")
    }
    image, draw = base_chart(
        "GDN2 operator throughput",
        "Combined forward + backward • BF16 • batch 4 • 64 heads × d128",
    )
    bounds = (150, 220, 1500, 730)
    draw_axes(draw, bounds, 4_000_000, 8, lambda value: f"{value / 1_000_000:.1f}M")
    draw.text((75, 460), "token-passes/s", fill=MUTED, font=font(18), anchor="mm")

    variants = (("cudnn", GREEN, "cuDNN FROST"), ("fla", PURPLE, "FLA Triton"))
    x_values = [2048, 4096, 8192, 16384, 32768]
    x_positions = [bounds[0] + index * (bounds[2] - bounds[0]) / 4 for index in range(5)]
    for x, sequence in zip(x_positions, x_values):
        text_center(draw, x, bounds[3] + 22, f"{sequence // 1024}K", 20, MUTED)
    text_center(draw, (bounds[0] + bounds[2]) / 2, 790, "sequence length", 19, MUTED)

    for variant, color, label in variants:
        points = []
        for x, row in zip(x_positions, by_variant[variant]):
            value = float(row["tokens_per_second"])
            y = bounds[3] - value / 4_000_000 * (bounds[3] - bounds[1])
            points.append((x, y))
        draw.line(points, fill=color, width=7, joint="curve")
        for index, ((x, y), row) in enumerate(zip(points, by_variant[variant])):
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=PANEL, outline=color, width=5)
            if index in (0, len(points) - 1):
                offset = -38 if variant == "cudnn" else 17
                text_center(draw, x, y + offset, value_label(float(row["tokens_per_second"])), 19, color, bold=True)
        legend_item(draw, 1040 if variant == "cudnn" else 1270, 173, color, label)

    speedups = []
    for cudnn, fla in zip(by_variant["cudnn"], by_variant["fla"]):
        speedups.append(float(cudnn["tokens_per_second"]) / float(fla["tokens_per_second"]))
    pill = f"cuDNN advantage: {min(speedups):.2f}–{max(speedups):.2f}× across 2K–32K"
    draw.rounded_rectangle((160, 170, 770, 211), 18, fill="#ECFCCB")
    draw.text((181, 178), pill, fill="#3F6212", font=font(19, bold=True))
    footer(draw, "Operator only—not a full model step • DGX Station GB300 • measured August 2026")
    image.save(CHART_DIR / "gdn2-operator-throughput.png", optimize=True)


def render_grouped_bars(
    filename: str,
    title: str,
    subtitle: str,
    categories: list[tuple[str, str]],
    values: dict[tuple[str, str], float],
    maximum: float,
    note: str,
    detail_lines: list[str],
) -> None:
    image, draw = base_chart(title, subtitle)
    bounds = (150, 220, 1500, 710)
    draw_axes(draw, bounds, maximum, 5, lambda value: f"{value / 1_000:.0f}k")
    draw.text((75, 445), "training tokens/s", fill=MUTED, font=font(18), anchor="mm")
    legend_item(draw, 1135, 173, BLUE, "1× GB300")
    legend_item(draw, 1325, 173, GREEN, "2× GB300")

    centers = [bounds[0] + (index + 0.5) * (bounds[2] - bounds[0]) / len(categories) for index in range(len(categories))]
    bar_width = 122
    for center, (key, label) in zip(centers, categories):
        for world_size, color, offset in (("1", BLUE, -bar_width - 7), ("2", GREEN, 7)):
            value = values[(key, world_size)]
            height = value / maximum * (bounds[3] - bounds[1])
            x0 = center + offset
            y0 = bounds[3] - height
            draw.rounded_rectangle((x0, y0, x0 + bar_width, bounds[3]), 9, fill=color)
            text_center(draw, x0 + bar_width / 2, y0 - 31, value_label(value), 20, color, bold=True)
        text_center(draw, center, bounds[3] + 24, label, 22, INK, bold=True)

    for index, line in enumerate(detail_lines):
        text_center(draw, centers[index], 790, line, 17, MUTED)
    footer(draw, note)
    image.save(CHART_DIR / filename, optimize=True)


def render_transformer_engine(rows: Iterable[dict[str, str]]) -> None:
    selected = [
        row
        for row in rows
        if row["family"] == "transformer_engine" and row["scope"] in {"one_node_fixed", "two_node_fixed"}
    ]
    values = {(row["variant"], row["world_size"]): float(row["tokens_per_second"]) for row in selected}
    render_grouped_bars(
        "transformer-engine-precision.png",
        "Transformer Engine precision comparison",
        "Full training step • 973M parameters • sequence 2048 • batch 64 per rank",
        [("bf16", "BF16"), ("fp8-delayed-dpa", "Delayed FP8 + DPA"), ("mxfp8", "MXFP8")],
        values,
        900_000,
        "Forward + backward + fused AdamW • fixed per-rank batch • measured August 2026",
        ["baseline", "1.68× BF16", "1.48× BF16"],
    )


def render_mamba(rows: Iterable[dict[str, str]]) -> None:
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["family"] == "transformer_engine" and row["variant"] == "bf16" and row["scope"] in {"one_node_fixed", "two_node_fixed"}:
            values[("te-bf16", row["world_size"])] = float(row["tokens_per_second"])
        if row["family"] == "mamba3" and row["scope"] in {"one_node_fixed", "two_node_fixed"}:
            values[(row["variant"], row["world_size"])] = float(row["tokens_per_second"])
    render_grouped_bars(
        "mamba3-vs-transformer.png",
        "Mamba-3 versus a similarly sized Transformer",
        "Complete training steps at sequence 2048 • approximately one billion parameters",
        [("te-bf16", "TE BF16"), ("siso", "Mamba-3 SISO"), ("mimo-r4", "Mamba-3 MIMO-r4")],
        values,
        550_000,
        "Different architectures; rates compare systems throughput, not model quality • measured August 2026",
        ["batch 64/rank", "batch 16/rank", "batch 64/rank"],
    )


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    render_gdn2(rows)
    render_transformer_engine(rows)
    render_mamba(rows)
    for path in sorted(CHART_DIR.glob("*.png")):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
