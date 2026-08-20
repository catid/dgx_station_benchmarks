#!/usr/bin/env python3
"""Render dependency-free SVG charts from the measured MiniMax H3 CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "data" / "1x-bf16" / "stage-timings.csv"
PERF = ROOT / "data" / "1x-bf16" / "performance.csv"
OUTPUT = ROOT / "charts" / "1x-bf16-stage-breakdown.svg"


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


with STAGES.open(newline="", encoding="utf-8") as handle:
    stages = next(row for row in csv.DictReader(handle) if row["request"] == "mean")
with PERF.open(newline="", encoding="utf-8") as handle:
    perf = next(csv.DictReader(handle))

latency = float(perf["latency_mean_s"])
values = [
    ("Text encode", float(stages["text_encode_s"]), "#5ac8fa"),
    ("Denoise", float(stages["denoise_s"]), "#0a84ff"),
    ("Decode", float(stages["decode_s"]), "#bf5af2"),
]
accounted = sum(value for _, value, _ in values)
values.append(("API / mux / other", max(0.0, latency - accounted), "#636366"))

x0, y0, width, height = 90.0, 118.0, 900.0, 78.0
cursor = x0
rects: list[str] = []
legend: list[str] = []
for index, (label, value, color) in enumerate(values):
    segment = width * value / latency
    rects.append(
        f'<rect x="{cursor:.2f}" y="{y0}" width="{segment:.2f}" '
        f'height="{height}" fill="{color}"/>'
    )
    if segment > 70:
        rects.append(
            f'<text x="{cursor + segment / 2:.2f}" y="{y0 + 47:.2f}" '
            f'text-anchor="middle" class="inside">{value:.2f}s</text>'
        )
    lx = 105 + (index % 2) * 440
    ly = 252 + (index // 2) * 42
    legend.append(f'<rect x="{lx}" y="{ly - 16}" width="22" height="22" fill="{color}"/>')
    legend.append(
        f'<text x="{lx + 34}" y="{ly + 1}" class="legend">'
        f'{esc(label)}: {value:.3f}s</text>'
    )
    cursor += segment

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="390" viewBox="0 0 1080 390">
<rect width="1080" height="390" fill="#111214"/>
<style>
  text {{ font-family: Inter, system-ui, sans-serif; fill: #f5f5f7; }}
  .title {{ font-size: 28px; font-weight: 700; }}
  .subtitle {{ font-size: 17px; fill: #b9bbc1; }}
  .inside {{ font-size: 18px; font-weight: 700; }}
  .legend {{ font-size: 18px; }}
  .foot {{ font-size: 15px; fill: #9b9da3; }}
</style>
<text x="90" y="52" class="title">MiniMax H3 · 1× GB300 warmed request</text>
<text x="90" y="82" class="subtitle">BF16 FL2VA · 1344×768 · 124 frames · 50 steps · mean of 3 VBench prompts</text>
<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="4" fill="#242528"/>
{''.join(rects)}
{''.join(legend)}
<text x="90" y="353" class="foot">Mean end-to-end latency: {latency:.2f}s · warmed peak HBM: {float(perf['peak_hbm_mb']):,.0f} MB · 3/3 requests succeeded</text>
</svg>
'''
OUTPUT.write_text(svg, encoding="utf-8")
print(OUTPUT)
