#!/usr/bin/env python3
"""Render dependency-free SVG charts from the measured MiniMax H3 CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "data" / "1x-bf16" / "stage-timings.csv"
PERF = ROOT / "data" / "1x-bf16" / "performance.csv"
OUTPUT = ROOT / "charts" / "1x-bf16-stage-breakdown.svg"
DURATION = ROOT / "data" / "duration-scaling" / "performance.csv"
INDEPENDENT = ROOT / "data" / "duration-scaling" / "independent-throughput.csv"
DURATION_OUTPUT = ROOT / "charts" / "duration-scaling.svg"
INDEPENDENT_OUTPUT = ROOT / "charts" / "15s-independent-throughput.svg"


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


with DURATION.open(newline="", encoding="utf-8") as handle:
    duration_rows = list(csv.DictReader(handle))

chart_x0, chart_y0 = 125.0, 100.0
chart_width, chart_height = 845.0, 350.0
maximum = max(float(row["client_e2e_mean_s"]) for row in duration_rows)
bar_slot = chart_width / len(duration_rows)
duration_marks: list[str] = []
for index, row in enumerate(duration_rows):
    value = float(row["client_e2e_mean_s"])
    bar_height = chart_height * value / maximum
    x = chart_x0 + index * bar_slot + bar_slot * 0.20
    bar_width = bar_slot * 0.60
    y = chart_y0 + chart_height - bar_height
    color = "#ff9f0a" if row["support_status"] != "official" else "#0a84ff"
    duration_marks.extend(
        [
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" rx="4" fill="{color}"/>',
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 12:.2f}" '
            f'text-anchor="middle" class="value">{value:.1f}s</text>',
            f'<text x="{x + bar_width / 2:.2f}" y="{chart_y0 + chart_height + 32:.2f}" '
            f'text-anchor="middle" class="axis">{float(row["requested_seconds"]):g}s</text>',
            f'<text x="{x + bar_width / 2:.2f}" y="{chart_y0 + chart_height + 54:.2f}" '
            f'text-anchor="middle" class="state">{esc(row["runtime_state"].replace("_", " "))}</text>',
        ]
    )

duration_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="570" viewBox="0 0 1080 570">
<rect width="1080" height="570" fill="#111214"/>
<style>
  text {{ font-family: Inter, system-ui, sans-serif; fill: #f5f5f7; }}
  .title {{ font-size: 28px; font-weight: 700; }}
  .subtitle {{ font-size: 16px; fill: #b9bbc1; }}
  .value {{ font-size: 17px; font-weight: 700; }}
  .axis {{ font-size: 18px; font-weight: 700; }}
  .state {{ font-size: 11px; fill: #a9abb1; }}
  .foot {{ font-size: 14px; fill: #9b9da3; }}
</style>
<text x="80" y="45" class="title">MiniMax H3 · native duration scaling on 1× GB300</text>
<text x="80" y="72" class="subtitle">Official BF16 FL2VA · 1344×768 · 50 steps · client end to end</text>
<line x1="{chart_x0}" y1="{chart_y0 + chart_height}" x2="{chart_x0 + chart_width}" y2="{chart_y0 + chart_height}" stroke="#515257"/>
{''.join(duration_marks)}
<rect x="80" y="532" width="15" height="15" fill="#0a84ff"/><text x="104" y="545" class="foot">official 4–15s</text>
<rect x="255" y="532" width="15" height="15" fill="#ff9f0a"/><text x="279" y="545" class="foot">unsupported one-line duration-cap experiment</text>
</svg>
'''
DURATION_OUTPUT.write_text(duration_svg, encoding="utf-8")
print(DURATION_OUTPUT)


with INDEPENDENT.open(newline="", encoding="utf-8") as handle:
    independent_rows = list(csv.DictReader(handle))

throughput_max = max(float(row["outputs_per_hour"]) for row in independent_rows)
throughput_marks: list[str] = []
for index, row in enumerate(independent_rows):
    value = float(row["outputs_per_hour"])
    x = 245.0 + index * 390.0
    height = 270.0 * value / throughput_max
    y = 390.0 - height
    throughput_marks.extend(
        [
            f'<rect x="{x:.2f}" y="{y:.2f}" width="210" height="{height:.2f}" rx="5" fill="#30d158"/>',
            f'<text x="{x + 105:.2f}" y="{y - 14:.2f}" text-anchor="middle" class="big">{value:.3f}</text>',
            f'<text x="{x + 105:.2f}" y="425" text-anchor="middle" class="label">{int(row["stations"])} station{("s" if row["stations"] != "1" else "")}</text>',
        ]
    )

independent_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="520" viewBox="0 0 1080 520">
<rect width="1080" height="520" fill="#111214"/>
<style>
  text {{ font-family: Inter, system-ui, sans-serif; fill: #f5f5f7; }}
  .title {{ font-size: 28px; font-weight: 700; }}
  .subtitle {{ font-size: 16px; fill: #b9bbc1; }}
  .big {{ font-size: 23px; font-weight: 700; }}
  .label {{ font-size: 19px; font-weight: 650; }}
  .foot {{ font-size: 14px; fill: #9b9da3; }}
</style>
<text x="80" y="48" class="title">MiniMax H3 · independent 15-second throughput</text>
<text x="80" y="77" class="subtitle">Same prompt and seed · one resident BF16 replica per station · 362 frames/output</text>
<line x1="150" y1="390" x2="930" y2="390" stroke="#515257"/>
{''.join(throughput_marks)}
<text x="540" y="472" text-anchor="middle" class="foot">Complete clips/hour · simultaneous pair makespan 719.296s · outputs byte-identical</text>
</svg>
'''
INDEPENDENT_OUTPUT.write_text(independent_svg, encoding="utf-8")
print(INDEPENDENT_OUTPUT)
