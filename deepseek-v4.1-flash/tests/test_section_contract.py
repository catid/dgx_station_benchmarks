#!/usr/bin/env python3
"""Offline consistency gates for the DeepSeek-V4.1-Flash publication section.

The section's rule: every configuration (lane) with complete, error-free rows is a series on the charts
and a column or row in the matching README table, accepted and diagnostic lanes alike. Placeholders may
only stand for lanes that have no rows yet; numbers must round from the CSV tables; the accepted versus
diagnostic distinction lives in publication_status/rankable and the lane ledger, never in what is drawn.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPOSITORY = ROOT.parent

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
ISLS = (16384, 32768, 65536, 131072)
CONCURRENCIES = (1, 4, 16)
CONCURRENCY_REFERENCE_ISL = 65536
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
DASH = "—"
LANE_TABLES = ("prefill.csv", "diagnostic-prefill.csv", "throughput.csv")
HEADLINE_LANE = "sglang_tp2_ep2_ar"

# Headline placeholder -> (lane, isl, concurrency, kind); accepted prefill rows only, as before.
REPLAY_LANE = "sglang_tp2_ep2_ar_replay"  # accepted and ranked, but not bit-identical to the exact reference
HEADLINE_PREFILL = (
    ("SGLANG_PREFILL_128K_C16", HEADLINE_LANE, 131072, 16, "rate"),
    ("SGLANG_PREFILL_16K_C1", HEADLINE_LANE, 16384, 1, "rate"),
    ("SGLANG_TTFT_16K_C1", HEADLINE_LANE, 16384, 1, "ttft"),
    ("SGLANG_REPLAY_PREFILL_128K_C16", REPLAY_LANE, 131072, 16, "rate"),
    ("SGLANG_REPLAY_PREFILL_16K_C1", REPLAY_LANE, 16384, 1, "rate"),
    ("SGLANG_REPLAY_TTFT_16K_C1", REPLAY_LANE, 16384, 1, "ttft"),
    ("SGLANG_REPLAY_TTFT_128K_C1", REPLAY_LANE, 131072, 1, "ttft"),
)
# Headline placeholder -> (lane, concurrency, kind); accepted decode rows only.
HEADLINE_DECODE = (
    ("SGLANG_DSPARK_USER_C1", "sglang_tp2_ep2_dspark", 1, "user"),
    ("SGLANG_DSPARK_ACCEPT_C1", "sglang_tp2_ep2_dspark", 1, "accept"),
    ("SGLANG_AR_USER_C1", HEADLINE_LANE, 1, "user"),
)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def header(name: str) -> tuple[str, ...]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return tuple(next(csv.reader(stream)))


def manifest_lanes() -> dict[str, dict]:
    return json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["lanes"]


def load_module(name: str, path: Path, stubs: dict[str, types.ModuleType] | None = None) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs or {}):
        spec.loader.exec_module(module)
    return module


def load_chart_renderer() -> types.ModuleType:
    """Load the renderer with tiny matplotlib stubs for data-policy tests."""
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []  # type: ignore[attr-defined]
    matplotlib.use = lambda _backend: None  # type: ignore[attr-defined]
    pyplot = types.ModuleType("matplotlib.pyplot")
    ticker = types.ModuleType("matplotlib.ticker")
    ticker.FuncFormatter = object  # type: ignore[attr-defined]
    return load_module(
        "deepseek_v4_1_flash_chart_policy",
        ROOT / "charts/render-charts.py",
        {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot, "matplotlib.ticker": ticker},
    )


def load_build_data() -> types.ModuleType:
    return load_module("deepseek_v4_1_flash_build_data", DATA / "build_data.py")


def accepted_prefill() -> dict[str, dict[tuple[int, int], dict[str, str]]]:
    grouped: dict[str, dict[tuple[int, int], dict[str, str]]] = defaultdict(dict)
    for row in rows("prefill.csv"):
        if row["publication_status"] == "accepted":
            grouped[row["lane"]][(int(row["isl_tokens"]), int(row["concurrency"]))] = row
    return grouped


def accepted_decode() -> dict[str, dict[int, dict[str, str]]]:
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows("throughput.csv"):
        if row["publication_status"] == "accepted":
            grouped[row["lane"]][int(row["concurrency"])] = row
    return grouped


def positive_lanes(names: tuple[str, ...], column: str) -> set[str]:
    """Lanes that have at least one complete, error-free row in the given tables."""
    lanes = set()
    for name in names:
        for row in rows(name):
            try:
                value = float(row[column])
            except ValueError:
                continue
            if math.isfinite(value) and value > 0 and row["num_errors"] == "0":
                lanes.add(row["lane"])
    return lanes


def table_after_heading(text: str, heading: str) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == heading)
    table: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            table.append(line)
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in table]
    body = [row for row in cells[1:] if not all(set(cell) <= set(":- ") for cell in row)]
    return cells[0], body


def section_text(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def unstyle(cell: str) -> str:
    cell = re.sub(r"<sub>.*?</sub>", "", cell)
    return cell.replace("**", "").replace("†", "").strip()


def placeholders_in(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def section_text_files() -> list[Path]:
    skip_suffixes = {".png", ".pyc"}
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix not in skip_suffixes and "__pycache__" not in path.parts
    )


def write_csv(path: Path, rows_: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows_:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_)


class SectionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.ledger = (ROOT / "notes/PLACEHOLDERS.md").read_text(encoding="utf-8")
        self.lanes = manifest_lanes()
        self.renderer = load_chart_renderer()
        self.short_labels = {
            self.renderer.short_label(lane["lane_label"]): lane_id for lane_id, lane in self.lanes.items()
        }

    # -- helpers -------------------------------------------------------------------------------------

    def assert_placeholder_listed(self, token: str) -> None:
        self.assertIn(token, self.ledger, f"{token} is not listed in notes/PLACEHOLDERS.md")

    def column_lane(self, cell: str) -> str:
        name = unstyle(cell)
        self.assertIn(name, self.short_labels,
                      f"README label {cell!r} is not the shortened lane_label of any lane in sources.json")
        return self.short_labels[name]

    def check_cell(self, cell: str, value: float | None, expected: str | None, lane_has_rows: bool, where: str) -> None:
        """A placeholder may only stand for a lane without rows; a dash only where no point exists; anything
        else must round from the CSV value the renderer would draw."""
        tokens = placeholders_in(cell)
        if tokens:
            for token in tokens:
                self.assert_placeholder_listed(token)
            self.assertFalse(lane_has_rows, f"{where}: placeholder {cell} stands where the lane already has rows")
            return
        if unstyle(cell) == DASH:
            self.assertIsNone(value, f"{where}: dash hides the measured value {expected}")
            return
        self.assertIsNotNone(value, f"{where}: cell {cell!r} has no CSV row behind it")
        self.assertTrue(unstyle(cell).startswith(expected), f"{where}: cell {cell!r} does not round from {expected}")

    def lane_columns(self, columns: list[str]) -> list[str]:
        lanes = [self.column_lane(label) for label in columns]
        self.assertEqual(len(lanes), len(set(lanes)), f"duplicate lane columns in {columns}")
        return lanes

    def assert_drawn_lanes_present(self, chart: str, table_lanes: list[str], series_lanes: dict[str, str]) -> None:
        drawn = self.renderer.chart_series()[chart]
        for label in drawn:
            label = re.sub(r"^\d+\. ", "", label)  # tuning-ladder series carry their step number
            lane = next((lane for lane_label, lane in series_lanes.items() if label.startswith(lane_label)), None)
            self.assertIsNotNone(lane, f"{chart}: series {label!r} is not a lane_label")
            self.assertIn(lane, table_lanes, f"{chart} draws {label!r} but the README table has no column/row for it")

    # -- renderer policy --------------------------------------------------------------------------------

    def test_renderer_filters_publication_statuses(self) -> None:
        renderer = self.renderer
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "prefill.csv").write_text(
                "publication_status,rankable,value\naccepted,true,1\nFAILED_BENCHMARK,false,999\n"
                "diagnostic,false,2\npending,false,3\n",
                encoding="utf-8",
            )
            self.assertEqual([row["value"] for row in renderer.accepted_rows(data / "prefill.csv")], ["1"])
            self.assertEqual([row["value"] for row in renderer.diagnostic_rows(data / "prefill.csv")], ["2"])
            self.assertEqual([row["value"] for row in renderer.drawable_rows(data / "prefill.csv")], ["1", "2"])
            self.assertEqual(renderer.accepted_rows(data / "missing.csv"), [])
        for value in ("0", "", "nan", "-5", "inf", "pending"):
            self.assertIsNone(renderer.positive(value), value)
        self.assertEqual(renderer.positive("26232.2"), 26232.2)
        self.assertEqual(len(renderer.CHART_NAMES), 8)
        self.assertEqual(renderer.short_label("SGLang TP2+EP2 · AR · Data Direct"), "SGLang AR · Data Direct")
        self.assertEqual(renderer.short_label("vLLM PP2 · AR"), "vLLM PP2 · AR")

    def test_renderer_draws_every_lane_with_rows(self) -> None:
        renderer = self.renderer
        series = renderer.chart_series()
        labels = {lane_id: lane["lane_label"] for lane_id, lane in self.lanes.items()}
        for lane in positive_lanes(("prefill.csv", "diagnostic-prefill.csv"), "aggregate_prompt_tokens_per_second"):
            self.assertIn(labels[lane], series["prefill-throughput.png"], f"{lane} has prefill rows but no series")
        for lane in positive_lanes(("prefill.csv", "diagnostic-prefill.csv"), "ttft_p50_seconds"):
            if any(row["lane"] == lane and row["concurrency"] == "1"
                   for name in ("prefill.csv", "diagnostic-prefill.csv") for row in rows(name)):
                self.assertIn(labels[lane], series["prefill-ttft.png"], f"{lane} has C1 rows but no TTFT series")
        for chart, column in (("decode-throughput.png", "aggregate_output_tokens_per_second"),
                              ("decode-per-user.png", "per_user_output_tokens_per_second_p50")):
            for lane in positive_lanes(("throughput.csv",), column):
                self.assertTrue(any(label.startswith(labels[lane]) for label in series[chart]),
                                f"{lane} has decode rows but no series on {chart}")
        for chart, labels_drawn in series.items():
            self.assertEqual(len(labels_drawn), len(set(labels_drawn)), f"{chart} draws a lane twice")

        # Synthetic tables: accepted and diagnostic lanes are drawn, a lane's accepted grid supersedes its own
        # spot check, and pending/failed rows never become a series.
        def lane(lane_id: str, order: int, status: str, **extra: str) -> dict[str, str]:
            return {"lane": lane_id, "lane_label": f"Lane {lane_id}", "series_order": str(order),
                    "publication_status": status, "rankable": "true" if status == "accepted" else "false",
                    "mode": "ar", "num_errors": "0", "ladder_step": "", "ladder_label": "", **extra}

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_csv(data / "prefill.csv", [
                lane("A", 2, "accepted", isl_tokens="65536", concurrency="16", aggregate_prompt_tokens_per_second="100",
                     ttft_p50_seconds="1.5", ladder_step="2", ladder_label="step two"),
                lane("A", 2, "accepted", isl_tokens="65536", concurrency="1", aggregate_prompt_tokens_per_second="90",
                     ttft_p50_seconds="1.0", ladder_step="2", ladder_label="step two"),
            ])
            write_csv(data / "diagnostic-prefill.csv", [
                lane("A", 2, "diagnostic", isl_tokens="65536", concurrency="1", aggregate_prompt_tokens_per_second="999",
                     ttft_p50_seconds="9", ladder_step="2", ladder_label="step two"),
                lane("B", 1, "diagnostic", isl_tokens="65536", concurrency="16", aggregate_prompt_tokens_per_second="50",
                     ttft_p50_seconds="3", ladder_step="1", ladder_label="step one"),
                lane("C", 3, "pending", isl_tokens="65536", concurrency="16", aggregate_prompt_tokens_per_second="70",
                     ttft_p50_seconds="1"),
                lane("D", 4, "FAILED_BENCHMARK", isl_tokens="65536", concurrency="16",
                     aggregate_prompt_tokens_per_second="0", ttft_p50_seconds="0"),
            ])
            write_csv(data / "throughput.csv", [
                lane("A", 2, "accepted", concurrency="1", aggregate_output_tokens_per_second="10",
                     per_user_output_tokens_per_second_p50="10", accept_length=""),
                lane("E", 5, "diagnostic", mode="dspark", concurrency="1", aggregate_output_tokens_per_second="12",
                     per_user_output_tokens_per_second_p50="12", accept_length="2.5"),
                lane("F", 6, "pending", concurrency="1", aggregate_output_tokens_per_second="99",
                     per_user_output_tokens_per_second_p50="99", accept_length=""),
            ])
            renderer.DATA = data
            try:
                prefill = renderer.prefill_series(renderer.RATE)
                self.assertEqual([entry["label"] for entry in prefill], ["Lane B", "Lane A"])
                self.assertEqual(prefill[1]["points"][1][65536], 90.0, "accepted grid must supersede the spot check")
                decode = renderer.decode_series(renderer.DECODE_AGGREGATE)
                self.assertEqual([renderer.decode_label(entry) for entry in decode],
                                 ["Lane A", "Lane E (accept length 2.50 @ C1)"])
                synthetic = renderer.chart_series()
                for chart, drawn in synthetic.items():
                    for absent in ("Lane C", "Lane D", "Lane F"):
                        self.assertFalse(any(absent in label for label in drawn), (chart, absent))
                reference, bars = renderer.ladder_bars()
                self.assertEqual(reference, (65536, 16))
                self.assertEqual([bar["label"] for bar in bars], ["1. Lane B", "2. Lane A"])
                self.assertEqual([bar["value"] for bar in bars], [50.0, 100.0])
            finally:
                renderer.DATA = DATA

    # -- tables ------------------------------------------------------------------------------------------

    def test_lane_labels_and_series_order_are_shared(self) -> None:
        orders = {lane_id: str(lane["series_order"]) for lane_id, lane in self.lanes.items()}
        self.assertEqual(len(set(orders.values())), len(orders), "series_order must be unique per lane")
        for lane_id, lane in self.lanes.items():
            self.assertTrue(lane["lane_label"], lane_id)
            self.assertEqual(self.renderer.short_label(lane["lane_label"]), self.renderer.short_label(lane["lane_label"]))
        for name in LANE_TABLES:
            for row in rows(name):
                self.assertEqual(row["lane_label"], self.lanes[row["lane"]]["lane_label"], (name, row["lane"]))
                self.assertEqual(row["series_order"], orders[row["lane"]], (name, row["lane"]))
        self.assertEqual(len(self.short_labels), len(self.lanes), "shortened lane labels must stay distinct")

    def test_prefill_rows_are_exact_complete_and_unique(self) -> None:
        build_data = load_build_data()
        self.assertEqual(header("prefill.csv"), build_data.PREFILL_FIELDS)
        self.assertEqual(header("diagnostic-prefill.csv"), build_data.DIAGNOSTIC_PREFILL_FIELDS)
        self.assertEqual(header("throughput.csv"), build_data.THROUGHPUT_FIELDS)
        accepted = rows("prefill.csv")
        self.assertTrue(all(row["publication_status"] == "accepted" for row in accepted))
        self.assertTrue(all(row["rankable"] == "true" for row in accepted))
        expected_grid = {(isl, c) for isl in ISLS for c in CONCURRENCIES}
        by_profile: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in accepted:
            by_profile[(row["lane"], row["profile"])].append(row)
        for (lane, profile), cells in by_profile.items():
            keys = {(int(row["isl_tokens"]), int(row["concurrency"])) for row in cells}
            self.assertEqual(len(cells), 12, f"{lane}/{profile} must publish exactly the 4 × 3 grid")
            self.assertEqual(keys, expected_grid, f"{lane}/{profile} grid is incomplete or has extras")
            for row in cells:
                c = int(row["concurrency"])
                self.assertEqual(row["model_id"], MODEL_ID)
                self.assertEqual(row["model_revision"], REVISION)
                self.assertEqual(row["num_errors"], "0")
                self.assertEqual(row["token_count_match"], "true")
                self.assertEqual(int(row["requests"]), max(4, 4 * c))
                self.assertEqual(int(row["actual_prompt_tokens_per_request"]), int(row["isl_tokens"]))
                rate = float(row["aggregate_prompt_tokens_per_second"])
                self.assertTrue(math.isfinite(rate) and rate > 0, row)
                recomputed = int(row["prompt_tokens_total"]) / float(row["wall_seconds"])
                self.assertLess(abs(recomputed / rate - 1), 1e-3, row)
                if row["server_prompt_tokens_delta"]:
                    self.assertEqual(row["server_prompt_tokens_delta"], row["prompt_tokens_total"])
                self.assertTrue(float(row["ttft_p50_seconds"]) > 0)
        for row in rows("diagnostic-prefill.csv"):
            self.assertEqual(row["publication_status"], "diagnostic")
            self.assertEqual(row["rankable"], "false")
            self.assertEqual(row["num_errors"], "0")
            self.assertTrue(row["diagnostic_status"])
            self.assertTrue(float(row["aggregate_prompt_tokens_per_second"]) > 0, row)
        if not accepted:
            self.assertFalse(
                any(row["status"] == "PASS_RANKABLE_PREFILL" for row in rows("qualification.csv")),
                "no accepted prefill rows, so no lane may be ledgered as PASS_RANKABLE_PREFILL",
            )

    def test_readme_tables_round_from_canonical_rows(self) -> None:
        self.assertIn(f"revision `{REVISION}`", self.readme)
        renderer = self.renderer
        labels = {lane["lane_label"]: lane_id for lane_id, lane in self.lanes.items()}
        rates = {entry["lane"]: entry for entry in renderer.prefill_series(renderer.RATE)}
        ttfts = {entry["lane"]: entry for entry in renderer.prefill_series(renderer.TTFT)}

        columns, body = table_after_heading(self.readme, "## Prefill throughput")
        lanes = self.lane_columns(columns[1:])
        self.assert_drawn_lanes_present("prefill-throughput.png", lanes, labels)
        for row in body:
            isl = int(row[0].rstrip("K")) * 1024
            for lane, cell in zip(lanes, row[1:]):
                entry = rates.get(lane)
                best = max((cells[isl] for cells in entry["points"].values() if isl in cells), default=None) \
                    if entry else None
                self.check_cell(cell, best, f"{best:,.0f}" if best is not None else None, entry is not None,
                                f"prefill {row[0]} {lane}")

        columns, body = table_after_heading(self.readme, "## Prefill scaling with concurrency")
        self.assertEqual(columns[1:], [f"C{c}" for c in CONCURRENCIES])
        lanes = self.lane_columns([row[0] for row in body])
        self.assert_drawn_lanes_present("prefill-concurrency.png", lanes, labels)
        for lane, row in zip(lanes, body):
            entry = rates.get(lane)
            for c, cell in zip(CONCURRENCIES, row[1:]):
                value = entry["points"].get(c, {}).get(CONCURRENCY_REFERENCE_ISL) if entry else None
                self.check_cell(cell, value, f"{value:,.0f}" if value is not None else None, entry is not None,
                                f"64K C{c} {lane}")

        columns, body = table_after_heading(self.readme, "## Time to first token")
        lanes = self.lane_columns(columns[1:])
        self.assert_drawn_lanes_present("prefill-ttft.png", lanes, labels)
        for row in body:
            isl = int(row[0].rstrip("K")) * 1024
            for lane, cell in zip(lanes, row[1:]):
                entry = ttfts.get(lane)
                value = entry["points"].get(1, {}).get(isl) if entry else None
                self.check_cell(cell, value, f"{value:.3f}s" if value is not None else None, entry is not None,
                                f"TTFT {row[0]} {lane}")

        for heading, chart, column in (
            ("## Decode throughput", "decode-throughput.png", renderer.DECODE_AGGREGATE),
            ("## Per-user decode speed", "decode-per-user.png", renderer.DECODE_PER_USER),
        ):
            decode = {entry["lane"]: entry for entry in renderer.decode_series(column)}
            columns, body = table_after_heading(self.readme, heading)
            lanes = self.lane_columns(columns[1:])
            self.assert_drawn_lanes_present(chart, lanes, labels)
            for row in body:
                concurrency = int(row[0])
                for lane, cell in zip(lanes, row[1:]):
                    entry = decode.get(lane)
                    value = entry["cells"].get(concurrency) if entry else None
                    self.check_cell(cell, value, f"{value:,.1f}" if value is not None else None, entry is not None,
                                    f"{heading} C{concurrency} {lane}")

        reference, bars = renderer.ladder_bars()
        columns, body = table_after_heading(self.readme, "## SGLang prefill tuning ladder")
        self.assertEqual(columns, ["Step", "Configuration", "Change", "Prompt tok/s", "vs step 1"])
        lanes = self.lane_columns([row[1] for row in body])
        self.assert_drawn_lanes_present("tuning-ladder.png", lanes, labels)
        by_lane = {bar["lane"]: bar for bar in bars}
        base = bars[0]["value"] if bars else None
        for lane, row in zip(lanes, body):
            bar = by_lane.get(lane)
            value = bar["value"] if bar else None
            self.check_cell(row[3], value, f"{value:,.0f}" if value is not None else None, bar is not None,
                            f"ladder {lane} rate")
            if bar is None:
                self.check_cell(row[4], None, None, False, f"ladder {lane} gain")
            elif value == base:
                self.assertEqual(unstyle(row[4]), DASH, f"ladder {lane} is the base step")
            else:
                self.assertEqual(unstyle(row[4]), f"{(value / base - 1) * 100:+.1f}%", f"ladder {lane} gain")
        if reference is not None:
            self.assertIn(f"{reference[0] // 1024}K prompts, C{reference[1]}",
                          section_text(self.readme, "## SGLang prefill tuning ladder"))

        fabric = {row["config_id"]: row for row in rows("fabric.csv") if row["in_place"] == "false"}
        columns, body = table_after_heading(self.readme, "## NCCL all-reduce bus bandwidth")
        for row in fabric.values():
            line = next((line for line in body if line[0] == row["label"]), None)
            self.assertIsNotNone(line, f"fabric config {row['config_id']} missing from README table")
            self.assertEqual(unstyle(line[-1]), f"{float(row['avg_busbw_gbps']):.1f}")

    def test_headline_binds_accepted_rows(self) -> None:
        headline = section_text(self.readme, "## Headline")
        prefill = accepted_prefill()
        decode = accepted_decode()
        for token, lane, isl, concurrency, kind in HEADLINE_PREFILL:
            point = prefill.get(lane, {}).get((isl, concurrency))
            if f"{{{{{token}}}}}" in headline:
                self.assert_placeholder_listed(f"{{{{{token}}}}}")
                self.assertIsNone(point, f"{token} placeholder hides accepted data")
            else:
                self.assertIsNotNone(point, f"{token} was replaced but no accepted row exists")
                expected = (f"{float(point['aggregate_prompt_tokens_per_second']):,.0f}" if kind == "rate"
                            else f"{float(point['ttft_p50_seconds']):.3f}s")
                self.assertIn(expected, headline)
        for token, lane, concurrency, kind in HEADLINE_DECODE:
            point = decode.get(lane, {}).get(concurrency)
            if f"{{{{{token}}}}}" in headline:
                self.assert_placeholder_listed(f"{{{{{token}}}}}")
                self.assertIsNone(point, f"{token} placeholder hides accepted data")
            else:
                self.assertIsNotNone(point, f"{token} was replaced but no accepted row exists")
                expected = (f"{float(point['per_user_output_tokens_per_second_p50']):,.1f}" if kind == "user"
                            else f"{float(point['accept_length']):.2f}")
                self.assertIn(expected, headline)
        best = max((float(row["aggregate_output_tokens_per_second"]) for cells in decode.values()
                    for row in cells.values()), default=None)
        if "{{BEST_DECODE_AGG}}" in headline:
            self.assert_placeholder_listed("{{BEST_DECODE_AGG}}")
            self.assertIsNone(best)
        else:
            self.assertIsNotNone(best)
            self.assertIn(f"**{best:,.1f} aggregate tok/s**", headline)
        _, bars = self.renderer.ladder_bars()
        steps = [bar for bar in bars if not bar["comparison"]]
        if "{{LADDER_DATADIRECT_REF}}" in headline:
            self.assert_placeholder_listed("{{LADDER_DATADIRECT_REF}}")
            self.assertLess(len(steps), 2, "ladder placeholders hide measured steps")
        else:
            self.assertGreaterEqual(len(steps), 2)
            self.assertIn(f"from **{steps[0]['value']:,.0f}** to **{steps[1]['value']:,.0f} prompt tok/s**", headline)
            self.assertIn(f"({(steps[1]['value'] / steps[0]['value'] - 1) * 100:+.1f}%)", headline)
        if len(steps) >= 3:  # every further step is quoted with its gain over step 1 and over the previous step
            for previous, step in zip(steps[1:], steps[2:]):
                self.assertIn(f"**{step['value']:,.0f}**", headline)
                self.assertIn(f"{(step['value'] / steps[0]['value'] - 1) * 100:+.1f}% versus", headline)
                self.assertIn(f"{(step['value'] / previous['value'] - 1) * 100:+.1f}% versus", headline)

    def test_fabric_and_kernel_tables_round_from_csv(self) -> None:
        fabric: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
        for row in rows("fabric.csv"):
            if row["in_place"] == "false":
                fabric[row["config_id"]][int(row["message_bytes"])] = row
        self.assertTrue(fabric)
        columns, body = table_after_heading(self.readme, "## NCCL all-reduce bus bandwidth")
        sizes = {"64 MiB": 64 << 20, "512 MiB": 512 << 20, "2 GiB": 2 << 30}
        self.assertEqual(columns, ["Configuration", *sizes, "Average"])
        self.assertEqual(len(body), len(fabric), "every fabric configuration is a README row, nothing else is")
        for line in body:
            cells = next((c for c in fabric.values() if next(iter(c.values()))["label"] == line[0]), None)
            self.assertIsNotNone(cells, f"README fabric row {line[0]!r} has no fabric.csv configuration")
            for label, cell in zip(columns[1:-1], line[1:-1]):
                self.assertEqual(unstyle(cell), f"{float(cells[sizes[label]]['busbw_gbps']):.1f}", (line[0], label))
            self.assertEqual(unstyle(line[-1]), f"{float(next(iter(cells.values()))['avg_busbw_gbps']):.1f}", line[0])

        profile: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows("profile.csv"):
            profile[(row["label"], row["node"])][row["category"]] = row
        self.assertTrue(profile)
        shown = {"NCCL": "nccl", "Attention": "attention", "GEMM": "gemm", "Norm/RoPE/quant": "norm/rope/quant"}
        columns, body = table_after_heading(self.readme, "## Where the GPU time goes")
        self.assertEqual(columns, ["Profile", "Node", *shown, "Other", "Total"])
        self.assertEqual(len(body), len(profile), "every profiled node is a README row, nothing else is")
        for line in body:
            key = (line[0], line[1])
            self.assertIn(key, profile, f"README kernel-time row {key} has no profile.csv rows")
            categories = profile[key]
            for label, cell in zip(columns[2:-2], line[2:-2]):
                row = categories[shown[label]]
                expected = f"{float(row['category_ms']):,.0f} ms"
                if label == "NCCL":
                    expected += f" · {float(row['category_pct']):.1f}%"
                self.assertEqual(unstyle(cell), expected, (key, label))
            other = round(sum(float(r["category_ms"]) for c, r in categories.items() if c not in shown.values()), 1)
            self.assertEqual(unstyle(line[-2]), f"{other:,.0f} ms", (key, "Other"))
            total = float(next(iter(categories.values()))["summed_kernel_ms"])
            self.assertEqual(unstyle(line[-1]), f"{total:,.0f} ms", (key, "Total"))

        headline = section_text(self.readme, "## Headline")
        for config, size in (("dual_rail_tuned_8ch", 2 << 30), ("single_rail_tuned", 2 << 30)):
            self.assertIn(f"**{float(fabric[config][size]['busbw_gbps']):.1f} GB/s**", headline, config)
        nccl_shares = sorted(
            float(categories["nccl"]["category_pct"]) for (_, node), categories in profile.items() if node == "node0"
        )
        self.assertIn(f"from **{nccl_shares[-1]:.1f}%** to **{nccl_shares[0]:.1f}%**", headline)

    def test_repository_overview_row_binds_headline(self) -> None:
        overview = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        proposed = (ROOT / "notes/ROOT_README_EDITS.md").read_text(encoding="utf-8")
        candidates = [line for line in overview.splitlines() if "[DeepSeek-V4.1-Flash]" in line]
        accepted = accepted_prefill()
        dspark = accepted_decode().get("sglang_tp2_ep2_dspark", {}).get(1)
        if not candidates:
            self.assertIn("| [DeepSeek-V4.1-Flash](deepseek-v4.1-flash/) |", proposed)
            self.assertIn('"deepseek_v4_1_flash_section_contract"', proposed)
            candidates = [line for line in proposed.splitlines() if "| [DeepSeek-V4.1-Flash](deepseek-v4.1-flash/) |" in line]
        row = candidates[0]
        self.assertIn("(deepseek-v4.1-flash/)", row)
        for lane in (HEADLINE_LANE, REPLAY_LANE):
            for isl, concurrency in ((16384, 1), (131072, 16)):
                point = accepted.get(lane, {}).get((isl, concurrency))
                if point is not None:
                    self.assertIn(f"{float(point['aggregate_prompt_tokens_per_second']):,.0f}", row, (lane, isl))
        if dspark is not None:
            self.assertIn(f"{float(dspark['per_user_output_tokens_per_second_p50']):,.1f}", row)
        for token in placeholders_in(row):
            self.assert_placeholder_listed(token)

    def test_unmeasured_lanes_publish_no_timing(self) -> None:
        build_data = load_build_data()
        self.assertEqual(build_data.check_tables(DATA), [])
        qualification = rows("qualification.csv")
        self.assertTrue(qualification)
        decode_lanes = {row["lane"] for row in rows("throughput.csv") if row["publication_status"] == "accepted"}
        prefill_lanes = {row["lane"] for row in rows("prefill.csv")}
        for row in qualification:
            self.assertEqual(row["rankable"] == "true", row["status"].startswith("PASS_RANKABLE"), row)
            if row["status"] in {"PENDING", "NOT_MEASURED", "NOT_ATTEMPTED"} or row["status"].startswith("FAILED"):
                metric = row["profile"].rsplit("_", 1)[-1]
                published = prefill_lanes if metric == "prefill" else decode_lanes
                self.assertNotIn(row["lane"], published, f"{row['profile']} is {row['status']} but has accepted rows")
                if metric == "decode":
                    self.assertFalse(any(r["lane"] == row["lane"] for r in rows("throughput.csv")),
                                     f"{row['profile']} is {row['status']} but throughput.csv has rows for it")
                elif row["lane"] in self.lanes:
                    self.assertFalse(any(r["lane"] == row["lane"] for r in rows("diagnostic-prefill.csv")
                                         if r["run_id"] not in self.spot_check_runs(row["lane"])),
                                     f"{row['profile']} is {row['status']} but diagnostic-prefill.csv has rows for it")
        self.assertTrue(any(row["status"] == "NOT_ATTEMPTED" and row["topology"] == "not_run" for row in qualification))
        self.assertEqual({(row["model_id"], row["model_revision"]) for row in qualification}, {(MODEL_ID, REVISION)})
        for row in rows("throughput.csv"):
            self.assertIn(row["publication_status"], {"accepted", "diagnostic"})
            self.assertEqual(row["rankable"] == "true", row["publication_status"] == "accepted", row["lane"])
        for name, column in (("throughput.csv", "aggregate_output_tokens_per_second"),
                             ("fabric.csv", "busbw_gbps"), ("profile.csv", "category_ms")):
            for row in rows(name):
                value = float(row[column])
                self.assertTrue(math.isfinite(value) and value > 0, (name, row))

    def spot_check_runs(self, lane: str) -> set[str]:
        """Diagnostic spot-check files a lane lists explicitly land in diagnostic-prefill.csv whatever its status."""
        return {Path(p).stem for p in self.lanes[lane]["prefill"].get("diagnostic_jsonl", [])}

    def test_checkpoint_summary_matches_config(self) -> None:
        checkpoint = json.loads((DATA / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["model_id"], MODEL_ID)
        self.assertEqual(checkpoint["revision"], REVISION)
        self.assertEqual(checkpoint["architecture"], "DeepseekV41ForCausalLM")
        self.assertEqual(checkpoint["model_type"], "deepseek_v41")
        self.assertEqual(checkpoint["safetensors_files"], 48)
        self.assertEqual(checkpoint["index_total_size_bytes"], 510286023000)
        self.assertEqual(checkpoint["safetensors_bytes"], 510296708312)
        self.assertEqual(checkpoint["index_tensor_entries"], 96085)
        self.assertEqual(checkpoint["precision"]["expert_dtype"], "fp4")
        self.assertEqual(checkpoint["precision"]["weight_block_size"], [32, 32])
        self.assertEqual(checkpoint["shape"]["num_hidden_layers"], 40)
        self.assertEqual(checkpoint["shape"]["dspark_block_size"], 5)
        self.assertFalse(checkpoint["vision_tower"]["used_in_benchmark"])
        self.assertIn("510,286,023,000 bytes", self.readme)

    def test_headline_is_a_deck_with_captioned_charts(self) -> None:
        renderer = self.renderer
        lines = self.readme.splitlines()
        self.assertLessEqual(len(lines), 170, "the headline README must stay a short deck")
        self.assertTrue(lines[0].startswith("# DeepSeek-V4.1-Flash on 2× NVIDIA GB300 DGX Stations"))
        for unrelated in ("MiniMax", "GLM", "Qwen", "Hy3", "Ornith"):
            self.assertNotIn(unrelated, self.readme)
        for name in renderer.CHART_NAMES:
            embeds = [index for index, line in enumerate(lines) if f"](charts/{name})" in line]
            self.assertEqual(len(embeds), 1, f"{name} must be embedded exactly once")
            caption = next(line for line in lines[embeds[0] + 1:] if line.strip())
            self.assertTrue(caption.startswith("*") and caption.endswith("*"), f"{name} needs an italic caption")
            self.assertTrue((ROOT / "charts" / name).is_file(), f"{name} is not rendered")
        self.assertIn("Details: [recipes/](recipes/) · [notes/](notes/) · [data/](data/)", self.readme)
        self.assertTrue(self.readme.rstrip().endswith("Return to the [repository overview](../)."))
        listed = set(re.findall(r"^\| `(\{\{[A-Z0-9_]+\}\})` \|", self.ledger, re.M))  # table rows, not the prose
        present: set[str] = set()
        for path in (ROOT / "README.md", ROOT / "notes/README.md", ROOT / "recipes/README.md", ROOT / "data/README.md",
                     ROOT / "notes/ROOT_README_EDITS.md"):
            for token in placeholders_in(path.read_text(encoding="utf-8")):
                self.assertIn(token, listed, f"{token} in {path.name} is missing from notes/PLACEHOLDERS.md")
                present.add(token)
        self.assertEqual(listed - present, set(), "notes/PLACEHOLDERS.md lists tokens that no file carries any more")

    def test_no_private_identifiers_anywhere_in_section(self) -> None:
        # Assembled from parts so this file itself stays grep-clean for the forbidden strings.
        forbidden = (
            "gem" + "ini1",
            "gem" + "ini2",
            "/home/" + "catid",
            "/mnt/" + "gem" + "ini2-models",
            "10.10" + ".69",
        )
        uuid = re.compile(r"\bGPU" r"-[0-9a-fA-F]{8}-")  # split so the pattern itself is not a scan hit
        for path in section_text_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in forbidden:
                self.assertNotIn(needle, text, f"{path.relative_to(ROOT)} leaks {needle!r}")
            self.assertIsNone(uuid.search(text), f"{path.relative_to(ROOT)} leaks a GPU UUID")


if __name__ == "__main__":
    unittest.main()
