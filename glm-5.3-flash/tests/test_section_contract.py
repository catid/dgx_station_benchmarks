#!/usr/bin/env python3
"""Offline consistency gates for the GLM-5.3-Flash publication section."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPOSITORY = ROOT.parent
PROFILES = ("TP2/MTP0", "TP2/MTP5", "TEP2/MTP5")
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
CONTEXTS = (8192, 65536, 131072)

EXTERNAL_FIELDS = (
    "source_status",
    "platform_label",
    "hardware",
    "gpu_count",
    "model_id",
    "model_revision",
    "runtime",
    "runtime_revision",
    "topology",
    "expert_parallelism",
    "overlay_repository",
    "overlay_revision",
    "mtp_tokens",
    "metric",
    "concurrency",
    "context_tokens",
    "tokens_per_second",
    "note",
)
EXTERNAL_BINDING = (
    "EXTERNAL_USER_SUPPLIED",
    "4x RTX PRO 6000 workstation",
    "NVIDIA RTX PRO 6000",
    "4",
    "zai-org/GLM-5.3-Flash",
    "NOT_SUPPLIED",
    "SM120 support overlay runtime",
    "NOT_SUPPLIED",
    "FOUR_GPU_SINGLE_SERVER_PARALLELISM_NOT_SUPPLIED",
    "NOT_SUPPLIED",
    "https://github.com/chriswritescode-dev/glm-5.3-flash-sm120",
    "NOT_SUPPLIED",
    "5",
)
EXPECTED_EXTERNAL_ROWS = tuple(
    EXTERNAL_BINDING + measurement
    for measurement in (
        ("decode", "1", "", "148.5", ""),
        ("decode", "2", "", "219.8", ""),
        ("decode", "4", "", "324.8", ""),
        ("decode", "8", "", "458.3", ""),
        (
            "decode",
            "10",
            "",
            "522.4",
            "External-only concurrency; recommended configuration caps execution at 10 sequences",
        ),
        ("prefill", "", "8192", "9936", ""),
        ("prefill", "", "65536", "10152", ""),
        ("prefill", "", "131072", "9892", ""),
    )
)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_chart_renderer() -> types.ModuleType:
    """Load the renderer with tiny matplotlib stubs for data-policy tests."""
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []  # type: ignore[attr-defined]
    matplotlib.use = lambda _backend: None  # type: ignore[attr-defined]
    pyplot = types.ModuleType("matplotlib.pyplot")
    ticker = types.ModuleType("matplotlib.ticker")
    ticker.FuncFormatter = object  # type: ignore[attr-defined]
    module_name = "glm_5_3_flash_chart_policy"
    path = ROOT / "charts/render-charts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load chart renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot, "matplotlib.ticker": ticker},
    ):
        spec.loader.exec_module(module)
    return module


class SectionContractTests(unittest.TestCase):
    def test_renderer_selects_measured_dgx_and_supplied_comparison_rows(self) -> None:
        renderer = load_chart_renderer()
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "external-rtx-pro-6000.csv").write_text(
                "source_status,metric,tokens_per_second\n"
                "EXTERNAL_USER_SUPPLIED,decode,1\n"
                "OTHER,decode,999\n"
                "EXTERNAL_USER_SUPPLIED,prefill,2\n",
                encoding="utf-8",
            )
            (data / "throughput.csv").write_text(
                "publication_status,value\nmeasured,1\nother,999\n",
                encoding="utf-8",
            )
            renderer.DATA = data
            self.assertEqual(
                [row["tokens_per_second"] for row in renderer.external_rows("decode")],
                ["1"],
            )
            self.assertEqual(
                [row["value"] for row in renderer.measured_rows(data / "throughput.csv")],
                ["1"],
            )

    def test_external_handoff_is_exact(self) -> None:
        external = rows("external-rtx-pro-6000.csv")
        self.assertEqual(len(external), 8)
        self.assertEqual(tuple(external[0]), EXTERNAL_FIELDS)
        self.assertEqual(
            tuple(tuple(row[field] for field in EXTERNAL_FIELDS) for row in external),
            EXPECTED_EXTERNAL_ROWS,
        )
        self.assertEqual(
            len({(row["metric"], row["concurrency"], row["context_tokens"]) for row in external}),
            len(external),
        )

    def test_all_completed_dgx_measurements_are_published(self) -> None:
        decode = rows("throughput.csv")
        prefill = rows("prefill.csv")
        self.assertEqual(len(decode), len(PROFILES) * len(CONCURRENCIES))
        self.assertEqual(len(prefill), len(PROFILES) * len(CONTEXTS))
        self.assertEqual({row["profile"] for row in decode + prefill}, set(PROFILES))
        self.assertEqual({int(row["concurrency"]) for row in decode}, set(CONCURRENCIES))
        self.assertEqual({int(row["nominal_context_tokens"]) for row in prefill}, set(CONTEXTS))
        self.assertTrue(all(row["publication_status"] == "measured" for row in decode + prefill))
        self.assertTrue(all(row["num_errors"] == "0" for row in decode + prefill))
        self.assertTrue(all(float(row["aggregate_output_tokens_per_second"]) > 0 for row in decode))
        self.assertTrue(all(float(row["prompt_tokens_per_second"]) > 0 for row in prefill))
        self.assertTrue(all((ROOT / row["result_file"]).is_file() for row in decode + prefill))

    def test_compact_rows_match_full_acquisition_records(self) -> None:
        decode_detail = {
            (row["run_id"], row["concurrency"]): row
            for row in rows("diagnostic-throughput.csv")
        }
        for row in rows("throughput.csv"):
            detail = decode_detail[(row["run_id"], row["concurrency"])]
            self.assertEqual(row["aggregate_output_tokens_per_second"], detail["aggregate_output_tokens_per_second"])
            self.assertEqual(row["effective_concurrency"], detail["effective_concurrency"])
            self.assertEqual(row["max_running_requests"], detail["max_running_requests"])
            self.assertEqual(row["saturation_observed"], detail["underfilled"])
            self.assertEqual(row["source_artifact_sha256"], detail["source_artifact_sha256"])

        prefill_detail = {
            (row["run_id"], row["nominal_context_tokens"]): row
            for row in rows("diagnostic-prefill.csv")
        }
        for row in rows("prefill.csv"):
            detail = prefill_detail[(row["run_id"], row["nominal_context_tokens"])]
            self.assertEqual(row["actual_prompt_tokens"], detail["actual_prompt_tokens"])
            self.assertEqual(row["samples"], detail["samples"])
            self.assertEqual(row["prompt_tokens_per_second"], detail["client_prompt_tokens_per_second"])
            self.assertEqual(row["source_artifact_sha256"], detail["source_artifact_sha256"])

    def test_full_records_preserve_saturation_and_measurement_status(self) -> None:
        decode = rows("diagnostic-throughput.csv")
        prefill = rows("diagnostic-prefill.csv")
        self.assertEqual(len(decode), 24)
        self.assertEqual(len(prefill), 9)
        self.assertTrue(all(row["measurement_status"] == "MEASURED" for row in decode + prefill))
        self.assertTrue(all(row["rankable"] == "true" for row in decode + prefill))
        self.assertTrue(any(row["underfilled"] == "true" for row in decode))
        self.assertTrue(all(math.isfinite(float(row["aggregate_output_tokens_per_second"])) for row in decode))
        self.assertTrue(all(math.isfinite(float(row["client_prompt_tokens_per_second"])) for row in prefill))

    def test_qualification_ledger_marks_completed_profiles_measured(self) -> None:
        qualification = rows("qualification.csv")
        measured = [row for row in qualification if row["profile"].endswith("_full")]
        self.assertEqual(len(measured), 3)
        self.assertTrue(all(row["status"].startswith("MEASURED") for row in measured))
        self.assertTrue(all(row["rankable"] == "true" for row in measured))

        identities = {(row["model_id"], row["model_revision"]) for row in qualification}
        self.assertEqual(
            identities,
            {
                ("zai-org/GLM-5.3-Flash", "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"),
                ("LibertAIDAI/GLM-5.3-Flash-NVFP4", "11d73216cd636238e82e1d77fe1042ffab36e7fa"),
                ("dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4", "d4d79fbbd474599db610b90a44b77497256ab518"),
            },
        )

    def test_readme_headlines_round_from_canonical_rows(self) -> None:
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        decode = rows("throughput.csv")
        by_decode = {
            (row["profile"], int(row["concurrency"])): float(row["aggregate_output_tokens_per_second"])
            for row in decode
        }
        for concurrency in CONCURRENCIES:
            values = [by_decode[(profile, concurrency)] for profile in PROFILES]
            expected = "| {} | {} | {} | {} |".format(
                concurrency, *(f"{value:,.1f}" for value in values)
            )
            self.assertIn(expected, section)

        prefill = rows("prefill.csv")
        by_prefill = {
            (row["profile"], int(row["nominal_context_tokens"])): float(row["prompt_tokens_per_second"])
            for row in prefill
        }
        for context in CONTEXTS:
            values = [by_prefill[(profile, context)] for profile in PROFILES]
            expected = "| {}K | {} | {} | {} |".format(
                context // 1024, *(f"{value:,.0f}" for value in values)
            )
            self.assertIn(expected, section)

        external = rows("external-rtx-pro-6000.csv")
        for row in external:
            if row["metric"] == "decode":
                self.assertIn(f'| C{row["concurrency"]} | {float(row["tokens_per_second"]):,.1f} |', section)
        revision = json.loads((DATA / "checkpoint.json").read_text(encoding="utf-8"))["revision"]
        self.assertIn(f"revision `{revision}`", section)
        self.assertLess(section.index("## 2× DGX Station GB300"), section.index("## 4× RTX PRO 6000 comparison"))

        overview = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        overview_row = next(line for line in overview.splitlines() if "[GLM-5.3-Flash]" in line)
        self.assertIn("(glm-5.3-flash/)", overview_row)

    def test_headline_and_graph_labels_are_plain_measurement_labels(self) -> None:
        headline = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for phrase in ("none qualifies", "failed test", "excluded", "unranked", "†"):
            self.assertNotIn(phrase, headline)
        self.assertNotIn("MiniMax", headline)
        self.assertNotIn("DS4F", headline)

        renderer = (ROOT / "charts/render-charts.py").read_text(encoding="utf-8")
        visible_lines = "\n".join(
            line for line in renderer.splitlines()
            if any(marker in line for marker in ("label=", "set_title", "set_xlabel", "set_ylabel", "annotate", "figure.text"))
        ).lower()
        for word in ("sealed", "source", "evidence", "qualification", "failed", "accepted", "rejected", "external"):
            self.assertNotIn(word, visible_lines)


if __name__ == "__main__":
    unittest.main()
