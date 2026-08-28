#!/usr/bin/env python3
"""Tests for importing the minimal Qwen NVFP4-4p89 result roots."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_importer():
    path = ROOT / "data/import-dgx-results.py"
    name = "qwen38_import_dgx_results"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


IMPORTER = load_importer()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_result(
    parent: Path,
    profile: str = "tp2-mtp3",
    *,
    concurrencies: tuple[int, ...] = IMPORTER.CONCURRENCIES,
    include_prefill: bool = True,
) -> Path:
    spec = IMPORTER.PROFILE_SPECS[profile]
    root = parent / f"qwen38-4p89-{profile}-test"
    write_json(root / "run-manifest.json", {
        "schema_version": 1,
        "run_id": root.name,
        "profile": profile,
        "model": {
            "repository": IMPORTER.MODEL_ID,
            "revision": IMPORTER.MODEL_REVISION,
            "quantization": "modelopt_mixed",
        },
        "runtime_image": IMPORTER.RUNTIME_IMAGE,
        "topology": {
            "nodes": spec.nodes,
            "tp": spec.tp_size,
            "ep": spec.ep_size,
        },
        "mtp": {"enabled": spec.mtp_tokens > 0, "steps": spec.mtp_tokens},
        "benchmark": {
            "commit": IMPORTER.BENCH_COMMIT,
            "decode_concurrency": list(IMPORTER.CONCURRENCIES),
            "input_tokens": 8192,
            "output_tokens": 1024,
        },
    })

    for concurrency in concurrencies:
        target = 5 * concurrency
        seconds = float(10 + concurrency)
        write_json(root / f"benchmark/raw/fixed/c{concurrency}.json", {
            "metadata": {
                "version": IMPORTER.BENCH_VERSION,
                "engine": "sglang",
                "model": IMPORTER.MODEL_ID,
                "max_tokens": 1024,
                "temperature": 0.0,
                "ignore_eos": True,
                "concurrency_levels": [concurrency],
                "context_lengths": [8192],
                "request_count": target,
                "warmup_request_count": concurrency,
            },
            "results": [{
                "concurrency": concurrency,
                "context_tokens": 8192,
                "request_count_target": target,
                "request_count": target,
                "completed_request_count": target,
                "num_errors": 0,
                "client_output_tokens": target * 1024,
                "measurement_seconds": seconds,
                "aggregate_tps": target * 1024 / seconds,
                "ttft_p50": 0.1 + concurrency / 1000,
                "inter_token_latency_p50": 0.005,
                "effective_concurrency": 42.0 if concurrency == 64 else float(concurrency),
                "queue_fraction": 0.75 if concurrency == 64 else 0.0,
                "capacity_limited": concurrency == 64,
            }],
        })

    if include_prefill:
        cells = {}
        for context in IMPORTER.PREFILL_CONTEXTS:
            prompt_tokens = context + 2
            ttft = context / 30000
            cells[str(context)] = {
                "prompt_tokens": prompt_tokens,
                "samples": 5,
                "tok_per_sec": prompt_tokens / ttft,
                "ttft_seconds": ttft,
            }
        write_json(root / "benchmark/raw/prefill/cold.json", {
            "metadata": {
                "version": IMPORTER.BENCH_VERSION,
                "engine": "sglang",
                "model": IMPORTER.MODEL_ID,
                "max_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "standalone_prefill": True,
                "prefill_only": True,
            },
            "prefill": cells,
        })
    (root / "benchmark/raw/STATUS.txt").write_text("COMPLETE\n", encoding="utf-8")
    (root / "STATUS.txt").write_text("COMPLETE_MEASURED_RAW\n", encoding="utf-8")
    return root


class ImportDgxResultsTests(unittest.TestCase):
    def test_imports_observed_cells_without_a_completion_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(
                Path(directory), concurrencies=(1, 64), include_prefill=False
            )
            profile, rows = IMPORTER.import_result(root)
        self.assertEqual(profile, "tp2-mtp3")
        self.assertEqual([int(row["concurrency"]) for row in rows], [1, 64])
        self.assertEqual(rows[-1]["effective_concurrency"], "42.0")
        self.assertEqual(rows[-1]["queue_fraction"], "0.75")
        self.assertEqual(rows[-1]["capacity_limited"], "true")
        self.assertEqual(rows[-1]["publication_status"], "MEASURED_CURRENT")

    def test_require_complete_rejects_a_partial_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(
                Path(directory), concurrencies=(1, 2), include_prefill=False
            )
            with self.assertRaisesRegex(ValueError, "missing decode cells"):
                IMPORTER.import_result(root, require_complete=True)

    def test_require_complete_rejects_an_unfinished_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(Path(directory))
            (root / "STATUS.txt").unlink()
            with self.assertRaisesRegex(ValueError, "result status"):
                IMPORTER.import_result(root, require_complete=True)

    def test_require_complete_rejects_an_unfinished_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(Path(directory))
            (root / "benchmark/raw/STATUS.txt").write_text(
                "RUNNING\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "benchmark client has not completed"):
                IMPORTER.import_result(root, require_complete=True)

    def test_complete_five_profile_import_has_55_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            imported = {}
            for profile in IMPORTER.PROFILE_SPECS:
                root = make_result(parent, profile)
                imported_profile, rows = IMPORTER.import_result(
                    root, require_complete=True
                )
                imported[imported_profile] = rows
        merged = IMPORTER.merge_rows([], imported)
        self.assertEqual(set(imported), set(IMPORTER.PROFILE_SPECS))
        self.assertEqual(len(merged), 55)
        self.assertEqual(len([row for row in merged if row["metric"] == "decode"]), 35)
        self.assertEqual(len([row for row in merged if row["metric"] == "prefill"]), 20)

    def test_tp1_profile_is_labeled_as_one_station(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(Path(directory), "tp1-mtp0")
            profile, rows = IMPORTER.import_result(root, require_complete=True)
        self.assertEqual(profile, "tp1-mtp0")
        self.assertEqual({row["platform_label"] for row in rows}, {"DGX Station"})
        self.assertEqual(
            {row["topology"] for row in rows},
            {"single_node_tp1_ep_disabled"},
        )
        self.assertEqual({row["notes"] for row in rows}, {"one_engine_on_one_station"})

    def test_first_new_import_drops_the_superseded_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(
                Path(directory), concurrencies=(1,), include_prefill=False
            )
            profile, rows = IMPORTER.import_result(root)
        old = dict.fromkeys(IMPORTER.FIELDS, "")
        old.update({
            "model_id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "profile": "NVFP4 TP1/MTP0",
            "metric": "decode",
            "concurrency": "1",
        })
        merged = IMPORTER.merge_rows([old], {profile: rows})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["model_id"], IMPORTER.MODEL_ID)

    def test_importing_one_profile_preserves_another_new_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first_root = make_result(parent, "tp2-mtp0")
            second_root = make_result(parent, "tep2-mtp3")
            first_profile, first_rows = IMPORTER.import_result(first_root)
            second_profile, second_rows = IMPORTER.import_result(second_root)
        merged = IMPORTER.merge_rows(first_rows, {second_profile: second_rows})
        self.assertEqual(
            {row["profile"] for row in merged},
            {
                IMPORTER.PROFILE_SPECS[first_profile].publication_profile,
                IMPORTER.PROFILE_SPECS[second_profile].publication_profile,
            },
        )

    def test_wrong_checkpoint_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(Path(directory))
            path = root / "run-manifest.json"
            manifest = json.loads(path.read_text())
            manifest["model"]["revision"] = "wrong"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "model revision differs"):
                IMPORTER.import_result(root)


if __name__ == "__main__":
    unittest.main()
