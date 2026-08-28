#!/usr/bin/env python3
"""Tests for the GLM-5.3 NVFP4 raw-result importer."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_importer():
    path = ROOT / "data/import-nvfp4-results.py"
    name = "glm53_import_nvfp4_results"
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


def make_result(parent: Path) -> Path:
    root = parent / "glm53-nvfp4-tp2-mtp0-test"
    write_json(root / "run-manifest.json", {
        "schema_version": 1,
        "run_id": root.name,
        "profile": IMPORTER.RUN_PROFILE,
        "model": {
            "repository": IMPORTER.MODEL_ID,
            "revision": IMPORTER.MODEL_REVISION,
            "weight_revision": IMPORTER.WEIGHT_REVISION,
            "runtime_config_sha256": IMPORTER.RUNTIME_CONFIG_SHA256,
            "quantization": "modelopt_fp4",
        },
        "runtime_image": IMPORTER.RUNTIME_IMAGE,
        "topology": {"nodes": 2, "tp": 2, "ep": 1},
        "mtp": {"enabled": False, "steps": 0},
        "benchmark": {
            "commit": IMPORTER.BENCH_COMMIT,
            "decode_concurrency": list(IMPORTER.CONCURRENCIES),
            "cold_prefill": ["8K", "64K", "128K"],
            "input_tokens": 8192,
            "measured_requests": "5*C",
            "output_tokens": 1024,
            "warmups": "C",
        },
    })
    for concurrency in IMPORTER.CONCURRENCIES:
        target = 5 * concurrency
        seconds = float(20 + concurrency)
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
                "num_completed": target,
                "num_errors": 0,
                "client_output_tokens": target * 1024,
                "measurement_seconds": seconds,
                "aggregate_source": "openai_completed_usage",
                "aggregate_tps": target * 1024 / seconds,
                "ttft_p50": 0.2,
                "inter_token_latency_p50": 0.008,
                "effective_concurrency": concurrency - 0.1,
                "max_running_reqs": concurrency,
                "underfilled": False,
                "capacity_limited": False,
                "server_accept_len_effective": 0.0,
                "server_steps_per_s": 0.0,
            }],
        })
    cells = {}
    for context in IMPORTER.PREFILL_CONTEXTS:
        prompt_tokens = context + 2
        ttft = prompt_tokens / 20000
        cells[str(context)] = {
            "prompt_tokens": prompt_tokens,
            "samples": 4,
            "tok_per_sec": 20000.0,
            "ttft_seconds": ttft,
            "method": "client",
            "server_validation": {"method": "", "tok_per_sec": 0.0},
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
    (root / "STATUS.retry.txt").write_text(
        "COMPLETE_MEASURED_RAW\ncleanup=PASS_FORCED_EXACT_NAMES\npostflight_retry=PASS\n",
        encoding="utf-8",
    )
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    (root / "runtime/cleanup-verdict.txt").write_text(
        "outcome=PASS_FORCED_EXACT_NAMES\n", encoding="utf-8"
    )
    (root / "postflight").mkdir(parents=True, exist_ok=True)
    (root / "postflight/verdict.retry.txt").write_text(
        "outcome=PASS\nlocal_used_mib=3\nremote_used_mib=6\n", encoding="utf-8"
    )
    return root


def make_dflash_result(parent: Path) -> Path:
    ar_root = make_result(parent)
    root = parent / "glm53-nvfp4-tp2-dflash2-test"
    ar_root.rename(root)

    manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = root.name
    manifest["profile"] = IMPORTER.DFLASH_RUN_PROFILE
    model = manifest.pop("model")
    manifest.pop("mtp")
    manifest.pop("runtime_image")
    manifest["target"] = {
        "repository": model["repository"],
        "revision": model["revision"],
        "weight_revision": model["weight_revision"],
        "quantization": model["quantization"],
        "kv_cache_dtype": "fp8_e4m3",
    }
    manifest["draft"] = {
        "repository": IMPORTER.DFLASH_DRAFT_ID,
        "revision": IMPORTER.DFLASH_DRAFT_REVISION,
        "proposed_tokens": IMPORTER.DFLASH_PROPOSED_TOKENS,
        "quantization": "unquant",
        "dtype": "bfloat16",
    }
    manifest["runtime"] = {
        "image": IMPORTER.DFLASH_RUNTIME_IMAGE,
        "source_commit": IMPORTER.DFLASH_RUNTIME_COMMIT,
    }
    prefill_source = "/results/glm53-nvfp4-tp2-mtp0-v1/benchmark/raw/prefill/cold.json"
    manifest["benchmark"]["cold_prefill"] = {
        "rerun": False,
        "source": prefill_source,
        "sha256": IMPORTER.AR_PREFILL_SHA256,
    }
    write_json(root / "run-manifest.json", manifest)

    (root / "benchmark/raw/prefill/cold.json").unlink()
    write_json(root / "benchmark/raw/prefill/reused-base.json", {
        "schema": "benchmark-prefill-reference/v1",
        "rerun": False,
        "source": prefill_source,
        "sha256": IMPORTER.AR_PREFILL_SHA256,
    })
    (root / "STATUS.retry.txt").unlink()
    (root / "STATUS.txt").write_text("COMPLETE_MEASURED_RAW\n", encoding="utf-8")
    (root / "postflight/verdict.retry.txt").rename(root / "postflight/verdict.txt")
    return root


def make_tp1_result(parent: Path) -> Path:
    root = make_result(parent)
    renamed = parent / "glm53-nvfp4-tp1-ar-test"
    root.rename(renamed)
    root = renamed
    manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    model = manifest.pop("model")
    manifest.pop("runtime_image")
    manifest.pop("mtp")
    manifest.update({
        "run_id": root.name,
        "profile": IMPORTER.TP1_AR_RUN_PROFILE,
        "target": {
            "repository": model["repository"],
            "revision": model["revision"],
            "quantization": model["quantization"],
            "kv_cache_dtype": "fp8_e4m3",
        },
        "draft": None,
        "runtime": {"image": IMPORTER.AR_RUNTIME_IMAGE},
        "topology": {"nodes": 1, "tp": 1, "pp": 1, "ep": 1},
    })
    manifest["benchmark"]["cold_prefill"] = {"measured": True, "source": None}
    write_json(root / "run-manifest.json", manifest)
    (root / "STATUS.retry.txt").rename(root / "STATUS.txt")
    (root / "runtime/cleanup-verdict.txt").write_text(
        "outcome=PASS_GRACEFUL_OR_ABSENT\n", encoding="utf-8"
    )
    (root / "postflight/verdict.retry.txt").rename(root / "postflight/verdict.txt")
    return root


def make_tp1_dflash_result(parent: Path) -> Path:
    root = make_tp1_result(parent)
    renamed = parent / "glm53-nvfp4-tp1-dflash2-test"
    root.rename(renamed)
    root = renamed
    manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = root.name
    manifest["profile"] = IMPORTER.TP1_DFLASH_RUN_PROFILE
    manifest["draft"] = {
        "repository": IMPORTER.DFLASH_DRAFT_ID,
        "revision": IMPORTER.DFLASH_DRAFT_REVISION,
        "proposed_tokens": IMPORTER.DFLASH_PROPOSED_TOKENS,
        "quantization": "unquant",
        "dtype": "bfloat16",
    }
    manifest["runtime"] = {"image": IMPORTER.DFLASH_RUNTIME_IMAGE}
    prefill_source = "/results/glm53-nvfp4-tp1-ar-v1/benchmark/raw/prefill/cold.json"
    manifest["benchmark"]["cold_prefill"] = {
        "measured": False,
        "source": prefill_source,
    }
    write_json(root / "run-manifest.json", manifest)
    (root / "benchmark/raw/prefill/cold.json").unlink()
    write_json(root / "benchmark/raw/prefill/reused-ar.json", {
        "schema": "benchmark-prefill-reference/v1",
        "rerun": False,
        "source": prefill_source,
        "sha256": IMPORTER.TP1_AR_PREFILL_SHA256,
    })
    return root


class ImportNvfp4ResultsTests(unittest.TestCase):
    def test_complete_tp1_profiles_map_single_station_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            ar = IMPORTER.import_result(make_tp1_result(parent))
        self.assertEqual({row["profile"] for row in ar["throughput"]},
                         {IMPORTER.TP1_AR_PUBLICATION_PROFILE})
        self.assertTrue(all(row["topology"] == IMPORTER.TP1_TOPOLOGY
                            for row in ar["throughput"] + ar["prefill"]))

        with tempfile.TemporaryDirectory() as directory:
            dflash = IMPORTER.import_result(make_tp1_dflash_result(Path(directory)))
        self.assertEqual({row["profile"] for row in dflash["throughput"]},
                         {IMPORTER.TP1_DFLASH_PUBLICATION_PROFILE})
        self.assertEqual(len(dflash["throughput"]), 7)
        self.assertEqual(dflash["prefill"], [])

    def test_complete_result_maps_all_raw_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            imported = IMPORTER.import_result(make_result(Path(directory)))
        self.assertEqual(len(imported["throughput"]), 7)
        self.assertEqual(len(imported["prefill"]), 3)
        self.assertEqual(len(imported["diagnostic_throughput"]), 7)
        self.assertEqual(len(imported["diagnostic_prefill"]), 3)
        self.assertEqual(
            {row["profile"] for row in imported["throughput"] + imported["prefill"]},
            {IMPORTER.PUBLICATION_PROFILE},
        )
        self.assertEqual(
            {row["model_revision"] for row in imported["throughput"]},
            {IMPORTER.MODEL_REVISION},
        )
        self.assertEqual(
            [int(row["completed_requests_in_window"])
             for row in imported["diagnostic_throughput"]],
            [5 * concurrency for concurrency in IMPORTER.CONCURRENCIES],
        )
        self.assertTrue(all(row["num_errors"] == "0" for row in imported["throughput"]))

    def test_result_requires_retained_successful_postflight_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_result(Path(directory))
            (root / "postflight/verdict.retry.txt").write_text(
                "outcome=BLOCKED_IDLE_GATE\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "postflight retry did not pass"):
                IMPORTER.import_result(root)

    def test_complete_dflash_result_maps_decode_without_duplicate_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            imported = IMPORTER.import_result(make_dflash_result(Path(directory)))
        self.assertEqual(len(imported["throughput"]), 7)
        self.assertEqual(len(imported["prefill"]), 0)
        self.assertEqual(len(imported["diagnostic_throughput"]), 7)
        self.assertEqual(len(imported["diagnostic_prefill"]), 0)
        self.assertEqual(
            {row["profile"] for row in imported["throughput"]},
            {IMPORTER.DFLASH_PUBLICATION_PROFILE},
        )
        self.assertEqual({row["mtp_tokens"] for row in imported["throughput"]}, {"0"})
        self.assertEqual(
            [int(row["completed_requests_in_window"])
             for row in imported["diagnostic_throughput"]],
            [5 * concurrency for concurrency in IMPORTER.CONCURRENCIES],
        )

    def test_dflash_result_requires_exact_draft_revision_and_direct_postflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = make_dflash_result(Path(directory))
            manifest = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
            manifest["draft"]["revision"] = "wrong"
            write_json(root / "run-manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "draft revision differs"):
                IMPORTER.import_result(root)

        with tempfile.TemporaryDirectory() as directory:
            root = make_dflash_result(Path(directory))
            (root / "postflight/verdict.txt").write_text(
                "outcome=BLOCKED_IDLE_GATE\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "DFlash2 postflight did not pass"):
                IMPORTER.import_result(root)

    def test_reimport_replaces_only_the_matching_publication_profile(self) -> None:
        official = {"profile": "TP2/MTP0", "run_id": "official"}
        old_ar = {"profile": IMPORTER.AR_PUBLICATION_PROFILE, "run_id": "old-ar"}
        old_dflash = {
            "profile": IMPORTER.DFLASH_PUBLICATION_PROFILE,
            "run_id": "old-dflash",
        }
        new_dflash = {
            "profile": IMPORTER.DFLASH_PUBLICATION_PROFILE,
            "run_id": "new-dflash",
        }
        merged = IMPORTER.replace_profile_rows(
            [official, old_ar, old_dflash], [new_dflash]
        )
        self.assertEqual(merged, [official, old_ar, new_dflash])

    def test_diagnostic_reimport_preserves_other_nvfp4_run(self) -> None:
        existing = [
            {"run_id": "native"},
            {"run_id": "ar"},
            {"run_id": "old-dflash"},
        ]
        imported = [{"run_id": "new-dflash"}]
        self.assertEqual(
            IMPORTER.replace_run_rows(existing, imported, {"old-dflash"}),
            [{"run_id": "native"}, {"run_id": "ar"}, {"run_id": "new-dflash"}],
        )


if __name__ == "__main__":
    unittest.main()
