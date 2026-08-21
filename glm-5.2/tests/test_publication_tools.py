#!/usr/bin/env python3
"""Synthetic acceptance/rejection tests for the GLM publication pipeline."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXTRACT = load_module("glm_extract_results", PACKAGE / "recipes" / "extract_results.py")
RENDER = load_module("glm_render_charts", PACKAGE / "recipes" / "render_charts.py")
UPDATE = load_module("glm_update_readme", PACKAGE / "recipes" / "update_readme.py")


class PublicationToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="glm52-publication-test-")
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.package = self.root / "package"
        (self.package / "data").mkdir(parents=True)
        (self.package / "charts").mkdir()
        shutil.copy2(PACKAGE / "README.md", self.package / "README.md")
        (self.package / "data" / "manual-quality-review.json").write_text(
            json.dumps({"tp2": {"status": "clean", "notes": "Synthetic fixture."}}) + "\n"
        )
        self.make_run("tp2")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def make_run(self, topology: str) -> None:
        settings = EXTRACT.TOPOLOGIES[topology]
        run = self.results / f"glm52-nvfp4-{topology}"
        runtime = run / "runtime"
        quality = run / "quality"
        runtime.mkdir(parents=True)
        quality.mkdir()
        (quality / "runtime").mkdir()
        results = []
        for concurrency in EXTRACT.CONCURRENCIES:
            results.append({
                "concurrency": concurrency, "context_tokens": 8192,
                "benchmark_mode": "duration", "measurement_seconds": 30.0,
                "client_output_tokens": 1024 * concurrency,
                "server_output_tokens": 1024 * concurrency,
                "aggregate_source": "openai_continuous_usage",
                "aggregate_tps": 100.0 * concurrency,
                "per_request_avg_tps": 100.0, "ttft_p50": 0.1,
                "inter_token_latency_p50": 0.01, "input_seq_len_avg": 8192.0,
                "output_seq_len_avg": 1024.0, "request_count": concurrency,
                "completed_request_count": concurrency, "num_errors": 0,
                "avg_running_reqs": concurrency, "max_running_reqs": concurrency,
                "effective_concurrency": concurrency, "avg_queue_reqs": 0,
                "max_queue_reqs": 0, "capacity_limited": False,
                "request_samples": [{
                    "completed": True, "input_tokens": 8192, "output_tokens": 1024,
                }],
            })
        benchmark = {
            "metadata": {
                "version": "test", "engine": "vllm", "model": "GLM-5.2-NVFP4",
                "timestamp": "2026-08-19T00:00:00", "decode_mode": "duration",
                "duration_per_test": 30.0, "max_tokens": 1024, "temperature": 0.0,
                "ignore_eos": True, "concurrency_levels": EXTRACT.CONCURRENCIES,
                "context_lengths": [8192],
            },
            "results": results,
            "prefill": {
                str(context): {
                    "prompt_tokens": context + 2, "tok_per_sec": float(context),
                    "ttft_seconds": 0.5, "samples": 2, "method": "client",
                }
                for context in EXTRACT.PREFILL_CONTEXTS
            },
        }
        (run / "llm-inference-bench.json").write_text(json.dumps(benchmark) + "\n")

        command = [
            "serve", "/model", "--served-model-name", "GLM-5.2-NVFP4",
            "--tensor-parallel-size", str(settings["tensor_parallel_size"]),
            "--pipeline-parallel-size", str(settings["pipeline_parallel_size"]),
            "--nnodes", "2", "--kv-cache-dtype", "fp8_e4m3",
            "--max-model-len", "135168", "--reasoning-parser", "glm45",
            "--tool-call-parser", "glm47", "--enable-prefix-caching",
            "--enable-auto-tool-choice", "--moe-backend", "flashinfer_cutedsl",
        ]
        command.append("--no-enable-flashinfer-autotune")
        if topology == "tp2":
            command.append("--enable-expert-parallel")
        for rank, host in enumerate(("node0", "node1")):
            ranked_command = command + ["--node-rank", str(rank)]
            inspect = [{
                "Config": {"Image": EXTRACT.RUNTIME_IMAGE, "Cmd": ranked_command},
                "Image": EXTRACT.RUNTIME_IMAGE_ID, "State": {"Status": "running"},
            }]
            (runtime / f"{host}-inspect.json").write_text(json.dumps(inspect) + "\n")
            autotune = "Skipping FlashInfer autotune because it is disabled"
            startup = "Application startup complete\n" if rank == 0 else ""
            (runtime / f"{host}-server.log").write_text(
                f"NCCL NET/IB mlx5_0\n{autotune}\n{startup}"
            )
            (runtime / f"{host}-nvidia-smi.csv").write_text("NVIDIA GB300\n")
        (runtime / "models.json").write_text(json.dumps({
            "data": [{"id": "GLM-5.2-NVFP4", "max_model_len": 135168}]
        }) + "\n")
        (runtime / "metrics-before.txt").write_text("vllm:num_requests_running 0\n")
        (runtime / "metrics-after.txt").write_text("vllm:num_requests_running 0\n")
        (runtime / "kv-cache-tokens.txt").write_text("2000000\n")
        (runtime / "llm-inference-bench-commit.txt").write_text(EXTRACT.BENCHMARK_COMMIT + "\n")
        (runtime / "lm-eval-commit.txt").write_text(EXTRACT.LM_EVAL_COMMIT + "\n")
        (runtime / "model-revision.txt").write_text(EXTRACT.MODEL_REVISION + "\n")

        outputs = []
        for index in range(4):
            text = f"Synthetic coherent answer number {index} with several distinct useful words."
            outputs.append({
                "prompt_index": index, "prompt": f"Prompt {index}", "finish_reason": "stop",
                "usage": {"completion_tokens": 12}, "reasoning_content": "", "content": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "analysis": EXTRACT.analyze(text),
            })
        (quality / "quality-audit.json").write_text(json.dumps({
            "model": "GLM-5.2-NVFP4", "settings": {"temperature": 0, "max_tokens": 4096},
            "outputs": outputs,
        }) + "\n")
        (quality / "provenance.json").write_text(json.dumps({
            "model_revision": EXTRACT.MODEL_REVISION,
            "runtime_image": EXTRACT.RUNTIME_IMAGE,
            "topology": topology,
            "expert_parallel": topology == "tp2",
            "moe_backend": "flashinfer_cutedsl",
            "flashinfer_autotune": False,
            "kv_cache_dtype": "bfloat16",
            "server_max_model_len": 8192,
            "audit_max_tokens": 4096,
        }) + "\n")
        (quality / "runtime" / "node0-server.log").write_text(
            "'max_model_len': 8192 'kv_cache_dtype': 'bfloat16' "
            "'tensor_parallel_size': 2 'enable_expert_parallel': True "
            "'enable_flashinfer_autotune': False 'moe_backend': 'flashinfer_cutedsl'\n"
        )
        (quality / "runtime" / "node1-server.log").write_text("BF16 KV rank 1\n")

    def test_accept_render_update_and_reject_without_stale_rows(self) -> None:
        EXTRACT.extract(self.results, self.package)
        self.assertEqual(len(self.csv_rows(self.package / "data" / "throughput.csv")), 8)
        self.assertEqual(len(self.csv_rows(self.package / "data" / "prefill.csv")), 3)
        self.assertEqual(len(self.csv_rows(self.package / "data" / "quality-audit.csv")), 4)
        self.assertTrue((self.package / "data" / "evidence" / "tp2-benchmark.json").is_file())
        benchmark_evidence = json.loads(
            (self.package / "data" / "evidence" / "tp2-benchmark.json").read_text()
        )
        runtime_evidence = json.loads(
            (self.package / "data" / "runtime" / "tp2.json").read_text()
        )
        self.assertEqual(
            benchmark_evidence["source_file"],
            "/results/glm52-nvfp4-tp2/llm-inference-bench.json",
        )
        self.assertEqual(
            runtime_evidence["source_directory"],
            "/results/glm52-nvfp4-tp2/runtime",
        )
        for path in [
            *(self.package / "data" / "evidence").glob("*.json"),
            *(self.package / "data" / "runtime").glob("*.json"),
        ]:
            public_text = path.read_text()
            self.assertNotIn(str(self.root), public_text)
            self.assertNotIn("/home/", public_text)
            self.assertNotIn("192.168.", public_text)
            self.assertNotRegex(public_text, r"GPU-[0-9a-fA-F-]{16,}")

        self.assertEqual(
            EXTRACT.scrub_public(
                "/home/operator/frontier-bench/glm52-staging/tokenizer"
            ),
            "/workspace/glm52-staging/tokenizer",
        )
        self.assertEqual(
            EXTRACT.scrub_public(
                "/home/operator/benchmarks/lm-evaluation-harness/lm_eval/task.py"
            ),
            "/workspace/lm-evaluation-harness/lm_eval/task.py",
        )
        private_example = ".".join(map(str, (192, 168, 200, 1)))
        self.assertEqual(EXTRACT.scrub_public(private_example), "192.0.2.1")

        RENDER.style()
        throughput = RENDER.read_rows(self.package / "data" / "throughput.csv")
        prefill = RENDER.read_rows(self.package / "data" / "prefill.csv")
        RENDER.render_throughput(throughput, self.package / "charts")
        RENDER.render_prefill(prefill, self.package / "charts")
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.package / "charts").glob("*.png")
        }
        RENDER.render_throughput(throughput, self.package / "charts")
        RENDER.render_prefill(prefill, self.package / "charts")
        self.assertEqual(hashes, {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.package / "charts").glob("*.png")
        })

        UPDATE.update(self.package)
        first = (self.package / "README.md").read_text()
        UPDATE.update(self.package)
        self.assertEqual(first, (self.package / "README.md").read_text())
        self.assertIn(
            "| **100.0** | **6,400.0** | **12,800.0** | "
            "**8,192 / 0.500s** | **65,536 / 0.500s** | "
            "**131,072 / 0.500s** |",
            first,
        )
        self.assertIn("| bfloat16 | — | — | — | 4/4 finished naturally | 0 | clean |", first)
        self.assertNotIn("operator-observed startup failure", first)
        self.assertNotIn("no fit", first.lower())
        self.assertNotIn("P10", first)
        manifest = json.loads(
            (self.package / "data" / "publication-manifest.json").read_text()
        )
        self.assertEqual(
            manifest["rejected_topologies"]["pp2"],
            EXTRACT.STARTUP_FAILURES["pp2"],
        )

        benchmark_path = self.results / "glm52-nvfp4-tp2" / "llm-inference-bench.json"
        benchmark = json.loads(benchmark_path.read_text())
        benchmark["results"][0]["num_errors"] = 1
        benchmark_path.write_text(json.dumps(benchmark) + "\n")
        EXTRACT.extract(self.results, self.package)
        self.assertEqual(self.csv_rows(self.package / "data" / "throughput.csv"), [])
        self.assertFalse((self.package / "data" / "evidence" / "tp2-benchmark.json").exists())


if __name__ == "__main__":
    unittest.main()
