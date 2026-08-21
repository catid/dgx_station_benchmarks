#!/usr/bin/env python3
"""CPU-only validation of manifests and fail-closed plan rules."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_plan import MANIFEST_DIR, PlanError, load_json, resolve, validate  # noqa: E402
from pp2_correctness_smoke import analyze  # noqa: E402
from postvalidate_pp2_correctness import validate_retained  # noqa: E402
from validate_run import validate_correctness_smoke  # noqa: E402


def must_reject(plan: dict, message: str) -> None:
    try:
        validate(plan)
    except PlanError:
        return
    raise AssertionError(f"unsafe plan was accepted: {message}")


def main() -> None:
    software = load_json(MANIFEST_DIR / "software.json")
    experiments = load_json(MANIFEST_DIR / "experiments.json")
    exclusions = load_json(MANIFEST_DIR / "exclusions.json")
    checkpoints = load_json(MANIFEST_DIR / "checkpoints.json")
    assert software["target"]["revision"] == "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
    assert all("@sha256:" in runtime["image"] for runtime in software["runtimes"].values())
    assert len(exclusions["exclusions"]) >= 10
    assert sum(item["disposition"] == "target" for item in checkpoints["checkpoints"]) == 1

    resolved = {}
    for profile_id, raw in experiments["profiles"].items():
        winner = "flashinfer_cutedsl" if raw.get("moe_backend") == "$WINNER_BACKEND" else None
        resolved[profile_id] = resolve(profile_id, winner)

    assert resolved["vllm-tp2-exact"]["moe_backend"] == "flashinfer_cutedsl"
    assert resolved["vllm-pp2-balanced"]["pp_layer_partition"] == "40,38"
    assert resolved["vllm-tp2-mtp5"]["speculation"]["num_speculative_tokens"] == 5
    assert resolved["vllm-tp2-mtp5"]["speculation"]["moe_backend"] == "flashinfer_cutlass"
    assert resolved["vllm-tp2-mtp5"]["vllm_use_v2_model_runner"] == 0
    split = resolved["vllm-tp2-mtp1-split-bootstrap"]
    assert split["moe_backend"] == "flashinfer_cutedsl"
    assert split["speculation"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "moe_backend": "flashinfer_cutlass",
    }
    assert split["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
    assert split["benchmark"]["minimum_kv_tokens"] == 24578
    short = resolved["vllm-tp2-mtp1-split-short-context"]
    assert short["max_model_len"] == 9216
    assert short["max_num_seqs"] == 8
    assert short["gpu_memory_utilization"] == 0.95
    assert short["cudagraph_capture_sizes"] == [1, 2, 4, 8]
    assert short["benchmark"]["minimum_kv_tokens"] == 16384
    assert short["cache_profile_id"] == "vllm-tp2-mtp1-split-bootstrap"
    smoke = resolved["vllm-pp2-block64-eager-smoke"]
    assert smoke["topology"] == "pp2"
    assert smoke["pp_layer_partition"] == "40,38"
    assert smoke["block_size"] == 64
    assert smoke["enforce_eager"] is True
    assert smoke["cudagraph_capture_sizes"] == []
    assert smoke["max_model_len"] == 16384
    assert smoke["benchmark"]["minimum_kv_tokens"] == 12802
    assert smoke["reportable"] is False
    assert resolved["sglang-pp2-balanced"]["speculation"]["method"] == "none"

    base = resolved["vllm-tp2-exact"]
    bad = copy.deepcopy(base)
    bad["cpu_offload_gb"] = 1
    must_reject(bad, "CPU offload")
    bad = copy.deepcopy(base)
    bad["topology"], bad["tp_size"], bad["pp_size"] = "pp2", 1, 2
    bad["expert_parallel_size"] = 1
    bad["pp_layer_partition"] = "39,39"
    must_reject(bad, "unbalanced PP")
    bad = copy.deepcopy(resolved["vllm-pp2-balanced"])
    bad["speculation"] = {"method": "mtp", "num_speculative_tokens": 1}
    must_reject(bad, "PP plus MTP")
    bad = copy.deepcopy(base)
    bad["moe_backend"] = "flashinfer_b12x"
    must_reject(bad, "SM120 backend on SM103")
    bad = copy.deepcopy(resolved["sglang-tp2-cutedsl"])
    bad["speculation"] = {"method": "mtp", "num_speculative_tokens": 1}
    must_reject(bad, "quarantined SGLang speculative TP2")
    for method in ("dspark", "dflash", "eagle3"):
        bad = copy.deepcopy(base)
        bad["speculation"] = {"method": method, "num_speculative_tokens": 1}
        must_reject(bad, f"excluded speculative method {method}")
    bad = copy.deepcopy(base)
    bad["moe_backend"] = "cutlass"
    bad["flashinfer_autotune"] = True
    must_reject(bad, "FlashInfer autotune on vLLM CUTLASS")
    bad = copy.deepcopy(split)
    del bad["speculation"]["moe_backend"]
    must_reject(bad, "CuTeDSL inherited by unquantized MTP draft")
    bad = copy.deepcopy(split)
    bad["speculation"]["moe_backend"] = "triton"
    must_reject(bad, "unaudited MTP draft backend")
    bad = copy.deepcopy(split)
    bad["cudagraph_capture_sizes"].append(32)
    must_reject(bad, "CUDA graph size above max sequences")
    bad = copy.deepcopy(short)
    bad["cache_profile_id"] = "../unsafe"
    must_reject(bad, "unsafe compiler-cache profile ID")
    bad = copy.deepcopy(smoke)
    bad["block_size"] = None
    must_reject(bad, "PP correctness smoke without block64")
    bad = copy.deepcopy(smoke)
    bad["enforce_eager"] = False
    bad["cudagraph_capture_sizes"] = [1, 2, 4]
    must_reject(bad, "PP correctness smoke without eager execution")
    bad = copy.deepcopy(smoke)
    bad["benchmark"]["mode"] = "headline"
    must_reject(bad, "PP correctness workload accidentally reportable")
    bad = copy.deepcopy(smoke)
    bad["reportable"] = True
    must_reject(bad, "PP correctness profile marked reportable")
    bad = copy.deepcopy(smoke)
    bad["gpu_memory_utilization"] = 0.91
    must_reject(bad, "PP correctness utilization drift")

    good_text = " ".join(
        f"Section{i} explains a distinct mathematical idea with careful historical context."
        for i in range(180)
    )
    assert analyze(good_text, 1024)["flagged"] is False
    assert analyze("A" * 100, 1024)["flagged"] is True
    settings = {
        "tokenize_target": 8192,
        "expected_api_prompt_tokens": 8192,
        "output_lengths": [1024, 4608],
        "repeats_per_length": 2,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "sequential": True,
    }
    outputs = []
    for output_tokens in (1024, 4608):
        for repeat in range(2):
            text = f"length{output_tokens} repeat{repeat} " + good_text
            digest = hashlib.sha256(text.encode()).hexdigest()
            analysis = analyze(text, output_tokens)
            assert analysis["flagged"] is False
            outputs.append(
                {
                    "requested_output_tokens": output_tokens,
                    "repeat": repeat,
                    "finish_reason": "length",
                    "usage": {
                        "prompt_tokens": 8192,
                        "completion_tokens": output_tokens,
                        "total_tokens": 8192 + output_tokens,
                    },
                    "exact_usage_checks": {
                        "prompt_tokens": True,
                        "completion_tokens": True,
                        "total_tokens": True,
                    },
                    "reasoning_content": "",
                    "content": text,
                    "combined_text": text,
                    "sha256": digest,
                    "analysis": analysis,
                    "passed": True,
                }
            )
    synthetic_correctness = {
        "schema_version": 1,
        "status": "passed",
        "settings": settings,
        "prompt_proof": {
            "tokenize_target": 8192,
            "tokenize_count": 8192,
            "messages_sha256": hashlib.sha256(
                json.dumps(
                    [{"role": "user", "content": "synthetic"}],
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "messages": [{"role": "user", "content": "synthetic"}],
        },
        "outputs": outputs,
        "failures": [],
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        quality = Path(temp_dir) / "quality"
        quality.mkdir()
        (quality / "pp2-correctness-smoke.json").write_text(
            json.dumps(synthetic_correctness)
        )
        corrected = validate_retained(quality / "pp2-correctness-smoke.json")
        (quality / "pp2-correctness-corrected.json").write_text(
            json.dumps(corrected)
        )
        correctness = validate_correctness_smoke(Path(temp_dir))
        assert correctness["correctness_outputs"] == 4
        assert correctness["non_byte_identical_pairs"] == 2
        synthetic_correctness["outputs"][-1]["sha256"] = "b" * 64
        (quality / "pp2-correctness-smoke.json").write_text(
            json.dumps(synthetic_correctness)
        )
        try:
            validate_correctness_smoke(Path(temp_dir))
        except ValueError:
            pass
        else:
            raise AssertionError("correctness validator accepted a forged retained hash")

    gate_script = Path(__file__).resolve().parent / "verify_split_mtp_pair.sh"
    gate_env = os.environ.copy()
    gate_env.update(
        {
            "CONTAINER_NAME": "glm52-deep-synthetic-split-mtp",
            "REMOTE_HOST": "synthetic-rank",
            "MOE_BACKEND": "flashinfer_cutedsl",
            "MTP_DRAFT_MOE_BACKEND": "flashinfer_cutlass",
            "MINIMUM_KV_TOKENS": "16384",
        }
    )
    command = subprocess.run(
        ["bash", str(gate_script), "--print-remote-inspect-command"],
        check=True,
        capture_output=True,
        text=True,
        env=gate_env,
    ).stdout.strip()
    inspect_template = "{{range .Config.Env}}{{println .}}{{end}}"
    expected_argv = [
        "docker",
        "inspect",
        "glm52-deep-synthetic-split-mtp",
        "--format",
        inspect_template,
    ]
    assert shlex.split(command) == expected_argv
    with tempfile.TemporaryDirectory() as temp_dir:
        stub = Path(temp_dir) / "docker"
        stub.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
        stub.chmod(0o755)
        synthetic_env = os.environ.copy()
        synthetic_env["PATH"] = f"{temp_dir}:{synthetic_env['PATH']}"
        parsed = subprocess.run(
            ["bash", "-c", command],
            check=True,
            capture_output=True,
            text=True,
            env=synthetic_env,
        ).stdout.splitlines()
        assert parsed == expected_argv[1:]

    # The global KV-token count is emitted only by EngineCore on the API rank.
    # Exercise the complete paired gate with a remote worker log that has its
    # local available-KV marker but deliberately has no global token marker.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        target_marker = "Using 'FLASHINFER_CUTEDSL' NvFp4 MoE backend"
        draft_marker = "Using FlashInfer CUTLASS Unquantized MoE backend"
        local_log = temp / "local.log"
        remote_log = temp / "remote.log"
        local_log.write_text(
            f"{target_marker}\n{draft_marker}\n"
            "Available KV cache memory: 8.08 GiB\n"
            "GPU KV cache size: 179,264 tokens\n"
        )
        remote_log.write_text(
            f"{target_marker}\n{draft_marker}\n"
            "Available KV cache memory: 8.09 GiB\n"
        )
        docker_stub = temp / "docker"
        docker_stub.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ $1 == logs ]]; then cat \"$STUB_LOCAL_LOG\"; "
            "elif [[ $1 == inspect ]]; then "
            "printf '%s\n' VLLM_USE_V2_MODEL_RUNNER=0; else exit 90; fi\n"
        )
        ssh_stub = temp / "ssh"
        ssh_stub.write_text(
            "#!/usr/bin/env bash\n"
            "shift\n"
            "if [[ $1 == docker && $2 == logs ]]; then cat \"$STUB_REMOTE_LOG\"; "
            "elif [[ $1 == docker\\ inspect* ]]; then "
            "printf '%s\n' VLLM_USE_V2_MODEL_RUNNER=0; else exit 91; fi\n"
        )
        docker_stub.chmod(0o755)
        ssh_stub.chmod(0o755)
        synthetic_env = gate_env.copy()
        synthetic_env.update(
            {
                "PATH": f"{temp_dir}:{synthetic_env['PATH']}",
                "STUB_LOCAL_LOG": str(local_log),
                "STUB_REMOTE_LOG": str(remote_log),
            }
        )
        gate = subprocess.run(
            ["bash", str(gate_script)],
            check=True,
            capture_output=True,
            text=True,
            env=synthetic_env,
        ).stdout
        assert "rank0 target=FLASHINFER_CUTEDSL" in gate
        assert "rank1 target=FLASHINFER_CUTEDSL" in gate
        assert "global_kv_tokens=179264 minimum_required=16384" in gate
        assert "gate passed before benchmark requests" in gate

    summary = {
        "profiles_validated": len(resolved),
        "exclusions_validated": len(exclusions["exclusions"]),
        "negative_rules_validated": 18,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
