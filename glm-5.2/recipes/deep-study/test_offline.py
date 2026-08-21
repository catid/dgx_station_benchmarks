#!/usr/bin/env python3
"""CPU-only validation of manifests and fail-closed plan rules."""

from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_plan import MANIFEST_DIR, PlanError, load_json, resolve, validate  # noqa: E402


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
        "negative_rules_validated": 13,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
