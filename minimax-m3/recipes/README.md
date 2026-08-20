# Reproducing MiniMax M3 NVFP4 on DGX Station GB300

The primary recipes target one DGX Station GB300 running Linux on ARM64 and
use NVIDIA's official ModelOpt NVFP4 checkpoint. A separate, dormant PP2
capacity recipe covers MiniMax's larger official MXFP8 checkpoint on two
stations. Both paths keep weights resident on GPU and do not use CPU offload.

## 0. License

Read the pinned [MiniMax Community License](https://huggingface.co/nvidia/MiniMax-M3-NVFP4/blob/901464083161bf8612a29ff7ad29914cd4ab4a85/LICENSE).
For commercial use it requires a prominent “Built with MiniMax M3” notice and
either a one-time email notice or prior written authorization, depending on
yearly revenue. It also prohibits several use categories. Set the explicit
acknowledgment only after handling those terms:

```bash
export MINIMAX_M3_LICENSE_ACCEPTED=YES
```

## 1. Download and verify the checkpoint

Install a current Hugging Face `hf` CLI and reserve at least 270 GB of free
disk space:

```bash
export MODEL_DIR=/absolute/path/to/MiniMax-M3-NVFP4
./download_model.sh
./verify_checkpoint.sh "$MODEL_DIR"
```

The scripts pin revision `901464083161bf8612a29ff7ad29914cd4ab4a85`
and verify 250,137,296,832 bytes, 88 safetensor shards, and hashes of the
configuration, index, quantization metadata, tokenizer, and license.

For the optional speculative row, download the GQA EAGLE3 draft separately:

```bash
export DRAFT_MODEL_DIR=/absolute/path/to/MiniMax-M3-EAGLE3-GQA
./download_draft.sh
./verify_draft.sh "$DRAFT_MODEL_DIR"
```

The draft model card declares MIT, and the repository also includes the
MiniMax M3 Community License covering target/base-model obligations. The recipe
pins revision `96692486b5fd38ebf8fd2a5f6bb53427d30819a8` and verifies the
6,149,993,396-byte repository.

NVFP4 fits one GB300, so this benchmark does not copy or shard it across the
two systems.

## 2. Runtime

Pull the pinned vLLM 0.27.1 ARM64 image on every system:

```bash
docker pull \
  vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

Do not substitute the old `vllm/vllm-openai:minimax-m3` vendor image or vLLM
0.26.x. They predate the MiniMax M3 NVFP4 inference-correctness fix merged in
vLLM PR 48929. A fresh runtime must pass the retained-output audit before any
throughput number is published.

Install the pinned benchmark tools on the client system:

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench tools/llm-inference-bench
git -C tools/llm-inference-bench checkout 0b4185b5b435e948b199c9077a00b084864aa963
python3 -m venv .bench-venv
.bench-venv/bin/pip install 'httpx>=0.27,<1' 'rich>=13,<15'

git clone https://github.com/EleutherAI/lm-evaluation-harness tools/lm-evaluation-harness
git -C tools/lm-evaluation-harness checkout 8a07e1110d060de48cfc7a9a7987b7659060b60b
python3 -m venv .eval-venv
.eval-venv/bin/pip install -e 'tools/lm-evaluation-harness[api]' \
  'transformers==4.57.6'
```

The API extra does not install Transformers, but the Hugging Face tokenizer
backend used by the WikiText-2 command requires it; keep the explicit pin.

## 3. Safety preflight

Measure a healthy post-boot idle baseline once, then run the executable gate
immediately before every launch:

```bash
export EXPECTED_IDLE_HBM_MIB=1000  # replace with this system's clean baseline
./preflight-idle-hbm.sh
```

The script checks the current boot's kernel log before making an NVIDIA query.
If it finds failed ATS-peer removal, nonzero PMA usage, `NV_ERR_INVALID_STATE`,
`RmInitAdapter failed`, an NVIDIA Xid, kernel oops, or soft lockup, stop without
issuing further NVIDIA ioctls. Never use a GPU reset, PCI unbind/rescan, or an
NVIDIA driver-module reload. If tens of GiB remain allocated with no compute,
UVM, or device-file owner, stop and arrange an operator-controlled normal
reboot; never compensate by increasing memory utilization.

The preflight prints the GB300 CDI selector. Export it for the launch rather
than assuming CUDA device zero:

```bash
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
```

## 4. One-station launch

Start with the conservative 95% memory profile and the official recipe's 262K
server window:

```bash
export MODEL_DIR=/absolute/path/to/MiniMax-M3-NVFP4
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
export THINKING_MODE=disabled  # disabled, adaptive, or enabled
export SERVER_PROFILE=throughput
export SPECULATIVE_MODE=none
./serve-1x.sh
curl --fail http://127.0.0.1:30000/health
```

The text benchmark deliberately passes `--language-model-only`; this omits the
vision tower from HBM and is not CPU offload. The throughput profile uses FP8
KV, a 262K model-length ceiling, prefix caching, and C1–C128 CUDA-graph sizes.
The larger ceiling leaves output-token headroom after the exact 128K prefill
prompt; the benchmark itself stops at 128K. If the initial load does not fit,
preserve the complete log and stop. Do not raise the memory fraction or modify
the driver to force a result.

If profiling proves the default model-length ceiling is larger than the
available one-GPU KV budget, preserve that failed startup as capacity evidence.
A separately labeled long-prefill profile may reduce the ceiling to exactly
128K input plus output headroom and capture only graph tiers that fit the
measured KV budget, for example:

```bash
export MAX_MODEL_LEN=132096
export CUDA_GRAPH_CAPTURE_SIZES=1,2,4,8,16
export MAX_CUDAGRAPH_CAPTURE_SIZE=16
```

Record any non-default memory fraction with the result. Change it only after a
clean teardown and idle-HBM preflight; never use it to mask retained HBM.

After the base matrix is complete, the optional EAGLE3-GQA row uses three
speculative tokens:

```bash
export SPECULATIVE_MODE=eagle3-gqa
export DRAFT_MODEL_DIR=/absolute/path/to/MiniMax-M3-EAGLE3-GQA
./serve-1x.sh
```

The draft adds about 5.73 GiB of weights and its own KV/cache overhead. A
single-GB300 launch is accepted only if it fits at the same conservative memory
setting and passes retained-output checks. Do not raise memory utilization to
force it.

For WikiText-2, stop the named container, re-run the safety gate, and launch a
small BF16-KV profile:

```bash
docker rm -f minimax-m3-vllm
export SERVER_PROFILE=ppl
export THINKING_MODE=disabled
./serve-1x.sh
```

## 5. Decode and cold-prefill sweep

Run each thinking mode from a separately launched server so the raw server
configuration proves the mode:

```bash
export BENCH_DIR=$PWD/tools/llm-inference-bench
export BENCH_PYTHON=$PWD/.bench-venv/bin/python
export RESULT_ROOT=$PWD/results/minimax-m3
./benchmark.sh disabled
```

Repeat after cleanly stopping, gating, and relaunching with `THINKING_MODE` set
to `adaptive` and `enabled`. The standard matrix is exact 8K input, up to 1K
output, 30 seconds per cell, C1–C128, plus exact cold 8K/64K/128K prefill.

## 6. Retained natural-output audit

For each thinking-mode server, retain natural outputs at C1, C64, and C128:

```bash
for concurrency in 1 64 128; do
  "$BENCH_PYTHON" ./natural_quality_audit.py \
    --thinking-mode "$THINKING_MODE" \
    --concurrency "$concurrency" \
    --output "$RESULT_ROOT/1x-nvfp4/${SPECULATIVE_MODE:-none}/$THINKING_MODE/natural-c${concurrency}.json"
done
```

The audit allows up to 4,096 output tokens so enabled-thinking responses have a
reasonable chance to reach a final answer. It uses the model-card sampling
recipe (temperature 1.0, top-p 0.95, top-k 40), saves full reasoning and answer
text, and flags empty output, repeated 8-grams, identical word runs, and
identical-character runs. Review all flags manually. Parser-separated reasoning
is included in the audit rather than silently discarded.

## 7. WikiText-2

With the `ppl` server profile and BF16 KV:

```bash
export LM_EVAL_DIR=$PWD/tools/lm-evaluation-harness
export LM_EVAL_BIN=$PWD/.eval-venv/bin/lm_eval
export MODEL_DIR=/absolute/path/to/MiniMax-M3-NVFP4
export RESULT_ROOT=$PWD/results/minimax-m3
./wikitext2.sh
```

WikiText-2 calls the raw completions/log-probability endpoint, so chat thinking
mode does not apply. Preserve the 62-document result JSON, samples, runtime
metadata, and server log.

## 8. MXFP8 two-station capacity path

Do not distribute NVFP4 merely to use another GPU: it fits one GB300, and the
current single data link would add communication to a model that does not need
capacity scaling.

The documented two-station path is PP2 for the larger official
`MiniMaxAI/MiniMax-M3-MXFP8` checkpoint. It is currently unmeasured. Each rank
needs a local copy of the pinned 443,776,005,285-byte repository, plus working
space. Do not delete other checkpoints or mount the model over the benchmark
rail merely to force this row. Once storage is explicitly available:

```bash
export MINIMAX_M3_LICENSE_ACCEPTED=YES
export MXFP8_MODEL_DIR=/absolute/path/to/MiniMax-M3-MXFP8
./download_mxfp8.sh
./verify_mxfp8.sh "$MXFP8_MODEL_DIR"
rsync -a --whole-file --partial --info=progress2 \
  --exclude='.cache/' "$MXFP8_MODEL_DIR/" node1:"$MXFP8_MODEL_DIR/"
ssh node1 "$PWD/verify_mxfp8.sh '$MXFP8_MODEL_DIR'"
```

Use a verified dedicated data interface and matching RDMA HCA/device. Run the
safety gate on both systems, then start rank 1 and rank 0 with the same absolute
model path:

```bash
# rank 1
export NODE_RANK=1 NODE0_IP=192.0.2.1 NODE1_IP=192.0.2.2
export FABRIC_IFACE=high_speed0 FABRIC_HCA=mlx5_0
export RDMA_DEVICE=/dev/infiniband/uverbs0
export MXFP8_MODEL_DIR=/absolute/path/to/MiniMax-M3-MXFP8
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
./serve-node-2x.sh

# rank 0: same settings, but NODE_RANK=0 and rank-0 GPU_DEVICE
```

The PP2 recipe disables distributed FlashInfer autotuning because pipeline
stages can enter different tuning sequences. FlashInfer remains enabled and
uses its fixed kernel selection. Preserve NCCL logs and verify the intended
RDMA transport before accepting a result.

## 9. Optional multimodal smoke test

Text results do not measure image or video understanding. After the full text
matrix, create a separate launch derived from `serve-1x.sh` with
`--language-model-only` removed and the official NVIDIA vision-encoder flags:

```text
--mm-encoder-attn-backend FLASHINFER
--mm-processor-cache-type shm
--mm-encoder-tp-mode data
```

Use fixed, redistributable image and short-video inputs. Record media checksum,
dimensions/duration, visual-token count, end-to-end and encoder latency, peak
HBM, and the full answer. Do not merge this smoke test with text-only decode or
prefill rows. If the resident vision path does not fit at the conservative
memory setting, record that outcome and stop; do not use CPU offload.

## 10. Cleanup

For the one-station NVFP4 run, remove only the named local container:

```bash
docker rm -f minimax-m3-vllm
```

For the optional PP2 MXFP8 capacity run, also remove the named container on
the second participating system:

```bash
ssh node1 docker rm -f minimax-m3-vllm
```

Then, while the driver is still known healthy, compare idle HBM with the clean
baseline and check for owners. Never reset a GPU to reclaim HBM. If accounting
does not return to normal, end GPU work and coordinate a normal reboot.
