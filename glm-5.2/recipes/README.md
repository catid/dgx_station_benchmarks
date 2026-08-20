# Reproducing GLM-5.2 NVFP4 on DGX Station GB300

These recipes start from a clean checkout and do not depend on the original
machine-specific absolute paths. Run them from this `recipes/` directory unless a
step says otherwise. The measured checkpoint cannot run wholly in one GB300's
HBM, so the one-station path ends after the capacity check. There is no CPU
offload path by design.

## 1. Prerequisites

Both nodes need:

- One server-class NVIDIA GB300 exposed to Docker through NVIDIA CDI
- NVIDIA driver 595.84 or a compatible newer server driver
- Docker, Python 3.12, Git, `curl`, `jq`, `rsync`, and OpenSSH
- Matplotlib for deterministic publication-chart rendering
- At least 500 GB free storage for this checkpoint
- The same absolute model and recipe paths on both nodes

The two benchmark nodes need a dedicated high-speed interface. The measured
setup used `enP1p3s0f0np0`, HCA `mlx5_0`, and MTU 9000. The addresses below are
examples for an isolated point-to-point `/30`; substitute your own subnet:

| Rank | Address |
| --- | --- |
| 0 / API node | `192.0.2.1/30` |
| 1 / worker | `192.0.2.2/30` |

Configure those addresses using your operating system's persistent network
configuration, then validate them without changing state:

```bash
FABRIC_IFACE=enP1p3s0f0np0 FABRIC_HCA=mlx5_0 ./check_fabric.sh 192.0.2.2
```

Use key-based SSH from rank 0 to rank 1. The examples call the worker
`node1-rail0`, but `REMOTE_HOST` may be any SSH alias that resolves over the
dedicated rail.

## 2. Download and verify the pinned checkpoint

On rank 0:

```bash
export MODEL_ROOT="$PWD/models"
./download_model.sh
./verify_checkpoint.py "$MODEL_ROOT/GLM-5.2-NVFP4"
./check_capacity.sh "$MODEL_ROOT/GLM-5.2-NVFP4"
```

`check_capacity.sh` intentionally exits 2 for the expected one-GB300 no-fit
result. It compares actual indexed weight-file bytes with the GB300 HBM reported
by `nvidia-smi`; it does not estimate compression from parameter count.

Copy the verified checkpoint and this repository to rank 1. `--whole-file` is
usually faster on an empty destination over a dedicated 400 GbE link:

```bash
rsync -a --whole-file --partial --info=progress2 \
  "$MODEL_ROOT/GLM-5.2-NVFP4/" \
  node1-rail0:"$MODEL_ROOT/GLM-5.2-NVFP4/"
rsync -a ../../glm-5.2/ \
  node1-rail0:"$PWD/../"
ssh node1-rail0 "$PWD/verify_checkpoint.py '$MODEL_ROOT/GLM-5.2-NVFP4'"
```

The second `rsync` assumes the checkout has the same absolute path on both
nodes. Adjust `REMOTE_RECIPE_DIR` later if it does not.

## 3. Install the pinned benchmarks

On rank 0:

```bash
mkdir -p tools
git clone https://github.com/local-inference-lab/llm-inference-bench.git \
  tools/llm-inference-bench
git -C tools/llm-inference-bench checkout \
  0b4185b5b435e948b199c9077a00b084864aa963
python3 -m venv .bench-venv
.bench-venv/bin/pip install 'httpx>=0.27,<1' 'rich>=13,<15'

git clone https://github.com/EleutherAI/lm-evaluation-harness.git \
  tools/lm-evaluation-harness
git -C tools/lm-evaluation-harness checkout \
  8a07e1110d060de48cfc7a9a7987b7659060b60b
python3 -m venv .eval-venv
.eval-venv/bin/pip install -e 'tools/lm-evaluation-harness[api]'
```

Pull the exact runtime image on both nodes:

```bash
docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
ssh node1-rail0 docker pull \
  vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

## 4. Launch the accepted TP2 profile

Set portable paths from rank 0:

```bash
export MODEL_DIR="$MODEL_ROOT/GLM-5.2-NVFP4"
export REMOTE_HOST=node1-rail0
export REMOTE_RECIPE_DIR="$PWD"
export BENCH_DIR="$PWD/tools/llm-inference-bench"
export BENCH_PYTHON="$PWD/.bench-venv/bin/python"
```

Only TP2/PP1 with expert parallelism produced an accepted run. For PP2, the
operator observed a severely imbalanced full-HBM state followed by repeated
small-allocation failures, but the original startup logs were not retained;
this is not checksummed failure evidence. The retained logs for a separate TP2
attempt show that the default TensorRT-LLM NVFP4 MoE backend exhausted HBM while
repacking weights. The accepted profile uses FlashInfer CuteDSL MoE, disables
distributed FlashInfer autotuning, caps the per-expert FP4 workspace at 32,768
tokens, and uses 93% HBM utilization. Rank 0 binds the API only to localhost;
rank 1 is headless.

```bash
PARALLEL_MODE=tp \
GPU_MEMORY_UTILIZATION=0.93 \
MOE_BACKEND=flashinfer_cutedsl \
FLASHINFER_AUTOTUNE=disabled \
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=32768 \
./launch_cluster.sh
./wait_for_server.sh
RESULT_DIR="$PWD/results/glm52-nvfp4-tp2/runtime" PHASE=before \
  BENCH_DIR="$BENCH_DIR" EVAL_DIR="$PWD/tools/lm-evaluation-harness" \
  ./collect_metadata.sh
LABEL=glm52-nvfp4-tp2 MAX_TOTAL_TOKENS=2000000 ./benchmark.sh
RESULT_DIR="$PWD/results/glm52-nvfp4-tp2/runtime" PHASE=after \
  BENCH_DIR="$BENCH_DIR" EVAL_DIR="$PWD/tools/lm-evaluation-harness" \
  ./collect_metadata.sh
./stop_cluster.sh
```

`MAX_TOTAL_TOKENS=2000000` bypasses the harness's pessimistic unshared-KV
precheck. It is not a hardware-capacity claim: every stream uses the same 8K
prompt, prefix caching is enabled, and the accepted C128 cell proved all 128
requests resident with zero queued requests, an 86.9% prefix-cache hit rate,
and 31.9% observed KV usage. The server reported 256,320 KV tokens.

Run the natural audit and WikiText-2 separately with BF16 KV. This avoids
conflating FP8-KV throughput tuning with the quality check. The checkpoint's
tokenizer metadata uses the Transformers-5 `TokenizersBackend` class; the
pinned lm-eval environment uses Transformers 4, so create a tokenizer-only
compatibility directory and verify a sample encoding before evaluation.

```bash
./make_transformers4_tokenizer.sh \
  "$MODEL_DIR" "$PWD/tokenizer-transformers4"
"$PWD/.eval-venv/bin/python" - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("tokenizer-transformers4")
assert t.encode("The sky is blue.", add_special_tokens=False) == [785, 12877, 374, 6303, 13]
assert t.eos_token_id == 154820
PY

PARALLEL_MODE=tp \
GPU_MEMORY_UTILIZATION=0.90 \
KV_CACHE_DTYPE=bfloat16 \
MAX_MODEL_LEN=8192 \
MAX_NUM_SEQS=4 \
MAX_NUM_BATCHED_TOKENS=4096 \
MOE_BACKEND=flashinfer_cutedsl \
FLASHINFER_AUTOTUNE=disabled \
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=4096 \
./launch_cluster.sh
./wait_for_server.sh
"$BENCH_PYTHON" ./quality_audit.py \
  --max-tokens 4096 \
  --output "$PWD/results/glm52-nvfp4-tp2/quality"
EVAL_PYTHON="$PWD/.eval-venv/bin/python" \
  EVAL_DIR="$PWD/tools/lm-evaluation-harness" \
  TOKENIZER_DIR="$PWD/tokenizer-transformers4" \
  KV_CACHE_DTYPE=bfloat16 \
  RESULT_DIR="$PWD/results/glm52-nvfp4-tp2/wikitext2" ./wikitext2.sh
./stop_cluster.sh
```

Do not run file transfers, fabric tests, or another GPU workload while measuring.
If a launch fails, preserve both complete startup logs before changing flags.
Remove only the two explicitly named benchmark containers. Never use a GPU,
PCI, or driver reset to recover HBM; arrange an operator-controlled reboot if
memory accounting remains nonzero after both containers and compute PIDs are
gone. Do not turn a failed cell into a zero.

## 5. Workload definitions

`benchmark.sh` creates one authoritative JSON and log containing every C1–C128
decode cell plus cold prefill. The decode workload is:

- Exactly 8,192 prompt tokens through the server tokenizer
- Exactly 1,024 generated tokens, temperature 0, EOS ignored
- C1, C2, C4, C8, C16, C32, C64, and C128
- A 30-second sustained window after warm-up for every concurrency
- Aggregate output tokens from vLLM's continuous OpenAI stream-usage counter,
  divided by the measured window

At high concurrency, a 30-second window can end before an individual stream
reaches its 1,024-token cap. Those in-flight streams still contribute their
observed output tokens to the sustained aggregate and are retained as
incomplete samples in the raw JSON; they are not mislabeled as completed 1K
requests.

Cold prefill targets 8K, 64K, and 128K exactly and reports client-observed
prompt tokens divided by TTFT.

Review every result for `num_errors`, completed request count, resolved/effective
concurrency, maximum running requests, and capacity-limited flags. The raw JSON
is authoritative; normalized CSVs should retain those fields.

## 6. Quality and WikiText-2

`quality_audit.py` sends four unrelated natural prompts at temperature 0 with
a 4,096-token cap. It
stores the full answer and reasoning text plus mechanical repetition metrics.
Flag an output for review if its repeated 8-gram fraction is at least 0.20, its
identical-word run reaches 4, it contains an identical-character run of 16, or
it is empty. Always inspect the saved text manually too. The accepted BF16-KV
audit ended naturally for all four prompts with complete final answers and no
mechanical flags. The original 1,024-token FP8-KV audit is retained separately
because two answers exhausted that cap before completing their final response.

`wikitext2.sh` runs the canonical document-level WikiText task through vLLM's
OpenAI-compatible completions API with a requested 2,048-token window. lm-eval
reserves one token for generation, so its result JSON records an effective
`max_length` of 2,047. Save all sample logs. Perplexity is a quality check for
this exact quantization and BF16 KV precision; it is not interchangeable with
an FP8-KV result.

## 7. Preserve evidence

Keep, at minimum:

- Both nodes' full startup logs and container command lines
- GPU/driver inventory, image ID, model and benchmark revisions
- Every raw `llm-inference-bench` JSON and log
- Full natural outputs and their repetition audit
- The complete `lm-evaluation-harness` result JSON and samples

Only add summarized performance values to the public README after checking that
all requests completed without errors and that the generated text is coherent.

## 8. Validate and publish result artifacts

The publication pipeline accepts only these exact result directory names:

```text
RESULTS_ROOT/glm52-nvfp4-pp2/
RESULTS_ROOT/glm52-nvfp4-tp2/
```

Each directory must contain `llm-inference-bench.json`,
`quality/quality-audit.json`, and `runtime/`. The runtime directory must retain
both container inspections and server logs, `/v1/models`, metrics before and
after the workload, KV capacity, both `nvidia-smi -q` captures, and these exact
one-line provenance files:

```text
model-revision.txt
llm-inference-bench-commit.txt
lm-eval-commit.txt
```

If a WikiText-2 result is included beneath `wikitext2/`, that directory must
also contain `model-revision.txt`, `runtime-image.txt`, `lm-eval-commit.txt`,
`kv-cache-dtype.txt`, and `batch-size.txt`. This prevents the extractor from
guessing whether perplexity used FP8 or BF16 KV.

After manually reading all four full natural outputs, record one of `clean`,
`flagged`, or `pending` in `data/manual-quality-review.json`. Then regenerate
the normalized CSVs, compact evidence, charts, and README blocks:

```bash
python3 -m pip install 'matplotlib>=3.8,<4'
CHART_PYTHON=python3 ./publish_results.sh "$RESULTS_ROOT"
python3 tests/test_publication_tools.py
```

`extract_results.py` rejects an entire topology unless all C1–C128 cells are
zero-error exact-8K sustained results with a 1,024-token request cap and valid
server usage counters, all three cold-prefill points are present, the natural
audit is internally consistent, and runtime provenance proves the pinned
checkpoint, image, commits, topology, and launch flags. A missing or rejected
topology contributes no table cells and no chart series.
