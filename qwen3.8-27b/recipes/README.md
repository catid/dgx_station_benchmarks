# Reproducing the Qwen3.8-27B benchmark

This recipe recreates the published fixed-length `llm-inference-bench` matrix on one NVIDIA GB300. It is written so another automation agent can execute it without relying on a host-specific directory layout.

The recipe has two server profiles because the original C1–C64 runs and the later C128 extension require different cache allocations:

| Profile | Use | Static HBM fraction | Server ceiling | Decode graphs | Prefill graphs |
| --- | --- | ---: | ---: | --- | --- |
| `baseline` | Published C1–C64 rows | 0.80 | 64 | 1–64 | Piecewise 256/1K/4K/8K |
| `c128` | Published C128 rows | 0.95 | 128 requested | Through the runtime-resolved ceiling | Disabled to leave HBM for BF16 decode cache |

At C128, `llm-inference-bench` requires at least `128 × (8192 + 1024) = 1,179,648` cache tokens. A 0.80 allocation exposes only 945,536 tokens for DFlash2 and is correctly rejected by the benchmark. The C128 profile exposes 1,195,136 tokens. DFlash2's BF16 Mamba state cache can still cap the number of simultaneously resident requests below 128, so treat C128 as offered client concurrency and inspect `capacity_limited_flag`, `effective_concurrency`, and `max_running_reqs` in the CSV.

## 1. Prerequisites

- A server-class NVIDIA GB300 with roughly 256 GB HBM
- NVIDIA driver 595.84 or a compatible newer server driver
- Docker configured with NVIDIA CDI (`nvidia-ctk cdi list` must show the GB300)
- Git, Git LFS, `curl`, `jq`, Python 3.12, and enough disk for the target plus three draft checkpoints
- Hugging Face access where required

Select the GB300 CDI device, not the display GPU:

```bash
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
nvidia-ctk cdi list | grep 'nvidia.com/gpu='
```

## 2. Download the pinned checkpoints

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install 'huggingface_hub[cli]'

export MODEL_ROOT="$PWD/models"
mkdir -p "$MODEL_ROOT"

.download-venv/bin/hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B"
.download-venv/bin/hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision adde41d8fde3a75dc905a7df0bd5088d2a44b5a1 \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-DFlash2"
.download-venv/bin/hf download kstoyanov99/Qwen3.8-27B-Dflash \
  --revision 0e6412afb974d65455703026ff4cfa9118ad13cd \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-DFlash1"
.download-venv/bin/hf download RadixArk/Qwen3.8-27B-DSpark \
  --revision 85ef153be924f17ce4bf62726954eeaa4a73e854 \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-DSpark"
.download-venv/bin/hf download huginnfork/Qwen3.8-27B-FP8 \
  --revision ee0358b3e33d7bedcd6db022c5039385b1ac72f2 \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-Huginn-FP8"
.download-venv/bin/hf download huginnfork/Qwen3.8-27B-NVFP4A16 \
  --revision 6916a5bb185e57c6e32bcffdc13a92fdea3b4095 \
  --local-dir "$MODEL_ROOT/Qwen3.8-27B-Huginn-NVFP4A16"
```

The main speculative-decoding matrix uses the unquantized BF16 target. The DSpark checkpoint was trained against the FP8 target, but that comparison deliberately verifies its proposals with the same BF16 target as every other SGLang mode. The two `huginnfork` checkpoints are unofficial quantizations and are scoped to the xhigh, prefill, WikiText-2, and retained-text audit sections. Measured rows are explicitly labeled as vLLM results and never filled from checkpoint-author data.

The Huginn FP8 checkpoint uses FP8 weights and activations for its 192 quantized MLP projections; vLLM selected `CutlassFP8ScaledMMLinearKernel`. The checkpoint named `NVFP4A16` is explicitly weight-only W4A16 (`input_activations: null`, 4-bit weights, group size 16), not a native W4A4 checkpoint. vLLM therefore selected `MarlinNvFp4LinearKernel` on this GB300. The compatible Humming W4A16 backend was also measured, but Marlin was faster. Do not describe these results as native FP4 tensor-core throughput.

## 3. Build the pinned SGLang image

The DFlash2 support used here is SGLang PR 35371 at commit `4cdb1dcc7ff725e3b4965c3f688c1107098a007e`, layered on the pinned ARM64 SGLang base digest.

```bash
sudo docker build \
  --file recipes/Dockerfile.sglang \
  --tag qwen3.8-dflash2-sglang:pr35371-4cdb1dc \
  recipes
```

Expected local image ID/digest from the measured machine: `sha256:c7c5da7c89b8aa73b6f6db5dd6b4587595e82b42759332227b55732549fb2545`.

## 4. Install the pinned benchmark

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git tools/llm-inference-bench
git -C tools/llm-inference-bench checkout 0b4185b5b435e948b199c9077a00b084864aa963
python3 -m venv .bench-venv
.bench-venv/bin/pip install -r tools/llm-inference-bench/requirements.txt
```

Point the supplied scripts at these paths:

```bash
export MODEL_ROOT="$PWD/models"
export BENCH_DIR="$PWD/tools/llm-inference-bench"
export BENCH_PYTHON="$PWD/.bench-venv/bin/python"
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
```

## 5. Run the fixed-length matrix

The server modes are `autoregressive`, `dflash1-community`, `dflash2`, `dspark`, and `mtp`. Start one mode at a time and wait for health:

```bash
recipes/serve.sh dflash2 baseline
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
CONCURRENCIES='1 2 4 8 16 32 64' recipes/benchmark.sh dflash2
sudo docker rm -f qwen3.8-benchmark

recipes/serve.sh dflash2 c128
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
CONCURRENCIES=128 recipes/benchmark.sh dflash2
sudo docker rm -f qwen3.8-benchmark
```

Repeat both blocks for all five modes. The runner applies all four published thinking settings (`none`, `low`, `medium`, `xhigh`) and writes one JSON plus log per cell under `results/qwen3.8-27b/<mode>/<thinking>/`.

The exact workload is:

- 8,192 prompt tokens targeted through the server tokenizer
- 1,024 generated tokens, temperature 0, EOS ignored
- Concurrency 1, 2, 4, 8, 16, 32, 64, and 128
- `5 × C` measured requests after `C` warm-up requests
- Aggregate output tokens divided by benchmark wall time

Validate every cell before aggregating:

```bash
find results/qwen3.8-27b -name 'c*.json' -print0 | while IFS= read -r -d '' result; do
  jq -e '.results | length == 1 and .[0].num_errors == 0' "$result" >/dev/null || echo "BAD $result"
done
```

Do not hide C128 queuing: retain `capacity_limited`, `effective_concurrency`, `max_running_reqs`, TTFT, and inter-token latency alongside aggregate throughput.

## 6. Run the unofficial FP8 and NVFP4A16 comparison

Pull the exact vLLM image used for these cross-runtime measurements:

```bash
sudo docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
```

The image reports vLLM v0.27.1, CUDA 13.0, and Linux/ARM64. The published quantized xhigh rows are **autoregressive only**. Run the complete C1–C128 matrix for each checkpoint:

```bash
recipes/serve-quant.sh fp8 autoregressive
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-quant.sh fp8 autoregressive
sudo docker rm -f qwen3.8-quant-benchmark
```

Repeat the block with `nvfp4`. Prefix caching must remain enabled because `llm-inference-bench` primes the fixed 8K prompt and this matrix is cached-prefill decode. Verify that each JSON has `5 × C` completed requests, zero errors, and the expected exact 8,192/1,024 token lengths.

Do not add the retained quantized MTP data to this table. The FP8 prefix-enabled MTP run stopped after C16, while the complete FP8 fallback and the partial NVFP4A16 run disabled prefix caching. Those results are neither complete nor methodologically interchangeable with the published AR rows. The MTP option remains in `serve-quant.sh` only for a future clean C1–C128 rerun using one consistent cache policy.

For the 8K/32K/64K/128K prefill curves, restart autoregressive mode with `MAX_NUM_BATCHED_TOKENS=16384`; this was the best tested long-context chunk ceiling. Use the follow-up runner below and retain client-observed prompt tok/s, TTFT, sample count, and the server-side confirmation that `cached_tokens` is zero.

## 7. Prefill and WikiText-2 checks

For the published prefill sweep, start autoregressive mode with the corresponding chunk/cache settings and run multiple samples. Report client prompt tokens divided by TTFT. The speed-only SGLang FP8-KV variants used fallback unit cache scales because checkpoint-specific scales were unavailable; the headline decode matrix, Huginn prefill, and perplexity use BF16 state.

The supplied follow-up runner reproduces the published NVFP4A16 prefill and PPL with the same settings used for FP8:

```bash
export RESULT_ROOT="$PWD/results/qwen3.8-27b-quant"
export LM_EVAL_DIR="$PWD/lm-evaluation-harness"

MAX_NUM_BATCHED_TOKENS=16384 recipes/serve-quant.sh nvfp4 autoregressive
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-quant-followups.sh nvfp4 prefill
recipes/benchmark-quant-followups.sh nvfp4 ppl
sudo docker rm -f qwen3.8-quant-benchmark
```

The PPL phase refuses to run unless `lm-evaluation-harness` is exactly at `8a07e1110d060de48cfc7a9a7987b7659060b60b`. Its result validator requires all 62 WikiText documents. Do not substitute either checkpoint's bundled author-side PPL JSON: it uses a different non-overlapping-chunk method.

The measured NVFP4A16 reference values are 8,897 / 8,891 / 8,608 / 7,939 prompt tok/s at 8K / 32K / 64K / 128K and 9.4434047753 word perplexity over all 62 documents. Treat materially different values as a reason to inspect the exact checkpoint revision, image digest, cold-cache validation, and BF16 cache precision before publishing.

The equivalent direct canonical PPL command is:

```bash
export TOKENIZER_DIR="$MODEL_ROOT/Qwen3.8-27B-Huginn-NVFP4A16"
"$LM_EVAL_DIR/.venv/bin/lm_eval" --model local-completions \
  --model_args "model=Qwen3.8-27B,base_url=http://127.0.0.1:30000/v1/completions,tokenizer=$TOKENIZER_DIR,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
  --batch_size 8 --tasks wikitext --confirm_run_unsafe_code --log_samples \
  --output_path results/wikitext2-qwen
```

For a standardized retained-text xhigh audit, first build the deterministic approximately 8K prompt from `EleutherAI/wikitext_document_level`, then run C1/C64/C128 for each quantized target:

```bash
"$LM_EVAL_DIR/.venv/bin/python" recipes/prepare_wikitext2_prompt.py \
  --tokenizer "$MODEL_ROOT/Qwen3.8-27B" \
  --target-tokens 8000 \
  --output "$PWD/results/qwen3.8-wikitext-8k-prompt.txt" \
  --metadata-output "$PWD/results/qwen3.8-wikitext-8k-prompt.json"

export PROMPT_FILE="$PWD/results/qwen3.8-wikitext-8k-prompt.txt"
sha256sum "$PROMPT_FILE"
# Expected with the pinned dataset/toolchain: 3aa36d4bacfa9ef48e1b1ff265d79f740a29e1c1bc0c50a300e3135f6653d918

recipes/serve-quant.sh nvfp4 autoregressive
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-quant-followups.sh nvfp4 natural-xhigh
sudo docker rm -f qwen3.8-quant-benchmark
```

Repeat with FP8. The runner uses temperature 0, xhigh thinking, EOS respected, 1,024 maximum output tokens, `5 × C` requests, and saves full text. [`audit_quant_outputs.py`](audit_quant_outputs.py) then records per-output hashes and repetition statistics without copying generated text into the audit file. It exits nonzero for missing text or if any output has four consecutive phrase repeats or a repeated 8-gram fraction ≥0.20. Retain failed audits; they are evidence, not disposable runs.

The existing fixed-length xhigh JSONs did not retain text. In the published standardized natural audit, the repetition checker intentionally returned exit code 2 for both checkpoints: 20/965 FP8 outputs and 12/965 NVFP4A16 outputs were flagged. Preserve that return code and the report as a quality finding. Those rates describe the retained natural continuation only and must not be assigned to the fixed 8K/1K rows.

If the private raw natural-decode results are available, recover and verify the
five published C128 rows with:

```bash
python3 recipes/extract_natural_c128.py /path/to/wikitext2-decode/qwen3.8-27b
```

The extractor accepts only the five exact retained source hashes. It validates
their workload, request set, and summary arithmetic, then deterministically
writes the public natural-decode CSV, the text-free per-output audit, and compact
source provenance. The raw generated text and machine diagnostics are not
copied into this package.

## 8. Sanity checks

- Confirm `/get_server_info` reports the intended `max_total_num_tokens` before each profile.
- Save `docker inspect qwen3.8-benchmark --format '{{json .Config.Cmd}}'` with the results.
- Save the complete startup log; it records the resolved Mamba cap and captured graph tiers.
- Reject runs with request errors or empty/degenerate output.
- Do not compare the FP8 prefill-only profile to BF16 decode/PPL without labeling the precision change.

## 9. Render the published charts

Create a pinned chart environment, then regenerate all eight images exclusively
from this package's checked-in data:

```bash
python3 -m venv .chart-venv
.chart-venv/bin/pip install -r recipes/render-requirements.txt
.chart-venv/bin/python recipes/render_charts.py
.chart-venv/bin/python recipes/render_quant_charts.py
```

`render_charts.py` writes `qwen-thinking-grid.png`,
`qwen-throughput-low.png`, `prefill-128k.png`, and
`wikitext2-quality-decode.png`. Before rendering, it rejects an incomplete
C1–C128 matrix, workload drift, request errors, altered fixed-length or natural
C128 source identities, or disagreement among the CSV, compact provenance, and
quality audit. `render_quant_charts.py` writes the four Huginn charts and applies
the corresponding quantized-source validations. Neither renderer reads data
outside this package.
