# Reproducing the DeepSeek-V4-Flash-0731 benchmark

This recipe recreates the published autoregressive and DSpark `llm-inference-bench` matrix on one NVIDIA GB300. It is self-contained and avoids the original machine's directory layout.

Two launch profiles preserve the measured methodology:

| Profile | Use | Static HBM fraction | Server ceiling | Decode graphs | Prefill graphs |
| --- | --- | ---: | ---: | --- | --- |
| `baseline` | Published C1–C64 rows | 0.88 | 64 | 1–64 | Runtime default |
| `c128` | Published C128 rows | 0.95 | 128 requested | 1–128 | Disabled to prioritize decode cache |

C128 is offered client concurrency. Always retain the benchmark's resolved `effective_concurrency`, `max_running_reqs`, queue metrics, and `capacity_limited_flag`; do not infer that all 128 requests were resident merely from the client setting.

## 1. Prerequisites

- A server-class NVIDIA GB300 with roughly 256 GB HBM
- NVIDIA driver 595.84 or a compatible newer server driver
- Docker configured with NVIDIA CDI
- Git, Git LFS, `curl`, `jq`, Python 3.12, and at least 170 GB free disk
- Hugging Face access where required

Select the GB300 rather than the display GPU:

```bash
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
nvidia-ctk cdi list | grep 'nvidia.com/gpu='
```

## 2. Download the pinned checkpoint

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install 'huggingface_hub[cli]'

export MODEL_ROOT="$PWD/models"
mkdir -p "$MODEL_ROOT"
.download-venv/bin/hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --local-dir "$MODEL_ROOT/DeepSeek-V4-Flash-0731"
```

This is the official 304B-total/13B-active checkpoint with native mixed MXFP4 experts and FP8 dense weights. Do not substitute a BF16 or third-party quantization and call it the same result.

## 3. Pull the pinned runtime

```bash
sudo docker pull lmsysorg/sglang@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0
sudo docker tag \
  lmsysorg/sglang@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0 \
  lmsysorg/sglang:v0.5.16
```

The included [`dspark_sps_tp1.json`](dspark_sps_tp1.json) is the measured single-GB300 DSpark steps-per-second calibration used by the server.

## 4. Install the pinned benchmark

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git tools/llm-inference-bench
git -C tools/llm-inference-bench checkout 0b4185b5b435e948b199c9077a00b084864aa963
python3 -m venv .bench-venv
.bench-venv/bin/pip install -r tools/llm-inference-bench/requirements.txt

export MODEL_ROOT="$PWD/models"
export BENCH_DIR="$PWD/tools/llm-inference-bench"
export BENCH_PYTHON="$PWD/.bench-venv/bin/python"
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
```

## 5. Run the matrix

The valid published modes are `autoregressive` and `dspark`:

```bash
recipes/serve.sh dspark baseline
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
CONCURRENCIES='1 2 4 8 16 32 64' recipes/benchmark.sh dspark
sudo docker rm -f deepseek-v4-benchmark

recipes/serve.sh dspark c128
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
CONCURRENCIES=128 recipes/benchmark.sh dspark
sudo docker rm -f deepseek-v4-benchmark
```

Repeat for `autoregressive`. The runner applies `low`, `high`, and `max` reasoning effort and writes one JSON plus log per cell under `results/deepseek-v4-flash-0731/<mode>/<effort>/`.

The exact fixed workload is:

- 8,192 prompt tokens targeted through the server tokenizer
- 1,024 generated tokens, temperature 0, EOS ignored
- Concurrency 1, 2, 4, 8, 16, 32, 64, and 128
- `5 × C` measured requests after `C` warm-up requests
- Aggregate output tokens divided by benchmark wall time

Validate every cell:

```bash
find results/deepseek-v4-flash-0731 -name 'c*.json' -print0 | while IFS= read -r -d '' result; do
  jq -e '.results | length == 1 and .[0].num_errors == 0' "$result" >/dev/null || echo "BAD $result"
done
```

## 6. Prefill, perplexity, and natural output

The published cold-prefill test used 8K chunks, the native FP8 hybrid cache, and prompts of 8K, 32K, 64K, and 128K. Run `llm_decode_bench.py --standalone-prefill --prefill-contexts 8k,32k,64k,128k`, repeat samples, and report client prompt tokens divided by TTFT.

For canonical WikiText-2 perplexity, pin `EleutherAI/lm-evaluation-harness` to `8a07e1110d060de48cfc7a9a7987b7659060b60b`:

```bash
lm_eval --model local-completions \
  --model_args "model=/model,base_url=http://127.0.0.1:30000/v1/completions,tokenizer=$MODEL_ROOT/DeepSeek-V4-Flash-0731,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
  --batch_size 8 --tasks wikitext --confirm_run_unsafe_code --log_samples \
  --output_path results/wikitext2-deepseek
```

SGLang v0.5.16's specialized DeepSeek hybrid cache requires uint8/FP8 storage. BF16 cache creation asserts, so the published PPL is explicitly labeled FP8 rather than silently substituting another runtime.

For natural decode, build a deterministic approximately 8K prompt from `EleutherAI/wikitext_document_level`, run `llm_decode_bench.py --completion-stats` at C1/C64/C128 with temperature 0 and EOS respected, and save full output text. Apply the same audit rule as the report: flag four consecutive phrase repeats or repeated 8-gram fraction ≥0.20.

The high-concurrency repetition caveat is part of the result. Do not publish C64 or C128 throughput without its corresponding audit rate.

The checked-in `data/wikitext2-natural-c128-source.json` is a sanitized run-level extraction of the original C128 artifacts. It records their SHA-256 identities, capture metadata, timing/token fields, and output hashes without publishing the generated text. The output hashes join directly to the C128 rows in `data/wikitext2-quality-audit.json`.

## 7. DFlash exclusion

Do not add a DeepSeek DFlash number using `RedHatAI/DeepSeek-V4-Flash-speculator.dflash` with this target. That draft was trained for an earlier preview and expects five full 16,384-wide auxiliary streams. Public vLLM supplied five collapsed 4,096-wide states for `0731`, and SGLang v0.5.16 does not support this combination. Padding or repeating states invalidates verification.

## 8. Sanity checks

- Save `/get_server_info`, the exact Docker command, and the complete startup log.
- Confirm the GB300 is the only device mounted in the container.
- Preserve cache precision, model revision, queue metrics, and request errors in the published data.
- Inspect natural outputs; a fast run with reasoning loops is not a quality-safe result.

## 9. Render the published charts

Create a pinned chart environment, then render all three images exclusively from this package's checked-in data:

```bash
python3 -m venv .chart-venv
.chart-venv/bin/pip install -r recipes/render-requirements.txt
.chart-venv/bin/python recipes/render_charts.py
```

The renderer writes `deepseek-throughput.png`, `prefill-128k.png`, and `wikitext2-quality-decode.png`. Before rendering, it rejects an incomplete C1–C128 matrix, workload drift, request errors, altered C128 throughput cells, mismatched raw-source identities, or any disagreement among the compact C128 provenance, natural-decode CSV, and per-output quality audit. No data outside this package is read.
