# Reproducing Ornith-1.5-397B on one DGX Station

This recipe recreates the published TP1 results on one server-class NVIDIA GB300. It pins the model, serving image, benchmark, and evaluation harness. Run only one server profile at a time; the 397B checkpoint leaves little spare HBM.

> **GB300 recovery safety:** Do not execute generic `suggested_reload` text
> retained inside the raw JSON; it is historical tool output, not an instruction.
> Never use GPU reset, unload or reload NVIDIA modules, or perform PCI
> unbind/rescan. Remove only the named containers. If GPU accounting or the
> driver remains unhealthy, stop GPU work and coordinate a controlled host
> reboot with the operator; never reboot automatically.

Run the commands below from the `ornith-1.5-397b/` experiment directory so the supplied `recipes/` paths resolve as written.

For the measured two-station PP2 and TP2 configurations, use the separate [two-station RDMA recipe](README-2x.md).

## 1. Prerequisites

- One NVIDIA DGX Station with a 256 GB GB300 and driver 595.84 or a compatible newer server driver
- Ubuntu/Arm64, Docker, NVIDIA CDI, Git, `curl`, `jq`, Python 3.12, and at least 240 GiB free storage
- Hugging Face access and enough download time for the 221.65 GiB checkpoint

Select the GB300 CDI device, not the display GPU:

```bash
nvidia-ctk cdi list
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
```

## 2. Download the pinned checkpoint

```bash
python3 -m venv .download-venv
.download-venv/bin/pip install 'huggingface_hub[cli]'

export MODEL_DIR="$PWD/models/Ornith-1.5-397B-NVFP4"
.download-venv/bin/hf download ornith-ai/Ornith-1.5-397B-NVFP4 \
  --revision 745c3c8236ca1dc6f3aced3a0c3e7508fd9d98b6 \
  --local-dir "$MODEL_DIR"
sha256sum "$MODEL_DIR/config.json" "$MODEL_DIR/model.safetensors.index.json"
```

Expected hashes:

```text
66976e69883bfbd07086e5117adf7d3611bf7ad39219958eb4e42c4edb863b09  config.json
c182191bd9d72391552f30b5e6f918228a38fb56ea5082374c62aefedae31295  model.safetensors.index.json
```

## 3. Pin the runtime and benchmark tools

```bash
sudo docker pull vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967

git clone https://github.com/local-inference-lab/llm-inference-bench.git tools/llm-inference-bench
git -C tools/llm-inference-bench checkout 0b4185b5b435e948b199c9077a00b084864aa963
python3 -m venv .bench-venv
.bench-venv/bin/pip install -r tools/llm-inference-bench/requirements.txt

git clone https://github.com/EleutherAI/lm-evaluation-harness.git tools/lm-evaluation-harness
git -C tools/lm-evaluation-harness checkout 8a07e1110d060de48cfc7a9a7987b7659060b60b
python3 -m venv .eval-venv
.eval-venv/bin/pip install -e 'tools/lm-evaluation-harness[api]'
.eval-venv/bin/pip install 'transformers==4.57.6'
```

Set paths from the directory in which you want results written:

```bash
export MODEL_DIR="$PWD/models/Ornith-1.5-397B-NVFP4"
export BENCH_DIR="$PWD/tools/llm-inference-bench"
export BENCH_PYTHON="$PWD/.bench-venv/bin/python"
export LM_EVAL_DIR="$PWD/tools/lm-evaluation-harness"
export LM_EVAL_PYTHON="$PWD/.eval-venv/bin/python"
export RESULT_DIR="$PWD/results/ornith-1.5-397b/1x"
export GPU_DEVICE='nvidia.com/gpu=GPU-REPLACE-WITH-GB300-UUID'
```

## 4. Sustained decode and output audit

Start the 32K/16-sequence FP8-KV profile:

```bash
recipes/serve-1x.sh throughput
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-1x.sh decode
recipes/benchmark-1x.sh quality
sudo docker inspect ornith15-397b-throughput > "$RESULT_DIR/throughput/container-inspect.json"
sudo docker rm -f ornith15-397b-throughput
```

The decode command runs 30-second sustained cells at C1, C2, C4, C8, and C16 with exact 8K prompts, 1,024 output tokens, temperature 0, and EOS ignored. This current `llm-inference-bench` sustained layer is different from its optional finite Burst/E2E layer.

The quality runner saves four normal-policy and four forced-length responses in full, then scores repetition. Read them: the automatic check catches obvious loops but is not a semantic judge.

## 5. Standalone prefill, including exact 128K

The long-context profile needs a little more HBM than throughput. Verify no stale process or container is holding the GB300 before launch:

```bash
nvidia-smi
recipes/serve-1x.sh prefill
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-1x.sh prefill
sudo docker inspect ornith15-397b-prefill > "$RESULT_DIR/prefill/container-inspect.json"
sudo docker rm -f ornith15-397b-prefill
```

The first two prefill runs target the benchmark's canonical 8K, 64K, and 128K tokenizer lengths. Because chat-template accounting adds two tokens at the API boundary, the last run targets 131,070 at `/tokenize` to measure exactly 131,072 API prompt tokens. Do not substitute the canonical 131,074-token result for that exact row.

## 6. Canonical WikiText-2 perplexity

Use the separate BF16-KV profile. This precision override is essential: `auto` selects FP8 KV from this checkpoint's quantization config.

```bash
recipes/serve-1x.sh wikitext2-bf16
until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done
recipes/benchmark-wikitext2-1x.sh
sudo docker inspect ornith15-397b-wikitext2-bf16 > "$RESULT_DIR/wikitext2-bf16-kv/container-inspect.json"
sudo docker rm -f ornith15-397b-wikitext2-bf16
```

The runner refuses to proceed unless the live container command contains `--kv-cache-dtype bfloat16`. It evaluates all 62 WikiText-2 documents with maximum length 2,048 and API batch size 4.

## 7. Regenerate publication data and charts

From this experiment directory, copy the compact artifacts into the matching filenames under `data/1x/raw/`, then run:

```bash
python3 recipes/extract_results.py
python3 -m venv .chart-venv
.chart-venv/bin/pip install 'matplotlib==3.11.1'
.chart-venv/bin/python recipes/render_charts.py
```

`extract_results.py` validates the expected rows, zero decode errors, all 62 WikiText documents, and eight complete audit outputs before rewriting the compact summaries. The checked-in raw filenames are the contract used by that script.

Two-station PP2/TP2 measurements are intentionally outside this one-station recipe. Their reserved schemas and boundary are documented in [`../data/2x/`](../data/2x/).
