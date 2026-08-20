# Secondary-node reproduction: official Qwen3.8-27B BF16

> **GB300 recovery safety:** Generic `suggested_reload` text in the retained raw JSON and log is provenance, not an instruction. Never use GPU reset, unload or reload NVIDIA modules, or unbind/rescan PCI devices. If the driver becomes unhealthy, stop GPU work and coordinate a controlled host reboot; never reboot automatically.

An identically configured secondary DGX Station (`node1`) independently
reproduced the official-weight, autoregressive WikiText-2 quality and natural
single-stream decode results from the primary DGX Station (`node0`).
The canonical perplexity values are bit-for-bit identical. Natural decode is
0.265% faster, well inside ordinary run-to-run variation, and all five
temperature-zero continuations are byte-for-byte identical to the original.

| Metric | node0 reference | node1 verification | Difference |
| --- | ---: | ---: | ---: |
| WikiText-2 word PPL | 9.2942006114 | 9.2942006114 | 0.000% |
| WikiText-2 byte PPL | 1.5172616882 | 1.5172616882 | 0.000% |
| WikiText-2 bits/byte | 0.6014699344 | 0.6014699344 | 0.000% |
| Natural C1 generation | 94.6478 tok/s | 94.8986 tok/s | +0.265% |
| Natural C1 mean TTFT | 0.231529 s | 0.229105 s | -1.047% |
| 8K prefill scout | 22,194.2 tok/s | 22,197.5 tok/s | +0.015% |

The natural run completed 5/5 requests with zero errors and exactly 1,024
tokens per request. The report's quality rule flagged 0/5 outputs. The maximum
repeated 8-gram fraction was 0.02790 and the maximum consecutive phrase-repeat
count was 1. Repeated runs are identical because this is temperature-zero
decoding; that cross-run identity is expected and is distinct from repetition
inside an output. Each output also has the exact same SHA-256 as the `node0`
reference:
`09bfd7b6b9a1a33c0e7685ebcabe22b887fd4f3f3bf154446d3a8f738d0cf2db`.

## Exact environment

- Host: secondary DGX Station (`node1`), Ubuntu 24.04.4 LTS, Linux 6.17.0-1029-nvidia-64k, ARM64
- GPU: NVIDIA GB300, 256,703 MiB, driver 595.84
- Model: `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Cache/state: BF16 KV and BF16 Mamba state
- SGLang image: `sha256:c7c5da7c89b8aa73b6f6db5dd6b4587595e82b42759332227b55732549fb2545`
- `llm-inference-bench`: `0b4185b5b435e948b199c9077a00b084864aa963`
- `lm-evaluation-harness`: `8a07e1110d060de48cfc7a9a7987b7659060b60b`

The copied checkpoint was checked against the source machine. Its `config.json`
SHA-256 is `191e0af2...0aab`; its safetensors index SHA-256 is
`77042094...9df`. Full values are in
[`checkpoint-manifest-sha256.txt`](checkpoint-manifest-sha256.txt).

## Exact run commands

The server used the same no-spec launch arguments as the published `node0`
baseline. The complete resolved argument vector is preserved in
[`server-command.json`](server-command.json), and the image ID is in
[`server-image-id.txt`](server-image-id.txt). The dataset-derived WikiText
prompt text is deliberately not redistributed; its construction metadata and
SHA-256 remain available so a locally reconstructed prompt can be verified.
After launching the server, reconstructing that prompt from the pinned dataset,
and waiting for `http://127.0.0.1:30000/health`, natural decode was run with:

```bash
export BENCH_PYTHON=/path/to/benchmark-venv/bin/python
export BENCH_DIR=/path/to/llm-inference-bench
export PROMPT_FILE=/path/to/qwen3.8-8k-prompt.txt

"$BENCH_PYTHON" "$BENCH_DIR/llm_decode_bench.py" \
  --host 127.0.0.1 --port 30000 --model Qwen3.8-27B \
  --completion-stats \
  --prompt-file "$PROMPT_FILE" \
  --profile-concurrency 1 --profile-runs 5 --max-tokens 1024 \
  --completion-stats-temperature 0 --completion-stats-correct-regex '' \
  --completion-stats-save-text --display-mode plain --no-hw-monitor \
  --no-resume --disable-thinking \
  --output natural-decode/c1.json
```

Canonical perplexity was then run with:

```bash
export LM_EVAL_DIR=/path/to/lm-evaluation-harness
export MODEL_ROOT=/path/to/models
cd "$LM_EVAL_DIR"
.venv/bin/lm_eval \
  --model local-completions \
  --model_args "model=Qwen3.8-27B,base_url=http://127.0.0.1:30000/v1/completions,tokenizer=$MODEL_ROOT/Qwen3.8-27B,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
  --batch_size 8 --tasks wikitext --confirm_run_unsafe_code \
  --log_samples --output_path perplexity
```

The per-document `--log_samples` JSONL was used for verification but is omitted
from the repository because it embeds the WikiText documents. The aggregate
result JSON and scalar comparison retain the values needed to audit the claim.

Re-run the saved-output audit from this directory with:

```bash
python3 audit_outputs.py natural-decode/c1.json quality-audit.json
```

## Raw artifacts

- [`natural-decode/c1.json`](natural-decode/c1.json) contains all timings and full generated text.
- [`natural-decode/c1.log`](natural-decode/c1.log) is the benchmark console log.
- The [prompt-construction metadata](natural-decode/qwen3.8-8k-prompt.json) and [SHA-256 record](natural-decode/prompt-sha256.txt) identify the exact dataset-derived prompt without republishing its text.
- [`perplexity/`](perplexity/) contains the canonical aggregate result JSON, package versions, commit, and log; the document samples are intentionally excluded.
- [`quality-audit.json`](quality-audit.json) records repetition statistics for every natural output.
- [`comparison.json`](comparison.json) preserves exact reference, verification, and delta values.
- [`server-startup.log`](server-startup.log), [`server-info.json`](server-info.json), and [`server-command.json`](server-command.json) preserve runtime details.
- [`environment.txt`](environment.txt) and [`nvidia-smi.csv`](nvidia-smi.csv) preserve host and GPU versions.
