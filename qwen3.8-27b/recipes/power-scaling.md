# Reproducing the GB300 power-scaling benchmark

This experiment runs Qwen3.8-27B independently on two DGX Stations. It is not tensor or pipeline parallel: each station serves the complete model on one GB300, so matching cells execute concurrently.

## Checkpoint and software

- Target: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Weights and KV/Mamba state: BF16; no CPU offload or external draft
- Native one-layer MTP: EAGLE, 3 speculative steps, top-k 1, 4 draft tokens
- SGLang: PR 35371 commit `4cdb1dcc7ff725e3b4965c3f688c1107098a007e` (`0.0.0.dev1+gc0b6474b4`)
- Runtime image ID: `sha256:c7c5da7c89b8aa73b6f6db5dd6b4587595e82b42759332227b55732549fb2545`
- `llm-inference-bench`: v0.4.29 at `0b4185b5b435e948b199c9077a00b084864aa963`

The checkpoint proof retained `config.json` SHA-256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` and index SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.

## Server profile

Use TP1, 262,144-token context, static memory fraction 0.80, max-running 64, BF16 KV/Mamba state, TRT-LLM MHA, FlashInfer linear attention, extra-buffer Mamba radix cache, and decode CUDA graphs at batches 1/2/4/8/16/32/64. The relevant SGLang arguments are:

```bash
python3 -m sglang.launch_server \
  --model-path /model --served-model-name Qwen3.8-27B \
  --trust-remote-code --tp-size 1 --context-length 262144 \
  --mem-fraction-static 0.80 --kv-cache-dtype bfloat16 \
  --chunked-prefill-size 8192 --max-running-requests 64 \
  --cuda-graph-max-bs-decode 64 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
  --attention-backend trtllm_mha \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer \
  --cuda-graph-backend-prefill tc_piecewise \
  --cuda-graph-max-bs-prefill 8192 \
  --cuda-graph-bs-prefill 256 1024 4096 8192 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port 30000
```

## Workload

Each request has exactly 8,192 input tokens and is forced to 1,024 output tokens at temperature 0 with low thinking and EOS ignored. Prefill is skipped in the client report. C1 uses one warmup plus five measured requests; C64 uses 64 warmups plus 320 measured requests.

```bash
C=64
python3 llm_decode_bench.py \
  --host 127.0.0.1 --port 30000 --model Qwen3.8-27B \
  --concurrency "$C" --contexts 8k \
  --request-count "$((C * 5))" --warmup-request-count "$C" \
  --max-tokens 1024 --temperature 0 --token-targeting exact \
  --skip-prefill --reasoning-effort low --display-mode plain \
  --no-hw-monitor --no-resume \
  --output result.json
```

## Balanced power schedule

Apply GPU-scope caps only after confirming the hardware-reported range and follow the repository's GB300 operating guide. The module limit remains at its 1,600 W default. Settle each cap for 30 seconds and wait for three consecutive cool/idle samples before entering a cell.

```text
Station A block 1: 1300,  800, 1000, 1100   C1 -> C64
Station A block 2: 1100, 1000,  800, 1300   C64 -> C1
Station B block 1:  800, 1100, 1300, 1000   C1 -> C64
Station B block 2: 1000, 1300, 1100,  800   C64 -> C1
```

Every cap/concurrency point therefore has four host×block observations. Run one resident server per station across the matrix, then restore the 1,300 W GPU default before graceful teardown.

## Metrics and validation

- Throughput: completed output tokens divided by measured wall time.
- Retained throughput: each cell divided by the 1,300 W cell in the same station and block; publish the median of four paired ratios.
- Board watts: 500 ms GPU-board telemetry; publish the median of the four per-cell sample medians.
- Efficiency: completed output tokens divided by trapezoidally integrated GPU-board joules inside the measured window; publish the median of four cell values.
- Uncertainty: retain absolute min/max and the inclusive IQR of paired retention in `power-scaling.csv` and provenance.

Every performance cell must complete 5/5 or 320/320 requests with exact token agreement, zero errors, no warmup timeout or underfill, and effective concurrency of at least 0.8 or 60. A C64 capacity advisory is acceptable only when all 320 requests and 327,680 output tokens complete, `max_running_reqs` reaches 64, and the other gates pass. Retain and compare a deterministic natural output at 1,300 W and 800 W on both stations.

## Run-specific audit note

All 32 scientific cells completed before teardown. An operator telemetry watcher then caused the first post-stop owner audit to fail closed. After the exact process was verified and stopped, only the cleanup audit and aggregation resumed: both stations reported 0 MiB idle HBM, restored 1,300/1,600 W limits, and clean current-boot journals. The original failure marker remains in the private manifest; no scientific cell was rerun or modified.
