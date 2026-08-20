# MiniMax H3 data

The accepted one-station BF16 result lives under [`1x-bf16/`](1x-bf16/):

- `performance.csv` is the normalized headline row;
- `stage-timings.csv` retains all three measured stage observations and their mean;
- `raw/benchmark.json` is the unmodified SGLang serving-benchmark output;
- `quality-seed1101.json` records stream metadata, signal checks, hashes, and manual review;
- `checksums.sha256` authenticates every retained result table, the normalized quality audit, the raw aggregate result, and all three published figures.

The performance table uses one row per measured configuration with these fields:

```text
run_id,topology,precision,task,width,height,frames,fps,requested_seconds,
encoded_seconds,num_inference_steps,prompts,concurrency,successful_requests,
failed_requests,benchmark_duration_s,latency_mean_s,latency_p50_s,
latency_p90_s,throughput_outputs_s,outputs_per_hour,
generated_frames_per_wall_s,generated_video_seconds_per_wall_second,
real_time_factor,peak_hbm_mb,quality_status
```

The exact launch commands and model/runtime pins are retained in the experiment and recipe READMEs. The current package retains the unmodified aggregate serving-benchmark JSON, but not the original server log, per-request stage source, or raw `ffprobe` JSON. Consequently, the headline end-to-end figures are independently traceable to raw data while the component-stage and normalized media fields have a weaker provenance trail. Future rows should retain those raw captures before publication.

Generated MP4 files are not committed unless their licensing and size make publication practical; cryptographic hashes, normalized media metadata, contact sheets, and spectrograms are retained instead.

The first accepted row contains three of three successful requests, 116.8647-second mean latency, and 125,268-MB warmed peak memory. No placeholder numbers are treated as measurements.
