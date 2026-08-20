# MiniMax H3 data

The warmed canonical one-station BF16 result lives under [`1x-bf16/`](1x-bf16/):

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

[`duration-scaling/`](duration-scaling/) retains the 10-second rocket and two
simultaneous, same-prompt/same-seed 15-second station runs. Each retained run
contains the submitted request, asynchronous job record, client clock, raw
`ffprobe` result, decode warning logs, file hash, and a normalized summary. The
two 15-second outputs are byte-identical, while their independent timing records
support both the per-station latency spread and measured two-station aggregate
throughput.

[`experimental-native30/`](experimental-native30/) contains the stock HTTP 400
rejection, static dynamic-shape audit, one-line duration-ceiling patch with
source hashes, and the single bounded patched request that completed in
2,454.441510 seconds client end to end. Its full media and timing audit is
retained under `run/`. This row is always labeled unsupported and is never
mixed into the official 4–15-second claims.

The exact launch commands and model/runtime pins are retained in the experiment
and recipe READMEs. The original canonical package retains the unmodified
aggregate serving-benchmark JSON, but not its original server log, per-request
stage source, or raw `ffprobe` JSON. Consequently, the canonical end-to-end
figures are independently traceable to raw data while that older component-stage
breakdown has weaker provenance. All new duration rows retain the raw captures
listed above.

Selected generated MP4 files are published in [`../samples/`](../samples/README.md)
with small animated WebP previews. Cryptographic hashes and normalized media
metadata live here.

The first accepted row contains three of three successful requests, 116.8647-second mean latency, and 125,268-MB warmed peak memory. No placeholder numbers are treated as measurements.
