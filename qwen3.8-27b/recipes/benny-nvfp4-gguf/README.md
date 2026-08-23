# BennyDaBall Qwen3.8 NVFP4/MTP on two GB300 stations

This workspace tests two independent, identical single-GPU serving instances.
It does not shard a model between the stations. The main matrix uses an 8,192
token input, a forced 1,024-token output, temperature zero, five measured waves,
and C1 through C128. Autoregressive and embedded-MTP runs are separate controls.
The harness intentionally sends a scout and then reuses the same prompt within
each cell so the primary matrix measures sustained decode with prefix caching.
The request stream timeout is 1,800 seconds. The harness default of 600 seconds
expired before first output for 67--71 C128 requests; those timeout-limited
artifacts are retained under `timeout600-diagnostic` and excluded from the
primary matrix.

The checkpoint is pinned to Hugging Face revision
`d8ad61f9211c00f3124fcd825f8934584065fca5`; the model file must hash to
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
llama.cpp is pinned to `6657ded4faa3b8450221119fc6b4d002e35104a2` and built for `sm_103a`
with CUDA 13.0.88 inside the same container image on each node.

Before every launch, run the kernel-log and idle-HBM safety gate from
`/home/catid/frontier-bench/SAFETY.md`. Remove the named container explicitly
after each mode. Never reset either GPU.

A separate `benny-nocache` C1 diagnostic is retained but excluded from the
primary matrix because disabling cache defeats the harness's decode-scout
methodology.
