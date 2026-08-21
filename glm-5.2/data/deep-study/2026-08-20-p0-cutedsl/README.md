# P0: accepted TP2 + EP2 CuTeDSL baseline

This frozen milestone is the accepted baseline for the GLM-5.2 deep study. It
uses the pinned NVIDIA NVFP4 checkpoint and vLLM image documented in
`resolved-plan.json`, with two GB300s, TP2 + EP2, FP8 E4M3 KV cache, no
speculation, and no CPU offload.

The structural validator accepted all eight 8K-input sustained-decode cells,
all three standalone prefill rows, the 261,952-token KV capacity, and four
natural outputs. The post-run kernel scan was clean and HBM returned to the
idle baseline. See `runtime-summary.json` for startup, capacity, network, and
teardown provenance.

The published benchmark JSON removes only machine-specific host labels and
generic driver-reconfiguration suggestions. Measurement fields are unchanged.

