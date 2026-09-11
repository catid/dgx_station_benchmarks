# PP2 + DSpark overlay for vllm/vllm-openai:deepseekv41-flash-0909

Stock vLLM in this image refuses DSpark speculative decoding under pipeline parallelism
(`dspark with pipeline parallel is not supported`, and the draft loader's own
`NotImplementedError`). This five-file source overlay lifts that for the TP1 × PP2 split, on top
of the two PP2 patches in [`../vllm/`](../vllm/) (which stay required and unchanged). Read
[`PLAN.md`](PLAN.md) first: it has the analysis, the file:line evidence, the risks, and the GPU
test protocol; it was written before the first GPU run.

**What it is and is not.** An experiment on top of the two PP2 patches, GPU-tested only on this
pair of stations with this image and checkpoint. It relays the draft block from the last pipeline
stage to the first over the existing PP sampled-token side channel, pads the sample broadcast to
a fixed width, lets the draft load its own input embedding on the last stage, and gives the
draft a PP1 parallel config so config validation accepts it. It requires async scheduling
(`VLLM_ASYNC_SCHEDULING=1`, the recipe default) and runs with adaptive verification **off**
(vLLM's validator rejects it under PP: confidences and cost curves exist only on the last stage),
so the resulting profile is fixed-K=5 DSpark, unlike the TP2 DSpark lane, which runs adaptive
verification on. Outcome and sanity checks are recorded in the section's
[`notes/`](../../../notes/).

Layout (image path with `/` replaced by `__`):

| Patched file | Image path | What it changes |
| --- | --- | --- |
| `vllm__v1__worker__gpu__pp_utils.py` | `vllm/v1/worker/gpu/pp_utils.py` | relay the draft block through the PP side channel; pad the sample broadcast |
| `vllm__v1__worker__gpu__model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | allow DSpark under PP (async scheduling only); hook the relay in |
| `vllm__v1__worker__gpu__spec_decode__dspark__utils.py` | `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` | drop the PP rejection; never alias a placeholder embedding |
| `vllm__models__deepseek_v4_1__nvidia__dspark.py` | `vllm/models/deepseek_v4_1/nvidia/dspark.py` | the draft loads `embed.weight` itself on the last stage |
| `vllm__config__speculative.py` | `vllm/config/speculative.py` | draft parallel config is PP1 for DSpark (config validation) |

Each has a `.orig` (byte-identical to the image) and a `.diff`. `mounts.txt` lists the seven
`--volume` lines relative to `$RECIPE_DIR` (the recipe directory, resolved by
`serve_vllm_node.sh` and `selfcheck.sh`): the two existing `../vllm/` PP2 patches plus these
five. `serve_vllm_node.sh` mounts them on both ranks when `VLLM_PATCH_PP_DSPARK=1`
(`config.env`), refuses `MODE=low-latency` under `VLLM_PARALLEL=pp` without it, forces
`enable_adaptive_verification:false` under PP, refuses `VLLM_ASYNC_SCHEDULING=0`, and
`launch_cluster.sh` forwards the flag to the remote rank.

`./selfcheck.sh` compiles every file and runs `selfcheck.py` inside the image with the mounts
applied and no GPU: imports, registry resolution, and the relay helpers with fake tensors
(36 checks, `ALL CHECKS PASSED` on the measured image). `SHA256SUMS` covers the patched files,
originals, diffs, `mounts.txt`, and `selfcheck.py` (`sha256sum -c SHA256SUMS`); the patched
files, originals, and diffs are byte-identical to the private working copies that were mounted
for the measured lane.
