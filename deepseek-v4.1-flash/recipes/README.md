# Reproducing the DeepSeek-V4.1-Flash benchmark

This page contains the runtime, topology, transport, launch, and benchmark
details kept out of the headline README. Every script here is generic: `node0`
is rank 0 (the API host, where the commands run), `node1` is rank 1, and the
addresses are examples on isolated point-to-point `/30` subnets. Substitute
your own SSH alias, interfaces, and addresses.

## Exact artifacts

| Item | Pin |
| --- | --- |
| Checkpoint | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Files | 48 safetensors shards, 510,296,708,312 bytes on disk (index `total_size` 510,286,023,000; 96,085 tensors) |
| Weight format | FP8 E4M3 dense, 32×32 blocks, `ue8m0` scales; FP4 experts (`quantization_config.expert_dtype`) |
| SGLang image tag | `lmsysorg/sglang:dev-dsv41` (arm64) |
| SGLang local image id | `sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349`, created 2026-09-10T05:33Z (a local build; not a Docker Hub digest) |
| SGLang commit | `NOT_RECORDED` (image labels say `unknown`; `sglang.__version__ == 0.0.0.dev0`) |
| SGLang source | unmerged `dsv4.1` branch, [sgl-project/sglang PR #38798](https://github.com/sgl-project/sglang/pull/38798) |
| vLLM image tag | `vllm/vllm-openai:deepseekv41-flash-0909` (arm64) |
| vLLM version | `0.1.dev20904+g179dd0fa9` (dev build from commit `179dd0fa9`; the only distribution of the V4.1 architecture) |
| vLLM local image id | `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d` (recorded by `serve_vllm_node.sh` into `logs/launch-vllm-rank<N>.txt` on the measured node) |
| Decode client | [`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |
| Prefill client | [`bench_prefill.py`](bench_prefill.py) in this directory (stdlib; `requests` optional) |
| NCCL fabric test | host NCCL 2.31.2, nccl-tests 2.19.7 `all_reduce_perf` |

The SGLang image must be rebuilt from the `dsv4.1` branch; `serve_node.sh`
(and `serve_vllm_node.sh` for vLLM) records `docker image inspect --format
'{{.Id}}'` into `logs/launch-rank<N>.txt` (`logs/launch-vllm-rank<N>.txt`),
and that id must match the pin above before a row is accepted.

## Two-station topology

- One GB300 per station. SGLang runs TP2 + EP2 (`--tp 2 --ep-size 2 --nnodes 2`);
  vLLM runs TP1 × PP2 by default (`VLLM_PARALLEL=pp`; the measured lane used
  vLLM's default even 20/20 layer split, `--language-model-only`, and the Engram
  tables in host memory via `--engram-config '{"cpu_offload": true}'`) or
  TP2 × PP1 (`VLLM_PARALLEL=tp`). Stock vLLM's DSpark runner rejects pipeline
  parallelism, so `MODE=low-latency` needs `VLLM_PARALLEL=tp`, or
  `VLLM_PARALLEL=pp` with the local five-file overlay
  (`VLLM_PATCH_PP_DSPARK=1`, [`patches/vllm-pp2-dspark/`](patches/vllm-pp2-dspark/)).
- Rank 0 (the API host) runs on node0 unless `SWAP_RANKS=1`, which starts
  rank 0 on node1 and rank 1 on node0 and swaps rail addresses, checkpoint
  paths, and the API bind address (clients then use `API_URL`, which every
  script reads from `config.env`). The vLLM lanes after PP2 were queued that
  way because vLLM's rank-0 teardown strands HBM on the host it runs on and
  only node1 has an automated reboot (see Safety).
- Two direct 400GbE ConnectX-8 RoCE rails between the stations. Rail 0
  (`FABRIC_IFACE`, `RANK0_IP`/`RANK1_IP`, example `192.168.200.1/30` and
  `.2/30`) carries the torch.distributed bootstrap; both rails are handed to
  NCCL (`NCCL_HCAS=mlx5_0,mlx5_1`) with the measured-best tuning
  `RING`/`SIMPLE`, 8 channels, 4 QPs per connection with data split across QPs.
- The 189 GiB Engram n-gram tables live in host memory on each station
  (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, `private` layout, pinned);
  `preflight.sh` requires more than 300 GiB `MemAvailable` on both hosts.
- The checkpoint must be visible on both ranks (`MODEL_DIR_RANK0`,
  `MODEL_DIR_RANK1`); the measured setup kept the copy on node1's NVMe and
  exported it to node0 over NFS on rail 1.

### Data Direct is required, and the container cannot provide it

The images ship Ubuntu rdma-core 50, whose `libmlx5` lacks
`mlx5dv_get_data_direct_sysfs_path`. NCCL inside such a container falls back to
the host-memory RDMA path and logs

```text
NCCL INFO dlvsym failed on mlx5dv_get_data_direct_sysfs_path - /lib/aarch64-linux-gnu/libmlx5.so: undefined symbol: mlx5dv_get_data_direct_sysfs_path
```

The fix is to stage the host's MOFED user libraries once per node
(`tools/stage_host_rdma.sh`, which refuses a `libmlx5` without the Data Direct
symbol) and let `serve_node.sh`/`serve_vllm_node.sh` bind-mount them
**over the image's real library files** (`HOST_RDMA=1`, the default), together
with the `/usr/lib/aarch64-linux-gnu/libibverbs` provider directory and
`/etc/libibverbs.d`. Placing the libraries beside the image's copies is not
enough: worker processes spawned with a scrubbed environment still load the
image `libmlx5`. The line that proves the fast path is active is

```text
NCCL INFO NET/IB: Data Direct DMA Interface is detected for device mlx5_0
```

(once per HCA, per rank) followed by `[send] via NET/IB/2/GDRDMA` channel
lines. `launch_cluster.sh` greps for both after `/health` answers.

### vLLM pipeline parallel needs two local source patches

The `deepseekv41-flash-0909` build cannot start DeepSeek-V4.1-Flash with
`--pipeline-parallel-size 2` as shipped. With `VLLM_PATCH_PP=1` (the default)
`serve_vllm_node.sh` bind-mounts two patched files from
[`patches/vllm/`](patches/vllm/) over the image on both ranks; each is kept
beside its unmodified `.orig`, and [`patches/vllm/README.md`](patches/vllm/README.md)
describes both. Without pipeline parallelism neither patch changes anything.

1. [`deepseek_v4_nvidia_model.py`](patches/vllm/deepseek_v4_nvidia_model.py)
   over `vllm/models/deepseek_v4/nvidia/model.py`. vLLM's model runner hands
   non-first pipeline-parallel ranks `input_ids=None` (and none during the
   memory-profile dummy run), and the stock V4 MoE raises `DeepSeek V4 vision
   MoE routing requires input_ids` even though the ids are only used to find
   image-span tokens for the vision routing bias. The patch routes every token
   as text when the bias exists but the ids are absent, which is exact under
   `--language-model-only` (there are no image tokens).
2. [`kv_cache_utils.py`](patches/vllm/kv_cache_utils.py) over
   `vllm/v1/core/kv_cache_utils.py` (unified diff in
   [`kv_cache_utils.diff`](patches/vllm/kv_cache_utils.diff)). With the patch
   above the launch got as far as KV-cache allocation and stage 1 died with
   `StopIteration` in `allocate_kv_cache` (`vllm/v1/worker/utils.py`). Root
   cause: `_project_kv_cache_groups_to_worker` leaves the global (unfiltered)
   `UniformTypeKVCacheSpecs` on a KV-cache group when a pipeline rank owns none
   of that group's layers. V4.1-Flash has one such group, the three
   `CircularBufferSpec` compressor caches of kv-source layers 2, 8, and 14,
   which all live on stage 0, so `get_kv_cache_config_from_groups` emitted
   tensors for stage-0 layers on stage 1 and the allocator found no group for
   them. The patch emits tensors only for the layers named in the group's own
   `layer_names`. The no-code alternative is an unbalanced split that puts a
   ratio-2 kv-source layer on both ranks (`VLLM_PP_LAYER_PARTITION=14,26` or
   `8,32`); the default even 20/20 split is the only balanced valid split,
   because layers 21–39 must stay with kv-source layer 20.

### vLLM PP2 + DSpark needs a five-file overlay on top of those patches

Stock vLLM rejects DSpark speculative decoding under pipeline parallelism, so
the TP2 DSpark lane is the only speculative vLLM lane the image supports as
shipped. `VLLM_PATCH_PP_DSPARK=1` additionally bind-mounts the five patched
files in [`patches/vllm-pp2-dspark/`](patches/vllm-pp2-dspark/) (each beside
its `.orig` and `.diff`; `mounts.txt` lists all seven `--volume` lines relative
to the recipe directory; `selfcheck.sh` is a CPU-only import and relay-helper
check inside the image). The overlay relays the draft block from the last
pipeline stage to the first over the existing PP sampled-token side channel,
pads the sample broadcast to a fixed width, lets the draft load its own input
embedding on the last stage, and gives the draft a PP1 parallel config so
config validation accepts it. It requires async scheduling
(`VLLM_ASYNC_SCHEDULING=1`) and runs with adaptive verification off (vLLM's
validator rejects it under PP), so the profile is fixed-K=5 DSpark. It is an
experiment on top of the two PP2 patches, GPU-tested only on this pair;
[`patches/vllm-pp2-dspark/README.md`](patches/vllm-pp2-dspark/README.md) and
[`PLAN.md`](patches/vllm-pp2-dspark/PLAN.md) describe it, and the section's
[`../notes/`](../notes/) record the equivalence check and sanity runs.

## Launch

1. On both nodes: pull/build the image, place the checkpoint, run
   `tools/stage_host_rdma.sh`, and check the recipe directory exists at the
   same absolute path (it is rsync'ed to node1 on every launch).
2. Edit or export knobs from [`config.env`](config.env) — that file is the
   complete list; nothing else is configurable.
3. `./preflight.sh` — idle-HBM gate (> 8 GiB retained fails), no stray GPU
   processes or containers, image present on both nodes, current-boot kernel
   NVIDIA signatures, 48/48 shards on both ranks, jumbo-frame ping on both
   rails, `rdma link` ACTIVE for every HCA in `NCCL_HCAS`, host memory.
4. `./launch_cluster.sh` — starts the remote rank over SSH, then the local
   rank, and waits for `$API_URL/health` (`http://127.0.0.1:30000` unless
   `SWAP_RANKS=1`). `MODE=low-latency` adds DSpark
   (`--speculative-algorithm DSPARK --speculative-dspark-block-size 5`;
   vLLM: `--speculative-config '{"method":"dspark","num_speculative_tokens":5,…}'`;
   under `VLLM_PARALLEL=pp` only with `VLLM_PATCH_PP_DSPARK=1`, which forces
   `enable_adaptive_verification:false`). `ENGINE=vllm` selects
   `serve_vllm_node.sh`; `ENGINE=vllm SWAP_RANKS=1` puts vLLM rank 0 on node1.
5. `./status.sh`, `./logs.sh 0 -f`, `./chat.sh "prompt"` to inspect.
6. `./bench_prefill.sh <label> --tag-env "<description>"` and
   `./bench_decode.sh <label>` (see the contract below);
   `python3 summarize_prefill.py results/prefill/*.jsonl`,
   `python3 summarize_decode.py results/decode/<run>`.
7. `tools/profile_prefill.sh 32768 <label>` for a kernel-time breakdown;
   `tools/run_nccl_fabric_bench.sh` for the host-side fabric sweep.
8. `./stop_cluster.sh` — saves both container logs, stops the containers
   gracefully (SIGTERM, 120 s) on both hosts, waits for the driver to release
   memory, prints idle HBM. Never resets a GPU. Then
   `tools/ensure_clean_rank1.sh` — if node1 still holds more than
   `THRESHOLD_MIB` (512) of HBM with no compute process, performs the
   preauthorized normal OS reboot of node1 and waits for a new boot id,
   docker, NFS, and ACTIVE rails (see [`../notes/`](../notes/) for why).
   node0 is never rebooted automatically; after an operator reboot of node0,
   `tools/after_reboot_node0.sh` remounts the checkpoint share
   (`MODEL_NFS_EXPORT`, if used), restages the host rdma-core libraries, and
   re-runs the preflight.
9. Publish: `python3 ../data/build_data.py --source-root <this directory>`
   from the section root, then `--check`.

Cold first launch takes about 11 minutes (376 s weight load, 285 s of it the
TileLang mHC compile); compiler caches persist under `CACHE_ROOT`.

## Benchmark contract

**Prefill** (`bench_prefill.py`, SGLang `/generate` or vLLM `/v1/completions`
with token-id prompts): every request carries exactly 16K, 32K, 64K, or 128K
unique random token ids (seeded per request, so no prefix cache can help);
`max_new_tokens=1`, temperature 0; `C ∈ {1, 4, 16}` requests held in flight;
the cache is flushed before every point; one unmeasured warm-up request; then
`max(4, 4 × C)` measured requests. Aggregate prompt tok/s is the sum of the
server-reported prompt tokens divided by the wave's wall time; the server's
`prompt_tokens_total` counter delta must equal the client total. Any request
error fails the sweep (exit 1) and the lane is not accepted. The server
context must exceed 131,073 tokens (`CONTEXT_LENGTH=262144`).

**Decode** (`bench_decode.sh` → `llm-inference-bench` finite-request layer):
8,192-token exact input, 1,024 forced output tokens, temperature 0, EOS
ignored, `C` warm-up requests then `5 × C` measured requests at each
concurrency, aggregate output tokens divided by benchmark wall time. DSpark
lanes retain the server-reported accept length and rate per cell. Underfilled
or errored cells are never published.

**Fabric** (`tools/run_nccl_fabric_bench.sh`): `all_reduce_perf -b 64M -e 2G
-f 2 -w 5 -n 20`, one GB300 per station, four rail/tuning configurations.

## Safety

> **GB300 safety:** never run `nvidia-smi --gpu-reset`, PCI unbind/rescan, or
> unload/reload NVIDIA modules. A failed distributed launch is cleaned up by
> removing the named containers on both hosts. If the driver remains unhealthy,
> stop GPU work and coordinate a host reboot with the operator. Never reboot it
> automatically.

`preflight.sh` refuses to launch when a GB300 retains more than 8 GiB with no
process attached; retained HBM is evidence to record, not something to
compensate for with a lower memory fraction. In the measured session node1
retained 4–5 GiB after every graceful SGLang stop and 35–69 GiB after a forced
removal, and even the small residue cost 52 GB at weight-load time, so the
recipe reboots node1 between lanes (`tools/ensure_clean_rank1.sh`, 512 MiB
threshold) instead of launching on top of it. vLLM's rank 0 strands HBM on
whichever host it runs on at every teardown, crash or clean stop (82,292 /
66,258 / 51,470 MiB in the measured session), which is why the later vLLM
lanes are launched with `SWAP_RANKS=1`: rank 0 then lives on node1, the only
node with an automated reboot. The recovery boundary and the
residual-HBM quirk are described in the
[DGX Station guide](../../dgx-station-guide/). Only passing rows are copied
into [`../data/`](../data/) by `build_data.py`.
