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
| vLLM local image id | {{VLLM_IMAGE_ID}} |
| Decode client | [`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |
| Prefill client | [`bench_prefill.py`](bench_prefill.py) in this directory (stdlib; `requests` optional) |
| NCCL fabric test | host NCCL 2.31.2, nccl-tests 2.19.7 `all_reduce_perf` |

The SGLang image must be rebuilt from the `dsv4.1` branch; `serve_node.sh`
records `docker image inspect --format '{{.Id}}'` into
`logs/launch-rank<N>.txt`, and that id must match the pin above before a row
is accepted.

## Two-station topology

- One GB300 per station. SGLang runs TP2 + EP2 (`--tp 2 --ep-size 2 --nnodes 2`);
  vLLM runs TP1 × PP2 by default (`VLLM_PARALLEL=pp`) or TP2 × PP1
  (`VLLM_PARALLEL=tp`, required for DSpark).
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

### vLLM pipeline parallel needs a one-line text-only patch

The `deepseekv41-flash-0909` build raises `DeepSeek V4 vision MoE routing
requires input_ids` on every non-first pipeline-parallel rank (vLLM hands those
ranks no `input_ids`, and none during the memory-profile dummy run). With
`VLLM_PATCH_PP=1` (the default) `serve_vllm_node.sh` bind-mounts
[`patches/vllm/deepseek_v4_nvidia_model.py`](patches/vllm/deepseek_v4_nvidia_model.py)
over `vllm/models/deepseek_v4/nvidia/model.py` inside the container; the only
change routes every token as text when the vision routing bias exists but
`input_ids` is `None`, which under `--language-model-only` changes nothing.
The unmodified file is kept beside it as `.orig`; see
[`patches/vllm/README.md`](patches/vllm/README.md).

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
4. `./launch_cluster.sh` — starts rank 1 over SSH, then rank 0, and waits for
   `http://127.0.0.1:30000/health`. `MODE=low-latency` adds DSpark
   (`--speculative-algorithm DSPARK --speculative-dspark-block-size 5`;
   vLLM: `--speculative-config '{"method":"dspark","num_speculative_tokens":5,…}'`).
   `ENGINE=vllm` selects `serve_vllm_node.sh`.
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
retained 4–5 GiB after every graceful stop and 35–69 GiB after a forced
removal, and even the small residue cost 52 GB at weight-load time, so the
recipe reboots node1 between lanes (`tools/ensure_clean_rank1.sh`, 512 MiB
threshold) instead of launching on top of it. The recovery boundary and the
residual-HBM quirk are described in the
[DGX Station guide](../../dgx-station-guide/). Only passing rows are copied
into [`../data/`](../data/) by `build_data.py`.
