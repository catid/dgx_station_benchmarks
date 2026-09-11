# CPU-only self-check for the PP2 + DSpark overlay. Run inside the image with the
# files from mounts.txt bind-mounted (see selfcheck.sh). No GPU is touched.
import importlib
import inspect
import sys
import types

import numpy as np
import torch

failures = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# 1. The patched modules import and carry the new symbols.
pp_utils = importlib.import_module("vllm.v1.worker.gpu.pp_utils")
check(hasattr(pp_utils, "_pad_sampled_tokens"), "pp_utils._pad_sampled_tokens present")
check(hasattr(pp_utils, "select_received_drafts"), "pp_utils.select_received_drafts present")
check(hasattr(pp_utils.PPHandler, "broadcast_drafts"), "PPHandler.broadcast_drafts present")
check("draft_tokens" in pp_utils.PendingRecv.__dataclass_fields__, "PendingRecv.draft_tokens field")
sig = inspect.signature(pp_utils.PPHandler.get_prev_sampled_outputs)
check("draft_tokens_to_update" in sig.parameters, "get_prev_sampled_outputs takes draft_tokens_to_update")

dspark_utils = importlib.import_module("vllm.v1.worker.gpu.spec_decode.dspark.utils")
src = inspect.getsource(dspark_utils.load_dspark_model)
check("does not support pipeline parallelism" not in src, "dspark utils: PP rejection removed")
check("PPMissingLayer" in src and "loads_own_embed_tokens" in src, "dspark utils: placeholder-embedding guard present")

dspark_model = importlib.import_module("vllm.models.deepseek_v4_1.nvidia.dspark")
src = inspect.getsource(dspark_model.DSparkDeepseekV4ForCausalLM.load_weights)
check('name == "embed.weight" and self.loads_own_embed_tokens' in src, "dspark model: loads embed.weight on last stage")
src = inspect.getsource(dspark_model.DSparkDeepseekV4ForCausalLM.__init__)
check("self.loads_own_embed_tokens = not get_pp_group().is_first_rank" in src, "dspark model: loads_own_embed_tokens flag")

mr = importlib.import_module("vllm.v1.worker.gpu.model_runner")
src = inspect.getsource(mr.GPUModelRunner.__init__)
check('self.speculative_config.method != "dspark"' in src, "model_runner: PP allowed for dspark only")
check("requires async" in src, "model_runner: sync scheduling rejected under PP+dspark")
src = inspect.getsource(mr.GPUModelRunner.sample_tokens)
check("self.pp_handler.broadcast_drafts(" in src, "model_runner: broadcasts drafts after propose")
src = inspect.getsource(mr.GPUModelRunner.update_pp_decode_requests)
check("get_prev_sampled_outputs(\n                self.req_states.draft_tokens\n            )" in src, "model_runner: hands draft buffer to slot consumer")

# 1b. The draft parallel config is PP1 for DSpark (draft config verification
#     rejects non-SupportsPP architectures under PP>1).
spec_mod = importlib.import_module("vllm.config.speculative")
from vllm.config.parallel import ParallelConfig
tgt = ParallelConfig(pipeline_parallel_size=2, tensor_parallel_size=1)
dp = spec_mod.SpeculativeConfig.create_draft_parallel_config(tgt, 1, "dspark")
check(dp.pipeline_parallel_size == 1 and dp.tensor_parallel_size == 1, "speculative: dspark draft parallel config is PP1/TP1 under PP2 target")
dp2 = spec_mod.SpeculativeConfig.create_draft_parallel_config(tgt, 1, "mtp")
check(dp2.pipeline_parallel_size == 2, "speculative: other methods keep the target PP size")
dp3 = spec_mod.SpeculativeConfig.create_draft_parallel_config(tgt, 1)
check(dp3.pipeline_parallel_size == 2, "speculative: method omitted keeps stock behaviour")
src_pi = inspect.getsource(spec_mod.SpeculativeConfig.__post_init__)
check("self.draft_tensor_parallel_size,\n                        self.method," in src_pi, "speculative: __post_init__ passes self.method")

# 2. The draft architecture resolves to the patched class.
from vllm.model_executor.models.registry import ModelRegistry

cls_name = "DSparkV41DraftModel"
try:
    model_cls = ModelRegistry._try_load_model_cls(cls_name)
except Exception as e:  # noqa: BLE001
    model_cls = None
    print("registry lookup raised:", repr(e))
check(model_cls is dspark_model.DSparkDeepseekV4ForCausalLM, f"registry: {cls_name} -> patched DSparkDeepseekV4ForCausalLM")

# 3. Pure helpers with fake CPU tensors.
K = 5
plain = torch.arange(3, dtype=torch.int64).view(3, 1)
padded = pp_utils._pad_sampled_tokens(plain, K + 1)
check(tuple(padded.shape) == (3, K + 1) and torch.equal(padded[:, 0], plain[:, 0]) and int(padded[:, 1:].abs().sum()) == 0,
      "_pad_sampled_tokens: [n,1] -> [n,K+1] zero padded")
full = torch.ones(3, K + 1, dtype=torch.int64)
check(pp_utils._pad_sampled_tokens(full, K + 1) is full, "_pad_sampled_tokens: full width passthrough")
try:
    pp_utils._pad_sampled_tokens(torch.ones(3, K + 2, dtype=torch.int64), K + 1)
    check(False, "_pad_sampled_tokens: rejects over-wide input")
except ValueError:
    check(True, "_pad_sampled_tokens: rejects over-wide input")

idx = np.array([7, 3, 9, 1], dtype=np.intp)
pos, rows_idx = pp_utils.select_received_drafts(idx, np.zeros(4, dtype=bool))
check(pos.tolist() == [0, 1, 2, 3] and rows_idx.tolist() == idx.tolist(), "select_received_drafts: no exclusion keeps every row")
pos, rows_idx = pp_utils.select_received_drafts(idx, np.array([False, True, False, True]))
check(pos.tolist() == [0, 2] and rows_idx.tolist() == [7, 9] and pos.dtype == np.intp, "select_received_drafts: drops excluded rows")

ib = types.SimpleNamespace(
    num_computed_tokens_np=np.array([10, 0, 100], dtype=np.int32),
    prefill_len_np=np.array([12, 50, 100], dtype=np.int32),
    num_scheduled_tokens=np.array([2, 20, 6], dtype=np.int32),
)
mask = pp_utils.compute_need_sampled_mask(ib)
check(mask is not None and mask.tolist() == [True, False, True], "compute_need_sampled_mask: final-chunk/decode rows only")
ib2 = types.SimpleNamespace(
    num_computed_tokens_np=np.array([0], dtype=np.int32),
    prefill_len_np=np.array([50], dtype=np.int32),
    num_scheduled_tokens=np.array([20], dtype=np.int32),
)
check(pp_utils.compute_need_sampled_mask(ib2) is None, "compute_need_sampled_mask: pure prefill chunk -> None (no relay)")

# 4. Slot-ring consume path on CPU: the relayed drafts land in the request state
#    for surviving rows only, and freed rows are skipped.
class FakeEvent:
    pass

class FakeStream:
    def __init__(self):
        self.waited = 0
    def wait_event(self, ev):
        self.waited += 1

pp_utils.async_copy_to_gpu = lambda a, device=None, out=None: torch.as_tensor(np.asarray(a))  # CPU stand-in

h = object.__new__(pp_utils.PPHandler)
h.device = torch.device("cpu")
h.main_stream = FakeStream()
h.req_idx_gen_np = np.zeros(16, dtype=np.int32)
h.num_speculative_steps = K
h.is_last_rank = False
from collections import deque
h.queue = deque([None, None])  # pp_size == 2 pre-seed

req_state_drafts = torch.zeros(16, K, dtype=torch.int64)
idx_np = np.array([4, 9, 2], dtype=np.intp)
slot = pp_utils.PendingRecv(
    event=FakeEvent(),
    sampled_tokens=torch.full((3, K + 1), 5, dtype=torch.int64),
    num_sampled=torch.ones(3, dtype=torch.int32),
    num_rejected=torch.zeros(3, dtype=torch.int32),
    idx_mapping=torch.as_tensor(idx_np),
    idx_mapping_np=idx_np,
    need_sampled_mask=np.array([True, True, False]),
    gen_at_receive_np=h.req_idx_gen_np[idx_np].copy(),
    draft_tokens=torch.tensor([[1] * K, [2] * K, [3] * K], dtype=torch.int64),
)
# step T: receive would have put the slot at queue[-1]
h.queue.popleft(); h.queue.append(slot)
# step T+1: consumes a None
check(h.get_prev_sampled_outputs(req_state_drafts) is None, "slot ring: step T+1 consumes pre-seeded None")
# request in slot 9 freed between receive and consume
h.on_req_idx_freed(9)
out = h.get_prev_sampled_outputs(req_state_drafts)
check(out is not None and out["idx_mapping"].tolist() == [4, -1, -1], "slot ring: freed + no-sample rows masked to -1 for post_update")
check(req_state_drafts[4].tolist() == [1] * K, "slot ring: drafts landed for surviving row")
check(int(req_state_drafts[9].abs().sum()) == 0 and int(req_state_drafts[2].abs().sum()) == 0, "slot ring: freed/no-sample rows untouched")
check(h.main_stream.waited == 1, "slot ring: main stream waited on the recv event exactly once")
check(len(h.queue) == 2 and h.queue[-1] is None, "slot ring: this step's slot re-reserved as None")

# all rows excluded -> None, but still waited (buffers may be in flight)
slot2 = pp_utils.PendingRecv(
    event=FakeEvent(), sampled_tokens=slot.sampled_tokens, num_sampled=slot.num_sampled,
    num_rejected=slot.num_rejected, idx_mapping=slot.idx_mapping, idx_mapping_np=idx_np,
    need_sampled_mask=np.array([False, False, False]), gen_at_receive_np=h.req_idx_gen_np[idx_np].copy(),
    draft_tokens=torch.full((3, K), 8, dtype=torch.int64),
)
h.queue.popleft(); h.queue.append(slot2)
h.get_prev_sampled_outputs(req_state_drafts)  # consumes None
before = req_state_drafts.clone()
check(h.get_prev_sampled_outputs(req_state_drafts) is None and torch.equal(before, req_state_drafts) and h.main_stream.waited == 2,
      "slot ring: all-excluded slot returns None, writes nothing, still waits")

# 5. Draft-name remap is untouched for mtp.* and still drops non-mtp names.
remap = dspark_model.DSparkDeepseekV4ForCausalLM._remap_dspark_name
fake = types.SimpleNamespace(model=types.SimpleNamespace(confidence_head=None))
check(remap(fake, "mtp.1.attn.wkv.weight") == "model.layers.1.attn.wkv.weight", "remap: mtp layer weight")
check(remap(fake, "mtp.0.main_proj.weight") == "model.main_proj.weight", "remap: mtp.0.main_proj")
check(remap(fake, "embed.weight") is None and remap(fake, "head.weight") is None, "remap: embed/head stay target-owned (embed handled before remap under PP)")

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
