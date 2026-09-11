#!/usr/bin/env python3
"""Summarize a torch-profiler Chrome trace (json or json.gz) from SGLang: GPU kernel time by category and top kernels.
Usage: analyze_trace.py trace.json[.gz] [--top 30]"""
import gzip, json, re, sys, collections
path = sys.argv[1]; top = int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 30
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f: data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data
CATS = [
  ("nccl", r"nccl|ncclDevKernel|AllReduce|allreduce|all_reduce|ReduceScatter|AllGather"),
  ("moe", r"moe|Moe|MoE|routing|topk_gating|expert|grouped_gemm|trtllm.*fp4|fp4_block_scale|silu_and_mul|swiglu"),
  ("engram", r"engram"),
  ("indexer/topk", r"indexer|mqa_logits|topk|top_k|sparse.*logits|candidate"),
  ("attention", r"flash_mla|flashmla|fmha|attention|attn|sparse_fwd|sm100_fmha|mla"),
  ("gemm", r"gemm|Gemm|GEMM|cutlass|cute|matmul|nvjet|Kernel_Cutlass|sm100_xmma|tensorop"),
  ("norm/rope/quant", r"norm|rope|Rope|quant|Quant|cast|hadamard|mhc|sinkhorn|sigmoid|softplus"),
  ("memcpy/memset", r"Memcpy|Memset|memcpy|memset"),
  ("elementwise/other-triton", r"triton|elementwise|vectorized|reduce_kernel|index_|gather|scatter|cat_|copy_|fill_|arange|cumsum|sort"),
]
def cat(name):
    for c, rx in CATS:
        if re.search(rx, name): return c
    return "other"
kern = collections.Counter(); kcount = collections.Counter(); bycat = collections.Counter()
t0 = min((e["ts"] for e in events if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")), default=0)
t1 = max((e["ts"]+e.get("dur",0) for e in events if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")), default=0)
for e in events:
    if e.get("ph") != "X": continue
    c = e.get("cat","")
    if c in ("kernel","gpu_memcpy","gpu_memset"):
        n = e["name"]; d = e.get("dur",0)
        kern[n] += d; kcount[n] += 1; bycat[cat(n)] += d
tot = sum(kern.values())
print(f"GPU kernel wall span: {(t1-t0)/1e3:.1f} ms; summed kernel time: {tot/1e3:.1f} ms; kernels: {sum(kcount.values())}")
print("\nBy category (ms, % of summed kernel time):")
for c, d in bycat.most_common(): print(f"  {c:<24} {d/1e3:9.1f}  {100*d/tot:5.1f}%")
print(f"\nTop {top} kernels (total ms, count, avg us):")
for n, d in kern.most_common(top): print(f"  {d/1e3:8.1f} ms  {kcount[n]:6d}x  {d/kcount[n]:8.1f} us  [{cat(n)}] {n[:110]}")
