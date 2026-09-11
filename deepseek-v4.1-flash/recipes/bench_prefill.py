#!/usr/bin/env python3
"""
bench_prefill.py -- prefill (prompt-processing) throughput benchmark for an
SGLang server's native /generate endpoint (e.g. DeepSeek-V4.1-Flash).

What it measures
  * Every request carries `isl` RANDOM token ids, uniform in [1000, 120000),
    generated from a seed derived from (seed, isl, concurrency, request index):
    reproducible across runs, but different for every request (and every
    point), so the radix/prefix cache cannot help.
  * max_new_tokens=1, so request latency == TTFT and prompt_tokens / latency is
    the per-request prefill rate. A wave of N concurrent requests gives
    aggregate prefill tok/s = sum(prompt_tokens) / wall time of the wave.
  * Cross-checks: meta_info.prompt_tokens from each response, the delta of any
    /metrics counter matching 'prompt_tokens_total' or 'prefill', and 1 Hz
    nvidia-smi util/power samples of the local GB300 (query only, never reset).

Examples
  python3 bench_prefill.py --dry-run --isl 8192,32768 --concurrency 1,4
  python3 bench_prefill.py --isl 8192,32768,131072 --concurrency 1,4,16 \
      --output results/prefill/run.jsonl --label "tp1 chunked16k" \
      --tag-env "sglang 0.5.x --tp 1 --chunked-prefill-size 16384"
  python3 bench_prefill.py --isl 131072 --concurrency 1 --requests-per-point 8 \
      --no-flush-cache --timeout 900 --wait-ready 600

Output: a compact table on stdout and one JSON object per (isl, concurrency)
point appended to --output. Exit status is non-zero if any request failed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

try:
    import requests
except ImportError:  # fall back to urllib (stdlib only)
    requests = None

TOKEN_LO, TOKEN_HI = 1000, 120000
GPU_MATCH = "GB300"


# ------------------------------------------------------------------ HTTP client
class Http:
    """Minimal JSON-over-HTTP client: `requests` if importable, else urllib."""

    def __init__(self, base: str, timeout: float):
        self.base, self.timeout = base.rstrip("/"), timeout
        self._local = threading.local()  # one requests.Session per worker thread

    def _session(self):
        if getattr(self._local, "s", None) is None:
            self._local.s = requests.Session()
        return self._local.s

    def _urllib(self, req, timeout) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    def get(self, path: str, timeout: float | None = None) -> tuple[int, str]:
        t = timeout or self.timeout
        if requests:
            r = self._session().get(self.base + path, timeout=t)
            return r.status_code, r.text
        return self._urllib(self.base + path, t)

    def post(self, path: str, body: bytes, timeout: float | None = None) -> tuple[int, str]:
        t, hdr = timeout or self.timeout, {"Content-Type": "application/json"}
        if requests:
            r = self._session().post(self.base + path, data=body, headers=hdr, timeout=t)
            return r.status_code, r.text
        return self._urllib(urllib.request.Request(self.base + path, data=body, headers=hdr, method="POST"), t)


def wait_ready(http: Http, seconds: float) -> None:
    """Poll /health until 200 (retrying connection errors) or give up."""
    deadline, last = time.monotonic() + seconds, "not tried"
    while True:
        try:
            code, _ = http.get("/health", timeout=5)
            if code == 200:
                return
            last = f"HTTP {code}"
        except Exception as e:  # connection refused, timeout, ...
            last = f"{type(e).__name__}: {str(e)[:100]}"
        if time.monotonic() >= deadline:
            sys.exit(f"error: {http.base}/health not ready after {seconds:.0f}s (last: {last})")
        print(f"  waiting for {http.base}/health ... ({last})", flush=True)
        time.sleep(2)


def server_info(http: Http) -> dict:
    """Best-effort snapshot of model path and a few server settings."""
    info: dict = {}
    wanted = {"/get_model_info": ("model_path",), "/get_server_info": ("version", "tp_size", "dp_size", "context_len",
              "chunked_prefill_size", "max_total_num_tokens", "max_running_requests", "max_prefill_tokens")}
    for path, keys in wanted.items():
        try:
            code, text = http.get(path, timeout=10)
            if code == 200:
                d = json.loads(text)
                info.update({k: d[k] for k in keys if k in d})
        except Exception:
            pass
    return info


def scrape_metrics(http: Http) -> dict[str, float] | None:
    """Sum Prometheus samples whose name matches prompt_tokens_total|prefill (labels merged). None if unavailable."""
    try:
        code, text = http.get("/metrics", timeout=10)
    except Exception:
        return None
    if code != 200:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if ("prompt_tokens_total" not in name and "prefill" not in name) or name.endswith(("_bucket", "_created")):
            continue
        rest = line.split("}", 1)[1] if "{" in line else line[len(name):]
        try:
            out[name] = out.get(name, 0.0) + float(rest.split()[0])
        except (ValueError, IndexError):
            continue
    return out


# ------------------------------------------------------------------ GPU sampler
class GpuSampler(threading.Thread):
    """1 Hz nvidia-smi --query-gpu sampler (read-only; never resets anything). Keeps only GPU_MATCH rows."""
    CMD = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt, self.util, self.power, self.name = threading.Event(), [], [], None

    def run(self):
        while not self.stop_evt.is_set():
            t0 = time.monotonic()
            try:
                out = subprocess.run(self.CMD, capture_output=True, text=True, timeout=5).stdout
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                return
            for row in out.splitlines():
                f = [x.strip() for x in row.split(",")]
                if len(f) >= 4 and GPU_MATCH in f[1]:
                    try:
                        u, p = float(f[2]), float(f[3])
                    except ValueError:  # "[N/A]"
                        continue
                    self.name = f[1]
                    self.util.append(u)
                    self.power.append(p)
            self.stop_evt.wait(max(0.0, 1.0 - (time.monotonic() - t0)))

    def stop(self) -> dict:
        self.stop_evt.set()
        self.join(timeout=10)
        mean = lambda xs: round(statistics.fmean(xs), 1) if xs else None
        return {"gpu_name": self.name, "gpu_samples": len(self.util),
                "gpu_util_mean_pct": mean(self.util), "gpu_util_max_pct": max(self.util, default=None),
                "gpu_power_mean_w": mean(self.power), "gpu_power_max_w": max(self.power, default=None)}


# ------------------------------------------------------------------ requests
ENGINE = {"name": "sglang", "model": None}  # set from --engine; vllm uses /v1/completions with token-id prompts


def make_body(seed: int, isl: int, conc: int, idx) -> bytes:
    """Pre-serialized request body with isl random token ids. Deterministic per (seed, isl, conc, idx)."""
    rng = random.Random(f"{seed}/{isl}/{conc}/{idx}")  # str seeds hash via sha512 -> stable across runs
    ids = rng.choices(range(TOKEN_LO, TOKEN_HI), k=isl)
    if ENGINE["name"] == "vllm":
        return json.dumps({"model": ENGINE["model"], "prompt": ids, "max_tokens": 1, "temperature": 0,
                           "stream": False}, separators=(",", ":")).encode()
    return json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                       "stream": False}, separators=(",", ":")).encode()


def run_one(http: Http, body: bytes, timeout: float) -> dict:
    t0 = time.perf_counter()
    try:
        path = "/v1/completions" if ENGINE["name"] == "vllm" else "/generate"
        code, text = http.post(path, body, timeout)
        t1 = time.perf_counter()
        if code != 200:
            return {"ok": False, "t0": t0, "t1": t1, "error": f"HTTP {code}: {text[:200]}"}
        resp = json.loads(text)
        if ENGINE["name"] == "vllm":
            u = resp.get("usage", {})
            return {"ok": True, "t0": t0, "t1": t1, "latency": t1 - t0, "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"), "e2e_latency": None}
        meta = (resp[0] if isinstance(resp, list) else resp).get("meta_info", {})
        return {"ok": True, "t0": t0, "t1": t1, "latency": t1 - t0, "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"), "e2e_latency": meta.get("e2e_latency")}
    except Exception as e:
        return {"ok": False, "t0": t0, "t1": time.perf_counter(), "error": f"{type(e).__name__}: {e}"}


def pct(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def run_point(args, http: Http, isl: int, conc: int, n_req: int) -> dict:
    if args.flush_cache:
        try:
            code, text = http.post("/reset_prefix_cache" if ENGINE["name"] == "vllm" else "/flush_cache", b"", timeout=60)
            assert code == 200, f"HTTP {code}: {text[:120]}"
        except Exception as e:
            print(f"  warn: /flush_cache failed: {str(e)[:160]}", flush=True)
    warm_bodies = [make_body(args.seed, isl, conc, f"warm{i}") for i in range(args.warmup)]
    bodies = [make_body(args.seed, isl, conc, i) for i in range(n_req)]
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for w in ex.map(lambda b: run_one(http, b, args.timeout), warm_bodies):
            if not w["ok"]:
                print(f"  warn: warmup request failed: {w['error']}", flush=True)

    m0 = scrape_metrics(http)
    gpu = GpuSampler()
    gpu.start()
    with ThreadPoolExecutor(max_workers=conc) as ex:  # exactly `conc` in flight until the queue drains
        res = list(ex.map(lambda b: run_one(http, b, args.timeout), bodies))
    gstats = gpu.stop()
    m1 = scrape_metrics(http)

    ok = [r for r in res if r["ok"]]
    bad = [r for r in res if not r["ok"]]
    wall = max(r["t1"] for r in res) - min(r["t0"] for r in res)
    tok = [r["prompt_tokens"] if r["prompt_tokens"] is not None else isl for r in ok]
    lat = sorted(r["latency"] for r in ok)
    per_req = [t / r["latency"] for t, r in zip(tok, ok)]
    e2e = [r["e2e_latency"] for r in ok if isinstance(r.get("e2e_latency"), (int, float))]
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": args.label,
           "tag_env": args.tag_env, "host": args.host, "port": args.port, "seed": args.seed,
           "isl": isl, "concurrency": conc, "requests": n_req, "warmup": args.warmup,
           "flush_cache": args.flush_cache, "ok": not bad, "errors": len(bad),
           "error_samples": [b["error"] for b in bad[:3]],
           "wall_s": round(wall, 4), "prompt_tokens_total": sum(tok), "prompt_tokens_expected": isl * len(ok),
           "token_count_match": all(t == isl for t in tok),
           "agg_prompt_tok_s": round(sum(tok) / wall, 1) if ok and wall > 0 else None,
           "ttft_mean_s": round(statistics.fmean(lat), 4) if lat else None,
           "ttft_p50_s": round(pct(lat, 0.5), 4) if lat else None,
           "ttft_p99_s": round(pct(lat, 0.99), 4) if lat else None,
           "ttft_min_s": round(lat[0], 4) if lat else None, "ttft_max_s": round(lat[-1], 4) if lat else None,
           "req_tok_s_mean": round(statistics.fmean(per_req), 1) if per_req else None,
           "req_tok_s_min": round(min(per_req), 1) if per_req else None, "req_tok_s_max": round(max(per_req), 1) if per_req else None,
           "server_e2e_latency_mean_s": round(statistics.fmean(e2e), 4) if e2e else None,
           "metrics_delta": {k: round(m1[k] - m0.get(k, 0.0), 3) for k in m1} if (m0 is not None and m1 is not None) else None,
           **gstats}
    return rec


# ------------------------------------------------------------------ CLI
HEADER = (f"{'isl':>7} {'conc':>4} {'n':>4} {'wall_s':>8} {'agg_tok/s':>10} {'ttft_mean':>9} {'ttft_p50':>9} "
          f"{'ttft_p99':>9} {'req_tok/s':>9} {'util%':>5} {'W':>6} {'metrics_prompt_d':>16} {'err':>3}")


def fmt_row(r: dict) -> str:
    f = lambda v, w, p=1: f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
    md = r.get("metrics_delta") or {}
    pt = next((md[k] for k in md if "prompt_tokens_total" in k), None)
    return (f"{r['isl']:>7} {r['concurrency']:>4} {r['requests']:>4} {f(r['wall_s'], 8, 2)} {f(r['agg_prompt_tok_s'], 10)} "
            f"{f(r['ttft_mean_s'], 9, 3)} {f(r['ttft_p50_s'], 9, 3)} {f(r['ttft_p99_s'], 9, 3)} {f(r['req_tok_s_mean'], 9)} "
            f"{f(r['gpu_util_mean_pct'], 5, 0)} {f(r['gpu_power_mean_w'], 6, 0)} {f(pt, 16, 0)} {r['errors']:>3}")


def parse_int_list(s: str, name: str) -> list[int]:
    try:
        vals = [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"error: --{name} must be a comma list of ints, got {s!r}")
    if not vals or any(v <= 0 for v in vals):
        sys.exit(f"error: --{name} values must be positive ints, got {s!r}")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="See the module docstring for examples.")
    p.add_argument("--engine", choices=["sglang", "vllm"], default="sglang", help="sglang: native /generate; vllm: /v1/completions with token-id prompt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--isl", default="8192,32768,131072", help="comma list of input lengths in tokens (default %(default)s)")
    p.add_argument("--concurrency", default="1,4,16", help="comma list of in-flight request counts (default %(default)s)")
    p.add_argument("--requests-per-point", type=int, default=None, help="measured requests per point (default 4*concurrency, min 4)")
    p.add_argument("--warmup", type=int, default=1, help="unmeasured warmup requests per point (default %(default)s)")
    p.add_argument("--flush-cache", action=argparse.BooleanOptionalAction, default=True, help="POST /flush_cache before each point")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds (default %(default)s)")
    p.add_argument("--wait-ready", type=float, default=120.0, help="seconds to wait for /health at startup (default %(default)s)")
    p.add_argument("--output", default=None, help="JSONL path; one line per point is appended")
    p.add_argument("--label", default="", help="free text stored in every result record")
    p.add_argument("--tag-env", default=None, help="free text describing the server config, stored in every record")
    p.add_argument("--seed", type=int, default=1234, help="base seed for token-id generation (default %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without contacting the server")
    a = p.parse_args()
    a.isl_list, a.conc_list = parse_int_list(a.isl, "isl"), parse_int_list(a.concurrency, "concurrency")
    if a.warmup < 0 or (a.requests_per_point is not None and a.requests_per_point < 1):
        sys.exit("error: --warmup must be >= 0 and --requests-per-point >= 1")
    return a


def main() -> int:
    args = parse_args()
    ENGINE["name"] = args.engine
    n_for = lambda c: max(4, args.requests_per_point if args.requests_per_point else 4 * c)
    points = [(isl, c) for isl in args.isl_list for c in args.conc_list]
    base = f"http://{args.host}:{args.port}"
    print(f"target={base} points={len(points)} warmup={args.warmup} flush_cache={args.flush_cache} "
          f"timeout={args.timeout:.0f}s seed={args.seed} output={args.output or '(none)'} label={args.label!r}")
    if args.dry_run:
        for isl, c in points:
            n = n_for(c)
            print(f"  isl={isl:>7} conc={c:>3} warmup={args.warmup} requests={n:>4} prompt_tokens={isl * n:>12,} "
                  f"body={len(make_body(args.seed, isl, c, 0)) / 1e6:.2f} MB/req")
        print(f"  total measured prompt tokens: {sum(isl * n_for(c) for isl, c in points):,}\n"
              "  dry run: nothing sent (nvidia-smi and /metrics untouched).")
        return 0

    http = Http(base, args.timeout)
    if ENGINE["name"] == "vllm":
        wait_ready(http, args.wait_ready)
        code, text = http.get("/v1/models", timeout=10)
        ENGINE["model"] = json.loads(text)["data"][0]["id"] if code == 200 else "default"
    wait_ready(http, args.wait_ready)
    info = server_info(http)
    print("server:", json.dumps(info) if info else "(no /get_model_info or /get_server_info)")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(HEADER, flush=True)
    failed = 0
    for isl, c in points:
        rec = run_point(args, http, isl, c, n_for(c))
        rec["server"] = info
        print(fmt_row(rec), flush=True)
        if not rec["token_count_match"]:
            print(f"  warn: server prompt_tokens != isl for some requests (total {rec['prompt_tokens_total']} vs "
                  f"expected {rec['prompt_tokens_expected']})", flush=True)
        for e in rec["error_samples"]:
            print(f"  error: {e}", flush=True)
        if args.output:
            with open(args.output, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        failed += rec["errors"]
    if failed:
        print(f"FAILED: {failed} request(s) failed", file=sys.stderr)
        return 1
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
