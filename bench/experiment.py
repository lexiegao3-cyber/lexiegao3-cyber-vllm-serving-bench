"""End-to-end experiment: for each prefix-caching mode, start the server, run the
throughput sweep, the shared-prefix sweep and the correctness collection, then
stop the server. Everything lands in one results directory.
"""
from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from pathlib import Path

from . import client, correctness
from .loadgen import run_closed_loop
from .metrics import summarize
from .server import ServerProcess, build_cmd
from .workloads import TokenPool, filler_text, random_prompts, shared_prefix_prompts


def _n_requests(c: int, cfg: dict) -> int:
    return max(cfg["min_requests"], cfg["requests_per_concurrency"] * c)


async def _token_pool(url: str, model: str, n_words: int) -> TokenPool:
    async with client.new_session() as s:
        return TokenPool(await client.tokenize(s, url, model, filler_text(n_words, seed=2024)))


async def throughput_sweep(url: str, model: str, pool: TokenPool, cfg: dict) -> list[dict]:
    rows = []
    async with client.new_session() as s:
        for c in cfg["concurrency"]:
            n = _n_requests(c, cfg)
            prompts = random_prompts(pool, n, cfg["input_len"], seed=c)
            warm = random_prompts(pool, cfg["warmup"], cfg["input_len"], seed=10_000 + c)
            res, wall = await run_closed_loop(s, url, model, prompts, cfg["output_len"], c,
                                              warmup_prompts=warm)
            row = {"concurrency": c, **summarize(res, wall)}
            print(f"  [throughput] c={c:<3} n={n:<4} TTFT p50={row['ttft_ms_p50']}ms "
                  f"E2E p95={row['e2e_ms_p95']}ms out={row['output_tok_s']} tok/s errors={row['errors']}",
                  flush=True)
            rows.append(row)
    return rows


async def prefix_sweep(url: str, model: str, pool: TokenPool, cfg: dict) -> list[dict]:
    """Shared-prefix workload. Each concurrency level uses *fresh* prefixes
    (seeded by level) so a level never benefits from blocks cached by the
    previous one: every group pays exactly one cold miss per level."""
    rows = []
    async with client.new_session() as s:
        for c in cfg["concurrency"]:
            reqs = shared_prefix_prompts(pool, cfg["num_requests"], cfg["num_groups"],
                                         cfg["shared_prefix_len"], cfg["suffix_len"], seed=500 + c)
            before = await client.prefix_cache_counters(s, url)
            res, wall = await run_closed_loop(s, url, model, [r.prompt for r in reqs],
                                              cfg["output_len"], c)
            after = await client.prefix_cache_counters(s, url)
            row = {"concurrency": c, **summarize(res, wall)}
            if after:
                q = after["queries"] - before.get("queries", 0)
                h = after["hits"] - before.get("hits", 0)
                row["cache_query_tokens"], row["cache_hit_tokens"] = q, h
                row["cache_hit_rate"] = round(h / q, 4) if q else 0.0
            print(f"  [prefix]     c={c:<3} TTFT p50={row['ttft_ms_p50']}ms p95={row['ttft_ms_p95']}ms "
                  f"E2E p95={row['e2e_ms_p95']}ms out={row['output_tok_s']} tok/s "
                  f"hit_rate={row.get('cache_hit_rate')}", flush=True)
            rows.append(row)
    return rows


def _env_info(cfg: dict) -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "vllm": sh("python -c 'import vllm; print(vllm.__version__)' 2>/dev/null"),
        "torch": sh("python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null"),
        "git_commit": sh("git rev-parse --short HEAD 2>/dev/null"),
        "model": cfg["server"]["model"],
    }


def run(cfg: dict, out_dir: Path, only: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "env.json").write_text(json.dumps(_env_info(cfg), indent=2))
    srv = cfg["server"]
    url = f"http://127.0.0.1:{srv['port']}"
    model = srv.get("served_model_name") or srv["model"]
    stages = only or ["throughput", "prefix", "correctness"]

    for mode in cfg["modes"]:
        print(f"\n=== mode: {mode} ===", flush=True)
        cmd = build_cmd(srv["cmd"], mode, model=srv["model"], port=srv["port"],
                        max_model_len=srv["max_model_len"], **srv.get("cmd_fields", {}))
        with ServerProcess(cmd, url, out_dir / f"server_{mode}.log", srv["startup_timeout_s"]):
            pool = asyncio.run(_token_pool(url, model, cfg["token_pool_words"]))
            result: dict = {"mode": mode, "cmd": cmd}
            if "throughput" in stages and mode in cfg["throughput"]["modes"]:
                result["throughput"] = asyncio.run(throughput_sweep(url, model, pool, cfg["throughput"]))
            if "prefix" in stages:
                result["prefix"] = asyncio.run(prefix_sweep(url, model, pool, cfg["prefix"]))
            (out_dir / f"bench_{mode}.json").write_text(json.dumps(result, indent=2))
            if "correctness" in stages:
                print("  [correctness] collecting ...", flush=True)
                rec = asyncio.run(correctness.collect(url, model, cfg["correctness"], srv["max_model_len"]))
                (out_dir / f"correctness_{mode}.json").write_text(json.dumps(rec, indent=2))
    print(f"\nDone. Results in {out_dir}\n"
          f"Next:  pytest tests/correctness --results {out_dir}\n"
          f"       python -m bench report --results {out_dir} --readme README.md")
