"""CLI.

  python -m bench run     --config configs/l4_qwen2.5-1.5b.yaml --out results/l4-qwen2.5-1.5b
  python -m bench load    --base-url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct -c 16
  python -m bench verify  --results results/l4-qwen2.5-1.5b
  python -m bench report  --results results/l4-qwen2.5-1.5b --readme README.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _cmd_run(a):
    from .experiment import run
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.modes:
        cfg["modes"] = a.modes
    run(cfg, Path(a.out), a.only)


def _cmd_load(a):
    """Ad-hoc load against an already-running server (no server management)."""
    from . import client
    from .loadgen import run_closed_loop, run_open_loop
    from .metrics import summarize
    from .workloads import TokenPool, filler_text, random_prompts

    async def go():
        async with client.new_session() as s:
            pool = TokenPool(await client.tokenize(s, a.base_url, a.model, filler_text(30000, 2024)))
            prompts = random_prompts(pool, a.num_requests, a.input_len, seed=0)
            if a.request_rate:
                res, wall = await run_open_loop(s, a.base_url, a.model, prompts, a.output_len, a.request_rate)
            else:
                res, wall = await run_closed_loop(s, a.base_url, a.model, prompts, a.output_len,
                                                  a.concurrency, warmup_prompts=prompts[:2])
            print(json.dumps(summarize(res, wall), indent=2))
    asyncio.run(go())


def _cmd_verify(a):
    out = Path(a.results) / "correctness_junit.txt"
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/correctness", "--results", a.results,
                           "-q", "-rA", "--no-header", "-p", "no:cacheprovider"],
                          capture_output=True, text=True)
    print(proc.stdout, proc.stderr)
    lines = [l for l in proc.stdout.splitlines() if l.startswith(("PASSED", "FAILED", "SKIPPED", "ERROR"))
             or l.startswith("=") and ("passed" in l or "failed" in l)]
    out.write_text("\n".join(lines) + "\n")
    sys.exit(proc.returncode)


def _cmd_report(a):
    from .report import main
    main(Path(a.results), Path(a.readme) if a.readme else None)


def main():
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="launch server per mode, benchmark, collect correctness evidence")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--modes", nargs="+", choices=["cache_on", "cache_off"])
    r.add_argument("--only", nargs="+", choices=["throughput", "prefix", "correctness"])
    r.set_defaults(fn=_cmd_run)

    l = sub.add_parser("load", help="ad-hoc concurrent load against a running server")
    l.add_argument("--base-url", default="http://127.0.0.1:8000")
    l.add_argument("--model", required=True)
    l.add_argument("-c", "--concurrency", type=int, default=8)
    l.add_argument("--request-rate", type=float, help="open-loop Poisson req/s instead of closed loop")
    l.add_argument("-n", "--num-requests", type=int, default=64)
    l.add_argument("--input-len", type=int, default=1024)
    l.add_argument("--output-len", type=int, default=128)
    l.set_defaults(fn=_cmd_load)

    v = sub.add_parser("verify", help="run correctness tests on a results dir")
    v.add_argument("--results", required=True)
    v.set_defaults(fn=_cmd_verify)

    rp = sub.add_parser("report", help="tables + charts (+ README splice)")
    rp.add_argument("--results", required=True)
    rp.add_argument("--readme")
    rp.set_defaults(fn=_cmd_report)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
