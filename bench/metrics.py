"""Aggregate per-request results into the numbers reported in the README."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .client import RequestResult


def pct(values: Iterable[float], q: float) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def summarize(results: list[RequestResult], wall_s: float) -> dict:
    """Summary for one (workload, concurrency) cell.

    output_tok_s : total generated tokens / wall-clock time of the measured phase
                   (system throughput, the number a capacity planner cares about)
    tpot_ms      : per-request decode speed, (E2E - TTFT) / (output_tokens - 1)
    """
    ok = [r for r in results if r.ok]
    ttft = [r.ttft for r in ok]
    e2e = [r.e2e for r in ok]
    itl = [x for r in ok for x in r.itl]
    tpot = [
        (r.e2e - r.ttft) / (r.output_tokens - 1)
        for r in ok
        if r.output_tokens and r.output_tokens > 1
    ]
    out_tokens = sum(r.output_tokens or 0 for r in ok)
    in_tokens = sum(r.prompt_tokens or 0 for r in ok)

    def ms(x):
        return None if x is None else round(x * 1000, 2)

    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "error_samples": [r.error for r in results if not r.ok][:3],
        "wall_s": round(wall_s, 3),
        "ttft_ms_p50": ms(pct(ttft, 50)),
        "ttft_ms_p95": ms(pct(ttft, 95)),
        "ttft_ms_p99": ms(pct(ttft, 99)),
        "e2e_ms_p50": ms(pct(e2e, 50)),
        "e2e_ms_p95": ms(pct(e2e, 95)),
        "e2e_ms_p99": ms(pct(e2e, 99)),
        "itl_ms_p50": ms(pct(itl, 50)),
        "itl_ms_p95": ms(pct(itl, 95)),
        "tpot_ms_mean": ms(float(np.mean(tpot))) if tpot else None,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "output_tok_s": round(out_tokens / wall_s, 1) if wall_s > 0 else None,
        "total_tok_s": round((in_tokens + out_tokens) / wall_s, 1) if wall_s > 0 else None,
        "req_s": round(len(ok) / wall_s, 3) if wall_s > 0 else None,
        "output_len_exact": all(r.output_tokens == ok[0].output_tokens for r in ok) if ok else None,
    }
