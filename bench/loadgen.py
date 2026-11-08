"""Closed-loop concurrent load generator.

Closed loop = exactly `concurrency` requests are in flight at all times; when one
finishes the next starts. This pins the batch size the server sees, which makes
"latency vs concurrency" curves easy to interpret. (An open-loop Poisson mode is
also provided for ad-hoc use.)
"""
from __future__ import annotations

import asyncio
import random
import time

import aiohttp

from .client import RequestResult, stream_completion


async def run_closed_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompts: list,
    max_tokens: int,
    concurrency: int,
    ignore_eos: bool = True,
    warmup_prompts: list | None = None,
) -> tuple[list[RequestResult], float]:
    """Run all prompts with `concurrency` workers. Returns (results, wall seconds).

    Warmup requests run first and are excluded from results (CUDA graphs,
    allocator and HTTP connections are warm before measuring).
    """
    if warmup_prompts:
        await asyncio.gather(*[
            stream_completion(session, base_url, model, p, max_tokens, ignore_eos)
            for p in warmup_prompts
        ])

    queue: asyncio.Queue = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))
    results: list[RequestResult | None] = [None] * len(prompts)

    async def worker():
        while True:
            try:
                i, p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results[i] = await stream_completion(session, base_url, model, p, max_tokens, ignore_eos)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(min(concurrency, len(prompts)))])
    wall = time.perf_counter() - t0
    return [r for r in results if r is not None], wall


async def run_open_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompts: list,
    max_tokens: int,
    request_rate: float,
    seed: int = 0,
    ignore_eos: bool = True,
) -> tuple[list[RequestResult], float]:
    """Poisson arrivals at `request_rate` req/s (concurrency is unbounded)."""
    rng = random.Random(seed)
    tasks = []
    t0 = time.perf_counter()
    for p in prompts:
        tasks.append(asyncio.create_task(
            stream_completion(session, base_url, model, p, max_tokens, ignore_eos)))
        await asyncio.sleep(rng.expovariate(request_rate))
    results = await asyncio.gather(*tasks)
    return list(results), time.perf_counter() - t0
