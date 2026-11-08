"""Async client for an OpenAI-compatible server (vLLM's `vllm serve`).

Timing is taken client-side with `time.perf_counter()`:
  * TTFT  = time from sending the request to the first streamed chunk that carries text
  * E2E   = time from sending the request to the end of the stream
  * ITL   = gaps between successive text-carrying chunks
Token counts come from the server's `usage` block (`stream_options.include_usage`),
not from counting chunks, because one SSE chunk can carry more than one token.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp


@dataclass
class RequestResult:
    ok: bool
    ttft: float | None = None          # seconds
    e2e: float | None = None           # seconds
    itl: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    text: str = ""
    error: str | None = None
    status: int | None = None
    start: float = 0.0                 # perf_counter at send
    end: float = 0.0                   # perf_counter at stream end
    n_chunks: int = 0
    malformed_chunks: int = 0


def new_session(timeout_s: float = 600) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_s),
        connector=aiohttp.TCPConnector(limit=0),   # never let the client cap concurrency
    )


async def _iter_sse(resp: aiohttp.ClientResponse):
    """Yield the JSON payload string of each `data:` line; stops at [DONE]."""
    buf = b""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip().decode()
            if data == "[DONE]":
                return
            yield data


async def stream_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str | list[int],
    max_tokens: int,
    ignore_eos: bool = True,
    extra: dict[str, Any] | None = None,
) -> RequestResult:
    """Streamed /v1/completions request with greedy decoding.

    `ignore_eos=True` (a vLLM sampling extension) forces exactly `max_tokens`
    output tokens, so every run measures the same amount of decode work.
    """
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }
    if extra:
        body.update(extra)
    res = RequestResult(ok=False)
    res.start = time.perf_counter()
    last = None
    parts: list[str] = []
    try:
        async with session.post(f"{base_url}/v1/completions", json=body) as resp:
            res.status = resp.status
            if resp.status != 200:
                res.error = (await resp.text())[:500]
                res.end = time.perf_counter()
                return res
            async for data in _iter_sse(resp):
                now = time.perf_counter()
                res.n_chunks += 1
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    res.malformed_chunks += 1
                    continue
                if obj.get("usage"):
                    res.prompt_tokens = obj["usage"].get("prompt_tokens")
                    res.output_tokens = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices") or []:
                    txt = ch.get("text") or ""
                    if txt:
                        if res.ttft is None:
                            res.ttft = now - res.start
                        elif last is not None:
                            res.itl.append(now - last)
                        last = now
                        parts.append(txt)
                    if ch.get("finish_reason"):
                        res.finish_reason = ch["finish_reason"]
        res.end = time.perf_counter()
        res.e2e = res.end - res.start
        res.text = "".join(parts)
        res.ok = res.ttft is not None
        if not res.ok:
            res.error = "stream ended without any text"
    except Exception as e:  # network errors, timeouts
        res.end = time.perf_counter()
        res.error = f"{type(e).__name__}: {e}"
    return res


async def chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    top_logprobs: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Non-streamed greedy chat completion with per-token logprobs.

    Returns a normalized record used by the correctness checks:
      {status, text, tokens, top: [[(tok, lp), ...], ...], finish_reason, usage, error}
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    async with session.post(f"{base_url}/v1/chat/completions", json=body) as resp:
        status = resp.status
        raw = await resp.text()
    if status != 200:
        return {"status": status, "error": raw[:500]}
    obj = json.loads(raw)
    ch = obj["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    return {
        "status": status,
        "text": ch["message"]["content"],
        "tokens": [c["token"] for c in content],
        "top": [[(t["token"], t["logprob"]) for t in c.get("top_logprobs", [])] for c in content],
        "finish_reason": ch.get("finish_reason"),
        "usage": obj.get("usage"),
    }


async def tokenize(session: aiohttp.ClientSession, base_url: str, model: str, text: str) -> list[int]:
    async with session.post(f"{base_url}/tokenize", json={"model": model, "prompt": text,
                                                          "add_special_tokens": False}) as resp:
        resp.raise_for_status()
        return (await resp.json())["tokens"]


async def detokenize(session: aiohttp.ClientSession, base_url: str, model: str, ids: list[int]) -> str:
    async with session.post(f"{base_url}/detokenize", json={"model": model, "tokens": ids}) as resp:
        resp.raise_for_status()
        return (await resp.json())["prompt"]


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+\-]+|NaN)$")


async def prefix_cache_counters(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    """Scrape vLLM's Prometheus /metrics for prefix-cache counters (in tokens).

    vLLM V1 exports `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total`
    (labels per engine). We sum across label sets and match by substring so minor
    renames across vLLM versions don't break the scrape. Returns {} if unavailable.
    """
    try:
        async with session.get(f"{base_url}/metrics") as resp:
            if resp.status != 200:
                return {}
            text = await resp.text()
    except Exception:
        return {}
    out = {"queries": 0.0, "hits": 0.0}
    found = False
    for line in text.splitlines():
        m = _METRIC_RE.match(line.strip())
        if not m:
            continue
        name, val = m.group(1), m.group(3)
        if name.endswith("_created") or "gpu_" in name or val == "NaN":
            continue
        if "prefix_cache_queries" in name:
            out["queries"] += float(val)
            found = True
        elif "prefix_cache_hits" in name:
            out["hits"] += float(val)
            found = True
    return out if found else {}
