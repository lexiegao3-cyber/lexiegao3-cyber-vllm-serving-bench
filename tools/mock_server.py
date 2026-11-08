"""A tiny *mock* OpenAI-compatible server that imitates the parts of `vllm serve`
this repo talks to. It exists ONLY so CI (no GPU) can exercise the whole
pipeline: server lifecycle, streaming parser, load generator, prefix-cache
metrics scrape, correctness collection, pytest checks and report generation.

Numbers produced against this server are synthetic and meaningless as
performance results. Real results come from a real vLLM run (see README).

Simulated behaviour:
  * whitespace "tokenizer" with stable ids (/tokenize, /detokenize)
  * prefill cost proportional to *uncached* prompt tokens, serialized across
    requests (one engine); decode step time grows with in-flight requests
  * block-level (16-token) prefix cache with hit/query counters on /metrics
  * deterministic greedy outputs (a function of the prompt only), so cache
    on/off and batching never change outputs -- as they shouldn't
  * HTTP 400 for prompts longer than --max-model-len
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
import zlib

from aiohttp import web

BLOCK = 16
WORDS = "alpha beta gamma delta server cache token batch model kernel memory queue river stone".split()


class Mock:
    def __init__(self, a):
        self.a = a
        self.vocab: dict[int, str] = {}
        self.blocks: set[int] = set()
        self.queries = self.hits = 0
        self.active = 0
        self.prefill_lock = asyncio.Lock()

    # ---------------------------------------------------------------- tokenizer
    def tok(self, text: str) -> list[int]:
        ids = []
        for w in text.split():
            i = zlib.crc32(w.encode()) % 150_000
            self.vocab[i] = w
            ids.append(i)
        return ids

    def detok(self, ids: list[int]) -> str:
        return " ".join(self.vocab.get(i, f"w{i}") for i in ids)

    # ---------------------------------------------------------------- engine
    async def prefill(self, ids: list[int]) -> int:
        """Simulate prefill; returns the number of prompt tokens served from cache."""
        n_full = len(ids) // BLOCK
        h, hit_blocks, hashes = 0, 0, []
        for b in range(n_full):
            h = zlib.crc32(str(ids[b * BLOCK:(b + 1) * BLOCK]).encode(), h)
            hashes.append(h)
        if self.a.prefix_caching:
            self.queries += len(ids)
            for hh in hashes:
                if hh in self.blocks:
                    hit_blocks += 1
                else:
                    break
            self.hits += hit_blocks * BLOCK
            self.blocks.update(hashes)
        uncached = len(ids) - hit_blocks * BLOCK
        async with self.prefill_lock:
            await asyncio.sleep(self.a.base_ms / 1000 + uncached * self.a.prefill_us / 1e6)
        return hit_blocks * BLOCK

    def step_s(self) -> float:
        return self.a.itl_ms / 1000 * (1 + 0.03 * self.active)

    def generate(self, ids: list[int], text: str, n: int, cached: int = 0) -> list[tuple[str, float]]:
        """Deterministic 'greedy' tokens + top1-top2 gap, function of the prompt only.

        With --inject-cache-bug, a request that hits the cache "sees" only the
        cached prefix (as if stale KV blocks from another request were reused), so
        its output changes -- the correctness suite must catch this.
        """
        seen = ids[:cached] if (self.a.inject_cache_bug and cached) else ids
        rng = random.Random(zlib.crc32(str(seen).encode()))
        out = []
        m = re.search(r"secret code is (\d+)", text)
        if m:
            out.append((m.group(1), 5.0))
        while len(out) < n:
            out.append((rng.choice(WORDS), rng.uniform(0.6, 6.0)))
        return out[:n]

    def too_long(self, n: int):
        if n > self.a.max_model_len:
            return web.json_response({"object": "error", "type": "BadRequestError", "code": 400,
                                      "message": f"This model's maximum context length is {self.a.max_model_len} "
                                                 f"tokens. However, you requested {n} tokens."}, status=400)
        return None


def make_app(a) -> web.Application:
    m = Mock(a)

    async def health(_):
        return web.Response(text="")

    async def models(_):
        return web.json_response({"object": "list", "data": [{"id": a.model, "object": "model"}]})

    async def tokenize(req):
        b = await req.json()
        ids = m.tok(b["prompt"])
        return web.json_response({"tokens": ids, "count": len(ids), "max_model_len": a.max_model_len})

    async def detokenize(req):
        return web.json_response({"prompt": m.detok((await req.json())["tokens"])})

    async def metrics(_):
        lines = []
        if a.prefix_caching:
            lines += [f'vllm:prefix_cache_queries_total{{engine="0",model_name="{a.model}"}} {float(m.queries)}',
                      f'vllm:prefix_cache_hits_total{{engine="0",model_name="{a.model}"}} {float(m.hits)}']
        lines.append(f'vllm:num_requests_running{{engine="0"}} {float(m.active)}')
        return web.Response(text="\n".join(lines) + "\n")

    async def completions(req):
        b = await req.json()
        p = b["prompt"]
        ids = p if isinstance(p, list) else m.tok(p)
        text = m.detok(ids) if isinstance(p, list) else p
        if (err := m.too_long(len(ids))):
            return err
        n = b.get("max_tokens", 16)
        m.active += 1
        try:
            cached = await m.prefill(ids)
            toks = m.generate(ids, text, n, cached)
            if not b.get("ignore_eos"):
                toks = toks[: max(1, len(toks) * 2 // 3)]
            finish = "length" if len(toks) == n else "stop"
            if not b.get("stream"):
                for _ in toks:
                    await asyncio.sleep(m.step_s())
                return web.json_response({"choices": [{"index": 0, "text": " ".join(t for t, _ in toks),
                                                       "finish_reason": finish}],
                                          "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks)}})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(req)
            for i, (t, _) in enumerate(toks):
                if i:
                    await asyncio.sleep(m.step_s())
                chunk = {"id": "cmpl-mock", "object": "text_completion", "created": int(time.time()),
                         "model": a.model,
                         "choices": [{"index": 0, "text": (" " if i else "") + t,
                                      "finish_reason": finish if i == len(toks) - 1 else None}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            usage = {"id": "cmpl-mock", "choices": [],
                     "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                               "total_tokens": len(ids) + len(toks)}}
            await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
            await resp.write_eof()
            return resp
        finally:
            m.active -= 1

    async def chat(req):
        b = await req.json()
        text = "".join(f"<|{x['role']}|> {x['content']} " for x in b["messages"]) + "<|assistant|>"
        ids = m.tok(text)
        if (err := m.too_long(len(ids))):
            return err
        n = b.get("max_tokens", 16)
        m.active += 1
        try:
            cached = await m.prefill(ids)
            toks = m.generate(ids, text, n, cached)
            for _ in toks:
                await asyncio.sleep(m.step_s())
        finally:
            m.active -= 1
        content = [{"token": t if i == 0 else " " + t, "logprob": -0.05,
                    "top_logprobs": [{"token": t if i == 0 else " " + t, "logprob": -0.05},
                                     {"token": " zzz", "logprob": -0.05 - g}]}
                   for i, (t, g) in enumerate(toks)]
        return web.json_response({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": a.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(c["token"] for c in content)},
                         "logprobs": {"content": content}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks)}})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.get("/health", health), web.get("/v1/models", models), web.get("/metrics", metrics),
                    web.post("/tokenize", tokenize), web.post("/detokenize", detokenize),
                    web.post("/v1/completions", completions), web.post("/v1/chat/completions", chat)])
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="mock-model")
    p.add_argument("--max-model-len", type=int, default=8192)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enable-prefix-caching", dest="prefix_caching", action="store_true", default=True)
    g.add_argument("--no-enable-prefix-caching", dest="prefix_caching", action="store_false")
    p.add_argument("--inject-cache-bug", action="store_true",
                   help="fault injection: corrupt outputs of requests that hit the prefix cache")
    p.add_argument("--base-ms", type=float, default=3.0, help="fixed per-request prefill overhead")
    p.add_argument("--prefill-us", type=float, default=15.0, help="prefill cost per uncached token (µs)")
    p.add_argument("--itl-ms", type=float, default=2.0, help="decode step time at batch size 0")
    a = p.parse_args()
    web.run_app(make_app(a), host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
