"""Collect the raw evidence for the correctness checks against a live server.

This module only *records* (prompts, outputs, logprobs, status codes, cache
counters) into JSON. The pass/fail logic lives in `tests/correctness/` so the
checks can be re-run, tightened or extended without touching the GPU again.
"""
from __future__ import annotations

import asyncio
import random

import aiohttp

from . import client
from .workloads import filler_text

FIXED_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List the first ten prime numbers, separated by commas.",
    "Explain what a KV cache is in a transformer, in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Translate to French: 'The server is running out of memory.'",
    "What is 17 * 23? Show the calculation.",
    "Give three tips for writing reliable unit tests.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
]

SHARED_DOC_QUESTIONS = [
    "How many sentences in the document above contain the word 'cache'? Give a rough estimate.",
    "What is the first word of the document above?",
    "Write a one-sentence summary of the document above.",
]


def _msgs(user: str, system: str | None = None) -> list[dict]:
    m = [{"role": "system", "content": system}] if system else []
    return m + [{"role": "user", "content": user}]


async def _fixed(session, url, model, cfg) -> dict:
    """Same prompts: repeated sequentially, then all at once (batched)."""
    max_t, k = cfg["max_tokens"], cfg["repeats"]
    seq = []
    for p in FIXED_PROMPTS:
        runs = [await client.chat(session, url, model, _msgs(p), max_t) for _ in range(k)]
        seq.append({"prompt": p, "runs": runs})
    batched = await asyncio.gather(*[client.chat(session, url, model, _msgs(p), max_t)
                                     for p in FIXED_PROMPTS])
    return {"sequential": seq, "batched": list(batched)}


async def _shared_prefix(session, url, model, cfg) -> dict:
    """Cold miss -> exact warm hit -> partial hit (same doc, new question).

    Prefix-cache counters are scraped around each request so the tests can
    prove the cache was actually exercised (cache_on) or bypassed (cache_off).
    """
    doc = filler_text(cfg["shared_doc_words"], seed=1234)
    system = "You are a careful assistant. Answer questions about the document.\n\nDOCUMENT:\n" + doc
    steps = [("cold", SHARED_DOC_QUESTIONS[0]), ("warm", SHARED_DOC_QUESTIONS[0]),
             ("partial", SHARED_DOC_QUESTIONS[1]), ("partial2", SHARED_DOC_QUESTIONS[2])]
    out = []
    for name, q in steps:
        before = await client.prefix_cache_counters(session, url)
        rec = await client.chat(session, url, model, _msgs(q, system), cfg["max_tokens"])
        after = await client.prefix_cache_counters(session, url)
        delta = {k: after[k] - before.get(k, 0) for k in after} if after else {}
        out.append({"step": name, "question": q, "result": rec, "cache_delta": delta})
    return {"steps": out}


async def _long_context(session, url, model, cfg, max_model_len) -> dict:
    """Needle-in-a-haystack at several context lengths and depths."""
    sample = filler_text(2000, seed=7)
    tok_per_word = len(await client.tokenize(session, url, model, sample)) / len(sample.split())
    lengths = sorted({int(x * max_model_len) if x <= 1 else int(x) for x in cfg["context_lengths"]})
    rng = random.Random(42)
    cases = []
    for L in lengths:
        budget = L - 64                                   # room for chat template + question
        words = filler_text(int(budget / tok_per_word) + 50, seed=L).split()
        # one calibration pass with the real tokenizer so the haystack hits its token budget
        n_tok = len(await client.tokenize(session, url, model, " ".join(words)))
        words = words[: int(len(words) * budget / n_tok)]
        for depth in cfg["depths"]:
            code = str(rng.randint(10000, 99999))
            pos = int(len(words) * depth)
            hay = " ".join(words[:pos] + [f"The secret code is {code}."] + words[pos:])
            q = hay + "\n\nWhat is the secret code mentioned in the text above? Reply with only the number."
            rec = await client.chat(session, url, model, _msgs(q), 16)
            cases.append({"target_tokens": L, "depth": depth, "code": code, "result": rec,
                          "found": rec.get("status") == 200 and code in (rec.get("text") or "")})
    # A prompt longer than the context window must be rejected cleanly (HTTP 400),
    # not truncated silently or crash the server.
    ids = await client.tokenize(session, url, model, filler_text(int(max_model_len * 1.3 / tok_per_word) + 50, 99))
    ids = ids[: max_model_len + 16]
    async with session.post(f"{url}/v1/completions",
                            json={"model": model, "prompt": ids, "max_tokens": 8}) as resp:
        overlong = {"prompt_tokens": len(ids), "status": resp.status, "body": (await resp.text())[:300]}
    healthy_after = False
    try:
        async with session.get(f"{url}/health") as r:
            healthy_after = r.status == 200
    except Exception:
        pass
    return {"tok_per_word": tok_per_word, "cases": cases, "overlong": overlong,
            "healthy_after_overlong": healthy_after}


async def _stream_contract(session, url, model) -> dict:
    """Streaming API shape: usage block, exact max_tokens, finish_reason, no bad chunks."""
    ids = await client.tokenize(session, url, model, filler_text(200, seed=3))
    ids = ids[:128]
    r = await client.stream_completion(session, url, model, ids, max_tokens=32, ignore_eos=True)
    return {"sent_prompt_tokens": len(ids), "ok": r.ok, "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens, "finish_reason": r.finish_reason,
            "n_chunks": r.n_chunks, "malformed_chunks": r.malformed_chunks, "error": r.error}


async def collect(url: str, model: str, cfg: dict, max_model_len: int) -> dict:
    async with client.new_session() as session:
        return {
            "fixed": await _fixed(session, url, model, cfg),
            "shared_prefix": await _shared_prefix(session, url, model, cfg),
            "long_context": await _long_context(session, url, model, cfg, max_model_len),
            "stream_contract": await _stream_contract(session, url, model),
        }
