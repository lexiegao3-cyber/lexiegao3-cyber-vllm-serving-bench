"""Workload generation.

Prompts for the performance benchmarks are sent as *token-id lists* (the
`/v1/completions` endpoint accepts `prompt: list[int]`), so every request has an
exact, known input length -- no re-tokenization drift. Token ids are sampled as
contiguous windows from a tokenized, deterministic filler corpus, so the
prompts look like natural text to the model rather than random vocab ids.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

_WORDS = (
    "the of and to in is was for on that with as by at from his her it an were are "
    "which this be or has had not but have its they after first new two their one "
    "system data model server request memory cache token latency throughput batch "
    "river mountain city music history science market energy policy network design "
    "student teacher engine signal value process result method research report"
).split()


def filler_text(n_words: int, seed: int = 0) -> str:
    """Deterministic pseudo-English filler text (same seed -> same text)."""
    rng = random.Random(seed)
    sentences, count = [], 0
    while count < n_words:
        k = rng.randint(8, 18)
        words = [rng.choice(_WORDS) for _ in range(k)]
        words[0] = words[0].capitalize()
        sentences.append(" ".join(words) + ".")
        count += k
    return " ".join(sentences)


@dataclass
class PrefixRequest:
    group: int            # which shared prefix this request uses
    prompt: list[int]     # prefix + unique suffix (token ids)


class TokenPool:
    """A long sequence of token ids from which prompt windows are sliced."""

    def __init__(self, ids: list[int]):
        if len(ids) < 64:
            raise ValueError("token pool too small")
        self.ids = ids

    def window(self, length: int, rng: random.Random) -> list[int]:
        if length > len(self.ids):
            reps = length // len(self.ids) + 1
            ids = self.ids * reps
        else:
            ids = self.ids
        start = rng.randrange(0, len(ids) - length + 1)
        return ids[start:start + length]


def random_prompts(pool: TokenPool, n: int, input_len: int, seed: int) -> list[list[int]]:
    """n independent prompts of exactly `input_len` tokens (no intended sharing)."""
    rng = random.Random(seed)
    return [pool.window(input_len, rng) for _ in range(n)]


def shared_prefix_prompts(
    pool: TokenPool,
    n: int,
    num_groups: int,
    prefix_len: int,
    suffix_len: int,
    seed: int,
) -> list[PrefixRequest]:
    """n prompts = one of `num_groups` shared prefixes + a unique suffix.

    Models the common production pattern of a long shared system prompt /
    document followed by a short per-user question. Requests are shuffled so
    groups interleave, like real traffic.
    """
    rng = random.Random(seed)
    prefixes = [pool.window(prefix_len, rng) for _ in range(num_groups)]
    reqs = [
        PrefixRequest(group=i % num_groups, prompt=prefixes[i % num_groups] + pool.window(suffix_len, rng))
        for i in range(n)
    ]
    rng.shuffle(reqs)
    return reqs
