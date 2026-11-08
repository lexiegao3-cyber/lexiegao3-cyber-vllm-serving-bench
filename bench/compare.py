"""Comparing two greedy generations of the same prompt.

Greedy decoding is *not* guaranteed to be bit-identical across different batch
compositions or kernel paths (e.g. prefix-cache hit vs. full prefill): floating-
point reductions are not associative, so logits can differ in the last bits.
That only changes the argmax when the top-2 candidates are nearly tied.

So the rule used throughout the tests is:
    two outputs agree  <=>  identical, OR they first diverge at a position where
    the reference's top-1 vs top-2 logprob gap is below `near_tie_nats`.
A divergence at a confident position (large gap) is a real bug -- e.g. a cache
returning KV blocks for the wrong prefix.
"""
from __future__ import annotations

import math


def first_divergence(a: list[str], b: list[str]) -> int | None:
    """Index of the first differing token, or None if the sequences are identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def top_gap(top: list[list], i: int) -> float:
    """top-1 minus top-2 logprob at position i (inf if unavailable)."""
    if i >= len(top) or len(top[i]) < 2:
        return math.inf
    lps = sorted((lp for _, lp in top[i]), reverse=True)
    return lps[0] - lps[1]


def compare(ref: dict, other: dict, near_tie_nats: float) -> dict:
    """Compare two chat records produced by `client.chat`."""
    idx = first_divergence(ref["tokens"], other["tokens"])
    if idx is None:
        return {"exact": True, "agree": True, "div_idx": None, "gap": None}
    gap = min(top_gap(ref.get("top", []), idx), top_gap(other.get("top", []), idx))
    return {
        "exact": False,
        "agree": gap <= near_tie_nats,
        "div_idx": idx,
        "gap": None if math.isinf(gap) else round(gap, 4),
        "ref_tail": "".join(ref["tokens"][idx:idx + 8]),
        "other_tail": "".join(other["tokens"][idx:idx + 8]),
    }
