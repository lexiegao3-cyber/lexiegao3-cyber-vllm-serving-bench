"""Fast tests for the pure-Python pieces (no server, no GPU)."""
import math

from bench.client import RequestResult
from bench.compare import compare, first_divergence, top_gap
from bench.metrics import pct, summarize
from bench.server import build_cmd
from bench.workloads import TokenPool, filler_text, random_prompts, shared_prefix_prompts


def test_filler_text_deterministic():
    assert filler_text(100, seed=1) == filler_text(100, seed=1)
    assert filler_text(100, seed=1) != filler_text(100, seed=2)
    assert len(filler_text(100).split()) >= 100


def test_random_prompts_exact_length_and_reproducible():
    pool = TokenPool(list(range(5000)))
    a = random_prompts(pool, 10, 256, seed=3)
    assert all(len(p) == 256 for p in a)
    assert a == random_prompts(pool, 10, 256, seed=3)


def test_pool_shorter_than_prompt_still_works():
    pool = TokenPool(list(range(100)))
    assert len(random_prompts(pool, 1, 1000, seed=0)[0]) == 1000


def test_shared_prefix_structure():
    pool = TokenPool(list(range(20000)))
    reqs = shared_prefix_prompts(pool, 40, num_groups=4, prefix_len=512, suffix_len=32, seed=0)
    assert len(reqs) == 40 and all(len(r.prompt) == 544 for r in reqs)
    by_group = {}
    for r in reqs:
        by_group.setdefault(r.group, set()).add(tuple(r.prompt[:512]))
    assert len(by_group) == 4 and all(len(v) == 1 for v in by_group.values())   # one prefix per group
    assert len({tuple(r.prompt[512:]) for r in reqs}) == 40                      # unique suffixes


def test_pct():
    assert pct([], 50) is None
    assert pct([1, 2, 3, 4], 50) == 2.5
    assert math.isclose(pct(range(101), 95), 95.0)


def test_summarize():
    rs = [RequestResult(ok=True, ttft=0.1 * i, e2e=1.0 + 0.1 * i, itl=[0.01] * 9,
                        prompt_tokens=100, output_tokens=10) for i in range(1, 11)]
    rs.append(RequestResult(ok=False, error="boom"))
    s = summarize(rs, wall_s=2.0)
    assert s["requests"] == 11 and s["ok"] == 10 and s["errors"] == 1
    assert s["output_tok_s"] == 50.0
    assert s["output_len_exact"] is True
    assert s["ttft_ms_p50"] == 550.0
    assert s["error_samples"] == ["boom"]


def test_first_divergence():
    assert first_divergence(["a", "b"], ["a", "b"]) is None
    assert first_divergence(["a", "b"], ["a", "c"]) == 1
    assert first_divergence(["a"], ["a", "b"]) == 1


def _rec(tokens, gaps):
    return {"tokens": tokens, "top": [[(t, -0.1), ("x", -0.1 - g)] for t, g in zip(tokens, gaps)]}


def test_compare_near_tie_is_tolerated():
    ref = _rec(["a", "b", "c"], [3.0, 0.05, 3.0])
    other = _rec(["a", "z", "q"], [3.0, 0.05, 3.0])
    c = compare(ref, other, near_tie_nats=0.5)
    assert c["exact"] is False and c["agree"] is True and c["div_idx"] == 1


def test_compare_confident_divergence_is_a_bug():
    ref = _rec(["a", "b"], [3.0, 4.0])
    other = _rec(["a", "z"], [3.0, 4.0])
    assert compare(ref, other, 0.5)["agree"] is False


def test_top_gap_missing():
    assert math.isinf(top_gap([], 0))


def test_build_cmd_sets_flag_explicitly_in_both_modes():
    t = "vllm serve {model} --port {port} {prefix_caching_flag}"
    assert build_cmd(t, "cache_on", model="m", port=1)[-1] == "--enable-prefix-caching"
    assert build_cmd(t, "cache_off", model="m", port=1)[-1] == "--no-enable-prefix-caching"
