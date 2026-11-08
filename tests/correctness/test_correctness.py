"""Correctness checks over the evidence recorded by `bench run`.

Agreement rule (see bench/compare.py): two greedy outputs agree if they are
identical, or first diverge only at a near-tie token (top-1 vs top-2 logprob gap
< thresholds.near_tie_nats). A confident divergence is treated as a bug.
"""
import json

import pytest

from bench.compare import compare

MODES = ["cache_on", "cache_off"]


def _need(runs, mode):
    if mode not in runs:
        pytest.skip(f"{mode} not run")
    return runs[mode]


def _all_ok(records):
    bad = [r for r in records if r.get("status") != 200]
    assert not bad, f"{len(bad)} requests failed, e.g. {bad[0].get('error')}"


# ---------------------------------------------------------------- fixed prompts
@pytest.mark.parametrize("mode", MODES)
def test_fixed_prompts_repeat_identically(runs, mode):
    """Same prompt, sent sequentially k times with temperature=0 -> identical output.
    Sequential requests see the same batch shape, so we require *exact* equality."""
    data = _need(runs, mode)["fixed"]["sequential"]
    for item in data:
        _all_ok(item["runs"])
        texts = {r["text"] for r in item["runs"]}
        assert len(texts) == 1, f"non-deterministic output for {item['prompt']!r}: {texts}"


@pytest.mark.parametrize("mode", MODES)
def test_batched_matches_sequential(runs, mode, tol):
    """Same prompts sent concurrently (batched together) vs one at a time."""
    data = _need(runs, mode)["fixed"]
    _all_ok(data["batched"])
    cmp = [compare(s["runs"][0], b, tol["near_tie_nats"]) for s, b in zip(data["sequential"], data["batched"])]
    bad = [(s["prompt"], c) for s, c in zip(data["sequential"], cmp) if not c["agree"]]
    assert not bad, f"confident divergence under batching: {bad}"


def test_fixed_prompts_cache_on_vs_off(runs, tol):
    """Prefix caching must not change what the model says."""
    on, off = _need(runs, "cache_on"), _need(runs, "cache_off")
    pairs = zip(on["fixed"]["sequential"], off["fixed"]["sequential"])
    cmp = [(a["prompt"], compare(b["runs"][0], a["runs"][0], tol["near_tie_nats"])) for a, b in pairs]
    bad = [(p, c) for p, c in cmp if not c["agree"]]
    exact = sum(c["exact"] for _, c in cmp) / len(cmp)
    assert not bad, f"cache on/off disagree: {bad}"
    assert exact >= tol["min_exact_match_rate"], f"exact-match rate {exact:.2f} too low"


# ---------------------------------------------------------------- shared prefix
def _steps(run):
    return {s["step"]: s for s in run["shared_prefix"]["steps"]}


@pytest.mark.parametrize("mode", MODES)
def test_shared_prefix_requests_succeed(runs, mode):
    _all_ok([s["result"] for s in _need(runs, mode)["shared_prefix"]["steps"]])


def test_warm_hit_matches_cold_miss(runs, tol):
    """Within cache_on: the 2nd identical request (served from cached KV blocks)
    must produce the same output as the 1st (full prefill)."""
    st = _steps(_need(runs, "cache_on"))
    c = compare(st["cold"]["result"], st["warm"]["result"], tol["near_tie_nats"])
    assert c["agree"], f"warm (cache hit) diverged from cold: {c}"


def test_shared_prefix_cache_on_vs_off(runs, tol):
    """Every shared-prefix request: cache_on output == cache_off output."""
    on, off = _steps(_need(runs, "cache_on")), _steps(_need(runs, "cache_off"))
    bad = {k: c for k in on if not (c := compare(off[k]["result"], on[k]["result"], tol["near_tie_nats"]))["agree"]}
    assert not bad, f"cache on/off disagree on shared-prefix requests: {bad}"


def test_prefix_cache_is_actually_hit(runs):
    """The A/B is only meaningful if the cache really is used when on ..."""
    st = _steps(_need(runs, "cache_on"))
    d = st["warm"]["cache_delta"]
    if not d:
        pytest.skip("server exposes no prefix-cache metrics")
    assert d["hits"] > 0, f"warm request produced no prefix-cache hits: {d}"
    assert st["partial"]["cache_delta"]["hits"] > 0, "same-document request did not reuse the prefix"


def test_prefix_cache_is_off_when_disabled(runs):
    """... and really bypassed when off."""
    st = _steps(_need(runs, "cache_off"))
    d = st["warm"]["cache_delta"]
    if not d:
        pytest.skip("server exposes no prefix-cache metrics (expected when caching is disabled)")
    assert d["hits"] == 0, f"cache_off server still reports prefix hits: {d}"


# ---------------------------------------------------------------- long context
@pytest.mark.parametrize("mode", MODES)
def test_long_context_requests_accepted(runs, mode):
    """Every long prompt is accepted and actually reaches its target length
    (catches silent truncation)."""
    cases = _need(runs, mode)["long_context"]["cases"]
    _all_ok([c["result"] for c in cases])
    for c in cases:
        got = c["result"]["usage"]["prompt_tokens"]
        assert got >= 0.85 * c["target_tokens"], f"prompt only {got} tokens, target {c['target_tokens']}"


@pytest.mark.parametrize("mode", MODES)
def test_needle_retrieval(runs, mode, tol):
    cases = _need(runs, mode)["long_context"]["cases"]
    acc = sum(c["found"] for c in cases) / len(cases)
    missed = [(c["target_tokens"], c["depth"]) for c in cases if not c["found"]]
    assert acc >= tol["needle_min_accuracy"], f"needle accuracy {acc:.2f}; missed (len, depth): {missed}"


def test_needle_results_same_with_and_without_cache(runs):
    on, off = _need(runs, "cache_on")["long_context"]["cases"], _need(runs, "cache_off")["long_context"]["cases"]
    diff = [(a["target_tokens"], a["depth"]) for a, b in zip(on, off) if a["found"] != b["found"]]
    assert not diff, f"needle found/missed differs between cache on and off at {diff}"


@pytest.mark.parametrize("mode", MODES)
def test_overlong_prompt_rejected_cleanly(runs, mode):
    lc = _need(runs, mode)["long_context"]
    assert lc["overlong"]["status"] == 400, f"expected HTTP 400 for over-length prompt, got {lc['overlong']}"
    assert lc["healthy_after_overlong"], "server unhealthy after an over-length request"


# ---------------------------------------------------------------- API contract
@pytest.mark.parametrize("mode", MODES)
def test_streaming_contract(runs, mode):
    s = _need(runs, mode)["stream_contract"]
    assert s["ok"], s["error"]
    assert s["malformed_chunks"] == 0
    assert s["prompt_tokens"] == s["sent_prompt_tokens"], "usage.prompt_tokens != tokens sent"
    assert s["output_tokens"] == 32, "ignore_eos + max_tokens=32 must yield exactly 32 tokens"
    assert s["finish_reason"] == "length"


# ---------------------------------------------------------------- regression vs golden
def test_matches_golden(runs, tol, golden_path, request):
    """Regression check: outputs for the fixed prompts vs a committed snapshot.
    Catches behaviour changes after upgrading vLLM / changing flags."""
    off = _need(runs, "cache_off")["fixed"]["sequential"]
    current = [{"prompt": x["prompt"], **{k: x["runs"][0][k] for k in ("text", "tokens", "top")}} for x in off]
    if request.config.getoption("--update-golden"):
        golden_path.parent.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(current, indent=1))
        pytest.skip(f"golden written to {golden_path}")
    if not golden_path.exists():
        pytest.skip(f"no golden file {golden_path.name}; create it with --update-golden")
    golden = json.loads(golden_path.read_text())
    bad = [(g["prompt"], c) for g, cur in zip(golden, current)
           if not (c := compare(g, cur, tol["near_tie_nats"]))["agree"]]
    assert not bad, f"outputs regressed vs golden: {bad}"
