"""Turn a results directory into markdown tables + PNG charts, and (optionally)
splice them into README.md between the RESULTS markers.

Every number written here is read from the JSON produced by `bench run`;
nothing is typed in by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
COLORS = {"cache_on": "#2a78d6", "cache_off": "#eb6834"}   # fixed order: on=slot1, off=slot2
LABELS = {"cache_on": "prefix caching ON", "cache_off": "prefix caching OFF"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _load(results: Path) -> dict:
    data = {}
    for mode in ("cache_on", "cache_off"):
        p = results / f"bench_{mode}.json"
        if p.exists():
            data[mode] = json.loads(p.read_text())
    return data


def _f(x, nd=1):
    return "–" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}")


def _style(ax, title, ylabel, xlabel="concurrent requests"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def _line_chart(data, key, metric, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150, facecolor=SURFACE)
    plotted = False
    for mode, d in data.items():
        rows = d.get(key) or []
        pts = [(r["concurrency"], r[metric]) for r in rows if r.get(metric) is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[mode], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[mode])
        plotted = True   # exact values live in the table right above the chart
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(bottom=0)
    _style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def _prefix_bars(data, path):
    modes = [m for m in ("cache_on", "cache_off") if data.get(m, {}).get("prefix")]
    if not modes:
        return False
    cs = [r["concurrency"] for r in data[modes[0]]["prefix"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150, facecolor=SURFACE)
    w = 0.38
    for i, mode in enumerate(modes):
        vals = [r["ttft_ms_p50"] or 0 for r in data[mode]["prefix"]]
        xs = [j + (i - (len(modes) - 1) / 2) * (w + 0.02) for j in range(len(cs))]
        bars = ax.bar(xs, vals, width=w, color=COLORS[mode], label=LABELS[mode])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([str(c) for c in cs])
    _style(ax, "Shared-prefix workload: median TTFT", "TTFT p50 (ms)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def build_markdown(results: Path, img_prefix: str) -> str:
    data = _load(results)
    cfg = json.loads((results / "config.json").read_text())
    env = json.loads((results / "env.json").read_text()) if (results / "env.json").exists() else {}
    t, p = cfg["throughput"], cfg["prefix"]
    md = []
    md.append(f"**Run:** `{results.as_posix()}` · model `{env.get('model', cfg['server']['model'])}` · "
              f"GPU `{env.get('gpu') or 'n/a'}` · vLLM `{env.get('vllm') or 'n/a'}` · "
              f"torch `{env.get('torch') or 'n/a'}` · {env.get('timestamp', '')}\n")

    # --- throughput sweep
    md.append(f"#### 1. Concurrency sweep — {t['input_len']} input / {t['output_len']} output tokens, "
              f"random (non-shared) prompts\n")
    for mode, d in data.items():
        rows = d.get("throughput")
        if not rows:
            continue
        md.append(f"*{LABELS[mode]}*\n")
        md.append("| concurrency | requests | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (ms) | E2E p95 (ms) "
                  "| TPOT mean (ms) | output tok/s | errors |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(f"| {r['concurrency']} | {r['requests']} | {_f(r['ttft_ms_p50'])} | {_f(r['ttft_ms_p95'])} "
                      f"| {_f(r['e2e_ms_p50'])} | {_f(r['e2e_ms_p95'])} | {_f(r['tpot_ms_mean'], 2)} "
                      f"| {_f(r['output_tok_s'])} | {r['errors']} |")
        md.append("")
    imgs = []
    if _line_chart(data, "throughput", "output_tok_s", "Output throughput vs concurrency",
                   "output tokens / s", results / "throughput.png"):
        imgs.append("throughput.png")
    if _line_chart(data, "throughput", "ttft_ms_p95", "TTFT p95 vs concurrency", "TTFT p95 (ms)",
                   results / "ttft_p95.png"):
        imgs.append("ttft_p95.png")

    # --- prefix A/B
    on = (data.get("cache_on") or {}).get("prefix") or []
    off = (data.get("cache_off") or {}).get("prefix") or []
    if on and off:
        md.append(f"#### 2. Prefix caching A/B — {p['num_groups']} shared prefixes × {p['shared_prefix_len']} "
                  f"tokens + {p['suffix_len']}-token unique suffix, {p['output_len']} output tokens, "
                  f"{p['num_requests']} requests per level\n")
        md.append("| concurrency | TTFT p50 on / off (ms) | TTFT speedup | TTFT p95 on / off (ms) "
                  "| E2E p95 on / off (ms) | output tok/s on / off | cache hit rate (on) |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        for a, b in zip(on, off):
            sp = (b["ttft_ms_p50"] / a["ttft_ms_p50"]) if a["ttft_ms_p50"] and b["ttft_ms_p50"] else None
            hr = a.get("cache_hit_rate")
            md.append(f"| {a['concurrency']} | {_f(a['ttft_ms_p50'])} / {_f(b['ttft_ms_p50'])} "
                      f"| {'–' if sp is None else f'{sp:.2f}×'} "
                      f"| {_f(a['ttft_ms_p95'])} / {_f(b['ttft_ms_p95'])} "
                      f"| {_f(a['e2e_ms_p95'])} / {_f(b['e2e_ms_p95'])} "
                      f"| {_f(a['output_tok_s'])} / {_f(b['output_tok_s'])} "
                      f"| {'–' if hr is None else f'{hr:.1%}'} |")
        md.append("")
        if _prefix_bars(data, results / "prefix_ttft.png"):
            imgs.append("prefix_ttft.png")

    for img in imgs:
        md.append(f"![{img}]({img_prefix}{img})")
    md.append("")

    junit = results / "correctness_junit.txt"
    if junit.exists():
        md.append("#### 3. Correctness checks\n")
        md.append("```\n" + junit.read_text().strip() + "\n```")
    return "\n".join(md)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{readme} has no {START} / {END} markers")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}",
                 text, flags=re.S)
    readme.write_text(new)


def main(results: Path, readme: Path | None) -> None:
    img_prefix = (results.as_posix().rstrip("/") + "/") if readme else ""
    block = build_markdown(results, img_prefix)
    (results / "summary.md").write_text(build_markdown(results, ""))
    print(block)
    if readme:
        update_readme(readme, block)
        print(f"\nREADME updated: {readme}")
