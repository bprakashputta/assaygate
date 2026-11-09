"""Self-contained HTML reports: run, comparison (gate) and judge calibration.

One file, no build step, no JS framework: charts are HTML/CSS (forest plots, bars, heatmaps)
plus one SVG ECDF per distribution, so text stays readable at phone width. A ~30-line script
adds hover tooltips; every chart also has a table view. Fonts load from Google Fonts with
system fallbacks, so the report still renders offline.

Colours: baseline = blue, candidate = signal orange (validated with the dataviz palette
checker on both paper surfaces); pass/fail always carries a text label, never colour alone.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from . import stats

if TYPE_CHECKING:
    from .calibrate import CalibrationResult
    from .gate import GateResult
    from .runner import Run

CSS = """
:root{--paper:#f5f2eb;--paper-2:#ece7dc;--card:#fbf9f4;--ink:#17160f;--ink-2:#4a463c;--ink-3:#6f695d;
--rule:#d9d2c3;--grid:#e4ddcf;--signal:#c8431a;--signal-ink:#a5360f;--signal-wash:#f4dccf;
--base:#3b5bdb;--cand:#c8431a;--seq:59,91,219;color-scheme:light;
--serif:"Instrument Serif","Iowan Old Style",Georgia,serif;
--sans:"Geist",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
--mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#12110d;--paper-2:#1b1a15;
--card:#181712;--ink:#ede7da;--ink-2:#bdb6a6;--ink-3:#8b8576;--rule:#2f2d25;--grid:#26241d;
--signal:#ff7a4a;--signal-ink:#ff9a72;--signal-wash:#3a1f14;--base:#6f8cf0;--cand:#e8602f;
--seq:111,140,240;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#12110d;--paper-2:#1b1a15;--card:#181712;--ink:#ede7da;--ink-2:#bdb6a6;
--ink-3:#8b8576;--rule:#2f2d25;--grid:#26241d;--signal:#ff7a4a;--signal-ink:#ff9a72;--signal-wash:#3a1f14;
--base:#6f8cf0;--cand:#e8602f;--seq:111,140,240;color-scheme:dark}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:400 15px/1.55 var(--sans);
-webkit-font-smoothing:antialiased;overflow-x:hidden}
.wrap{max-width:1120px;margin:0 auto;padding:0 clamp(16px,4vw,40px)}
.mono,.label,code,td.num,th.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.label{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}
code{font-size:.86em;background:var(--paper-2);padding:1px 5px;border-radius:3px}
header.top{border-bottom:1px solid var(--rule)}
header.top .wrap{display:flex;justify-content:space-between;align-items:center;height:52px;gap:12px}
.brand{font-family:var(--mono);font-size:13px;color:var(--ink-2)}.brand b{color:var(--ink);font-weight:500}
.hero{padding:40px 0 28px;border-bottom:1px solid var(--rule)}
h1{font-family:var(--serif);font-weight:400;font-size:clamp(34px,5.4vw,60px);line-height:1;
letter-spacing:-.015em;margin:14px 0 18px;max-width:28ch}
h1 em{font-style:italic;color:var(--signal)}
h2{font-family:var(--serif);font-weight:400;font-size:clamp(26px,3vw,34px);line-height:1.05;margin:0 0 6px}
h3{font-size:15px;font-weight:500;margin:0 0 10px}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;color:var(--ink-2);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--rule);border-radius:999px;
padding:4px 12px;font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase}
.pill.fail{border-color:var(--signal);color:var(--signal-ink);background:var(--signal-wash)}
.pill.pass{color:var(--ink)}
.pill i{width:8px;height:8px;border-radius:50%;background:currentColor;display:inline-block}
section{padding:34px 0;border-bottom:1px solid var(--rule)}
.lede{color:var(--ink-2);max-width:72ch;margin:0 0 22px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));border-top:1px solid var(--rule);
border-left:1px solid var(--rule);margin-top:26px}
.tile{padding:16px 18px;border-right:1px solid var(--rule);border-bottom:1px solid var(--rule);min-width:0}
.tile .v{font-size:28px;font-weight:500;letter-spacing:-.01em;margin:6px 0 2px}
.tile .d{font-family:var(--mono);font-size:12.5px;color:var(--ink-2)}
.tile.fail .d b{color:var(--signal-ink)}
.status{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase}
.status.fail{color:var(--signal-ink)}.status.pass{color:var(--ink-3)}
.forest{display:grid;grid-template-columns:minmax(110px,180px) 1fr minmax(120px,170px);gap:0 14px;align-items:center}
.forest .row{display:contents}
.forest .name{font-size:13.5px;padding:10px 0;border-top:1px solid var(--rule);overflow-wrap:anywhere}
.forest .val{font-family:var(--mono);font-size:12px;color:var(--ink-2);text-align:right;padding:10px 0;
border-top:1px solid var(--rule)}
.forest .plot{position:relative;height:40px;border-top:1px solid var(--rule)}
.forest .axis{position:relative;height:22px}
.forest .axis span{position:absolute;top:4px;transform:translateX(-50%);font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.vline{position:absolute;top:0;bottom:0;width:1px;background:var(--ink-3)}
.vline.limit{background:var(--signal);opacity:.55}
.whisker{position:absolute;top:50%;height:2px;margin-top:-1px;border-radius:1px}
.dot{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;
box-shadow:0 0 0 2px var(--paper)}
.ok .whisker,.ok .dot{background:var(--ink)}.bad .whisker,.bad .dot{background:var(--signal)}
.b .whisker,.b .dot{background:var(--base)}.c .whisker,.c .dot{background:var(--cand)}
.hit{position:absolute;inset:0;cursor:default}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink-2);margin:0 0 12px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.key{width:10px;height:10px;border-radius:50%;display:inline-block}.key.b{background:var(--base)}
.key.c{background:var(--cand)}.key.line{width:16px;height:2px;border-radius:1px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:28px}
.ecdf{position:relative}
.ecdf svg{display:block;width:100%;height:190px;overflow:visible}
.ecdf .gl{stroke:var(--grid);stroke-width:1;vector-effect:non-scaling-stroke}
.ecdf .ln{fill:none;stroke-width:2;vector-effect:non-scaling-stroke;stroke-linejoin:round;stroke-linecap:round}
.ecdf .ln.b{stroke:var(--base)}.ecdf .ln.c{stroke:var(--cand)}
.ecdf .xt{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-top:4px}
.ecdf .cross{position:absolute;top:0;height:190px;width:1px;background:var(--ink-3);display:none;pointer-events:none}
.yl{position:absolute;left:0;font-family:var(--mono);font-size:11px;color:var(--ink-3);transform:translateY(-50%)}
.bars{display:grid;grid-template-columns:minmax(110px,180px) 1fr 64px;gap:8px 14px;align-items:center}
.bars .track{height:14px;position:relative}
.bars .fill{position:absolute;left:0;top:0;bottom:0;border-radius:0 4px 4px 0;background:var(--base)}
.bars .fill.em{background:var(--signal)}
.bars .n{font-family:var(--mono);font-size:12px;text-align:right}
.heat{display:grid;gap:2px;font-family:var(--mono);font-size:12px}
.heat .cell{padding:10px 6px;text-align:center;border-radius:3px}
.heat .hd{color:var(--ink-3);font-size:11px;text-align:center;padding:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{font-weight:500;text-align:left;color:var(--ink-3);font-family:var(--mono);font-size:11px;letter-spacing:.05em;
text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--ink)}
td{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
td.num,th.num{text-align:right;white-space:nowrap}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
td .out{max-width:34ch;overflow-wrap:anywhere;white-space:pre-wrap;font-size:12.5px}
td .out.bad{color:var(--signal-ink)}
details{margin-top:18px}summary{cursor:pointer;font-family:var(--mono);font-size:12px;color:var(--ink-2)}
.tip{position:fixed;z-index:9;pointer-events:none;background:var(--ink);color:var(--paper);font:12px/1.4 var(--mono);
padding:6px 9px;border-radius:6px;max-width:300px;display:none;white-space:pre-line}
footer{padding:26px 0 40px;color:var(--ink-3);font-size:12.5px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.filters button{font:12px var(--mono);border:1px solid var(--rule);background:transparent;color:var(--ink-2);
border-radius:999px;padding:5px 12px;cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
@media (max-width:620px){.forest{grid-template-columns:96px 1fr 92px;gap:0 8px}.bars{grid-template-columns:96px 1fr 52px}
.tile .v{font-size:24px}}
"""

JS = """
(function(){var t=document.createElement('div');t.className='tip';document.body.appendChild(t);
function show(e,s){t.textContent=s;t.style.display='block';var x=e.clientX+14,y=e.clientY+14;
var r=t.getBoundingClientRect();if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
t.style.left=x+'px';t.style.top=y+'px'}
document.addEventListener('mousemove',function(e){var el=e.target.closest('[data-tip]');if(el){show(e,el.getAttribute('data-tip'))}
else if(!e.target.closest('.ecdf'))t.style.display='none'});
document.querySelectorAll('.ecdf').forEach(function(box){var d=JSON.parse(box.getAttribute('data-series'));
var svg=box.querySelector('svg'),cr=box.querySelector('.cross');
box.addEventListener('mousemove',function(e){var r=svg.getBoundingClientRect();var f=(e.clientX-r.left)/r.width;
if(f<0||f>1){cr.style.display='none';t.style.display='none';return}
var x=d.min+f*(d.max-d.min);cr.style.display='block';cr.style.left=(e.clientX-box.getBoundingClientRect().left)+'px';
var lines=[d.fmt.replace('{}',x.toFixed(d.dp))];d.series.forEach(function(s){var n=0;for(var i=0;i<s.v.length;i++){if(s.v[i]<=x)n++}
lines.push(s.name+': '+(100*n/s.v.length).toFixed(0)+'% of items at or below')});show(e,lines.join('\\n'))});
box.addEventListener('mouseleave',function(){cr.style.display='none';t.style.display='none'})});
document.querySelectorAll('.filters').forEach(function(f){f.addEventListener('click',function(e){var b=e.target.closest('button');if(!b)return;
f.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});var m=b.getAttribute('data-metric');
document.querySelectorAll(f.getAttribute('data-target')+' tr[data-metric]').forEach(function(r){r.style.display=(m==='*'||r.getAttribute('data-metric')===m)?'':'none'})})});
})();
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600'
    '&family=Geist+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap">'
)


def e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def page(title: str, body: str, subtitle: str = "") -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{e(title)}</title><meta name='description' content='{e(subtitle)}'>"
        f"{FONTS}<style>{CSS}</style></head><body>"
        "<header class='top'><div class='wrap'><span class='brand'><b>assaygate</b>"
        f" / {e(subtitle)}</span><span class='label'>report</span></div></header>"
        f"<main class='wrap'>{body}</main>"
        "<footer class='wrap'>Generated by assaygate. Every number on this page is computed "
        "from the run artifacts it names.</footer>"
        f"<script>{JS}</script></body></html>"
    )


# ----------------------------------------------------------------------------- primitives


def _scale(lo: float, hi: float, pad: float = 0.08) -> tuple[float, float]:
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _pct(v: float, lo: float, hi: float) -> float:
    return 100 * (v - lo) / (hi - lo)


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / max(1, n - 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = np.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-12:
        out.append(round(float(v), 10))
        v += step
    return out


def forest(
    rows: list[dict[str, Any]],
    *,
    ref: float,
    limits: list[float] | None = None,
    fmt: str = "{:+.3f}",
    tick_fmt: str = "{:+.2f}",
    cls_key: str = "cls",
) -> str:
    """Dot-and-whisker rows sharing one x-axis. rows: name, est, lo, hi, cls, tip, val."""
    vals = [r["lo"] for r in rows] + [r["hi"] for r in rows] + [ref] + (limits or [])
    lo, hi = _scale(min(vals), max(vals))
    out = ["<div class='forest' role='img' aria-label='estimates with confidence intervals'>"]
    for r in rows:
        left, right = _pct(r["lo"], lo, hi), _pct(r["hi"], lo, hi)
        lines = f"<div class='vline' style='left:{_pct(ref, lo, hi):.2f}%'></div>"
        for lim in limits or []:
            lines += f"<div class='vline limit' style='left:{_pct(lim, lo, hi):.2f}%'></div>"
        out.append(
            f"<div class='row'><div class='name'>{e(r['name'])}</div>"
            f"<div class='plot {e(r.get(cls_key, 'ok'))}'>{lines}"
            f"<div class='whisker' style='left:{left:.2f}%;width:{max(right - left, 0.4):.2f}%'></div>"
            f"<div class='dot' style='left:{_pct(r['est'], lo, hi):.2f}%'></div>"
            f"<div class='hit' data-tip='{e(r['tip'])}'></div></div>"
            f"<div class='val'>{e(r.get('val', fmt.format(r['est'])))}</div></div>"
        )
    ticks = "".join(
        f"<span style='left:{_pct(t, lo, hi):.2f}%'>{e(tick_fmt.format(t))}</span>"
        for t in _ticks(lo, hi)
    )
    out.append(f"<div></div><div class='axis'>{ticks}</div><div></div></div>")
    return "".join(out)


def ecdf(series: list[tuple[str, str, list[float]]], *, unit_fmt: str, dp: int, title: str) -> str:
    """Empirical CDF lines. series: (name, css class b|c, values)."""
    allv = [v for _, _, vs in series for v in vs]
    if not allv:
        return ""
    lo, hi = float(min(allv)), float(max(allv))
    if hi == lo:
        hi = lo + 1
    paths = []
    for name, cls, vs in series:
        xs = np.sort(np.asarray(vs, dtype=float))
        n = len(xs)
        pts = [f"0,{100:.2f}"] if xs[0] > lo else []
        for i, x in enumerate(xs):
            px = 100 * (x - lo) / (hi - lo)
            pts.append(f"{px:.2f},{100 - 100 * i / n:.2f}")
            pts.append(f"{px:.2f},{100 - 100 * (i + 1) / n:.2f}")
        pts.append(f"100,{100 - 100:.2f}")
        paths.append(
            f"<polyline class='ln {cls}' points='{' '.join(pts)}'><title>{e(name)}</title></polyline>"
        )
    grid = "".join(
        f"<line class='gl' x1='0' x2='100' y1='{y}' y2='{y}'/>" for y in (0, 25, 50, 75, 100)
    )
    ylabels = "".join(
        f"<span class='yl' style='top:{y * 1.9:.0f}px'>{100 - y}%</span>" for y in (0, 50, 100)
    )
    data = {
        "min": lo,
        "max": hi,
        "dp": dp,
        "fmt": unit_fmt,
        "series": [{"name": n, "v": [round(float(v), 6) for v in vs]} for n, _, vs in series],
    }
    xt = "".join(
        f"<span>{e(unit_fmt.replace('{}', f'{t:.{dp}f}'))}</span>" for t in np.linspace(lo, hi, 4)
    )
    return (
        f"<div><h3>{e(title)}</h3><div class='ecdf' data-series='{e(json.dumps(data))}' "
        f"role='img' aria-label='{e(title)} cumulative distribution'>"
        f"<div style='position:relative;padding-left:38px'>{ylabels}"
        f"<svg viewBox='0 0 100 100' preserveAspectRatio='none'>{grid}{''.join(paths)}</svg>"
        f"<div class='xt'>{xt}</div></div><div class='cross'></div></div></div>"
    )


def legend(items: list[tuple[str, str]]) -> str:
    return (
        "<div class='legend'>"
        + "".join(f"<span><i class='key {c}'></i>{e(n)}</span>" for n, c in items)
        + "</div>"
    )


# ----------------------------------------------------------------------------- run / compare


def _fmt_value(kind: str, v: float) -> str:
    if kind == "cost":
        return f"${v:.5f}"
    if kind == "latency":
        return f"{v:,.0f} ms"
    return f"{v:.3f}"


def render_compare_html(result: GateResult, baseline: Run, candidate: Run) -> str:
    fails = [c for c in result.checks if c.status == "fail"]
    verdict = "fail" if not result.passed else "pass"
    headline = (
        f"Gate <em>failed</em>: {len(fails)} of {len(result.checks)} checks"
        if fails
        else f"Gate passed: {len(result.checks)} checks, no regression"
    )
    bm, cm = baseline.manifest, candidate.manifest
    hero = (
        f"<div class='hero'><span class='pill {verdict}'><i></i>{verdict}</span>"
        f"<h1>{headline}</h1><div class='meta'>"
        f"<span>suite <code>{e(cm.suite)}</code></span>"
        f"<span>baseline <code>{e(bm.label or bm.run_id)}</code></span>"
        f"<span>candidate <code>{e(cm.label or cm.run_id)}</code></span>"
        f"<span class='mono'>{result.n_paired} paired items · {int(result.level * 100)}% CIs</span>"
        "</div><div class='tiles'>"
    )
    for ch in result.checks:
        delta = f"{ch.delta:+.3f}" if ch.kind == "quality" else f"x{ch.delta:.2f}"
        ci = (
            f"[{ch.ci_low:+.3f}, {ch.ci_high:+.3f}]"
            if ch.kind == "quality"
            else f"[x{ch.ci_low:.2f}, x{ch.ci_high:.2f}]"
        )
        hero += (
            f"<div class='tile {ch.status}'><div class='label'>{e(ch.name)}</div>"
            f"<div class='v'>{e(_fmt_value(ch.kind, ch.candidate))}</div>"
            f"<div class='d'><b>{delta}</b> {ci}</div>"
            f"<div class='status {ch.status}'>{ch.status} · {e(ch.reason.split(';')[0])}</div></div>"
        )
    hero += "</div></div>"

    q = [c for c in result.checks if c.kind == "quality"]
    qrows = [
        {
            "name": c.name,
            "est": c.delta,
            "lo": c.ci_low,
            "hi": c.ci_high,
            "cls": "bad" if c.status == "fail" else "ok",
            "tip": f"{c.name}\n{c.baseline:.3f} -> {c.candidate:.3f}\ndelta {c.delta:+.3f} "
            f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]\np={c.p_value:.3g}\n{c.status}: {c.rule}",
            "val": f"{c.delta:+.3f}",
        }
        for c in q
    ]
    margins = sorted({-g["max_drop"] for g in (cm.gate.get("quality") or []) if g.get("max_drop")})
    quality = (
        "<section><span class='label'>Quality</span><h2>Per-metric change, candidate − baseline</h2>"
        "<p class='lede'>Dot = mean paired difference; line = 95% bootstrap CI over items. "
        "A metric fails when its whole interval sits left of zero (or of the allowed margin, "
        "shown as a thin orange line). Failing rows are drawn in orange.</p>"
        + (forest(qrows, ref=0.0, limits=margins) if qrows else "<p>No shared metrics.</p>")
        + "</section>"
    )

    ids = sorted({r.id for r in baseline.records} & {r.id for r in candidate.records})
    b_by = {r.id: r for r in baseline.records}
    c_by = {r.id: r for r in candidate.records}
    lat_b = [b_by[i].latency_ms for i in ids]
    lat_c = [c_by[i].latency_ms for i in ids]
    cost_b = [b_by[i].cost_usd * 1000 for i in ids]
    cost_c = [c_by[i].cost_usd * 1000 for i in ids]
    rc = [c for c in result.checks if c.kind != "quality"]
    rrows = [
        {
            "name": "cost / item" if c.kind == "cost" else "p95 latency",
            "est": c.delta,
            "lo": c.ci_low,
            "hi": c.ci_high,
            "cls": "bad" if c.status == "fail" else "ok",
            "tip": f"{c.name}\n{_fmt_value(c.kind, c.baseline)} -> {_fmt_value(c.kind, c.candidate)}"
            f"\nratio x{c.delta:.2f} [x{c.ci_low:.2f}, x{c.ci_high:.2f}]\n{c.status}: {c.rule}",
            "val": f"x{c.delta:.2f}",
        }
        for c in rc
    ]
    lims = []
    for key, sub in (("cost", "max_increase"), ("latency", "p95_max_increase")):
        g = cm.gate.get(key) or {}
        if g.get(sub) is not None:
            lims.append(1 + g[sub])
    ops = (
        "<section><span class='label'>Cost and latency</span><h2>What the candidate costs to run</h2>"
        "<p class='lede'>Ratios candidate ÷ baseline with paired bootstrap CIs (orange lines = "
        "allowed increase). Below: the full per-item distributions as cumulative curves; hover "
        "to read the share of items at or below a value.</p>"
        + (forest(rrows, ref=1.0, limits=lims, tick_fmt="x{:.2f}") if rrows else "")
        + "<div style='height:22px'></div>"
        + legend([("baseline", "b"), ("candidate", "c")])
        + "<div class='grid2'>"
        + ecdf(
            [("baseline", "b", lat_b), ("candidate", "c", lat_c)],
            unit_fmt="{} ms",
            dp=0,
            title="Latency per item",
        )
        + (
            ecdf(
                [("baseline", "b", cost_b), ("candidate", "c", cost_c)],
                unit_fmt="${}/1k",
                dp=2,
                title="System cost per item ($ per 1,000 items)",
            )
            if max(cost_b + cost_c, default=0) > 0
            else ""
        )
        + "</div></section>"
    )

    metrics = sorted({r.metric for r in result.regressions + result.improvements})
    buttons = "<button data-metric='*' aria-pressed='true'>all</button>" + "".join(
        f"<button data-metric='{e(m)}' aria-pressed='false'>{e(m)}</button>" for m in metrics
    )
    rows = "".join(
        f"<tr data-metric='{e(r.metric)}'><td class='mono'>{e(r.id)}</td><td>{e(r.metric)}</td>"
        f"<td class='num'>{r.baseline:.2f} → {r.candidate:.2f}</td>"
        f"<td><div class='out'>{e(r.baseline_output)}</div></td>"
        f"<td><div class='out bad'>{e(r.candidate_output)}</div></td>"
        f"<td><div class='out'>{e(r.expected)}</div></td></tr>"
        for r in result.regressions[:300]
    )
    diff = (
        "<section><span class='label'>Failures diff</span>"
        f"<h2>{len(result.regressions)} item-level regressions, {len(result.improvements)} improvements</h2>"
        "<p class='lede'>Items whose score went down from baseline to candidate. Read these "
        "before trusting any aggregate: they show what actually broke.</p>"
        f"<div class='filters' data-target='#regs'>{buttons}</div>"
        "<div class='scroll'><table id='regs'><thead><tr><th>item</th><th>metric</th>"
        "<th class='num'>score</th><th>baseline output</th><th>candidate output</th>"
        f"<th>expected</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
    )

    table = "".join(
        f"<tr><td>{e(c.name)}</td><td class='num'>{e(_fmt_value(c.kind, c.baseline))}</td>"
        f"<td class='num'>{e(_fmt_value(c.kind, c.candidate))}</td>"
        f"<td class='num'>{c.delta:+.4f}</td><td class='num'>[{c.ci_low:.4f}, {c.ci_high:.4f}]</td>"
        f"<td class='num'>{'' if c.p_value is None else f'{c.p_value:.3g}'}</td>"
        f"<td>{e(c.rule)}</td><td><span class='status {c.status}'>{c.status}</span> {e(c.reason)}</td></tr>"
        for c in result.checks
    )
    data = (
        "<section><span class='label'>Table view</span><h2>All checks</h2><div class='scroll'><table>"
        "<thead><tr><th>check</th><th class='num'>baseline</th><th class='num'>candidate</th>"
        "<th class='num'>delta / ratio</th><th class='num'>95% CI</th><th class='num'>p</th>"
        f"<th>rule</th><th>result</th></tr></thead><tbody>{table}</tbody></table></div>"
        f"<p class='lede' style='margin-top:14px'>Baseline run <code>{e(bm.run_id)}</code> "
        f"({bm.n_items} items, {bm.n_errors} errors, git {e((bm.git_sha or '')[:8])}) · candidate "
        f"<code>{e(cm.run_id)}</code> ({cm.n_items} items, {cm.n_errors} errors, git "
        f"{e((cm.git_sha or '')[:8])}). Dataset sha256 "
        f"{'matches' if bm.dataset_sha256 == cm.dataset_sha256 else 'DIFFERS'}.</p></section>"
    )
    return page(
        f"{cm.suite} gate report",
        hero + quality + ops + diff + data,
        subtitle=f"{cm.suite} · gate",
    )


def render_run_html(run: Run) -> str:
    m = run.manifest
    rows = []
    for metric in m.metrics:
        vals = np.array([r.scores[metric].value for r in run.records if metric in r.scores])
        ci = stats.bootstrap_ci(vals)
        rows.append(
            {
                "name": metric,
                "est": ci.estimate,
                "lo": ci.low,
                "hi": ci.high,
                "cls": "b",
                "tip": f"{metric}\nmean {ci.estimate:.3f} [{ci.low:.3f}, {ci.high:.3f}]\nn={vals.size}",
                "val": f"{ci.estimate:.3f}",
            }
        )
    lat = [r.latency_ms for r in run.records]
    cost = [r.cost_usd * 1000 for r in run.records]
    hero = (
        f"<div class='hero'><span class='pill'>run</span><h1>{e(m.suite)}</h1><div class='meta'>"
        f"<span><code>{e(m.label or m.run_id)}</code></span><span class='mono'>{m.n_items} items · "
        f"{m.n_errors} errors · {m.duration_s:.1f} s</span><span class='mono'>system $"
        f"{m.totals.get('system_cost_usd', 0):.4f} · judge ${m.totals.get('eval_cost_usd', 0):.4f}"
        f"</span><span class='mono'>p95 {np.percentile(lat, 95):,.0f} ms</span></div></div>"
    )
    body = (
        "<section><span class='label'>Quality</span><h2>Metric means with 95% bootstrap CIs</h2>"
        + forest(rows, ref=0.0, fmt="{:.3f}", tick_fmt="{:.2f}")
        + "</section><section><span class='label'>Distributions</span><h2>Latency and cost</h2>"
        "<div class='grid2'>"
        + ecdf([("this run", "b", lat)], unit_fmt="{} ms", dp=0, title="Latency per item")
        + (
            ecdf([("this run", "b", cost)], unit_fmt="${} per 1k", dp=2, title="Cost per item")
            if max(cost, default=0) > 0
            else ""
        )
        + "</div></section>"
    )
    failed = [(r, name, s) for r in run.records for name, s in r.scores.items() if not s.passed]
    trs = "".join(
        f"<tr data-metric='{e(name)}'><td class='mono'>{e(r.id)}</td><td>{e(name)}</td>"
        f"<td class='num'>{s.value:.2f}</td><td><div class='out'>{e(str(r.output)[:400])}</div></td>"
        f"<td><div class='out'>{e(r.expected)}</div></td><td><div class='out'>{e(s.detail)}</div></td></tr>"
        for r, name, s in failed[:400]
    )
    body += (
        f"<section><span class='label'>Failures</span><h2>{len(failed)} failed item-metric pairs</h2>"
        "<div class='scroll'><table><thead><tr><th>item</th><th>metric</th><th class='num'>score</th>"
        f"<th>output</th><th>expected</th><th>detail</th></tr></thead><tbody>{trs}</tbody></table>"
        "</div></section>"
    )
    return page(f"{m.suite} run report", hero + body, subtitle=f"{m.suite} · run")


# ----------------------------------------------------------------------------- calibration


def render_calibration_html(result: CalibrationResult, payload: dict[str, Any]) -> str:
    vs = result.variants
    first = vs[0]
    # Feature the configuration declared final *before* the test run, never the best on test.
    final_name = payload.get("final_variant")
    best = next((v for v in vs if v.name == final_name), vs[-1])
    hh = payload.get("human_human", {})
    hh_tie = hh.get("with_tie")
    hero = (
        f"<div class='hero'><span class='pill'>judge calibration</span>"
        f"<h1>Judge–human agreement <em>{100 * first.agreement['estimate']:.0f}% → "
        f"{100 * best.agreement['estimate']:.0f}%</em></h1><div class='meta'>"
        f"<span>{e(result.name)}</span><span class='mono'>N={result.n_items} pairs · judge "
        f"{e(payload.get('judge_model'))}</span>"
        + (
            f"<span class='mono'>human–human {100 * hh_tie:.0f}% (with ties)</span>"
            if hh_tie
            else ""
        )
        + "</div><div class='tiles'>"
    )
    tiles = [
        (
            "agreement, naive",
            f"{100 * first.agreement['estimate']:.1f}%",
            f"kappa {first.kappa['estimate']:.3f}",
        ),
        (
            f"agreement, {best.name}",
            f"{100 * best.agreement['estimate']:.1f}%",
            f"kappa {best.kappa['estimate']:.3f}",
        ),
        (
            "naive: flips on swap",
            f"{100 * (1 - first.consistency):.0f}% of pairs",
            "final judge swaps and is order-invariant",
        ),
        ("judge cost / item", f"${best.cost_per_item_usd:.5f}", f"{best.name}"),
    ]
    for label, v, d in tiles:
        hero += f"<div class='tile'><div class='label'>{e(label)}</div><div class='v'>{e(v)}</div><div class='d'>{e(d)}</div></div>"
    hero += "</div></div>"

    rows = [
        {
            "name": v.name,
            "est": v.agreement["estimate"],
            "lo": v.agreement["low"],
            "hi": v.agreement["high"],
            "cls": "c" if v is best else "b",
            "tip": f"{v.name}\nagreement {100 * v.agreement['estimate']:.1f}% "
            f"[{100 * v.agreement['low']:.1f}, {100 * v.agreement['high']:.1f}]\nkappa "
            f"{v.kappa['estimate']:.3f}\njudge ties {100 * v.judge_tie_rate:.0f}%",
            "val": f"{100 * v.agreement['estimate']:.1f}%",
        }
        for v in vs
    ]
    ladder = (
        "<section><span class='label'>Technique ladder</span><h2>Agreement with human experts, 3-way (A / B / tie)</h2>"
        "<p class='lede'>Rows 1–6 each add one technique to the row above; rows 6a–6c change one "
        "thing relative to row 6 (the final judge, orange). Dot = agreement, line = 95% Wilson "
        "interval. Grey line = how often two individual <em>humans</em> agree on these pairs; the "
        "labels are majority votes, which are less noisy than one vote, so a judge can exceed it.</p>"
        + forest(rows, ref=hh_tie if hh_tie else 0.5, fmt="{:.1%}", tick_fmt="{:.0%}")
        + "</section>"
    )

    def bar_rows(key: str, fmt: str) -> str:
        out = ""
        for v in vs:
            val = getattr(v, key)
            if val is None:
                continue
            out += (
                f"<div>{e(v.name)}</div><div class='track' data-tip='{e(v.name)}: {fmt.format(val)}'>"
                f"<div class='fill{' em' if v is best else ''}' style='width:{100 * val:.1f}%'></div></div>"
                f"<div class='n'>{fmt.format(val)}</div>"
            )
        return out

    bias = (
        "<section><span class='label'>Position bias</span><h2>Does the verdict survive swapping the order?</h2>"
        "<p class='lede'>Every pair was judged twice, once as (A, B) and once as (B, A). Share of "
        "pairs where the two single calls agree. Variants that swap combine both calls, so their "
        "final verdict is order-invariant by construction; this measures the call underneath.</p>"
        "<div class='bars'>"
        + bar_rows("consistency", "{:.0%}")
        + "</div><p class='lede' style='margin-top:22px'>Among inconsistent pairs, how often the "
        "response shown <em>first</em> won. Grey line = 50%, no direction.</p>"
        + forest(
            [
                {
                    "name": v.name,
                    "est": v.first_position_rate,
                    "lo": v.first_position_rate,
                    "hi": v.first_position_rate,
                    "cls": "c" if v is best else "b",
                    "tip": f"{v.name}: first-shown won {100 * v.first_position_rate:.0f}% of "
                    f"{round((1 - v.consistency) * v.n)} inconsistent pairs",
                    "val": f"{100 * v.first_position_rate:.0f}%",
                }
                for v in vs
                if v.first_position_rate is not None
            ],
            ref=0.5,
            fmt="{:.0%}",
            tick_fmt="{:.0%}",
        )
        + "</section>"
    )

    vrows = []
    for v in vs:
        if v.verbosity_judge_longer is None:
            continue
        lo_, hi_ = sorted([v.verbosity_judge_longer, v.verbosity_human_longer or 0])
        vrows.append(
            {
                "name": v.name,
                "est": v.verbosity_judge_longer,
                "lo": lo_,
                "hi": hi_,
                "cls": "c",
                "tip": f"{v.name}\njudge picks longer {100 * v.verbosity_judge_longer:.0f}%\n"
                f"humans pick longer {100 * (v.verbosity_human_longer or 0):.0f}%\nn={v.verbosity_n}",
                "val": f"{100 * v.verbosity_judge_longer:.0f}% vs {100 * (v.verbosity_human_longer or 0):.0f}%",
            }
        )
    hl = vs[0].verbosity_human_longer or 0.5
    verbosity = (
        "<section><span class='label'>Verbosity bias</span><h2>How often the longer answer wins</h2>"
        f"<p class='lede'>Pairs where one answer is at least 1.5x longer and both humans and judge "
        f"picked a winner. Grey line = humans ({100 * hl:.0f}%); orange dot = the judge; the "
        "segment between them is the bias.</p>"
        + (forest(vrows, ref=hl, fmt="{:.0%}", tick_fmt="{:.0%}") if vrows else "")
        + "</section>"
    )

    def heat(v: Any) -> str:
        mat = np.array(v.confusion)
        mx = mat.max() or 1
        cells = "<div class='hd'></div>" + "".join(
            f"<div class='hd'>judge {lab}</div>" for lab in result.labels
        )
        for i, lab in enumerate(result.labels):
            cells += f"<div class='hd' style='text-align:right'>human {lab}</div>"
            for j, lab2 in enumerate(result.labels):
                a = 0.08 + 0.82 * mat[i, j] / mx
                ink = "#fff" if a > 0.5 else "var(--ink)"
                cells += (
                    f"<div class='cell' style='background:rgba(var(--seq),{a:.2f});color:{ink}' "
                    f"data-tip='human {lab}, judge {lab2}: {mat[i, j]} pairs'>{mat[i, j]}</div>"
                )
        return (
            f"<div><h3>{e(v.name)}</h3><div class='heat' style='grid-template-columns:auto repeat(3,1fr)'>"
            f"{cells}</div></div>"
        )

    confusion = (
        "<section><span class='label'>Confusion</span><h2>Where the judge and the humans disagree</h2>"
        "<p class='lede'>Rows = human verdict, columns = judge verdict; darker = more pairs.</p>"
        f"<div class='grid2'>{heat(first)}{heat(best)}</div></section>"
    )

    trs = ""
    for v in vs:
        d = v.vs_first
        dtxt = (
            f"{100 * d['delta']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}] p={d['mcnemar_p']:.3g}"
            if d
            else "–"
        )
        trs += (
            f"<tr><td>{e(v.name)}</td><td class='num'>{100 * v.agreement['estimate']:.1f}%</td>"
            f"<td class='num'>{v.kappa['estimate']:.3f} [{v.kappa['low']:.3f}, {v.kappa['high']:.3f}]</td>"
            f"<td class='num'>{'–' if v.non_tie_agreement is None else f'{100 * v.non_tie_agreement:.1f}%'}</td>"
            f"<td class='num'>{100 * v.judge_tie_rate:.0f}%</td><td class='num'>{100 * v.consistency:.0f}%</td>"
            f"<td class='num'>{dtxt}</td><td class='num'>${v.cost_per_item_usd:.5f}</td>"
            f"<td class='mono' style='font-size:11.5px'>{e(', '.join(v.prompt_files))}</td></tr>"
        )
    table = (
        "<section><span class='label'>Table view</span><h2>All variants</h2><div class='scroll'><table><thead><tr>"
        "<th>variant</th><th class='num'>agree</th><th class='num'>kappa [95% CI]</th>"
        "<th class='num'>non-tie agree</th><th class='num'>judge ties</th><th class='num'>swap-consistent</th>"
        "<th class='num'>Δ agree vs naive (pp)</th><th class='num'>$/item</th><th>prompt files</th></tr></thead>"
        f"<tbody>{trs}</tbody></table></div><p class='lede' style='margin-top:14px'>Human tie rate "
        f"{100 * first.human_tie_rate:.0f}%. Paired deltas use a bootstrap over items and McNemar's exact test.</p></section>"
    )
    return page(
        "Judge calibration report",
        hero + ladder + bias + verbosity + confusion + table,
        subtitle="judge calibration",
    )
