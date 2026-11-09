# 0005. Self-contained HTML reports with HTML/CSS charts

Status: accepted · 2026-11-08

## Context
The report is a CI artifact: it is downloaded, attached to PRs, opened offline, sometimes on a phone. A JS charting bundle adds weight and a build step; an SVG scaled into a narrow viewport shrinks its text to unreadable sizes.

## Decision
One HTML file per report with inline CSS and ~30 lines of vanilla JS (tooltips, ECDF crosshair, filter chips). Forest plots, bars and heatmaps are HTML/CSS positioned by percentage, so labels stay at their CSS size at any width; ECDFs are SVG with `vector-effect: non-scaling-stroke`. Light and dark themes come from the portfolio's paper/ink tokens; the two series colours (baseline blue, candidate orange) were validated with the dataviz palette checker on both paper surfaces. Pass/fail is always spelled out in text, never colour alone, and every chart has a table view. Fonts load from Google Fonts with system fallbacks.

## Consequences
No interactive zoom or brushing; for that, the JSON artifacts are there to load into a notebook.
