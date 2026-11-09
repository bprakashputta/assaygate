"""Command line: ``assaygate run | compare | calibrate | report | serve``.

Exit codes: 0 = ok / gate passed, 1 = gate failed, 2 = usage or configuration error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from .calibrate import calibrate as run_calibration
from .calibrate import load_calibration_config
from .config import GateConfig, load_suite
from .gate import compare as gate_compare
from .gate import to_markdown
from .llm import LLMClient, RateLimiter, ResponseCache, SpendLedger
from .report import render_calibration_html, render_compare_html, render_run_html
from .runner import load_run, run_suite

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LLM evaluation and regression gating: scorers, calibrated judges, CI gates.",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@app.command()
def run(
    suite: Annotated[Path, typer.Argument(help="suite YAML")],
    runs_dir: Annotated[Path, typer.Option(help="where run folders are written")] = Path("runs"),
    label: Annotated[str | None, typer.Option(help="human label, e.g. 'main' or 'pr-42'")] = None,
    run_id: Annotated[str | None, typer.Option(help="fixed run id (default: timestamp)")] = None,
    html: Annotated[bool, typer.Option(help="also write report.html")] = True,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run a suite and write records.jsonl + manifest.json (+ report.html)."""
    _setup_logging(verbose)
    _load_dotenv()
    cfg = load_suite(suite)
    result = asyncio.run(run_suite(cfg, runs_dir=runs_dir, label=label, run_id=run_id))
    out = runs_dir / result.manifest.run_id
    if html:
        (out / "report.html").write_text(render_run_html(result))
    t = result.manifest.totals
    metrics = ", ".join(f"{m}={t.get('mean_' + m, 0):.3f}" for m in result.manifest.metrics)
    typer.echo(
        f"run {result.manifest.run_id}: {result.manifest.n_items} items, "
        f"{result.manifest.n_errors} errors, {metrics}, system ${t['system_cost_usd']:.4f}, "
        f"judge ${t['eval_cost_usd']:.4f}"
    )
    typer.echo(str(out))


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="baseline run folder")],
    candidate: Annotated[Path, typer.Argument(help="candidate run folder")],
    suite: Annotated[
        Path | None, typer.Option(help="take gate thresholds from this suite YAML")
    ] = None,
    markdown: Annotated[Path | None, typer.Option(help="write a PR-comment summary")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the full result")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML gate report")] = None,
) -> None:
    """Gate a candidate run against a baseline. Exit 1 if any check fails."""
    b, c = load_run(baseline), load_run(candidate)
    gate = load_suite(suite).gate if suite else GateConfig.model_validate(c.manifest.gate or {})
    try:
        result = gate_compare(b, c, gate)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    md = to_markdown(result)
    typer.echo(md)
    if markdown:
        markdown.write_text(md)
    if json_out:
        json_out.write_text(result.model_dump_json(indent=2))
    if html:
        html.write_text(render_compare_html(result, b, c))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(md)
    raise typer.Exit(0 if result.passed else 1)


@app.command()
def calibrate(
    config: Annotated[Path, typer.Argument(help="calibration YAML")],
    out: Annotated[Path, typer.Option(help="output folder")] = Path("calibration"),
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Measure judge variants against human labels (agreement, kappa, position/verbosity bias)."""
    _setup_logging(verbose)
    _load_dotenv()
    cfg = load_calibration_config(config)
    base = Path(cfg.base_dir)
    llm = LLMClient(
        ledger=SpendLedger(base / cfg.ledger_path, cfg.spend_cap_usd),
        cache=ResponseCache(base / cfg.cache_path) if cfg.cache_path else None,
        limiter=RateLimiter(cfg.rpm),
    )
    result = asyncio.run(run_calibration(cfg, llm))
    out.mkdir(parents=True, exist_ok=True)
    payload = {"judge_model": cfg.variants[0].llm.get("model", "claude-haiku-5-5")}
    (out / "calibration.json").write_text(result.model_dump_json(indent=2))
    (out / "report.html").write_text(render_calibration_html(result, payload))
    for v in result.variants:
        typer.echo(
            f"{v.name:<24} agree {100 * v.agreement['estimate']:5.1f}%  kappa "
            f"{v.kappa['estimate']:.3f}  swap-consistent {100 * v.consistency:4.0f}%  "
            f"${v.cost_per_item_usd:.5f}/item"
        )
    typer.echo(f"spent ${llm.ledger.spent_usd:.4f} total (ledger) · {out / 'report.html'}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="run folder")],
    baseline: Annotated[Path | None, typer.Option(help="baseline run to compare with")] = None,
    out: Annotated[Path | None, typer.Option(help="output file")] = None,
) -> None:
    """Render a run (or a baseline-vs-candidate comparison) as self-contained HTML."""
    cand = load_run(run_dir)
    if baseline:
        b = load_run(baseline)
        html = render_compare_html(gate_compare(b, cand), b, cand)
    else:
        html = render_run_html(cand)
    target = out or run_dir / ("compare.html" if baseline else "report.html")
    target.write_text(html)
    typer.echo(str(target))


@app.command()
def serve(
    runs_dir: Annotated[Path, typer.Option(help="runs folder to browse")] = Path("runs"),
    host: str = "127.0.0.1",
    port: int = 8100,
) -> None:
    """Serve run artifacts and reports over HTTP (FastAPI)."""
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(runs_dir), host=host, port=port, log_config=None)


@app.command()
def scorers() -> None:
    """List the registered scorer types."""
    from .scorers import available_scorers

    typer.echo(json.dumps(available_scorers()))


if __name__ == "__main__":  # pragma: no cover
    app()
