"""A small FastAPI service over a runs folder: browse runs, render reports, gate on demand.

Endpoints
  GET  /health                       liveness + runs folder status
  GET  /api/runs                     run manifests, newest first
  GET  /api/runs/{run_id}            one manifest + per-metric means
  GET  /api/compare?baseline=&candidate=   gate result as JSON (thresholds from the candidate)
  GET  /runs/{run_id}                HTML run report
  GET  /compare?baseline=&candidate= HTML gate report
  GET  /                             index of runs (HTML)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from .gate import GateResult, compare
from .report import e, page, render_compare_html, render_run_html
from .runner import Manifest, Run, load_run

log = logging.getLogger("assaygate.server")
_SAFE_ID = re.compile(r"^[\w.\-]+$")


def create_app(runs_dir: Path | str = "runs") -> FastAPI:
    root = Path(runs_dir)
    app = FastAPI(title="assaygate", version="0.1.0")

    @app.middleware("http")
    async def access_log(request: Request, call_next):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        response = await call_next(request)
        log.info(
            json.dumps(
                {
                    "event": "http",
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            )
        )
        return response

    def _load(run_id: str) -> Run:
        if not _SAFE_ID.match(run_id):
            raise HTTPException(400, "invalid run id")
        path = root / run_id
        if not (path / "manifest.json").exists():
            raise HTTPException(404, f"run {run_id} not found")
        return load_run(path)

    def _manifests() -> list[Manifest]:
        if not root.exists():
            return []
        out = []
        for p in root.iterdir():
            mf = p / "manifest.json"
            if mf.exists():
                out.append(Manifest.model_validate_json(mf.read_text()))
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "runs_dir": str(root), "runs": len(_manifests())}

    @app.get("/api/runs")
    def list_runs() -> list[Manifest]:
        return _manifests()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> Manifest:
        return _load(run_id).manifest

    @app.get("/api/compare")
    def api_compare(baseline: str, candidate: str) -> GateResult:
        try:
            return compare(_load(baseline), _load(candidate))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(run_id: str) -> str:
        return render_run_html(_load(run_id))

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(baseline: str, candidate: str) -> str:
        b, c = _load(baseline), _load(candidate)
        try:
            return render_compare_html(compare(b, c), b, c)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        ms = _manifests()
        rows = "".join(
            f"<tr><td class='mono'><a href='/runs/{e(m.run_id)}'>{e(m.run_id)}</a></td>"
            f"<td>{e(m.suite)}</td><td>{e(m.label)}</td><td class='num'>{m.n_items}</td>"
            f"<td class='num'>{m.n_errors}</td><td class='mono'>"
            + e(", ".join(f"{k[5:]}={v:.3f}" for k, v in m.totals.items() if k.startswith("mean_")))
            + f"</td><td class='mono'>{e(m.created_at)}</td></tr>"
            for m in ms
        )
        pairs = ""
        if len(ms) >= 2:
            pairs = (
                "<p class='lede'>Compare the two newest: "
                f"<a href='/compare?baseline={e(ms[1].run_id)}"
                f"&candidate={e(ms[0].run_id)}'>{e(ms[1].run_id)} → {e(ms[0].run_id)}</a></p>"
            )
        body = (
            "<div class='hero'><span class='pill'>runs</span><h1>Evaluation runs</h1>"
            f"<div class='meta'><span class='mono'>{len(ms)} runs in {e(root)}</span></div></div>"
            f"<section>{pairs}<div class='scroll'><table><thead><tr><th>run</th><th>suite</th>"
            "<th>label</th><th class='num'>items</th><th class='num'>errors</th><th>means</th>"
            f"<th>created</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
        )
        return page("assaygate runs", body, subtitle="runs")

    return app
