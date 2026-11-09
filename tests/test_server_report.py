from __future__ import annotations

from fastapi.testclient import TestClient

from assaygate.gate import compare
from assaygate.report import render_compare_html, render_run_html
from assaygate.runner import save_run
from assaygate.server import create_app

from .helpers import pair


def test_server_endpoints(tmp_path):
    b, c = pair(drop=0.05)
    save_run(b, tmp_path / "base")
    save_run(c, tmp_path / "cand")
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json() == {"status": "ok", "runs_dir": str(tmp_path), "runs": 2}
    assert {m["run_id"] for m in client.get("/api/runs").json()} == {"base", "cand"}
    assert client.get("/api/runs/base").json()["n_items"] == 300
    g = client.get("/api/compare", params={"baseline": "base", "candidate": "cand"}).json()
    assert g["passed"] is False
    assert "Gate <em>failed</em>" in client.get("/compare?baseline=base&candidate=cand").text
    assert client.get("/runs/base").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/..%2Fetc").status_code in (400, 404)


def test_reports_are_self_contained_and_escape_outputs():
    b, c = pair(drop=0.05)
    c.records[0].output = "<script>alert(1)</script>"
    c.records[0].scores["accuracy"].value = 0.0
    html = render_compare_html(compare(b, c), b, c)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert html.startswith("<!doctype html>") and "prefers-color-scheme:dark" in html
    assert "Failures diff" in html and "Table view" in html
    assert "<html" in render_run_html(b)
