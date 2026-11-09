from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
import yaml
from typer.testing import CliRunner

from assaygate.cli import app
from assaygate.config import SuiteConfig, load_suite
from assaygate.runner import load_run, run_suite
from assaygate.targets import Item, render_template

from .sut import ANSWERS


def write_suite(tmp_path: Path, ref: str, n_repeat: int = 15, **extra) -> Path:
    rows = []
    for k in range(n_repeat):
        for q, a in ANSWERS.items():
            rows.append({"id": f"{k}-{q}", "input": q, "expected": a})
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = {
        "name": "qa",
        "dataset": {"path": "data.jsonl"},
        "target": {"type": "callable", "ref": ref},
        "scorers": [{"type": "exact", "name": "accuracy"}, {"type": "token_f1"}],
        "gate": {"quality": [{"metric": "accuracy"}], "latency": None},
        "execution": {"concurrency": 4, "ledger_path": "spend.jsonl", "cache_path": None},
        **extra,
    }
    path = tmp_path / f"suite-{ref.split(':')[1]}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


async def test_run_suite_writes_artifacts(tmp_path):
    suite = load_suite(write_suite(tmp_path, "tests.sut:good"))
    run = await run_suite(suite, runs_dir=tmp_path / "runs", run_id="r1")
    assert run.manifest.n_items == 60 and run.manifest.totals["mean_accuracy"] == 1.0
    assert run.manifest.totals["system_cost_usd"] == pytest.approx(0.006)
    again = load_run(tmp_path / "runs" / "r1")
    assert again.records[0].scores["accuracy"].passed
    assert again.manifest.metrics == ["accuracy", "token_f1"]


async def test_async_and_failing_targets(tmp_path):
    run = await run_suite(
        load_suite(write_suite(tmp_path, "tests.sut:slow_async")), runs_dir=tmp_path
    )
    assert run.manifest.totals["mean_accuracy"] == 1.0
    broken = await run_suite(
        load_suite(write_suite(tmp_path, "tests.sut:broken")), runs_dir=tmp_path
    )
    assert broken.manifest.n_errors == 60
    assert all(r.scores["accuracy"].value == 0 for r in broken.records)


def test_cli_run_compare_exit_codes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    good = write_suite(tmp_path, "tests.sut:good")
    bad = write_suite(tmp_path, "tests.sut:bad")
    r1 = runner.invoke(app, ["run", str(good), "--run-id", "base", "--runs-dir", "runs"])
    r2 = runner.invoke(app, ["run", str(bad), "--run-id", "cand", "--runs-dir", "runs"])
    r3 = runner.invoke(app, ["run", str(good), "--run-id", "same", "--runs-dir", "runs"])
    assert r1.exit_code == 0 and r2.exit_code == 0 and r3.exit_code == 0, r1.output + r2.output
    assert (tmp_path / "runs" / "base" / "report.html").exists()

    fail = runner.invoke(
        app,
        [
            "compare",
            "runs/base",
            "runs/cand",
            "--markdown",
            "pr.md",
            "--html",
            "gate.html",
            "--json",
            "gate.json",
        ],
    )
    assert fail.exit_code == 1, fail.output
    assert "FAILED" in (tmp_path / "pr.md").read_text()
    assert json.loads((tmp_path / "gate.json").read_text())["passed"] is False
    assert "Gate <em>failed</em>" in (tmp_path / "gate.html").read_text()

    ok = runner.invoke(app, ["compare", "runs/base", "runs/same"])
    assert ok.exit_code == 0, ok.output

    rep = runner.invoke(app, ["report", "runs/cand", "--baseline", "runs/base"])
    assert rep.exit_code == 0 and (tmp_path / "runs" / "cand" / "compare.html").exists()


def test_env_expansion_in_suite(tmp_path, monkeypatch):
    p = write_suite(tmp_path, "tests.sut:good")
    text = p.read_text().replace("data.jsonl", "${DATA_FILE:-data.jsonl}")
    p.write_text(text)
    assert load_suite(p).dataset.path == "data.jsonl"
    monkeypatch.setenv("DATA_FILE", "other.jsonl")
    assert load_suite(p).dataset.path == "other.jsonl"


def test_template_rendering_keeps_types():
    item = Item(id="7", input={"q": 1}, fields={"meta": {"lang": "en"}})
    out = render_template({"x": "{{input}}", "s": "id={{id}} {{meta.lang}}", "l": ["{{id}}"]}, item)
    assert out == {"x": {"q": 1}, "s": "id=7 en", "l": ["7"]}


@respx.mock
async def test_http_target(tmp_path):
    respx.post("http://sut.local/classify").mock(
        side_effect=lambda req: httpx.Response(
            200, json={"label": ANSWERS[json.loads(req.content)["text"]], "cost_usd": 2e-5}
        )
    )
    cfg = yaml.safe_load(write_suite(tmp_path, "tests.sut:good", n_repeat=2).read_text())
    cfg["target"] = {
        "type": "http",
        "url": "http://sut.local/classify",
        "body": {"text": "{{input}}"},
        "output_path": "label",
        "cost_path": "cost_usd",
    }
    suite = SuiteConfig.model_validate({**cfg, "base_dir": str(tmp_path)})
    run = await run_suite(suite, runs_dir=tmp_path / "runs")
    assert run.manifest.totals["mean_accuracy"] == 1.0
    assert run.manifest.totals["system_cost_usd"] == pytest.approx(8 * 2e-5)


def test_duplicate_ids_rejected(tmp_path):
    from assaygate.data import load_items

    (tmp_path / "d.jsonl").write_text('{"id": 1, "input": "a"}\n{"id": 1, "input": "b"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_items(tmp_path / "d.jsonl")
