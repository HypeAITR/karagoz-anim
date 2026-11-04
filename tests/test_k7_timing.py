"""Acceptance tests for K-7 (timing spans and the timing report)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import test_draw
import yaml

from karagoz import cli
from karagoz.studio import timing_log
from karagoz.studio.draw import plan_draw, run_draw
from karagoz.studio.events import EventLog
from karagoz.studio.profile import load_spans, summarize


def _mini_production(root: Path) -> Path:
    """The spec mini-production from tests/test_draw.py (shot sh01, lines L01/L02)."""
    prod = root / "prod"
    episode = prod / test_draw.EP
    (episode / "shots" / "sh01").mkdir(parents=True)
    (episode / "audio").mkdir()
    (prod / "referans").mkdir()
    style = yaml.safe_load((test_draw.FIX / "style.yaml").read_text(encoding="utf-8"))
    style["live"].update(test_draw.STYLE_EXTRA)
    (prod / "style.yaml").write_text(yaml.safe_dump(style, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(test_draw.FIX / name, episode / name)
    for name in ("poses.yaml", "xsheet.yaml"):
        shutil.copy(test_draw.FIX / name, episode / "shots" / "sh01" / name)
    for lid, text in (("L01", "Yarın pazar, Ali."), ("L02", "Su ver bana.")):
        n = len(text)
        starts = [round(i * 0.1, 6) for i in range(n)]
        ends = [round((i + 1) * 0.1, 6) for i in range(n)]
        align = {"version": 1, "line": lid, "text": text, "characters": list(text), "starts": starts, "ends": ends}
        (episode / "audio" / f"{lid}.align.json").write_text(json.dumps(align), encoding="utf-8")
    test_draw.person((240, 200, 160)).save(prod / "referans" / "ali.png")
    test_draw.person((230, 190, 170)).save(prod / "referans" / "ayse.png")
    (prod / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: false }\n",
        encoding="utf-8",
    )
    (prod / "locations.yaml").write_text(
        "version: 1\nlocations:\n  salon: { description: 'a small test living room' }\n", encoding="utf-8"
    )
    return prod


def _span_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and obj.get("payload_type") == "timing.span":
            rows.append(obj)
    return rows


def test_nested_spans_record_the_enclosing_parent(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    with (
        timing_log.activate(log),
        timing_log.span("outer", task="sh01/layout"),
        timing_log.span("inner", task="sh01/layout", attempt=1),
    ):
        pass

    rows = _span_rows(tmp_path / "events.jsonl")
    assert len(rows) == 2
    assert all(r["payload_type"] == "timing.span" for r in rows)
    by_name = {r["payload"]["name"]: r["payload"] for r in rows}
    outer, inner = by_name["outer"], by_name["inner"]
    assert set(outer) == {"name", "parent", "start", "end", "sec", "attrs"}
    assert outer["parent"] is None
    assert (inner["name"], inner["parent"]) == ("inner", "outer")
    assert inner["attrs"] == {"attempt": 1}
    assert inner["end"] >= inner["start"] and inner["sec"] >= 0.0


def test_span_without_an_active_log_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    with timing_log.span("ghost", task="sh01/layout"), timing_log.span("ghost-child"):
        pass
    assert not target.exists()
    with timing_log.activate(EventLog(target)), timing_log.span("real"):
        pass
    assert [r["payload"]["name"] for r in _span_rows(target)] == ["real"]


def test_fake_run_gives_expected_totals_critical_path_and_parallelism(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    timing_log.record(
        "task:sh01/layout",
        0.0,
        10.0,
        log=log,
        task="sh01/layout",
        department="layout",
        carrier="api",
        attempt=1,
        ok=True,
    )
    timing_log.record(
        "model:layout",
        0.5,
        9.0,
        log=log,
        task="sh01/layout",
        parent="task:sh01/layout",
        model="m",
        input_tokens=100,
        output_tokens=50,
        retries=1,
    )
    timing_log.record("draw:wave:1", 1.0, 4.0, log=log, parent="task:sh01/layout", jobs=6)
    timing_log.record("draw:wave:2", 4.0, 8.0, log=log, parent="task:sh01/layout", jobs=6)
    summary = summarize(load_spans(path))
    assert summary["spans"] == 4
    assert summary["wall_sec"] == pytest.approx(10.0)
    by_name = summary["by_name"]
    assert by_name["draw:wave:1"]["count"] == 1
    assert by_name["draw:wave:1"]["total_sec"] == pytest.approx(3.0)
    assert by_name["draw:wave:1"]["share"] == pytest.approx(0.3)
    assert by_name["model:layout"]["total_sec"] == pytest.approx(8.5)
    assert summary["critical_path"] == ["task:sh01/layout", "model:layout"]
    assert summary["parallelism"]["draw"]["total_sec"] == pytest.approx(7.0)
    assert summary["parallelism"]["draw"]["parallelism"] == pytest.approx(0.7)
    assert summary["slowest"][0]["name"] == "task:sh01/layout"


def test_profile_command_writes_the_table_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "events.jsonl"
    log = EventLog(events)
    timing_log.record("task:sh01/layout", 0.0, 10.0, log=log, ok=True)
    timing_log.record("model:layout", 0.5, 9.0, log=log, parent="task:sh01/layout", retries=0)
    out = tmp_path / "report.md"
    assert cli.main(["profile", str(events), "--out", str(out)]) == 0
    table = out.read_text(encoding="utf-8")
    assert "| span | count | total s | mean s | max s | share |" in table
    assert "`task:sh01/layout`" in table
    assert "## Critical path" in table
    assert "`task:sh01/layout` → `model:layout`" in table
    assert "## Parallelism by stage" in table
    assert "## Slowest single spans" in table
    data = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert data["spans"] == 2
    assert data["wall_sec"] == pytest.approx(10.0)
    assert data["critical_path"] == ["task:sh01/layout", "model:layout"]
    assert data["by_name"]["model:layout"]["total_sec"] == pytest.approx(8.5)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["profile", str(events)]) == 0
    assert "Timing profile" in capsys.readouterr().out


def test_fake_image_draw_logs_one_span_per_draw_job(tmp_path: Path) -> None:
    prod = _mini_production(tmp_path)
    jobs = plan_draw(prod, test_draw.EP, "sh01")
    assert len(jobs) > 0
    log = EventLog(tmp_path / "events.jsonl")
    with timing_log.activate(log), timing_log.span("draw-task"):
        report = run_draw(jobs, test_draw.FakeImage())
    assert not report.failed
    assert len(report.done) == len(jobs)
    spans = _span_rows(tmp_path / "events.jsonl")
    job_spans = [s for s in spans if s["payload"]["name"].startswith("draw:job:")]
    assert len(job_spans) == len(jobs)
    wave_parents = set()
    for span_row in job_spans:
        payload = span_row["payload"]
        assert payload["parent"].startswith("draw:wave:")
        assert payload["end"] >= payload["start"]
        wave_parents.add(payload["parent"])
    wave_spans = [s for s in spans if s["payload"]["name"] in wave_parents]
    assert len(wave_spans) == len(wave_parents) > 0
    assert {s["payload"]["parent"] for s in wave_spans} == {"draw-task"}


def test_studio_task_runs_write_spans_without_manual_activation(tmp_path: Path) -> None:
    """Reviewer: a real run never calls ``activate``; the studio must activate its own log per task."""
    import inspect

    from karagoz.studio import orchestrator

    src = inspect.getsource(orchestrator.Studio)
    assert src.count("activate(self.log)") >= 2
