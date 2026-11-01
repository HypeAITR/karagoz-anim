"""WP-49 items 5-6: regions --episode cost event, draw SAM spend line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from karagoz import cli
from karagoz.studio.events import EventLog


class CountingSam:
    def __init__(self, client: Any = None) -> None:
        self.client = client
        self.calls = 0

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Any]:
        from karagoz.meta.sam import Segment

        self.calls += 1
        if concept not in ("sofa", "head"):
            return []
        frame = (1920, 1080)
        if isinstance(image, Path):
            with Image.open(image) as img:
                frame = img.size
        box = (536, 462, 1427, 848) if concept == "sofa" else (150, 50, 250, 150)
        x1, y1, x2, y2 = box
        return [Segment(box, frame, 0.9, np.full((y2 - y1, x2 - x1), 200, np.uint8))]


def _salon(prod: Path) -> Path:
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (200, 200, 200)).save(loc / "background.png")
    return loc


def test_wp49_regions_episode_appends_cost_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.meta.sam

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    prod = tmp_path / "prod"
    _salon(prod)
    log_path = prod / "ep" / "run" / "events.jsonl"
    EventLog(log_path).append("ep/director", "state", old="ready", new="running")
    before = len(log_path.read_text(encoding="utf-8").splitlines())

    assert cli.main(["regions", str(prod), "salon", "sofa", "--episode", "ep"]) == 0
    out = capsys.readouterr().out
    assert "sam: 3 calls, ~0.0075 $" in out
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == before + 1
    event = json.loads(lines[-1])
    assert event["kind"] == "cost"
    assert event["task"] == "salon/regions"
    assert event["model"] == "sam-3.1"
    assert event["units"] == 3
    assert event["usd"] == 0.0075


def test_wp49_regions_without_episode_writes_no_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.meta.sam

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    prod = tmp_path / "prod"
    _salon(prod)
    log_path = prod / "ep" / "run" / "events.jsonl"
    EventLog(log_path).append("ep/director", "state", old="ready", new="running")
    before = log_path.read_text(encoding="utf-8")

    assert cli.main(["regions", str(prod), "salon", "sofa"]) == 0
    assert "sam: 3 calls, ~0.0075 $" in capsys.readouterr().out
    assert log_path.read_text(encoding="utf-8") == before


def test_wp49_regions_episode_without_events_file_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.meta.sam

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    prod = tmp_path / "prod"
    _salon(prod)

    assert cli.main(["regions", str(prod), "salon", "sofa", "--episode", "ep"]) == 0
    assert "sam: 3 calls, ~0.0075 $" in capsys.readouterr().out
    assert not (prod / "ep" / "run" / "events.jsonl").exists()


def test_wp49_draw_landmark_measuring_prints_sam_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.engine.landmarks
    import karagoz.meta.sam
    import karagoz.studio.draw
    import karagoz.studio.orchestrator

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    monkeypatch.setattr(karagoz.studio.draw, "link_location_background", lambda *a: None)
    monkeypatch.setattr(karagoz.studio.draw, "plan_draw", lambda *a: [])
    monkeypatch.setattr(karagoz.studio.draw, "retire_stale", lambda *a: [])
    monkeypatch.setattr(karagoz.studio.draw, "sam_face_of", lambda *a: lambda p: None)

    def fake_run(jobs: Any, image: Any, **kw: Any) -> Any:
        return karagoz.studio.draw.DrawReport(done=[], cached=[], failed=[], calls=0, images=0)

    monkeypatch.setattr(karagoz.studio.draw, "run_draw", fake_run)
    monkeypatch.setattr(karagoz.studio.orchestrator, "_shot_has_regions", lambda *a: True)

    prod = tmp_path / "prod"
    d = prod / "ep" / "assets" / "sh01" / "ali"
    d.mkdir(parents=True)
    Image.new("RGB", (400, 600), (255, 255, 255)).save(d / "p1.png")

    assert cli.main(["draw", str(prod), "ep", "sh01"]) == 0
    out = capsys.readouterr().out
    assert "1 landmarks measured" in out
    assert "sam: 7 calls, ~0.0175 $" in out
