"""WP-44 item 1: one cost counter (MetaClient hook) + CLI SAM cost reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from karagoz import cli
from karagoz.meta.client import MetaClient
from karagoz.studio.budget import Budget
from karagoz.studio.events import EventLog


class FakeTransport:
    def __init__(self, *responses: tuple[int, dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, body))
        return self.responses.pop(0)


def test_hook_three_calls_charge_budget_and_log_once(tmp_path: Path) -> None:
    from karagoz.studio.orchestrator import install_cost_hook

    budget = Budget(unit_cap_usd=10.0, episode_cap_usd=10.0)
    log = EventLog(tmp_path / "run" / "events.jsonl")
    transport = FakeTransport(
        (200, {"data": [{"b64_json": "eA=="}]}),
        (200, {"data": [{"b64_json": "eA=="}, {"b64_json": "eA=="}]}),
        (200, {"output": []}),
    )
    client = MetaClient(api_key="k", transport=transport, sleep=lambda s: None)
    install_cost_hook(client, budget, log, "sh01/draw")

    client.post("/images/generations", {"model": "muse-image-1.0", "prompt": "a", "n": 1}, "t1")
    client.post("/images/edits", {"model": "muse-image-1.0", "prompt": "b", "n": 2}, "t2")
    client.post("/responses", {"model": "sam-3.1", "input": []}, "t3")

    # 1 image + 2 images + 1 SAM call = 0.01 + 0.02 + 0.0025, charged exactly once.
    assert budget.total() == pytest.approx(0.0325)
    assert budget.spent == {"sh01": pytest.approx(0.0325)}
    costs = [e for e in log.read() if e["kind"] == "cost"]
    assert len(costs) == 3
    assert [c["task"] for c in costs] == ["sh01/draw"] * 3
    assert [c["model"] for c in costs] == ["muse-image-1.0", "muse-image-1.0", "sam-3.1"]
    assert [c["units"] for c in costs] == [1, 2, 1]
    assert sum(c["usd"] for c in costs) == pytest.approx(0.0325)
    assert [r.cost_usd for r in client.records] == pytest.approx([0.01, 0.02, 0.0025])


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


def test_cli_regions_reports_sam_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.meta.sam

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    prod = tmp_path / "prod"
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (200, 200, 200)).save(loc / "background.png")

    assert cli.main(["regions", str(prod), "salon", "sofa"]) == 0
    out = capsys.readouterr().out
    assert "sam: 3 calls, ~0.0075 $" in out


def test_cli_landmarks_reports_cost_and_appends_to_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.meta.sam

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    prod = tmp_path / "prod"
    d = prod / "ep" / "assets" / "sh01" / "ali"
    d.mkdir(parents=True)
    Image.new("RGB", (400, 600), (255, 255, 255)).save(d / "p1.png")
    log_path = prod / "ep" / "run" / "events.jsonl"
    EventLog(log_path).append("ep/director", "state", old="ready", new="running")
    before = len(log_path.read_text(encoding="utf-8").splitlines())

    assert cli.main(["landmarks", str(prod), "ep", "sh01"]) == 0
    out = capsys.readouterr().out
    assert "1 drawings measured" in out
    assert "sam: 7 calls, ~0.0175 $" in out
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == before + 1
    event = json.loads(lines[-1])
    assert event["kind"] == "cost"
    assert event["task"] == "sh01/landmarks"
    assert event["model"] == "sam-3.1"
    assert event["units"] == 7
    assert event["usd"] == 0.0175


def test_cli_landmarks_without_events_file_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import karagoz.meta.sam

    monkeypatch.setattr(karagoz.meta.sam, "Sam", CountingSam)
    prod = tmp_path / "prod"
    d = prod / "ep" / "assets" / "sh01" / "ali"
    d.mkdir(parents=True)
    Image.new("RGB", (400, 600), (255, 255, 255)).save(d / "p1.png")

    assert cli.main(["landmarks", str(prod), "ep", "sh01"]) == 0
    out = capsys.readouterr().out
    assert "1 drawings measured" in out
    assert "sam: 7 calls, ~0.0175 $" in out
    assert not (prod / "ep" / "run" / "events.jsonl").exists()
