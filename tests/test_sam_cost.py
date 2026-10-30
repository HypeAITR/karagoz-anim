"""Acceptance tests for WP-36 (SAM calls are booked). Written by the maintainer; do not edit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from test_draw import EP, FakeImage, prod  # noqa: F401  (prod is a pytest fixture)
from test_draw_landmarks import HeadSam
from test_location_draw import FakeSam, add_objects

from karagoz.meta.sam import USD_PER_CALL, Segment
from karagoz.studio.draw import USD_PER_IMAGE
from karagoz.studio.orchestrator import code_work
from karagoz.studio.tasks import Task


class CountingHeadSam(HeadSam):
    pass


def test_draw_task_books_sam_calls(prod: Path) -> None:  # noqa: F811
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    regions = {"version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080], "scale": None,
               "objects": [], "places": []}  # fmt: skip
    (loc / "regions.yaml").write_text(yaml.safe_dump(regions), encoding="utf-8")
    sam, image = CountingHeadSam(), FakeImage()
    events: list[dict[str, Any]] = []
    r = code_work(prod, EP, Task(id="sh01/draw"), image=image, sam=sam, on_event=events.append)
    assert r.ok and r.final_text is not None and "landmarks" in r.final_text
    assert sam.calls > 0
    assert r.cost_usd == pytest.approx(len(image.calls) * USD_PER_IMAGE + sam.calls * USD_PER_CALL)
    notes = [e["payload"]["text"] for e in events if e.get("payload_type") == "code.note"]
    assert any(n.startswith(f"sam: {sam.calls} calls") for n in notes)


class CountingFakeSam(FakeSam):
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls += 1
        return super().segment(image, concept, tag=tag)


def test_location_task_books_sam_calls(prod: Path) -> None:  # noqa: F811
    add_objects(prod)
    sam, image = CountingFakeSam(), FakeImage()
    r = code_work(prod, EP, Task(id="salon/location"), image=image, sam=sam)
    assert r.ok, r.error
    assert r.final_text == "1 objects, 1 places"
    assert sam.calls > 0
    assert r.cost_usd == pytest.approx(len(image.calls) * USD_PER_IMAGE + sam.calls * USD_PER_CALL)
