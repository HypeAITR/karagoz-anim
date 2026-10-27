"""Acceptance tests for WP-F6, draw side: one background per location. Written by the maintainer; do not edit.

Uses the mini production and the fake image model of the WP-09a tests (`test_draw.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from test_draw import EP, FakeImage, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.meta.sam import Segment
from karagoz.spec import Regions, load
from karagoz.studio.draw import link_location_background, location_objects, plan_draw, plan_location_draw
from karagoz.studio.orchestrator import code_work
from karagoz.studio.tasks import Task


def add_objects(prod: Path) -> None:  # noqa: F811
    (prod / "locations.yaml").write_text(
        "version: 1\nlocations:\n"
        "  salon: { description: 'a small test living room', objects: [sofa, 3, coffee table] }\n",
        encoding="utf-8",
    )


def test_location_objects(prod: Path) -> None:  # noqa: F811
    assert location_objects(prod, "salon") == []
    add_objects(prod)
    assert location_objects(prod, "salon") == ["sofa", "coffee table"]  # strings only
    assert location_objects(prod, "mutfak") == []


def test_plan_location_draw(prod: Path) -> None:  # noqa: F811
    (job,) = plan_location_draw(prod, "salon")
    shot_bg = next(j for j in plan_draw(prod, EP, "sh01") if j.kind == "background")
    assert job.kind == "background" and job.out == prod / "locations" / "salon" / "background.png"
    assert job.prompt == shot_bg.prompt and job.size == "1920x1080" and job.reasoning == "high" and job.wave == 1
    assert plan_location_draw(prod, "mutfak") == []


def test_shot_uses_the_location_background(prod: Path) -> None:  # noqa: F811
    a = prod / EP / "assets" / "sh01"
    assert link_location_background(prod, EP, "sh01") is None
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(loc / "background.png")
    assert all(j.kind != "background" for j in plan_draw(prod, EP, "sh01"))
    assert link_location_background(prod, EP, "sh01") == a / "background.png"
    assert (a / "background.png").read_bytes() == (loc / "background.png").read_bytes()


class FakeSam:
    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        boxes = {"sofa": (536, 462, 1427, 848), "seat cushion": (618, 652, 986, 756)}
        if concept not in boxes:
            return []
        x1, y1, x2, y2 = boxes[concept]
        return [Segment((x1, y1, x2, y2), (1920, 1080), 0.95, np.full((y2 - y1, x2 - x1), 200, np.uint8))]


def test_code_work_location(prod: Path) -> None:  # noqa: F811
    add_objects(prod)
    image = FakeImage()
    r = code_work(prod, EP, Task(id="salon/location"), image=image, sam=FakeSam())
    assert r.ok, r.error
    loc = prod / "locations" / "salon"
    assert Image.open(loc / "background.png").size == (1920, 1080) and (loc / "manifest.json").is_file()
    regions = load(loc / "regions.yaml")
    assert isinstance(regions, Regions) and [p.name for p in regions.places] == ["sofa"]
    assert r.final_text == "1 objects, 1 places"
    assert [c[0] for c in image.calls] == ["generate"]
    r2 = code_work(prod, EP, Task(id="salon/location"), image=image, sam=FakeSam())
    assert r2.ok and len(image.calls) == 1  # background cached by the draw manifest
