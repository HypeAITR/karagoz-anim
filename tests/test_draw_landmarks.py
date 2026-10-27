"""Maintainer integration test (3b): with a region map, the draw task measures the landmarks of what it drew."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from test_draw import EP, FakeImage, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.meta.sam import Segment
from karagoz.studio.orchestrator import code_work
from karagoz.studio.tasks import Task


class HeadSam:
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls += 1
        if concept != "head":
            return []
        return [Segment((350, 200, 650, 560), (1024, 1536), 0.9, np.full((360, 300), 200, np.uint8))]


def test_draw_measures_landmarks_only_with_regions(prod: Path) -> None:  # noqa: F811
    sam = HeadSam()
    r = code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), sam=sam)
    assert r.ok and sam.calls == 0  # no region map: nothing measured
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    regions = {"version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080], "scale": None,
               "objects": [], "places": []}  # fmt: skip
    (loc / "regions.yaml").write_text(yaml.safe_dump(regions), encoding="utf-8")
    r = code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), sam=sam)
    assert r.ok and r.final_text is not None and "landmarks" in r.final_text
    a = prod / EP / "assets" / "sh01" / "ali"
    assert (a / "p1.landmarks.json").is_file() and (a / "p1-p2.1.landmarks.json").is_file()
    assert not list(a.glob("*.mouth-*.landmarks.json"))
