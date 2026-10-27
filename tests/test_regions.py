"""Acceptance tests for WP-10 (region map from a location's background). Written by the maintainer; do not edit.

The fake SAM answers with rectangles at the boxes measured on the real slice background (D30), slightly rounded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from karagoz.engine.regions import (
    CUSHION_MIN_CONF,
    FLAT,
    KNOWN_HEIGHT_CM,
    OBJECT_MIN_CONF,
    SEATABLE,
    STAND_OFFSET_CM,
    build_regions,
    object_names,
)
from karagoz.meta.sam import Segment

W, H = 1920, 1080

ANSWERS: dict[str, list[tuple[tuple[int, int, int, int], float | None]]] = {
    "sofa": [((536, 462, 1427, 848), 0.96)],
    "seat cushion": [
        ((988, 652, 1350, 756), 0.91),
        ((618, 652, 986, 756), 0.91),
        ((638, 466, 987, 652), 0.59),  # back cushion, low confidence
        ((990, 463, 1347, 654), 0.85),  # back cushion, confident but above the sofa's middle
        ((1600, 700, 1700, 760), 0.95),  # outside the sofa
    ],
    "sofa armrest": [((1307, 581, 1428, 848), 0.93), ((536, 582, 672, 846), 0.93)],
    "coffee table": [((652, 782, 1296, 977), 0.93)],
    "tv cabinet": [((1536, 624, 1883, 979), None)],
    "rug": [((192, 861, 1677, 1032), 0.96), ((0, 0, 50, 50), 0.3)],
    "chair": [((100, 500, 300, 900), 0.9)],
}


def seg(box: tuple[int, int, int, int], conf: float | None) -> Segment:
    x1, y1, x2, y2 = box
    return Segment(box, (W, H), conf, np.full((y2 - y1, x2 - x1), 200, np.uint8))


class FakeSam:
    def __init__(self, answers: dict[str, list[tuple[tuple[int, int, int, int], float | None]]] | None = None) -> None:
        self.answers = ANSWERS if answers is None else answers
        self.calls: list[tuple[str, str]] = []

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls.append((concept, tag))
        return [seg(b, c) for b, c in self.answers.get(concept, [])]


@pytest.fixture
def bg(tmp_path: Path) -> Path:
    d = tmp_path / "locations" / "salon"
    d.mkdir(parents=True)
    p = d / "background.png"
    Image.new("RGB", (W, H), (240, 230, 210)).save(p)
    return p


def test_constants() -> None:
    assert SEATABLE["sofa"] == 45.0 and KNOWN_HEIGHT_CM["coffee table"] == 42.0 and "rug" in FLAT
    assert OBJECT_MIN_CONF == 0.5 and CUSHION_MIN_CONF == 0.8 and STAND_OFFSET_CM == 8.0


def test_object_names() -> None:
    assert object_names("coffee table", [(0, 0, 10, 10)]) == ["coffee_table"]
    assert object_names("sofa armrest", [(900, 0, 990, 10), (0, 0, 10, 10)]) == [
        "sofa_armrest_right",
        "sofa_armrest_left",
    ]
    assert object_names("sofa", [(500, 0, 600, 1), (0, 0, 10, 1), (200, 0, 300, 1)]) == [
        "sofa_right",
        "sofa_left",
        "sofa_center",
    ]
    assert object_names("x", [(30, 0, 40, 1), (0, 0, 10, 1), (10, 0, 20, 1), (20, 0, 30, 1)]) == [
        "x_4",
        "x_1",
        "x_2",
        "x_3",
    ]


def test_salon(bg: Path) -> None:
    sam = FakeSam()
    out = bg.parent
    r = build_regions(bg, "salon", ["sofa", "coffee table", "tv cabinet", "rug"], sam, out)
    assert r == yaml.safe_load((out / "regions.yaml").read_text(encoding="utf-8"))
    assert list(r) == ["version", "location", "image", "size", "scale", "objects", "places"]
    assert r["version"] == 1 and r["location"] == "salon" and r["image"] == "background.png" and r["size"] == [W, H]
    assert r["scale"] == {"px_per_cm": 3.2, "source": "sofa", "known_cm": 45.0}  # (848 - 704) / 45
    objs = {o["name"]: o for o in r["objects"]}
    assert [o["name"] for o in r["objects"]] == [
        "sofa", "sofa_armrest_left", "sofa_armrest_right", "coffee_table", "tv_cabinet", "rug",
    ]  # fmt: skip
    assert list(objs["sofa"]) == ["name", "kind", "box", "floor_y", "confidence", "mask", "front"]
    assert objs["sofa"] == {"name": "sofa", "kind": "sofa", "box": [536, 462, 1427, 848], "floor_y": 848,
                            "confidence": 0.96, "mask": "masks/sofa.png", "front": False}  # fmt: skip
    assert objs["sofa_armrest_left"]["kind"] == "sofa armrest" and objs["sofa_armrest_left"]["floor_y"] == 846
    assert objs["sofa_armrest_left"]["front"] and objs["sofa_armrest_right"]["front"]
    assert objs["coffee_table"]["front"] and objs["tv_cabinet"]["front"]  # nearer than the sofa
    assert objs["tv_cabinet"]["confidence"] == 1.0
    assert not objs["rug"]["front"]  # flat
    assert r["places"] == [
        {"name": "sofa_left", "object": "sofa", "kind": "seat", "seat": [802, 704], "floor": [802, 874]},
        {"name": "sofa_right", "object": "sofa", "kind": "seat", "seat": [1169, 704], "floor": [1169, 874]},
    ]
    assert ("seat cushion", "regions:salon:seat cushion") in sam.calls and (
        "sofa armrest",
        "regions:salon:sofa armrest",
    ) in sam.calls
    m = np.array(Image.open(out / "masks" / "coffee_table.png"))
    assert m.shape == (H, W) and m.dtype == np.uint8 and set(np.unique(m)) == {0, 255}
    assert m[800, 700] == 255 and m[700, 700] == 0
    assert sorted(p.name for p in (out / "masks").iterdir()) == sorted(f"{n}.png" for n in objs)


def test_scale_from_known_height_and_no_seats(bg: Path) -> None:
    r = build_regions(bg, "salon", ["coffee table", "rug"], FakeSam(), bg.parent)
    assert r["scale"] == {"px_per_cm": pytest.approx(4.643, abs=0.001), "source": "coffee_table", "known_cm": 42.0}
    assert r["places"] == [] and all(not o["front"] for o in r["objects"])


def test_seat_without_cushions_and_no_scale(bg: Path) -> None:
    r = build_regions(bg, "oda", ["chair"], FakeSam({"chair": ANSWERS["chair"]}), bg.parent)
    assert r["scale"] is None
    assert r["places"] == [
        {"name": "chair", "object": "chair", "kind": "seat", "seat": [200, 700], "floor": [200, 900]}
    ]


def test_nothing_found(bg: Path) -> None:
    r = build_regions(bg, "salon", ["sofa"], FakeSam({}), bg.parent)
    assert r["objects"] == [] and r["places"] == [] and r["scale"] is None
