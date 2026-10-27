"""Acceptance tests for WP-12a (landmarks of every drawing). Written by the maintainer; do not edit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from karagoz.engine.landmarks import (
    CONCEPTS,
    LANDMARK_VERSION,
    NEAR_PX,
    body_drawings,
    image_digest,
    measure,
    measure_shot,
)
from karagoz.meta.sam import Segment

W, H = 400, 600


def rect(box: tuple[int, int, int, int], conf: float | None) -> Segment:
    x1, y1, x2, y2 = box
    return Segment(box, (W, H), conf, np.full((y2 - y1, x2 - x1), 220, np.uint8))


def pants(conf: float | None = 0.9) -> Segment:
    """Box (140, 300, 260, 560): full width down to row 499, then only columns 140..179 down to row 549."""
    m = np.zeros((260, 120), np.uint8)
    m[:200, :] = 220
    m[200:250, :40] = 220
    m[255, 0] = 100  # soft pixel below the threshold (row 555): ignored
    return Segment((140, 300, 260, 560), (W, H), conf, m)


def answers() -> dict[str, list[Segment]]:
    return {
        "head": [rect((0, 0, 10, 10), 0.3), rect((150, 50, 250, 150), 0.88)],
        "face": [rect((170, 80, 230, 140), None)],
        "foot": [rect((210, 550, 260, 590), 0.93), rect((150, 540, 200, 580), 0.9)],
        "shoe": [rect((150, 540, 200, 580), 0.95)],
        "pants": [pants(0.9)],
        "hand": [rect((300, 200, 330, 230), 0.86), rect((100, 400, 130, 430), 0.8)],
        "cup": [rect((305, 180, 325, 210), 0.789)],
    }


class FakeSam:
    def __init__(self, answers: dict[str, list[Segment]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls.append((concept, tag))
        return list(self.answers.get(concept, []))


@pytest.fixture
def drawing(tmp_path: Path) -> Path:
    p = tmp_path / "p1.png"
    Image.new("RGB", (W, H), (255, 255, 255)).save(p)
    return p


def test_constants() -> None:
    assert LANDMARK_VERSION == 1 and NEAR_PX == 8
    assert CONCEPTS == ("head", "face", "foot", "shoe", "pants", "hand", "cup")


def test_measure(drawing: Path) -> None:
    sam = FakeSam(answers())
    r = measure(drawing, sam)
    assert [c for c, _ in sam.calls] == list(CONCEPTS)
    assert ("head", "landmarks:p1.png:head") in sam.calls
    assert r["version"] == 1 and r["image"] == "p1.png" and r["size"] == [W, H] and r["facing"] == "right"
    assert r["sha256"] == image_digest(drawing)
    assert r["head"] == {"box": [150, 50, 250, 150], "confidence": 0.88}
    assert r["face"] == {"box": [170, 80, 230, 140], "confidence": 1.0}
    assert r["feet_source"] == "foot"
    assert [f["box"] for f in r["feet"]] == [[150, 540, 200, 580], [210, 550, 260, 590]]
    assert r["foot_low"] == [234.5, 589]  # rows 581..589 belong to the right foot only
    assert r["pants"] == {"box": [140, 300, 260, 560], "confidence": 0.9}
    assert r["hip"] == [154.5, 549]  # back (left) quarter: columns 140..169, lowest row 549
    assert [h["box"] for h in r["hands"]] == [[100, 400, 130, 430], [300, 200, 330, 230]]
    assert r["cups"] == [{"box": [305, 180, 325, 210], "confidence": 0.79}]
    assert json.loads(json.dumps(r)) == r


def test_facing_left_takes_the_other_quarter(drawing: Path) -> None:
    r = measure(drawing, FakeSam(answers()), facing="left")
    assert r["facing"] == "left" and r["hip"] == [244.5, 499]


def test_shoes_when_no_feet_and_missing_parts(drawing: Path) -> None:
    a = answers()
    a["foot"] = []
    a["pants"] = []
    a["head"] = []
    r = measure(drawing, FakeSam(a))
    assert r["feet_source"] == "shoe" and r["foot_low"] == [174.5, 579]
    assert r["hip"] is None and r["pants"] is None and r["head"] is None
    a["shoe"] = []
    r = measure(drawing, FakeSam(a))
    assert r["feet"] == [] and r["feet_source"] is None and r["foot_low"] is None


def test_cache(drawing: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sam = FakeSam(answers())
    first = measure(drawing, sam, cache_dir=cache)
    assert (cache / f"{image_digest(drawing)}-right.json").is_file()
    n = len(sam.calls)
    copy = tmp_path / "other-name.png"
    copy.write_bytes(drawing.read_bytes())
    second = measure(copy, sam, cache_dir=cache)
    assert len(sam.calls) == n  # same bytes: no SAM call
    assert second == {**first, "image": "other-name.png"}
    measure(drawing, sam, facing="left", cache_dir=cache)
    assert len(sam.calls) == 2 * n  # another facing is another measurement


def test_body_drawings_and_measure_shot(tmp_path: Path) -> None:
    prod = tmp_path / "prod"
    a = prod / "ep" / "assets" / "sh01"
    (a / "naime").mkdir(parents=True)
    (a / "recep").mkdir()
    Image.new("RGB", (1920, 1080), (1, 2, 3)).save(a / "background.png")
    names = ["p1.png", "p1-p2.1.png", "p1.mouth-A.png", "p1.mouth-A.variant.png", "p1.eyes-closed.png"]
    for i, name in enumerate(names):
        Image.new("RGB", (W, H), (255, 255 - i, 255)).save(a / "naime" / name)
    Image.new("RGB", (W, H), (200, 200, 200)).save(a / "recep" / "p1.png")
    (prod / "characters.yaml").write_text(
        "version: 1\ncharacters:\n  naime: { label: Naime, natural_facing: left }\n  recep: { label: Recep }\n",
        encoding="utf-8",
    )
    assert [p.name for p in body_drawings(a, "naime")] == ["p1-p2.1.png", "p1.png"]
    sam = FakeSam(answers())
    written = measure_shot(prod, "ep", "sh01", sam, workers=2)
    assert [p.relative_to(a).as_posix() for p in written] == [
        "naime/p1-p2.1.landmarks.json",
        "naime/p1.landmarks.json",
        "recep/p1.landmarks.json",
    ]
    nm = json.loads((a / "naime" / "p1.landmarks.json").read_text(encoding="utf-8"))
    rc = json.loads((a / "recep" / "p1.landmarks.json").read_text(encoding="utf-8"))
    assert nm["facing"] == "left" and nm["hip"] == [244.5, 499] and rc["facing"] == "right"
    assert len(sam.calls) == 3 * len(CONCEPTS)
    assert (prod / ".cache" / "landmarks").is_dir()
    measure_shot(prod, "ep", "sh01", sam)
    assert len(sam.calls) == 3 * len(CONCEPTS)  # cached
