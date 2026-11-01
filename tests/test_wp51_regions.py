"""WP-51 extra tests: regions cushion queries and duplicate object names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from karagoz.engine.regions import build_regions
from karagoz.meta.sam import Segment
from karagoz.spec import Regions


def _seg(box: tuple[int, int, int, int], conf: float | None) -> Segment:
    x1, y1, x2, y2 = box
    return Segment(box, (1920, 1080), conf, np.full((y2 - y1, x2 - x1), 200, np.uint8))


class TwoArmchairSam:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls.append(concept)
        if concept == "armchair":
            return [_seg((100, 400, 500, 800), 0.96), _seg((600, 400, 1000, 800), 0.95)]
        return []


def test_seat_cushion_queried_once_for_two_seats(tmp_path: Path) -> None:
    bg = tmp_path / "background.png"
    Image.new("RGB", (1920, 1080), (240, 230, 210)).save(bg)
    sam = TwoArmchairSam()
    build_regions(bg, "salon", ["armchair"], sam, tmp_path / "out")
    assert sam.calls.count("seat cushion") == 1


class OneChairSam:
    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        if concept == "chair":
            return [_seg((536, 462, 1427, 848), 0.96)]
        return []


def test_duplicate_object_names_become_unique_and_validate(tmp_path: Path) -> None:
    bg = tmp_path / "background.png"
    Image.new("RGB", (1920, 1080), (240, 230, 210)).save(bg)
    out = tmp_path / "out"
    result = build_regions(bg, "salon", ["chair", "chair"], OneChairSam(), out)
    names = [o["name"] for o in result["objects"]]
    assert len(set(names)) == len(names) == 2
    data: Any = yaml.safe_load((out / "regions.yaml").read_text(encoding="utf-8"))
    Regions.model_validate(data)
    for name in names:
        assert (out / "masks" / f"{name}.png").is_file()
