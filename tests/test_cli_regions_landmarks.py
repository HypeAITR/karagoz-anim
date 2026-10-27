"""Maintainer integration tests: `karagoz regions` and `karagoz landmarks` wired to WP-10 / WP-12a."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import karagoz.meta.client
import karagoz.meta.sam
from karagoz import cli
from karagoz.meta.sam import Segment
from karagoz.spec import Regions, load

BOXES = {
    "sofa": (536, 462, 1427, 848),
    "seat cushion": (618, 652, 986, 756),
    "head": (150, 50, 250, 150),
    "foot": (150, 540, 200, 580),
    "pants": (140, 300, 260, 560),
}


class FakeSam:
    def __init__(self, client: Any = None) -> None:
        self.client = client

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        if concept not in BOXES:
            return []
        x1, y1, x2, y2 = BOXES[concept]
        frame = Image.open(image).size if isinstance(image, Path) else (1920, 1080)
        return [Segment((x1, y1, x2, y2), frame, 0.9, np.full((y2 - y1, x2 - x1), 200, np.uint8))]


@pytest.fixture(autouse=True)
def fake_sam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(karagoz.meta.sam, "Sam", FakeSam)
    monkeypatch.setattr(karagoz.meta.client, "MetaClient", lambda: None)


def test_regions_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    loc = tmp_path / "prod" / "locations" / "salon"
    assert cli.main(["regions", str(tmp_path / "prod"), "salon", "sofa"]) == 1
    assert "draw the location first" in capsys.readouterr().err
    loc.mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (200, 200, 200)).save(loc / "background.png")
    assert cli.main(["regions", str(tmp_path / "prod"), "salon", "sofa", "coffee table"]) == 0
    out = capsys.readouterr().out
    assert "scale: 3.2 px/cm from sofa" in out and "place sofa (seat)" in out and "object sofa" in out
    assert isinstance(load(loc / "regions.yaml"), Regions)


def test_landmarks_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prod = tmp_path / "prod"
    d = prod / "ep" / "assets" / "sh01" / "ali"
    d.mkdir(parents=True)
    Image.new("RGB", (400, 600), (255, 255, 255)).save(d / "p1.png")
    assert cli.main(["landmarks", str(prod), "ep", "sh01"]) == 0
    assert "1 drawings measured" in capsys.readouterr().out
    lm = json.loads((d / "p1.landmarks.json").read_text(encoding="utf-8"))
    assert lm["head"]["box"] == [150, 50, 250, 150] and lm["hip"] == [154.5, 559]
