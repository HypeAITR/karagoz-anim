"""Acceptance tests for WP-11b (the render places every drawing with the solver). Maintainer's; do not edit.

Uses the mini production of the WP-09b tests (`test_shot.py`): background (60, 60, 80); ali (brown figure) p1/p2;
ayse (green figure with a red marker on its right side) p1, mirrored in the shot. Every figure is the same 400x600
drawing: outline box (150, 100)-(250, 550); head box (150, 100, 250, 200) -> head height 100, opaque height 451.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image
from test_shot import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.engine.landmarks import measure_shot
from karagoz.engine.shot import render_shot
from karagoz.meta.sam import Segment

A = 100 / 451  # head ratio of the figure


def lm() -> dict[str, Any]:
    box = {"box": [150, 100, 250, 200], "confidence": 0.9}
    return {"version": 1, "image": "x.png", "sha256": "0", "size": [400, 600], "facing": "right", "head": box,
            "face": box, "feet": [], "feet_source": "foot", "foot_low": [200, 549], "pants": None, "hip": [162, 549],
            "hands": [], "cups": []}  # fmt: skip


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def setup_placement(p: Path) -> None:
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true, height_cm: 180 }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true, height_cm: 160 }\n",
        encoding="utf-8",
    )
    loc = p / "locations" / "salon"
    (loc / "masks").mkdir(parents=True)
    table = np.zeros((1080, 1920), np.uint8)
    table[600:700, 900:1100] = 255
    Image.fromarray(table).save(loc / "masks" / "sehpa.png")
    Image.fromarray(np.zeros((1080, 1920), np.uint8)).save(loc / "masks" / "koltuk.png")
    regions = {
        "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
        "scale": {"px_per_cm": 3.0, "source": "koltuk", "known_cm": 45},
        "objects": [
            {"name": "koltuk", "kind": "sofa", "box": [536, 462, 1427, 848], "floor_y": 848, "confidence": 0.9,
             "mask": "masks/koltuk.png", "front": False},
            {"name": "sehpa", "kind": "coffee table", "box": [900, 600, 1100, 700], "floor_y": 978, "confidence": 0.9,
             "mask": "masks/sehpa.png", "front": True},
        ],
        "places": [
            {"name": "koltuk", "object": "koltuk", "kind": "seat", "seat": [980, 704], "floor": [980, 874]},
            {"name": "kapi", "object": "koltuk", "kind": "floor", "floor": [300, 900]},
        ],
    }  # fmt: skip
    (loc / "regions.yaml").write_text(yaml.safe_dump(regions), encoding="utf-8")
    edit(p / EP / "shots.yaml", "{ name: ayse, place: koltuk", "{ name: ayse, place: kapi")
    edit(p / EP / "shots" / "sh01" / "poses.yaml", "omuzları görünüyor.\"\n        support: sit",
         "omuzları görünüyor.\"\n        support: stand")  # fmt: skip
    a = p / EP / "assets" / "sh01"
    for rel in ("ali/p1", "ali/p2", "ayse/p1", "ali/ref", "ayse/ref"):
        (a / f"{rel}.landmarks.json").write_text(json.dumps(lm()), encoding="utf-8")


def px(frame: Path, x: int, y: int) -> np.ndarray:
    return np.array(Image.open(frame).convert("RGB")).astype(int)[y, x]


def near(c: np.ndarray, rgb: tuple[int, int, int], tol: int = 30) -> bool:
    return int(np.abs(c - np.array(rgb)).sum()) < tol


def test_placement_mode(prod: Path) -> None:  # noqa: F811
    setup_placement(prod)
    r = render_shot(prod, EP, "sh01")
    a = prod / EP / "assets" / "sh01"
    placement = json.loads((a / "placement.json").read_text(encoding="utf-8"))
    ali, ayse = placement["characters"]["ali"]["p1"], placement["characters"]["ayse"]["p1"]
    s_ali, s_ayse = A * 180 * 3 / 100, A * 160 * 3 / 100
    assert ali["scale"] == pytest.approx(s_ali, abs=1e-4) and (ali["x"], ali["y"]) == (786, 47)
    assert ali["anchor"] == "hip" and ali["occluders"] == ["sehpa"] and ali["depth"] == 848
    assert ayse["scale"] == pytest.approx(s_ayse, abs=1e-4) and (ayse["x"], ayse["y"]) == (87, 316)
    assert ayse["mirror"] is True and ayse["anchor"] == "foot" and ayse["depth"] == 900
    assert any(w.startswith("ali p1: feet") and "while sitting on koltuk" in w for w in r.warnings)
    assert not any("has no placement" in w for w in r.warnings)  # missing inbetweens use their key's placement

    f0 = prod / EP / "render" / "sh01" / "frames" / "00000.png"
    assert near(px(f0, 1000, 400), (200, 120, 80))  # ali, on the seat, at his own scale
    assert near(px(f0, 1000, 650), (60, 60, 80))  # the table mask is in front of him: background shows
    assert near(px(f0, 300, 600), (90, 160, 90))  # ayse standing at the door place
    assert near(px(f0, 215, 561), (255, 0, 0), tol=120)  # her marker is on the left: mirrored
    assert near(px(f0, 1000, 150), (60, 60, 80))  # above ali's head: nothing


def test_without_regions_the_render_is_unchanged(prod: Path) -> None:  # noqa: F811
    r = render_shot(prod, EP, "sh01")
    assert r.frames == 100
    assert not (prod / EP / "assets" / "sh01" / "placement.json").exists()


def test_without_a_reference_landmark_the_render_is_unchanged(prod: Path) -> None:  # noqa: F811
    setup_placement(prod)
    (prod / EP / "assets" / "sh01" / "ayse" / "ref.landmarks.json").unlink()
    render_shot(prod, EP, "sh01")
    assert not (prod / EP / "assets" / "sh01" / "placement.json").exists()


class FakeSam:
    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        if concept != "head":
            return []
        return [Segment((150, 100, 250, 200), (400, 600), 0.9, np.full((100, 100), 200, np.uint8))]


def test_measure_shot_measures_the_references(prod: Path) -> None:  # noqa: F811
    written = measure_shot(prod, EP, "sh01", FakeSam(), workers=1)
    a = prod / EP / "assets" / "sh01"
    names = {p.relative_to(a).as_posix() for p in written}
    assert {"ali/ref.landmarks.json", "ayse/ref.landmarks.json", "ali/p1.landmarks.json"} <= names
    ref = json.loads((a / "ali" / "ref.landmarks.json").read_text(encoding="utf-8"))
    assert ref["head"]["box"] == [150, 100, 250, 200] and ref["image"] == "ali.png"
