"""Acceptance tests for WP-38 (world flags: objects and characters against real-world sizes). Do not edit.

T-62, second kind: the gate flags when the measured scene does not fit the world (a sofa or TV far too big for the
characters' heights). It flags, it never rejects. Scale fixture: 3.211 px/cm, measured on the sofa (D30).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_gate import setup_shot

from karagoz.engine.gate import KINDS, WORLD_SIZES, WORLD_TOL, gate_shot, world_flags
from karagoz.spec import Regions

PX_PER_CM = 3.211


def obj(name: str, kind: str, height_px: int, top: int = 300) -> dict[str, Any]:
    return {"name": name, "kind": kind, "box": [100, top, 300, top + height_px], "floor_y": top + height_px + 1,
            "confidence": 0.9, "mask": f"masks/{name}.png", "front": False}  # fmt: skip


def regions(*objects: dict[str, Any], scale: bool = True) -> Regions:
    sofa = obj("sofa", "sofa", 385, top=462)  # 120 cm: outside the sofa range, but it is the scale source
    return Regions.model_validate(
        {
            "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
            "scale": {"px_per_cm": PX_PER_CM, "source": "sofa", "known_cm": 45} if scale else None,
            "objects": [sofa, *objects],
            "places": [{"name": "koltuk", "object": "sofa", "kind": "seat", "seat": [802, 704], "floor": [802, 874]}],
        }
    )  # fmt: skip


def test_world_is_the_last_kind_and_the_table_is_exposed() -> None:
    assert KINDS == ("prop_count", "reverse", "jump", "scale_jump", "placement", "world")
    assert WORLD_TOL == 1.5
    assert WORLD_SIZES["door"] == (180, 230) and WORLD_SIZES["coffee table"] == (30, 60)
    assert all(lo < hi for lo, hi in WORLD_SIZES.values())


def test_a_tv_far_too_big_for_the_scale_is_flagged() -> None:
    flags = world_flags(regions(obj("tv", "tv", 700), obj("tv_small", "television", 150)), {})
    assert [(f.kind, f.character, f.drawings) for f in flags] == [("world", None, ("tv",))]
    f = flags[0]
    height = 700 / PX_PER_CM  # 218 cm, a TV is 30-120 cm
    assert f.size == pytest.approx(height / 120, abs=0.01)
    assert "tv" in f.message and "218 cm" in f.message and "30" in f.message and "120" in f.message
    assert f.id == "world:-:tv" and f.shas == ()


def test_far_too_small_is_flagged_and_the_longest_kind_name_wins() -> None:
    flags = world_flags(regions(obj("sehpa", "coffee table", 25), obj("masa", "dining table", 250)), {})
    assert [f.drawings for f in flags] == [("sehpa",)]  # 7.8 cm coffee table; the 78 cm dining table fits "table"
    assert "coffee table" in flags[0].message
    assert flags[0].size == pytest.approx(30 / (25 / PX_PER_CM), abs=0.01)


def test_scale_source_unknown_kinds_and_missing_scale_are_not_flagged() -> None:
    assert world_flags(regions(obj("rug", "rug", 900)), {}) == []  # rugs have no height range
    assert world_flags(regions(obj("tv", "tv", 700), scale=False), {"ali": 165}) == []
    assert world_flags(None, {"ali": 165}) == []


def test_within_the_tolerance_is_not_flagged() -> None:
    hi_px = int(120 * WORLD_TOL * PX_PER_CM) - 2  # just under 1.5 x the top of the TV range
    assert world_flags(regions(obj("tv", "tv", hi_px)), {}) == []


def test_a_character_taller_than_a_door_is_flagged() -> None:
    door = obj("kapi", "door", 560)  # 174 cm: inside the door range with the tolerance
    flags = world_flags(regions(door), {"ali": 180.0, "ayse": 155.0})
    assert [(f.kind, f.character, f.drawings) for f in flags] == [("world", "ali", ("kapi",))]
    assert flags[0].size == pytest.approx(180 / (560 / PX_PER_CM), abs=0.01)
    assert "ali" in flags[0].message and "180 cm" in flags[0].message and "174 cm" in flags[0].message


def test_gate_shot_adds_world_flags_from_the_location_and_character_heights(prod: Path) -> None:  # noqa: F811
    setup_shot(prod, cups_in_ib1=1)
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    data = regions(obj("tv", "tv", 700), obj("kapi", "door", 500)).model_dump(mode="json")
    (loc / "regions.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    chars = yaml.safe_load((prod / "characters.yaml").read_text(encoding="utf-8"))
    chars["characters"]["ali"]["height_cm"] = 165
    (prod / "characters.yaml").write_text(yaml.safe_dump(chars, allow_unicode=True), encoding="utf-8")
    out = gate_shot(prod, EP, "sh01")
    ids = [f["id"] for f in out["flags"]]
    assert "world:-:tv" in ids and "world:ali:kapi" in ids  # 500 px door = 156 cm, ali is 165 cm
    assert json.loads((prod / EP / "review" / "sh01" / "gate.json").read_text(encoding="utf-8")) == out


def test_gate_shot_without_a_region_map_has_no_world_flags(prod: Path) -> None:  # noqa: F811
    setup_shot(prod, cups_in_ib1=1)
    assert [f for f in gate_shot(prod, EP, "sh01")["flags"] if f["kind"] == "world"] == []
