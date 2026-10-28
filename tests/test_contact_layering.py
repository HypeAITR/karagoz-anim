"""Acceptance tests for WP-11e (contact layering). Written by the maintainer; do not edit.

Salon as in the WP-11/11c tests: coffee_table is a front object, box (652, 782)-(1296, 977), floor_y 978; sofa_right
seat (1168, 704), floor (1168, 874). Drawings are 1280x1920; the sitting hip [520, 1150] lands on the seat, scale
124 / (300 * 2). A part rests on an object when it is lifted more than half a head (62 px) above the floor of the
place and its screen box overlaps the box of the object.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from PIL import Image

from karagoz.engine.compose import _composite_layers_placement
from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo, solve, to_json
from karagoz.spec import Regions

REGIONS = Regions.model_validate(
    {
        "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
        "scale": {"px_per_cm": 3.211, "source": "sofa", "known_cm": 45},
        "objects": [
            {"name": "sofa", "kind": "sofa", "box": [536, 462, 1427, 847], "floor_y": 848, "confidence": 0.96,
             "mask": "masks/sofa.png", "front": False},
            {"name": "coffee_table", "kind": "coffee table", "box": [652, 782, 1296, 977], "floor_y": 978,
             "confidence": 0.93, "mask": "masks/coffee_table.png", "front": True},
        ],
        "places": [
            {"name": "sofa_left", "object": "sofa", "kind": "seat", "seat": [802, 704], "floor": [802, 874]},
            {"name": "sofa_right", "object": "sofa", "kind": "seat", "seat": [1168, 704], "floor": [1168, 874]},
        ],
    }
)  # fmt: skip

FLOOR_FOOT = [633, 1602, 945, 1796]  # on screen ~36 px above the floor: on the floor, behind the table
TABLE_FOOT = [700, 1400, 950, 1600]  # ~77 px up, overlaps the table: rests on it
HAND_ON_TABLE = [100, 1450, 250, 1560]  # ~85 px up, overlaps the table: rests on it
HAND_IN_LAP = [100, 900, 250, 1000]  # lifted but above the table box: does not reach it


def lm(feet: list[list[int]] | None = None, hands: list[list[int]] | None = None) -> dict[str, Any]:
    entries = lambda boxes: [{"box": b, "confidence": 0.6} for b in (boxes or [])]  # noqa: E731
    return {
        "version": 1, "image": "x.png", "size": [1280, 1920], "facing": "right",
        "head": {"box": [500, 100, 800, 700], "confidence": 0.9},
        "face": {"box": [550, 300, 790, 600], "confidence": 0.9},
        "feet": entries(feet), "feet_source": "foot" if feet else None,
        "foot_low": [789, 1796] if feet else None, "pants": None, "hip": [520, 1150], "hands": entries(hands),
        "cups": [],
    }  # fmt: skip


def recep(
    feet: list[list[int]] | None,
    hands: list[list[int]] | None = None,
    rests: tuple[str, ...] = ("coffee_table",),
    mirror: bool = False,
) -> CharacterInput:
    info = PoseInfo(support="sit", rests_on=rests)
    return CharacterInput(
        name="recep",
        target_head_px=124.0,
        mirror=mirror,
        place="sofa_right",
        poses={"p1": info, "p2": info},
        landmarks={"p1": lm(feet, hands), "p1-p2.1": lm(feet, hands), "p2": lm(feet, hands)},
        head_per_face=2.0,
    )


def screen_box(p: DrawingPlacement, box: list[int], mirror: bool = False, w: int = 1280) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    if mirror:
        x0, x1 = w - x1, w - x0
    return (
        math.floor(p.x + x0 * p.scale) - 6,
        math.floor(p.y + y0 * p.scale) - 6,
        math.ceil(p.x + x1 * p.scale) + 6,
        math.ceil(p.y + y1 * p.scale) + 6,
    )


def solved(c: CharacterInput) -> tuple[dict[str, DrawingPlacement], list[str]]:
    placements, warnings = solve([c], REGIONS)
    return placements["recep"], warnings


def test_holes_is_an_optional_last_field() -> None:
    p = DrawingPlacement(scale=1.0, x=0, y=0, mirror=False, anchor="hip", occluders=(), depth=0)
    assert p.holes == ()


def test_fixture_geometry() -> None:
    p = solved(recep([FLOOR_FOOT, TABLE_FOOT]))[0]["p1"]
    assert p.anchor == "hip" and p.scale == pytest.approx(124 / 600)
    assert p.y + 1150 * p.scale == pytest.approx(704, abs=1)
    assert 874 - (p.y + FLOOR_FOOT[3] * p.scale) < 62 < 874 - (p.y + TABLE_FOOT[3] * p.scale)


def test_table_stays_in_front_and_only_the_resting_foot_is_over_it() -> None:
    p = solved(recep([FLOOR_FOOT, TABLE_FOOT]))[0]["p1"]
    assert "coffee_table" in p.occluders
    assert p.holes == (("coffee_table", screen_box(p, TABLE_FOOT)),)


def test_part_order_does_not_matter() -> None:
    a = solved(recep([FLOOR_FOOT, TABLE_FOOT]))[0]["p1"]
    b = solved(recep([TABLE_FOOT, FLOOR_FOOT]))[0]["p1"]
    assert a.holes == b.holes


def test_a_hand_on_the_table_is_over_it_too() -> None:
    p = solved(recep([FLOOR_FOOT, TABLE_FOOT], hands=[HAND_IN_LAP, HAND_ON_TABLE]))[0]["p1"]
    assert p.holes == (
        ("coffee_table", screen_box(p, TABLE_FOOT)),
        ("coffee_table", screen_box(p, HAND_ON_TABLE)),
    )


def test_inbetween_uses_its_own_parts() -> None:
    c = recep([FLOOR_FOOT, TABLE_FOOT])
    own = [700, 1380, 960, 1580]
    c.landmarks["p1-p2.1"] = lm([FLOOR_FOOT, own])
    p = solved(c)[0]["p1-p2.1"]
    assert "coffee_table" in p.occluders
    assert p.holes == (("coffee_table", screen_box(p, own)),)


def test_mirrored_drawing_mirrors_the_hole() -> None:
    p = solved(recep([FLOOR_FOOT, TABLE_FOOT], mirror=True))[0]["p1"]
    assert p.holes == (("coffee_table", screen_box(p, TABLE_FOOT, mirror=True)),)


def test_no_part_reaches_the_object_keeps_it_in_front_with_a_warning() -> None:
    placements, warnings = solved(recep([FLOOR_FOOT], hands=[HAND_IN_LAP]))
    p = placements["p1"]
    assert "coffee_table" in p.occluders and p.holes == ()
    assert "recep p1: rests on coffee_table but no foot or hand reaches it" in warnings


def test_no_parts_at_all_keeps_todays_rule() -> None:
    placements, warnings = solved(recep(None))
    p = placements["p1"]
    assert "coffee_table" not in p.occluders and p.holes == ()
    assert not [w for w in warnings if "reaches it" in w]


def test_without_rests_on_nothing_changes() -> None:
    placements, warnings = solved(recep([FLOOR_FOOT, TABLE_FOOT], rests=()))
    p = placements["p1"]
    assert "coffee_table" in p.occluders and p.holes == ()
    assert not [w for w in warnings if "reaches it" in w]


def test_to_json_writes_holes() -> None:
    placements = {"recep": solved(recep([FLOOR_FOOT, TABLE_FOOT]))[0]}
    data = to_json(placements, [])
    p1 = data["characters"]["recep"]["p1"]
    assert p1["holes"] == [{"object": "coffee_table", "box": list(placements["recep"]["p1"].holes[0][1])}]
    assert all(isinstance(v, int) for v in p1["holes"][0]["box"])
    assert data["characters"]["recep"]["p2"]["holes"]


def _layer(holes: tuple[tuple[str, tuple[int, int, int, int]], ...]) -> np.ndarray:
    bg = Image.new("RGB", (100, 100), (10, 20, 30))
    char = Image.new("RGBA", (100, 100), (200, 0, 0, 255))
    table = np.zeros((100, 100), dtype=bool)
    table[50:, :] = True  # the table covers the lower half
    pl = DrawingPlacement(
        scale=1.0, x=0, y=0, mirror=False, anchor="hip", occluders=("coffee_table",), depth=0, holes=holes
    )
    return np.array(_composite_layers_placement(bg, [(char, pl)], {"coffee_table": table}))


def test_compose_keeps_the_part_inside_the_hole_over_the_object() -> None:
    out = _layer((("coffee_table", (60, 50, 80, 70)),))
    assert tuple(out[20, 20]) == (200, 0, 0)  # above the table: the character
    assert tuple(out[60, 20]) == (10, 20, 30)  # behind the table: the table (background) again
    assert tuple(out[60, 70]) == (200, 0, 0)  # inside the hole: the resting part stays over the table
    assert tuple(out[75, 70]) == (10, 20, 30)  # just below the hole: the table again


def test_compose_two_holes_on_one_object() -> None:
    out = _layer((("coffee_table", (10, 50, 20, 60)), ("coffee_table", (60, 50, 80, 70))))
    assert tuple(out[55, 15]) == (200, 0, 0) and tuple(out[60, 70]) == (200, 0, 0)
    assert tuple(out[55, 40]) == (10, 20, 30)


def test_compose_hole_of_another_object_does_not_open_this_one() -> None:
    out = _layer((("armrest", (60, 50, 80, 70)),))
    assert tuple(out[60, 70]) == (10, 20, 30)
