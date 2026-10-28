"""Acceptance tests for WP-11 (placement solver), updated for WP-11c (Deniz p1 crouches). Do not edit.

The region map is the slice's salon as measured by SAM (D30, WP-10); drawings are 1280x1920 like Muse's.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from karagoz.engine.placement import (
    FEET_GAP_HEADS,
    INBETWEEN_RE,
    KEY_RE,
    OCCLUDE_MARGIN,
    CharacterInput,
    DrawingPlacement,
    PoseInfo,
    solve,
    target_head_px,
    to_json,
)
from karagoz.spec import Regions

REGIONS = Regions.model_validate(
    {
        "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
        "scale": {"px_per_cm": 3.211, "source": "sofa", "known_cm": 45},
        "objects": [
            {"name": "sofa", "kind": "sofa", "box": [536, 462, 1427, 847], "floor_y": 848, "confidence": 0.96,
             "mask": "masks/sofa.png", "front": False},
            {"name": "sofa_armrest_left", "kind": "sofa armrest", "box": [536, 582, 672, 846], "floor_y": 847,
             "confidence": 0.93, "mask": "masks/sofa_armrest_left.png", "front": True},
            {"name": "coffee_table", "kind": "coffee table", "box": [652, 782, 1296, 977], "floor_y": 978,
             "confidence": 0.93, "mask": "masks/coffee_table.png", "front": True},
            {"name": "rug", "kind": "rug", "box": [192, 861, 1677, 1032], "floor_y": 980, "confidence": 0.96,
             "mask": "masks/rug.png", "front": False},
        ],
        "places": [
            {"name": "sofa_left", "object": "sofa", "kind": "seat", "seat": [802, 704], "floor": [802, 874]},
            {"name": "sofa_right", "object": "sofa", "kind": "seat", "seat": [1168, 704], "floor": [1168, 874]},
            {"name": "kapi", "object": "sofa", "kind": "floor", "floor": [200, 900]},
        ],
    }
)  # fmt: skip


def lm(
    head: list[int] | None = None,
    face: list[int] | None = None,
    hip: list[float] | None = None,
    foot: list[float] | None = None,
    size: tuple[int, int] = (1280, 1920),
) -> dict[str, Any]:
    box = lambda b: None if b is None else {"box": b, "confidence": 0.9}  # noqa: E731
    return {"version": 1, "image": "x.png", "size": list(size), "facing": "right", "head": box(head),
            "face": box(face), "feet": [], "feet_source": "foot", "foot_low": foot, "pants": None, "hip": hip,
            "hands": [], "cups": []}  # fmt: skip


NAIME = CharacterInput(
    name="naime",
    target_head_px=118.0,
    mirror=False,
    place="sofa_left",
    poses={"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="stand")},
    landmarks={
        "p1": lm(head=[400, 100, 700, 690], face=[450, 300, 690, 595], hip=[500, 1700], foot=[800, 1750]),
        "p2": lm(head=[500, 300, 760, 772], face=[520, 500, 760, 736], hip=[600, 1500], foot=[640, 1852]),
        "p1-p2.1": lm(head=[420, 200, 700, 700], hip=[550, 1600], foot=[700, 1840]),
        "p2-p1.1": lm(face=[450, 300, 690, 600]),  # no head, hip or foot: placed between the keys
    },
)
RECEP = CharacterInput(
    name="recep",
    target_head_px=124.0,
    mirror=True,
    place="sofa_right",
    poses={"p1": PoseInfo(support="sit", rests_on=("coffee_table",))},
    landmarks={"p1": lm(head=[300, 100, 560, 596], hip=[500, 1700], foot=[900, 1800])},
)
DENIZ = CharacterInput(
    name="deniz",
    target_head_px=100.0,
    mirror=False,
    place="kapi",
    poses={"p1": PoseInfo(support="crouch")},  # p2 has no support: a floor place, so he stands (WP-11c: own run)
    landmarks={
        "p1": lm(head=[0, 0, 100, 200], face=[0, 0, 100, 100], foot=[50, 380], size=(100, 400)),
        "p2": lm(face=[0, 0, 100, 150], foot=[50, 400], size=(100, 400)),
    },
)
VELI = CharacterInput(
    "veli", 50.0, False, "balkon", {}, {"p1": lm(head=[0, 0, 100, 100], foot=[10, 90], size=(100, 100))}
)
ZEKI = CharacterInput("zeki", 50.0, False, None, {}, {"p1": lm(head=[0, 0, 100, 100], foot=[10, 90], size=(100, 100))})


@pytest.fixture(scope="module")
def result() -> tuple[dict[str, dict[str, DrawingPlacement]], list[str]]:
    return solve([NAIME, RECEP, DENIZ, VELI, ZEKI], REGIONS)


def test_constants_and_target() -> None:
    assert FEET_GAP_HEADS == 0.25 and OCCLUDE_MARGIN == 10
    assert KEY_RE.match("p12") and not KEY_RE.match("p1-p2.1")
    m = INBETWEEN_RE.match("p4-p5.3")
    assert m is not None and m.groups() == ("p4", "p5", "3")
    assert target_head_px(155, 3.211, 0.238) == pytest.approx(118.45, abs=0.01)


def test_sitting_key_on_the_seat(result: Any) -> None:
    p = result[0]["naime"]["p1"]
    assert p == DrawingPlacement(scale=pytest.approx(0.2), x=702, y=364, mirror=False, anchor="hip",
                                 occluders=("sofa_armrest_left", "coffee_table"), depth=848)  # fmt: skip


def test_standing_key_on_the_floor(result: Any) -> None:
    p = result[0]["naime"]["p2"]
    assert (p.scale, p.x, p.y, p.anchor, p.depth) == (pytest.approx(0.25), 642, 411, "foot", 874)
    assert p.occluders == ("coffee_table",)  # the armrest is behind someone standing in front of the sofa


def test_inbetween_anchored_on_the_least_moving_point(result: Any) -> None:
    p = result[0]["naime"]["p1-p2.1"]
    # hip moves 83 px between the keys, the head 103, the foot 171: the hip is lerped to (797, 745)
    assert (p.scale, p.x, p.y, p.anchor) == (pytest.approx(0.236), 667, 367, "hip")
    assert (p.depth, p.occluders) == (874, ("coffee_table",))  # k = 1 > n / 2: key B's


def test_inbetween_without_points_is_lerped(result: Any) -> None:
    p = result[0]["naime"]["p2-p1.1"]
    assert (p.scale, p.x, p.y, p.anchor) == (pytest.approx(0.225), 672, 388, "fallback")
    assert (p.depth, p.occluders) == (848, ("sofa_armrest_left", "coffee_table"))


def test_mirrored_character_resting_on_the_table(result: Any) -> None:
    p = result[0]["recep"]["p1"]
    assert (p.scale, p.x, p.y, p.mirror, p.anchor) == (pytest.approx(0.25), 973, 279, True, "hip")
    assert p.occluders == ("sofa_armrest_left",) and p.depth == 848  # the table carries his foot: not in front


def test_face_stands_in_for_a_missing_head(result: Any) -> None:
    p = result[0]["deniz"]["p2"]
    assert (p.scale, p.x, p.y, p.anchor) == (pytest.approx(1 / 3), 183, 767, "foot")
    assert p.depth == 900 and p.occluders == ("coffee_table",)


def test_unknown_or_missing_place(result: Any) -> None:
    v = result[0]["veli"]["p1"]
    assert (v.scale, v.x, v.y, v.anchor, v.depth, v.occluders) == (pytest.approx(0.5), 955, 970, "fallback", 1015, ())
    assert result[0]["zeki"]["p1"].anchor == "fallback"


def test_warnings(result: Any) -> None:
    assert result[1] == [
        "naime p1: feet 160 px above the floor while sitting on sofa_left",
        "recep p1: feet 145 px above the floor while sitting on sofa_right",
        "veli p1: unknown place 'balkon'",
        "zeki p1: no place",
    ]


def test_json_and_determinism(result: Any) -> None:
    j = to_json(*result)
    assert j == json.loads(json.dumps(j))
    occ = ["sofa_armrest_left", "coffee_table"]
    expected = {"scale": 0.2, "x": 702, "y": 364, "mirror": False, "anchor": "hip", "occluders": occ, "depth": 848}
    assert j["characters"]["naime"]["p1"] == expected
    assert j["warnings"] == result[1]
    assert solve([NAIME, RECEP, DENIZ, VELI, ZEKI], REGIONS) == result
