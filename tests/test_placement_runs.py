"""Acceptance tests for WP-11c (scale from the face, one scale per settled run). Written by the maintainer; do not edit.

Same salon as the WP-11 tests: sofa_left seat (802, 704), floor (802, 874). Drawings are 1280x1920.
"""

from __future__ import annotations

from typing import Any

import pytest

from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo, solve
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

S1 = 118 / 600  # run p1..p2: median of the face scales 118/600, 118/580, 118/620, 118/600
S3 = 118 / 480  # run p3 (standing)


def lm(
    head: list[int] | None = None,
    face: list[int] | None = None,
    hip: list[float] | None = None,
    foot: list[float] | None = None,
) -> dict[str, Any]:
    box = lambda b: None if b is None else {"box": b, "confidence": 0.9}  # noqa: E731
    return {"version": 1, "image": "x.png", "size": [1280, 1920], "facing": "right", "head": box(head),
            "face": box(face), "feet": [], "feet_source": "foot", "foot_low": foot, "pants": None, "hip": hip,
            "hands": [], "cups": []}  # fmt: skip


def naime(head_per_face: float | None = 2.0) -> CharacterInput:
    return CharacterInput(
        name="naime",
        target_head_px=118.0,
        mirror=False,
        place="sofa_left",
        poses={"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit"), "p3": PoseInfo(support="stand")},
        landmarks={
            # the head box wanders with hair and tilt (590..700); the face box is steadier
            "p1": lm(head=[400, 100, 700, 690], face=[450, 300, 690, 600], hip=[500, 1700], foot=[800, 1750]),
            "p1-p2.1": lm(head=[400, 50, 700, 700], face=[450, 310, 690, 600], hip=[510, 1702], foot=[790, 1750]),
            "p1-p2.2": lm(head=[400, 100, 700, 700], face=[450, 290, 690, 600], hip=[490, 1698]),
            "p2": lm(head=[400, 80, 700, 700], face=[450, 300, 690, 600], hip=[505, 1700], foot=[805, 1752]),
            "p2-p3.1": lm(head=[400, 0, 700, 700], face=[450, 320, 690, 600], hip=[480, 1600], foot=[600, 1850]),
            "p2-p3.2": lm(head=[400, 300, 700, 700], face=[450, 400, 690, 600], hip=[520, 1550], foot=[620, 1851]),
            "p2-p3.3": lm(),  # no box, no point
            "p3": lm(head=[500, 300, 760, 780], face=[520, 500, 760, 740], hip=[600, 1500], foot=[640, 1852]),
        },
        head_per_face=head_per_face,
    )


@pytest.fixture(scope="module")
def placed() -> dict[str, DrawingPlacement]:
    return solve([naime()], REGIONS)[0]["naime"]


def test_head_per_face_is_an_optional_last_field() -> None:
    c = CharacterInput("veli", 50.0, False, None, {}, {})
    assert c.head_per_face is None
    assert naime().head_per_face == 2.0


def test_one_scale_for_the_sitting_run(placed: dict[str, DrawingPlacement]) -> None:
    for name in ("p1", "p1-p2.1", "p1-p2.2", "p2"):
        assert placed[name].scale == pytest.approx(S1), name


def test_run_keeps_every_hip_on_the_seat(placed: dict[str, DrawingPlacement]) -> None:
    hips = {"p1": (500, 1700), "p1-p2.1": (510, 1702), "p1-p2.2": (490, 1698), "p2": (505, 1700)}
    for name, (hx, hy) in hips.items():
        p = placed[name]
        assert p.anchor == "hip", name
        assert p.x + hx * p.scale == pytest.approx(802, abs=1), name
        assert p.y + hy * p.scale == pytest.approx(704, abs=1), name
    assert (placed["p1"].x, placed["p1"].y) == (704, 370)
    assert (placed["p1-p2.1"].x, placed["p1-p2.1"].y) == (702, 369)
    assert (placed["p1-p2.2"].depth, placed["p1-p2.2"].occluders) == (848, ("coffee_table",))


def test_standing_run_scale_from_the_face(placed: dict[str, DrawingPlacement]) -> None:
    p = placed["p3"]
    assert (p.scale, p.x, p.y, p.anchor, p.depth) == (pytest.approx(S3), 645, 419, "foot", 874)


def test_transition_inbetween_keeps_its_face_scale_between_the_runs(placed: dict[str, DrawingPlacement]) -> None:
    p = placed["p2-p3.1"]
    assert p.scale == pytest.approx(118 / 560) and p.anchor == "foot"
    # sit -> stand (WP-11d): the foot is lerped a quarter of the way between the keys' feet on screen,
    # (861.3, 714.6) -> (802.3, 874.3)
    assert p.x + 600 * p.scale == pytest.approx(846.57, abs=1)
    assert p.y + 1850 * p.scale == pytest.approx(754.49, abs=1)


def test_transition_inbetween_scale_is_clamped(placed: dict[str, DrawingPlacement]) -> None:
    assert placed["p2-p3.2"].scale == pytest.approx(S3)  # its face says 118/400: clamped to the larger key's scale


def test_transition_inbetween_without_points_is_lerped(placed: dict[str, DrawingPlacement]) -> None:
    p = placed["p2-p3.3"]
    assert (p.scale, p.x, p.y, p.anchor) == (pytest.approx(S1 + 0.75 * (S3 - S1)), 660, 407, "fallback")
    assert (p.depth, p.occluders) == (874, ("coffee_table",))


def test_ratio_from_the_drawings_when_not_given() -> None:
    marks = {
        "p1": lm(head=[0, 0, 100, 600], face=[0, 0, 100, 300], hip=[500, 1700]),  # head/face 2.0
        "p2": lm(head=[0, 0, 100, 330], face=[0, 0, 100, 300], hip=[500, 1700]),  # 1.1
        "p3": lm(head=[0, 0, 100, 900], face=[0, 0, 100, 300], hip=[500, 1700]),  # 3.0
    }
    sit = {k: PoseInfo(support="sit") for k in marks}
    median_ratio = solve([CharacterInput("a", 118.0, False, "sofa_left", sit, marks)], REGIONS)[0]["a"]
    assert {n: round(p.scale, 5) for n, p in median_ratio.items()} == dict.fromkeys(marks, round(118 / 600, 5))
    given = solve([CharacterInput("a", 118.0, False, "sofa_left", sit, marks, 1.0)], REGIONS)[0]["a"]
    assert {n: round(p.scale, 5) for n, p in given.items()} == dict.fromkeys(marks, round(118 / 300, 5))


def test_head_box_when_there_is_no_face() -> None:
    marks = {"p1": lm(head=[0, 0, 100, 590], hip=[500, 1700])}
    p = solve([CharacterInput("a", 118.0, False, "sofa_left", {"p1": PoseInfo(support="sit")}, marks)], REGIONS)
    assert (p[0]["a"]["p1"].scale, p[0]["a"]["p1"].anchor) == (pytest.approx(0.2), "hip")


def test_a_new_place_starts_a_new_run() -> None:
    marks = {
        "p1": lm(face=[0, 0, 100, 300], head=[0, 0, 100, 600], hip=[500, 1700]),
        "p2": lm(face=[0, 0, 100, 250], head=[0, 0, 100, 500], hip=[500, 1700]),
    }
    poses = {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit", on="sofa_right")}
    p = solve([CharacterInput("a", 118.0, False, "sofa_left", poses, marks, 2.0)], REGIONS)[0]["a"]
    assert p["p1"].scale == pytest.approx(118 / 600) and p["p2"].scale == pytest.approx(118 / 500)
    assert p["p2"].x + 500 * p["p2"].scale == pytest.approx(1168, abs=1)


def test_runs_follow_numeric_key_order() -> None:
    # numeric order p2, p9, p10: the standing p9 splits the two sitting keys (string order would join p10 and p2)
    marks = {
        "p2": lm(face=[0, 0, 100, 300], hip=[500, 1700]),
        "p9": lm(face=[0, 0, 100, 240], foot=[640, 1852]),
        "p10": lm(face=[0, 0, 100, 250], hip=[500, 1700]),
    }
    poses = {"p2": PoseInfo(support="sit"), "p9": PoseInfo(support="stand"), "p10": PoseInfo(support="sit")}
    p = solve([CharacterInput("a", 118.0, False, "sofa_left", poses, marks, 2.0)], REGIONS)[0]["a"]
    assert p["p2"].scale == pytest.approx(118 / 600) and p["p10"].scale == pytest.approx(118 / 500)


def test_deterministic() -> None:
    assert solve([naime()], REGIONS) == solve([naime()], REGIONS)
