"""Acceptance tests for WP-11f (contact continuity). Written by the maintainer; do not edit.

Uses the WP-11c fixture (sofa_left seat (802, 704), floor (802, 874)): p1, p2 sit, p3 stands. A standing key next to a
seated key at the same place keeps its feet where the seated key had them on screen (x), on the floor line (y).
"""

from __future__ import annotations

import pytest
from test_placement_runs import REGIONS, lm, naime

from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo, solve


def solved(c: CharacterInput) -> dict[str, DrawingPlacement]:
    return solve([c], REGIONS)[0][c.name]


def foot_on_screen(c: CharacterInput, p: DrawingPlacement, drawing: str) -> tuple[float, float]:
    fx, fy = c.landmarks[drawing]["foot_low"]
    w = c.landmarks[drawing]["size"][0]
    if c.mirror:
        fx = w - fx
    return (p.x + fx * p.scale, p.y + fy * p.scale)


def with_poses(c: CharacterInput, poses: dict[str, PoseInfo], mirror: bool | None = None) -> CharacterInput:
    return CharacterInput(
        c.name, c.target_head_px, c.mirror if mirror is None else mirror, c.place, poses, c.landmarks, c.head_per_face
    )


def test_getting_up_keeps_the_feet() -> None:
    c = naime()
    out = solved(c)
    seated = foot_on_screen(c, out["p2"], "p2")
    standing = foot_on_screen(c, out["p3"], "p3")
    assert standing[0] == pytest.approx(seated[0], abs=1)
    assert standing[1] == pytest.approx(874, abs=1)
    assert out["p3"].anchor == "foot"


def test_sitting_down_keeps_the_feet() -> None:
    c = naime()
    rev = CharacterInput(
        c.name, c.target_head_px, c.mirror, c.place,
        {"p1": PoseInfo(support="stand"), "p2": PoseInfo(support="sit")},
        {"p1": c.landmarks["p3"], "p2": c.landmarks["p2"]}, c.head_per_face,
    )  # fmt: skip
    out = solved(rev)
    assert foot_on_screen(rev, out["p1"], "p1")[0] == pytest.approx(foot_on_screen(rev, out["p2"], "p2")[0], abs=1)


def test_standing_alone_stays_on_the_place_point() -> None:
    c = naime()
    alone = CharacterInput(
        c.name, c.target_head_px, c.mirror, c.place, {"p3": PoseInfo(support="stand")},
        {"p3": c.landmarks["p3"]}, c.head_per_face,
    )  # fmt: skip
    p = solved(alone)["p3"]
    assert foot_on_screen(alone, p, "p3") == (pytest.approx(802, abs=1), pytest.approx(874, abs=1))


def test_seated_neighbour_with_feet_up_does_not_move_the_feet() -> None:
    c = with_poses(
        naime(),
        {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit", rests_on=("coffee_table",)),
         "p3": PoseInfo(support="stand")},
    )  # fmt: skip
    p = solved(c)["p3"]
    assert foot_on_screen(c, p, "p3")[0] == pytest.approx(802, abs=1)


def test_neighbour_at_another_place_does_not_move_the_feet() -> None:
    c = with_poses(
        naime(),
        {
            "p1": PoseInfo(support="sit"),
            "p2": PoseInfo(support="sit"),
            "p3": PoseInfo(support="stand", on="sofa_right"),
        },
    )
    p = solved(c)["p3"]
    assert foot_on_screen(c, p, "p3")[0] == pytest.approx(1168, abs=1)


def test_previous_seated_key_wins_over_the_next() -> None:
    c = naime()
    marks = dict(c.landmarks)
    marks["p4"] = lm(head=[400, 80, 700, 700], face=[450, 300, 690, 600], hip=[505, 1700], foot=[905, 1752])
    c4 = CharacterInput(
        c.name, c.target_head_px, c.mirror, c.place,
        {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit"), "p3": PoseInfo(support="stand"),
         "p4": PoseInfo(support="sit")},
        marks, c.head_per_face,
    )  # fmt: skip
    out = solved(c4)
    assert foot_on_screen(c4, out["p3"], "p3")[0] == pytest.approx(foot_on_screen(c4, out["p2"], "p2")[0], abs=1)
    assert foot_on_screen(c4, out["p4"], "p4")[0] != pytest.approx(foot_on_screen(c4, out["p2"], "p2")[0], abs=5)


def test_mirrored_character_keeps_the_feet_too() -> None:
    c = with_poses(naime(), naime().poses, mirror=True)
    out = solved(c)
    assert foot_on_screen(c, out["p3"], "p3")[0] == pytest.approx(foot_on_screen(c, out["p2"], "p2")[0], abs=1)


def test_seated_keys_are_unchanged() -> None:
    out = solved(naime())
    for name in ("p1", "p2"):
        assert out[name].anchor == "hip"
        hx, hy = naime().landmarks[name]["hip"]
        assert out[name].x + hx * out[name].scale == pytest.approx(802, abs=1)
        assert out[name].y + hy * out[name].scale == pytest.approx(704, abs=1)
