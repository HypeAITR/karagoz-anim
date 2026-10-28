"""Acceptance tests for WP-11d (feet planted while sitting down / getting up; head/face ratio from the reference).

Written by the maintainer; do not edit. Same salon as the WP-11 tests: sofa_left seat (802, 704), floor (802, 874).
Naime stands at p1 (feet on the floor) and sits at p2 (hip on the seat). Her head moves least between the two keys
(5 px on screen), the feet move most (171 px) - the D41 case where a head anchor lifted the crouching inbetween.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_placement_runs import REGIONS, lm
from test_render_placement import A, setup_placement
from test_shot import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo, solve
from karagoz.engine.shot import render_shot

S1, S2, SI = 118 / 480, 118 / 600, 118 / 560


def naime(p2: PoseInfo, inbetween: dict[str, Any] | None = None) -> CharacterInput:
    ib = inbetween or lm(head=[480, 300, 760, 860], face=[500, 400, 740, 680], hip=[520, 1600], foot=[700, 1800])
    return CharacterInput(
        "naime", 118.0, False, "sofa_left", {"p1": PoseInfo(support="stand"), "p2": p2},
        {
            "p1": lm(head=[500, 300, 760, 780], face=[520, 500, 760, 740], hip=[600, 1500], foot=[640, 1852]),
            "p2": lm(head=[343, 600, 623, 1200], face=[400, 700, 640, 1000], hip=[500, 1700], foot=[800, 1750]),
            "p1-p2.1": ib,
        },
        2.0,
    )  # fmt: skip


def placed(c: CharacterInput) -> dict[str, DrawingPlacement]:
    return solve([c], REGIONS)[0]["naime"]


def test_keys_are_as_before() -> None:
    p = placed(naime(PoseInfo(support="sit")))
    assert (p["p1"].scale, p["p1"].x, p["p1"].y, p["p1"].anchor) == (pytest.approx(S1), 704, 419, "foot")  # WP-11f
    assert (p["p2"].scale, p["p2"].x, p["p2"].y, p["p2"].anchor) == (pytest.approx(S2), 704, 370, "hip")


def test_sitting_down_keeps_the_feet_on_the_floor_line() -> None:
    p = placed(naime(PoseInfo(support="sit")))["p1-p2.1"]
    assert p.anchor == "foot" and p.scale == pytest.approx(SI)
    # key feet on screen (WP-11f): (861.3, 874.3) standing, (861.3, 714.2) seated: the feet do not slide
    assert p.x + 700 * p.scale == pytest.approx(861.3, abs=1)
    assert p.y + 1800 * p.scale == pytest.approx(794.2, abs=1)
    assert (p.x, p.y) == (714, 415)


def test_getting_up_is_the_same_rule() -> None:
    c = naime(PoseInfo(support="sit"))
    rev = CharacterInput(c.name, c.target_head_px, c.mirror, c.place, {"p1": PoseInfo(support="sit"),
                         "p2": PoseInfo(support="stand")}, {"p1": c.landmarks["p2"], "p2": c.landmarks["p1"],
                         "p1-p2.1": c.landmarks["p1-p2.1"]}, 2.0)  # fmt: skip
    assert placed(rev)["p1-p2.1"].anchor == "foot"


def test_feet_on_a_table_keep_the_least_moving_point() -> None:
    p = placed(naime(PoseInfo(support="sit", rests_on=("coffee_table",))))["p1-p2.1"]
    assert p.anchor == "head"


def test_without_a_foot_keeps_the_least_moving_point() -> None:
    ib = lm(head=[480, 300, 760, 860], face=[500, 400, 740, 680], hip=[520, 1600])
    assert placed(naime(PoseInfo(support="sit"), ib))["p1-p2.1"].anchor == "head"


# --- shot.py: head_per_face from the reference ------------------------------------------------------------------


def test_reference_head_per_face_reaches_the_solver(prod: Path) -> None:  # noqa: F811
    setup_placement(prod)
    a = prod / EP / "assets" / "sh01"
    ref = json.loads((a / "ali" / "ref.landmarks.json").read_text(encoding="utf-8"))
    ref["face"]["box"] = [150, 150, 250, 200]  # face 50 px in a 100 px head: head/face = 2
    (a / "ali" / "ref.landmarks.json").write_text(json.dumps(ref), encoding="utf-8")
    render_shot(prod, EP, "sh01")
    placement = json.loads((a / "placement.json").read_text(encoding="utf-8"))
    # the drawings' face box is 100 px: 100 * 2 = a 200 px head, so half the scale of WP-11b's test
    assert placement["characters"]["ali"]["p1"]["scale"] == pytest.approx(A * 180 * 3 / 100 / 2, abs=1e-4)
    assert placement["characters"]["ayse"]["p1"]["scale"] == pytest.approx(A * 160 * 3 / 100, abs=1e-4)


def test_reference_without_face_falls_back_to_the_drawings(prod: Path) -> None:  # noqa: F811
    setup_placement(prod)
    a = prod / EP / "assets" / "sh01"
    ref = json.loads((a / "ali" / "ref.landmarks.json").read_text(encoding="utf-8"))
    ref["face"] = None
    (a / "ali" / "ref.landmarks.json").write_text(json.dumps(ref), encoding="utf-8")
    render_shot(prod, EP, "sh01")
    placement = json.loads((a / "placement.json").read_text(encoding="utf-8"))
    assert placement["characters"]["ali"]["p1"]["scale"] == pytest.approx(A * 180 * 3 / 100, abs=1e-4)
