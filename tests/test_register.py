"""Acceptance tests for WP-27 (drawings aligned by the picture of the still part). Written by the maintainer.

Unit tests: 200x300 figures at (100, 100), scale 1 — an upper block (rows 20..180, cols 60..140) and a lower block
(rows 180..290, cols 50..150); the hip landmark is at row 180. Target head 120 px -> R = 24 px.
Integration: the WP-11b mini production (`test_render_placement.py`), with ali's p2 picture moved 10 px right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from test_placement_runs import REGIONS
from test_render_placement import setup_placement
from test_shot import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo
from karagoz.engine.register import refine
from karagoz.engine.shot import render_shot


def fig(lower: tuple[int, int] = (0, 0), upper_dx: int = 0) -> Image.Image:
    a = np.zeros((300, 200, 4), np.uint8)
    a[20:180, 60 + upper_dx : 140 + upper_dx] = (200, 120, 80, 255)
    dx, dy = lower
    a[180 + dy : 290 + dy, 50 + dx : 150 + dx] = (40, 40, 120, 255)
    return Image.fromarray(a)


def lm() -> dict[str, Any]:
    return {"size": [200, 300], "hip": [100, 180], "foot_low": [100, 289], "feet": [], "hands": []}


def char(poses: dict[str, PoseInfo], names: list[str]) -> CharacterInput:
    return CharacterInput("ali", 120.0, False, "sofa_left", poses, {n: lm() for n in names}, None)


def placed(names: list[str], **kw: Any) -> dict[str, dict[str, DrawingPlacement]]:
    return {
        "ali": {
            n: DrawingPlacement(scale=1.0, x=100, y=100, mirror=False, anchor="hip", occluders=(), depth=848, **kw)
            for n in names
        }
    }


SIT = {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit")}
NAMES = ["p1", "p1-p2.1", "p2"]


def run(
    images: dict[str, Image.Image], poses: dict[str, PoseInfo] = SIT, **kw: Any
) -> tuple[dict[str, Any], list[str]]:
    names = list(images)
    out, warnings = refine([char(poses, names)], placed(names, **kw), REGIONS, lambda c, d: images.get(d))
    return out["ali"], warnings


def test_a_shifted_inbetween_in_a_seated_run_is_pulled_back() -> None:
    out, warnings = run({"p1": fig(), "p1-p2.1": fig((15, 0)), "p2": fig()})
    assert (out["p1-p2.1"].x, out["p1-p2.1"].y) == (85, 100)
    assert (out["p1"].x, out["p1"].y) == (100, 100) and (out["p2"].x, out["p2"].y) == (100, 100)
    assert warnings == []


def test_vertical_offset_too() -> None:
    out, _ = run({"p1": fig(), "p1-p2.1": fig(), "p2": fig((0, 8))})
    assert (out["p2"].x, out["p2"].y) == (100, 92)


def test_a_moving_upper_body_is_not_a_reason_to_move() -> None:
    out, warnings = run({"p1": fig(), "p1-p2.1": fig(upper_dx=30), "p2": fig()})
    assert (out["p1-p2.1"].x, out["p1-p2.1"].y) == (100, 100) and warnings == []


def test_standing_run_aligns_the_feet() -> None:
    stand = {"p1": PoseInfo(support="stand"), "p2": PoseInfo(support="stand")}
    out, _ = run({"p1": fig(), "p1-p2.1": fig((-12, 0)), "p2": fig()}, poses=stand)
    assert (out["p1-p2.1"].x, out["p1-p2.1"].y) == (112, 100)


def test_transition_inbetween_moves_sideways_only() -> None:
    up = {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="stand")}
    out, _ = run({"p1": fig(), "p1-p2.1": fig((10, 10)), "p2": fig()}, poses=up)
    assert (out["p1-p2.1"].x, out["p1-p2.1"].y) == (90, 100)


def test_too_far_is_left_as_placed_with_a_warning() -> None:
    out, warnings = run({"p1": fig(), "p1-p2.1": fig((40, 0)), "p2": fig()})
    assert (out["p1-p2.1"].x, out["p1-p2.1"].y) == (100, 100)
    assert "ali p1-p2.1: alignment needs more than 24 px, left as placed" in warnings


def test_holes_move_with_the_drawing() -> None:
    out, _ = run({"p1": fig(), "p1-p2.1": fig((15, 0)), "p2": fig()}, holes=(("sehpa", (10, 20, 30, 40)),))
    assert out["p1-p2.1"].holes == (("sehpa", (-5, 20, 15, 40)),)
    assert out["p1"].holes == (("sehpa", (10, 20, 30, 40)),)


def test_missing_picture_is_left_alone_and_inputs_are_not_modified() -> None:
    names = ["p1", "p1-p2.1", "p2"]
    images = {"p1": fig(), "p2": fig((15, 0))}
    before = placed(names)
    out, _ = refine([char(SIT, names)], before, REGIONS, lambda c, d: images.get(d))
    assert (out["ali"]["p1-p2.1"].x, out["ali"]["p2"].x) == (100, 85)
    assert before["ali"]["p2"].x == 100


def test_mirrored_drawing() -> None:
    names = ["p1", "p2"]
    images = {"p1": fig(), "p2": fig((15, 0))}  # in the drawing +15; mirrored on screen: -15
    pl = {
        "ali": {
            n: DrawingPlacement(scale=1.0, x=100, y=100, mirror=True, anchor="hip", occluders=(), depth=848)
            for n in names
        }
    }
    c = CharacterInput("ali", 120.0, True, "sofa_left", SIT, {n: lm() for n in names}, None)
    out, _ = refine([c], pl, REGIONS, lambda c_, d: images.get(d))
    assert out["ali"]["p2"].x == 115


def test_render_uses_the_refined_placements(prod: Path) -> None:  # noqa: F811
    setup_placement(prod)
    for name in ("p1", "p2"):  # the hip in the middle of the figure: the seated band is its lower half
        path = prod / EP / "assets" / "sh01" / "ali" / f"{name}.landmarks.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["hip"] = [200, 350]
        path.write_text(json.dumps(data), encoding="utf-8")
    p2 = prod / EP / "assets" / "sh01" / "ali" / "p2.png"
    arr = np.array(Image.open(p2).convert("RGB"))
    moved = np.full_like(arr, 255)
    moved[:, 10:] = arr[:, :-10]  # the figure 10 px to the right inside the picture
    Image.fromarray(moved).save(p2)
    render_shot(prod, EP, "sh01")
    pl = json.loads((prod / EP / "assets" / "sh01" / "placement.json").read_text(encoding="utf-8"))["characters"]
    p1, p2_ = pl["ali"]["p1"], pl["ali"]["p2"]
    shift = 10 * p1["scale"]
    # WP-45: ali is mirrored (he faces ayse), so the in-picture shift lands on screen with the opposite sign.
    assert p2_["x"] == pytest.approx(p1["x"] + shift, abs=1.5)
    assert p2_["y"] == p1["y"]
