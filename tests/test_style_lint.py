"""Acceptance tests for WP-30 (style lint from style.yaml `lint:`). Written by the maintainer; do not edit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from test_xsheet import ALIGNS, FIX, align, style_dict

from karagoz.engine.xsheet import Exposure, Window, check_timing, style_lint
from karagoz.spec import Style

LINT = {
    "pose_per_line_max": {"value": 4, "source": "test"},
    "hand_static_max": {"value": 48, "source": "test"},
    "twos_share_warn": {"value": 0.8, "source": "test"},
}


def lint_style(**only: float) -> Style:
    d = style_dict()
    d["lint"] = {k: {"value": v, "source": "test"} for k, v in only.items()} if only else LINT
    return Style.model_validate(d)


def no_lint_style() -> Style:
    return Style.model_validate(style_dict())


def exposure(body: list[str | None], windows: list[Window] | None = None, char: str = "ali") -> Exposure:
    n = len(body)
    return Exposure(shot="sh01", length=n, body={char: body}, eyes={char: ["open"] * n}, windows=windows or [])


def moves(hits: list[int], char: str = "ali") -> list[Window]:
    return [Window(f"{char}.body", i, "transition", h - 4, h, h) for i, h in enumerate(hits)]


def alternate(n: int, every: int) -> list[str | None]:
    return [("p1" if (i // every) % 2 == 0 else "p2") for i in range(n)]


def codes(findings: list) -> list[tuple[str, str, str]]:  # type: ignore[type-arg]
    return [(f.level, f.code, f.path) for f in findings]


def test_without_lint_group_nothing_runs() -> None:
    exp = exposure(["p1"] * 200, moves([10, 20, 30, 40, 50]))
    assert style_lint(exp, no_lint_style(), {"ali": [("L01", 0, 60)]}, {}) == []


def test_pose_too_many() -> None:
    exp = exposure(alternate(100, 10), moves([10, 20, 30, 40, 50]))
    speaking = {"ali": [("L01", 0, 60)]}
    f = style_lint(exp, lint_style(pose_per_line_max=4), speaking, {})
    assert codes(f) == [("warn", "pose.too_many", "tracks.ali.body")]
    assert "5" in f[0].message and "L01" in f[0].message
    assert style_lint(exp, lint_style(pose_per_line_max=5), speaking, {}) == []
    # moves outside the line are not counted
    assert style_lint(exp, lint_style(pose_per_line_max=4), {"ali": [("L01", 0, 45)]}, {}) == []


def test_hand_static() -> None:
    style = lint_style(hand_static_max=48)
    f = style_lint(exposure(["p1"] * 60), style, {}, {})
    assert codes(f) == [("warn", "hand.static", "tracks.ali.body")]
    assert "60" in f[0].message and "p1" in f[0].message
    assert style_lint(exposure(["p1"] * 48), style, {}, {}) == []
    assert style_lint(exposure([None] * 100), style, {}, {}) == []  # no drawing is not a held drawing
    two = style_lint(exposure(["p1"] * 50 + ["p2"] * 10 + ["p3"] * 50), style, {}, {})
    assert [f.code for f in two] == ["hand.static", "hand.static"]


def test_mostly_twos_from_body_changes() -> None:
    style = lint_style(twos_share_warn=0.8)
    speaking = {"ali": [("L01", 0, 40)]}
    f = style_lint(exposure(alternate(40, 2)), style, speaking, {})
    assert codes(f) == [("warn", "cadence.mostly_twos", "tracks.ali.body")]
    assert style_lint(exposure(alternate(40, 3)), style, speaking, {}) == []
    # too few changes to judge (fewer than 8 inner intervals)
    assert style_lint(exposure(alternate(16, 2)), style, {"ali": [("L01", 0, 16)]}, {}) == []
    # not speaking: not judged
    assert style_lint(exposure(alternate(40, 2)), style, {}, {}) == []


def test_mostly_twos_counts_the_mouth() -> None:
    style = lint_style(twos_share_warn=0.8)
    speaking = {"ali": [("L01", 0, 40)]}
    mouth = {"ali": [("A" if (i // 2) % 2 == 0 else "D") for i in range(40)]}
    f = style_lint(exposure(["p1"] * 40), style, speaking, mouth)
    assert codes(f) == [("warn", "cadence.mostly_twos", "tracks.ali.body")]
    mouth3 = {"ali": [("A" if (i // 3) % 2 == 0 else "D") for i in range(40)]}
    assert style_lint(exposure(["p1"] * 40), style, speaking, mouth3) == []


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    ep = prod / "ep"
    (ep / "shots" / "sh01").mkdir(parents=True)
    (ep / "audio").mkdir()
    d = style_dict()
    d["lint"] = LINT
    (prod / "style.yaml").write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    for name in ("poses.yaml", "xsheet.yaml"):
        shutil.copy(FIX / name, ep / "shots" / "sh01" / name)
    for k, v in ALIGNS.items():
        (ep / "audio" / f"{k}.align.json").write_text(json.dumps(align(k, v).model_dump()), encoding="utf-8")
    return ep


def test_check_timing_runs_style_lint(episode: Path) -> None:
    # fixture: ali holds p1 on frames 0..60 (61 frames) before the move to p2; the existing blink warning stays
    f = check_timing(episode)
    assert [(x.code, x.path) for x in f if x.level == "warn"] == [
        ("hand.static", "tracks.ali.body"),
        ("timing.blink_in_transition", "tracks.ali.eyes[0]"),
    ]
    hs = next(x for x in f if x.code == "hand.static")
    assert "61" in hs.message and hs.file == "shots/sh01/xsheet.yaml"
