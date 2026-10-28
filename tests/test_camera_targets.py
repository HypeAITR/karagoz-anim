"""Acceptance tests for WP-14b (camera targets from the drawing at the event, SAM face box). Maintainer's; do not edit.

Fixture timing (WP-02/WP-03 tests): shot sh01 = [0, 100); L01 = [8, 48); L02 = [60, 88).
Ali sits until frame 59 and stands from frame 60 (his box and face move up 400 px); a second character, Ayse,
stays put.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw
from test_camera import BOXES, approx_box, ev, style, timing, track
from test_render_placement import setup_placement
from test_shot import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.engine import shot as shot_module
from karagoz.engine.camera import FACE_PART, Box, camera_track, framing_box
from karagoz.engine.compose import ShotAssets, character_bboxes
from karagoz.engine.framing import frame_targets
from karagoz.engine.placement import DrawingPlacement
from karagoz.engine.shot import render_shot
from karagoz.engine.xsheet import Exposure

SIT = {"ali": (800, 600, 1000, 1000), "ali.face": (850, 620, 950, 700), "ayse": (1300, 300, 1500, 1000)}
STAND = {"ali": (800, 200, 1000, 1000), "ali.face": (850, 220, 950, 300), "ayse": (1300, 300, 1500, 1000)}
TARGETS: list[dict[str, Any]] = [dict(SIT) for _ in range(60)] + [dict(STAND) for _ in range(40)]


def run(
    events: list[Any], targets: list[dict[str, Any]] | None = TARGETS, framing: str = "wide"
) -> tuple[list[Box], list[str]]:
    s = style()
    return camera_track(events, framing, "sh01", 100, timing(s), s, BOXES, frame_targets=targets)


# --- camera.py -----------------------------------------------------------------------------------------------------


def test_without_frame_targets_nothing_changes() -> None:
    events = [ev(at="L01.start", hold="medium", target="ali.face"), ev(at="L02.start", move="pan", target="ayse",
              until="L02.end"), ev(at="L02.end", move="shake")]  # fmt: skip
    assert run(events, None) == track(events)


def test_face_target_follows_the_standing_character() -> None:
    boxes, warn = run([ev(at="L02.start", hold="medium_close", target="ali.face")])
    assert boxes[59] == Box(0, 0, 1920, 1080)
    assert approx_box(boxes[60], framing_box("medium_close", STAND["ali.face"])) and boxes[99] == boxes[60]
    assert warn == []


def test_body_target_at_the_event_frame() -> None:
    sitting, _ = run([ev(at="L01.start", hold="medium", target="ali")])  # frame 8: still sitting
    assert approx_box(sitting[8], framing_box("medium", SIT["ali"]))
    standing, _ = run([ev(at="L02.start", hold="medium", target="ali")])
    assert approx_box(standing[60], framing_box("medium", STAND["ali"]))


def test_move_target_at_its_last_frame() -> None:
    targets = [dict(SIT) for _ in range(87)] + [dict(STAND) for _ in range(13)]  # stands up at frame 87
    boxes, _ = run([ev(at="L02.start", move="push_in", target="ali.face", until="L02.end")], targets)
    assert approx_box(boxes[87], framing_box("medium", STAND["ali.face"]))  # end = 88 -> resolved at frame 87


def test_face_missing_uses_the_top_of_that_frames_body() -> None:
    targets = [{"ali": STAND["ali"], "ayse": STAND["ayse"]} for _ in range(100)]
    boxes, _ = run([ev(at="L02.start", hold="medium", target="ali.face")], targets)
    x0, y0, x1, y1 = STAND["ali"]
    assert approx_box(boxes[60], framing_box("medium", (x0, y0, x1, y0 + round(FACE_PART * (y1 - y0)))))


def test_no_target_is_the_union_of_bodies_at_the_frame() -> None:
    boxes, _ = run([ev(at="L02.start", hold="medium")])
    assert approx_box(boxes[60], framing_box("medium", (800, 200, 1500, 1000)))


def test_character_off_screen_at_the_frame_falls_back() -> None:
    targets = [{"ayse": SIT["ayse"]} for _ in range(100)]
    boxes, warn = run([ev(at="L02.start", hold="medium", target="ali")], targets)
    assert approx_box(boxes[60], framing_box("medium", BOXES["ali"])) and warn == []
    _, warn = run([ev(at="L02.start", hold="medium", target="veli")], targets)
    assert warn == ["unknown camera target 'veli' at 'L02.start', using all characters"]


# --- framing.py ----------------------------------------------------------------------------------------------------


def figure() -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((150, 100, 250, 550), fill=(200, 120, 80), outline=(0, 0, 0), width=6)
    return img


def lm(face: list[int] | None, size: list[int] | None = None) -> dict[str, Any]:
    return {"version": 1, "image": "x.png", "size": size or [400, 600], "facing": "right", "head": None,
            "face": None if face is None else {"box": face, "confidence": 0.9}, "feet": [], "feet_source": None,
            "foot_low": None, "pants": None, "hip": None, "hands": [], "cups": []}  # fmt: skip


def place(x: int, y: int, mirror: bool = False) -> DrawingPlacement:
    return DrawingPlacement(scale=0.5, x=x, y=y, mirror=mirror, anchor="hip", occluders=(), depth=900)


@pytest.fixture
def assets(tmp_path: Path) -> ShotAssets:
    root = tmp_path / "assets" / "sh01"
    (root / "ali").mkdir(parents=True)
    for name in ("p1", "p2", "p1-p2.1"):
        figure().save(root / "ali" / f"{name}.png")
    (root / "ali" / "p1.landmarks.json").write_text(json.dumps(lm([150, 100, 230, 200])), encoding="utf-8")
    (root / "ali" / "p2.landmarks.json").write_text(json.dumps(lm([300, 200, 460, 400], [800, 1200])), "utf-8")
    return ShotAssets(root)


def exposure(body: list[str | None]) -> Exposure:
    return Exposure(shot="sh01", length=len(body), body={"ali": body}, eyes={"ali": ["open"] * len(body)}, windows=[])


def test_frame_targets_body_and_face(assets: ShotAssets) -> None:
    pls = {"ali": {"p1": place(100, 50), "p2": place(100, 300), "p1-p2.1": place(100, 200)}}
    exp = exposure(["p1", "p1-p2.1", "p2", None])
    ft = frame_targets(exp, assets, ["ali"], pls)
    assert len(ft) == 4
    first = character_bboxes(exp, assets, ["ali"], placements=pls)["ali"]
    assert ft[0]["ali"] == first
    assert ft[0]["ali.face"] == (175, 100, 215, 150)  # (150, 100, 230, 200) * 0.5 + (100, 50)
    assert "ali.face" not in ft[1]  # p1-p2.1 has no landmarks
    assert ft[1]["ali"][1] == first[1] + 150
    assert ft[2]["ali.face"] == (175, 350, 215, 400)  # measured on an 800x1200 copy: halved first
    assert ft[3] == {}


def test_frame_targets_mirrored(assets: ShotAssets) -> None:
    pls = {"ali": {"p1": place(100, 50, mirror=True)}}
    ft = frame_targets(exposure(["p1"]), assets, ["ali"], pls)
    assert ft[0]["ali.face"] == (185, 100, 225, 150)  # x (150, 230) -> (170, 250) on the mirrored 400 px drawing


def test_frame_targets_missing_inbetween_uses_the_render_fallback(assets: ShotAssets) -> None:
    (assets.root / "ali" / "p1-p2.1.png").unlink()
    pls = {"ali": {"p1": place(100, 50), "p2": place(100, 300)}}
    ft = frame_targets(exposure(["p1", "p1-p2.1", "p2"]), assets, ["ali"], pls)
    assert ft[1] == ft[2]  # one inbetween: k = 1 > n / 2 -> p2, as the render does


# --- shot.py -------------------------------------------------------------------------------------------------------


def capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = shot_module.camera_track

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(shot_module, "camera_track", spy)
    return calls


def test_render_passes_frame_targets_in_placement_mode(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    setup_placement(prod)
    calls = capture(monkeypatch)
    r = render_shot(prod, EP, "sh01")
    ft = calls[0]["frame_targets"]
    assert len(ft) == r.frames and "ali" in ft[0] and "ali.face" in ft[0]


def test_render_without_placement_is_unchanged(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    calls = capture(monkeypatch)
    render_shot(prod, EP, "sh01")
    assert calls[0].get("frame_targets") is None


def test_xsheet_face_event_renders(prod: Path) -> None:  # noqa: F811
    setup_placement(prod)
    xs_path = prod / EP / "shots" / "sh01" / "xsheet.yaml"
    xs = yaml.safe_load(xs_path.read_text(encoding="utf-8"))
    xs["camera"] = [{"at": "L02.start", "hold": "close", "target": "ali.face"}]
    xs_path.write_text(yaml.safe_dump(xs, allow_unicode=True), encoding="utf-8")
    r = render_shot(prod, EP, "sh01")
    assert not any("camera" in w for w in r.warnings)
