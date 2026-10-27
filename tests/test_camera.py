"""Acceptance tests for WP-14 (the render follows the camera track). Written by the maintainer; do not edit.

Fixture timing (see WP-02/WP-03 tests): shot sh01 = [0, 100); L01 = [8, 48); L02 = [60, 88).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from karagoz.engine.camera import (
    COVER,
    FRAMING_ORDER,
    HEADROOM,
    MAX_ZOOM,
    Box,
    camera_track,
    ease,
    framing_box,
    full_frame,
    target_bbox,
)
from karagoz.engine.compose import ShotAssets, character_bboxes, render_frames
from karagoz.engine.shot import render_shot
from karagoz.engine.timing import EpisodeTiming, resolve_episode
from karagoz.engine.xsheet import Exposure
from karagoz.spec import Align, CameraEvent, Lines, Shots, Style, check_episode

FIX = Path(__file__).parent / "fixtures" / "spec"
FULL = Box(0, 0, 1920, 1080)
ALI = (800, 200, 1000, 1000)
AYSE = (1300, 300, 1500, 1000)
BOXES = {"ali": ALI, "ayse": AYSE}


def style(camera: bool = True) -> Style:
    d = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    if camera:
        d["live"]["camera"] = {
            "ease_min": {"value": 16, "source": "test"},
            "pan_max_px": {"value": 11.4, "source": "test"},
            "shake": {"value": 12, "source": "test"},
            "overscan": {"value": 0.016, "source": "test"},
        }
    return Style.model_validate(d)


def align(line_id: str, text: str) -> Align:
    n = len(text)
    return Align.model_validate(
        {"version": 1, "line": line_id, "text": text, "characters": list(text),
         "starts": [round(i * 0.1, 6) for i in range(n)], "ends": [round((i + 1) * 0.1, 6) for i in range(n)]}
    )  # fmt: skip


def timing(st: Style | None = None) -> EpisodeTiming:
    lines = Lines.model_validate(yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8")))
    shots = Shots.model_validate(yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8")))
    aligns = {"L01": align("L01", "Yarın pazar, Ali."), "L02": align("L02", "Su ver bana.")}
    return resolve_episode(lines, shots, aligns, st or style())


def ev(**kw: object) -> CameraEvent:
    return CameraEvent.model_validate(kw)


def track(
    events: list[CameraEvent] | None, framing: str = "wide", st: Style | None = None
) -> tuple[list[Box], list[str]]:
    s = st or style()
    return camera_track(events, framing, "sh01", 100, timing(s), s, BOXES)


def inside(b: Box) -> bool:
    return b.x >= -1e-6 and b.y >= -1e-6 and b.x + b.w <= 1920 + 1e-6 and b.y + b.h <= 1080 + 1e-6


def approx_box(a: Box, b: Box, tol: float = 0.01) -> bool:
    return all(abs(p - q) <= tol for p, q in zip((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h), strict=True))


# --- presets -------------------------------------------------------------------------------------------------------


def test_constants() -> None:
    assert FRAMING_ORDER == ["wide", "medium", "medium_close", "close", "extreme_close"]
    assert COVER["close"] == 0.28 and HEADROOM == 0.10 and MAX_ZOOM == 3.0


def test_wide_and_no_target_are_full_frame() -> None:
    assert full_frame() == FULL
    assert framing_box("wide", ALI) == FULL
    assert framing_box("close", None) == FULL


def test_medium_box_from_target() -> None:
    target = (800, 100, 1100, 1000)  # 300 x 900
    b = framing_box("medium", target)
    h = 0.55 * 900 / 0.9
    w = h * 1920 / 1080
    assert approx_box(b, Box(950 - w / 2, 100 - 0.1 * h, w, h))
    assert b.w / b.h == pytest.approx(16 / 9)


def test_close_is_limited_by_max_zoom() -> None:
    b = framing_box("close", ALI)  # 0.28 * 800 / 0.9 = 249 px high -> wider than 3x zoom allows
    assert approx_box(b, Box(900 - 320, 200 - 36, 640, 360))


def test_box_is_shifted_inside_the_frame() -> None:
    b = framing_box("close", (1800, 0, 1920, 1000))
    assert approx_box(b, Box(1280, 0, 640, 360))
    assert inside(b)


def test_too_large_box_becomes_full_frame() -> None:
    assert framing_box("medium", (0, -500, 1920, 1600)) == FULL


def test_target_bbox() -> None:
    assert target_bbox(None, BOXES) == (800, 200, 1500, 1000)
    assert target_bbox(None, {}) is None
    assert target_bbox("ayse", BOXES) == AYSE
    assert target_bbox("ali.face", BOXES) == (800, 200, 1000, 200 + 240)
    with pytest.raises(KeyError):
        target_bbox("veli", BOXES)


def test_ease() -> None:
    assert ease(0.5, "linear") == 0.5
    assert ease(0.25, "slow") == pytest.approx(3 * 0.25**2 - 2 * 0.25**3)
    assert ease(0.25, None) == ease(0.25, "slow")
    assert ease(0.5, "fast") == pytest.approx(0.75)
    assert ease(-1, "linear") == 0 and ease(2, "fast") == 1


# --- track ---------------------------------------------------------------------------------------------------------


def test_no_events_follow_the_shot_framing() -> None:
    boxes, warnings = track(None)
    assert len(boxes) == 100 and set(boxes) == {FULL} and warnings == []
    boxes, _ = track(None, framing="close")
    assert set(boxes) == {framing_box("close", (800, 200, 1500, 1000))}


def test_hold_at_shot_in_wins_over_shot_framing() -> None:
    boxes, _ = track([ev(at="shot.in", hold="wide")], framing="close")
    assert set(boxes) == {FULL}


def test_hold_cuts_at_its_frame() -> None:
    boxes, _ = track([ev(at="shot.in", hold="wide"), ev(at="L01.start", hold="close", target="ali")])
    assert set(boxes[:8]) == {FULL}
    assert set(boxes[8:]) == {framing_box("close", ALI)}


def test_push_in_eases_to_the_next_closer_framing() -> None:
    boxes, _ = track([ev(at="shot.in", hold="wide"), ev(at="shot.in", move="push_in", target="ali", until="L01.start")])
    end = framing_box("medium", ALI)
    assert approx_box(boxes[7], end) and all(approx_box(b, end) for b in boxes[8:])
    widths = [b.w for b in boxes[:8]]
    assert widths[0] < 1920 and all(a > b for a, b in zip(widths, widths[1:], strict=False))
    assert all(inside(b) for b in boxes)


def test_push_in_without_until_uses_ease_min() -> None:
    boxes, _ = track([ev(at="shot.in", move="push_in", target="ali")])
    end = framing_box("medium", ALI)
    assert not approx_box(boxes[14], end) and approx_box(boxes[15], end)
    st = style(camera=False)  # no camera group: same default
    boxes2, _ = track([ev(at="shot.in", move="push_in", target="ali")], st=st)
    assert approx_box(boxes2[15], end)


def test_two_push_ins_step_through_the_framings() -> None:
    events = [ev(at="shot.in", move="push_in", target="ali", until="shot.in + 4f"),
              ev(at="shot.in + 10f", move="push_in", target="ali", until="shot.in + 14f")]  # fmt: skip
    boxes, _ = track(events)
    assert approx_box(boxes[9], framing_box("medium", ALI))
    assert approx_box(boxes[13], framing_box("medium_close", ALI))


def test_pull_out_at_wide_warns() -> None:
    boxes, warnings = track([ev(at="shot.in", move="pull_out")])
    assert set(boxes) == {FULL}
    assert any("pull_out" in w for w in warnings)


def test_pull_out_from_close() -> None:
    boxes, _ = track([ev(at="shot.in", hold="close", target="ali"), ev(at="L01.start", move="pull_out", target="ali")])
    assert approx_box(boxes[0], framing_box("close", ALI))
    assert approx_box(boxes[8 + 15], framing_box("medium_close", ALI))


def test_pan_keeps_size_and_moves_to_target() -> None:
    events = [
        ev(at="shot.in", hold="close", target="ali"),
        ev(at="L01.start", move="pan", target="ayse", until="L01.end"),
    ]
    boxes, _ = track(events)
    start, end = boxes[0], boxes[47]
    assert end.w == pytest.approx(start.w) and end.h == pytest.approx(start.h) and end.y == pytest.approx(start.y)
    assert end.x + end.w / 2 == pytest.approx(1400)


def test_fast_pan_warns() -> None:
    events = [
        ev(at="shot.in", hold="close", target="ali"),
        ev(at="L01.start", move="pan", target="ayse", until="L01.start + 4f"),
    ]
    _, warnings = track(events)
    assert any("pan" in w for w in warnings)


def test_shake_moves_then_returns() -> None:
    boxes, _ = track([ev(at="shot.in", hold="wide"), ev(at="L01.start", move="shake")])
    assert set(boxes[:8]) == {FULL} and set(boxes[8 + 12 :]) == {FULL}
    shaken = boxes[8 : 8 + 12]
    assert all(b.w == pytest.approx(1920 * (1 - 0.016)) for b in shaken)
    assert len({(round(b.x, 3), round(b.y, 3)) for b in shaken}) > 3
    assert all(inside(b) for b in shaken)


def test_unknown_target_warns_and_uses_everyone() -> None:
    boxes, warnings = track([ev(at="shot.in", hold="close", target="veli")])
    assert set(boxes) == {framing_box("close", (800, 200, 1500, 1000))}
    assert any("veli" in w for w in warnings)


def test_deterministic() -> None:
    events = [ev(at="shot.in", move="push_in", target="ali"), ev(at="L01.start", move="shake")]
    assert track(events) == track(events)


# --- compositor ----------------------------------------------------------------------------------------------------


def figure(colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((150, 100, 250, 550), fill=colour, outline=(0, 0, 0), width=6)
    return img


def assets(tmp_path: Path) -> ShotAssets:
    root = tmp_path / "a"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (40, 40, 60)).save(root / "background.png")
    figure((200, 120, 80)).save(root / "ali" / "p1.png")
    return ShotAssets(root)


def test_character_bboxes_match_the_render(tmp_path: Path) -> None:
    a = assets(tmp_path)
    exp = Exposure(shot="sh01", length=1, body={"ali": ["p1"]}, eyes={"ali": ["open"]}, windows=[])
    boxes = character_bboxes(exp, a, ["ali"], scales={"ali": 1.0})
    render_frames(exp, {}, a, ["ali"], tmp_path / "f", scales={"ali": 1.0})
    px = np.array(Image.open(tmp_path / "f" / "00000.png").convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(px - np.array([40, 40, 60])).sum(axis=2) > 30)
    x0, y0, x1, y1 = boxes["ali"]
    assert abs(x0 - xs.min()) <= 1 and abs(x1 - (xs.max() + 1)) <= 1
    assert abs(y0 - ys.min()) <= 1 and abs(y1 - (ys.max() + 1)) <= 1


def test_full_frame_camera_writes_the_same_bytes(tmp_path: Path) -> None:
    a = assets(tmp_path)
    exp = Exposure(shot="sh01", length=2, body={"ali": ["p1", "p1"]}, eyes={"ali": ["open", "open"]}, windows=[])
    render_frames(exp, {}, a, ["ali"], tmp_path / "f1")
    render_frames(exp, {}, a, ["ali"], tmp_path / "f2", camera=[FULL, FULL])
    for name in ("00000.png", "00001.png"):
        assert (tmp_path / "f1" / name).read_bytes() == (tmp_path / "f2" / name).read_bytes()


def test_camera_box_crops_and_scales(tmp_path: Path) -> None:
    a = assets(tmp_path)
    exp = Exposure(shot="sh01", length=2, body={"ali": ["p1", "p1"]}, eyes={"ali": ["open", "open"]}, windows=[])
    bb = character_bboxes(exp, a, ["ali"])["ali"]
    close = framing_box("medium", bb)
    render_frames(exp, {}, a, ["ali"], tmp_path / "f", camera=[FULL, close])

    def body_width(name: str) -> int:
        px = np.array(Image.open(tmp_path / "f" / name).convert("RGB")).astype(int)
        xs = np.nonzero((np.abs(px - np.array([200, 120, 80])).sum(axis=2) < 30).any(axis=0))[0]
        return int(xs.max() - xs.min())

    img = Image.open(tmp_path / "f" / "00001.png")
    assert img.size == (1920, 1080)
    zoom = 1920 / close.w
    assert body_width("00001.png") == pytest.approx(zoom * body_width("00000.png"), rel=0.05)  # closer camera


# --- render_shot + check -------------------------------------------------------------------------------------------


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    p = tmp_path / "prod"
    ep = p / "ep"
    a = ep / "assets" / "sh01"
    for d in (ep / "shots" / "sh01", ep / "audio", a / "ali", a / "ayse", p / "referans"):
        d.mkdir(parents=True)
    st = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    st["live"]["transition"]["anticipate"] = {"value": 4, "source": "test"}
    st["live"]["blink"].update({"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}})
    (p / "style.yaml").write_text(yaml.safe_dump(st, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    shutil.copy(FIX / "poses.yaml", ep / "shots" / "sh01" / "poses.yaml")
    xs = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    xs["tracks"]["ayse.body"] = [{"at": "shot.in", "pose": "p1"}]
    xs["camera"] = [{"at": "shot.in", "hold": "wide"}, {"at": "L02.start", "hold": "close", "target": "ali"}]
    (ep / "shots" / "sh01" / "xsheet.yaml").write_text(yaml.safe_dump(xs, allow_unicode=True), encoding="utf-8")
    for lid, text in (("L01", "Yarın pazar, Ali."), ("L02", "Su ver bana.")):
        n = len(text)
        (ep / "audio" / f"{lid}.align.json").write_text(align(lid, text).model_dump_json(), encoding="utf-8")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d={n * 0.1}", "-q:a", "4",
                        str(ep / "audio" / f"{lid}.mp3")], check=True)  # fmt: skip
    Image.new("RGB", (1920, 1080), (60, 60, 80)).save(a / "background.png")
    for pose in ("p1", "p2"):
        figure((200, 120, 80)).save(a / "ali" / f"{pose}.png")
    figure((90, 160, 90)).save(a / "ayse" / "p1.png")
    figure((200, 120, 80)).save(p / "referans" / "ali.png")
    figure((90, 160, 90)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }\n",
        encoding="utf-8",
    )
    return p


def test_render_shot_cuts_to_close(prod: Path) -> None:
    render_shot(prod, "ep", "sh01")
    d = prod / "ep" / "render" / "sh01" / "frames"

    def ali_width(name: str) -> int:
        px = np.array(Image.open(d / name).convert("RGB")).astype(int)
        xs = np.nonzero((np.abs(px - np.array([200, 120, 80])).sum(axis=2) < 30).any(axis=0))[0]
        return int(xs.max() - xs.min()) if xs.size else 0

    assert ali_width("00059.png") > 0 and ali_width("00060.png") > 2.5 * ali_width("00059.png")  # 3x close-up
    green = np.array(Image.open(d / "00090.png").convert("RGB")).astype(int)
    assert not ((np.abs(green - np.array([90, 160, 90])).sum(axis=2) < 30).any())  # ayse is out of the close-up


def test_check_unknown_camera_target(prod: Path) -> None:
    xpath = prod / "ep" / "shots" / "sh01" / "xsheet.yaml"
    assert not [f for f in check_episode(prod / "ep") if f.code == "camera.unknown_target"]
    xpath.write_text(xpath.read_text(encoding="utf-8").replace("target: ali", "target: veli.face"), encoding="utf-8")
    found = [f for f in check_episode(prod / "ep") if f.code == "camera.unknown_target"]
    assert len(found) == 1 and found[0].level == "error" and found[0].path == "camera[1].target"
    assert found[0].file == "shots/sh01/xsheet.yaml"
