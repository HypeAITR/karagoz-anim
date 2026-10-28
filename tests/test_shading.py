"""Acceptance tests for WP-21 (contact shadow + ambient tone, placement mode). Written by the maintainer; do not edit.

One 400x600 figure (150..250 x 100..550) placed at scale 1 with its top-left at (500, 300): on screen it spans
x 650..750, y 400..850. Its feet box (150, 500, 250, 550) is (650, 800, 750, 850) on screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from karagoz.engine.compose import ShotAssets, render_frames
from karagoz.engine.placement import DrawingPlacement
from karagoz.engine.shading import (
    AMBIENT,
    SHADOW_FLAT,
    SHADOW_OPACITY,
    SHADOW_WIDTH,
    ambient_tint,
    apply_shadow,
    background_tint,
    shadow_alpha,
)
from karagoz.engine.xsheet import Exposure

GREY = (128, 128, 128)
WHITE_SHIRT = (250, 250, 250)


def figure() -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 250, 550), fill=WHITE_SHIRT, outline=(0, 0, 0), width=6)
    return img


def landmarks(feet: list[list[int]]) -> dict[str, object]:
    return {"version": 1, "image": "x.png", "size": [400, 600], "facing": "right", "head": None, "face": None,
            "feet": [{"box": b, "confidence": 0.9} for b in feet], "feet_source": "foot", "foot_low": None,
            "pants": None, "hip": None, "hands": [], "cups": []}  # fmt: skip


def shot(tmp_path: Path, bg: tuple[int, int, int], feet: list[list[int]] | None) -> ShotAssets:
    root = tmp_path / "assets" / "sh01"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), bg).save(root / "background.png")
    figure().save(root / "ali" / "p1.png")
    if feet is not None:
        (root / "ali" / "p1.landmarks.json").write_text(json.dumps(landmarks(feet)), encoding="utf-8")
    return ShotAssets(root)


def render(assets: ShotAssets, out: Path, occluders: dict[str, np.ndarray] | None = None) -> np.ndarray:
    exp = Exposure(shot="sh01", length=2, body={"ali": ["p1", "p1"]}, eyes={"ali": ["open", "open"]}, windows=[])
    occ = tuple(occluders) if occluders else ()
    pl = DrawingPlacement(scale=1.0, x=500, y=300, mirror=False, anchor="foot", occluders=occ, depth=900)
    render_frames(exp, {}, assets, ["ali"], out, placements={"ali": {"p1": pl}}, occluders=occluders)
    assert (out / "00000.png").read_bytes() == (out / "00001.png").read_bytes()  # the hold is still one frame
    return np.array(Image.open(out / "00000.png").convert("RGB")).astype(int)


# --- shading.py ----------------------------------------------------------------------------------------------------


def test_constants() -> None:
    assert (SHADOW_OPACITY, SHADOW_WIDTH, SHADOW_FLAT, AMBIENT) == (0.35, 0.6, 0.12, 0.12)


def test_background_tint() -> None:
    assert background_tint(Image.new("RGB", (10, 10), (200, 100, 60))) == pytest.approx((200 / 120, 100 / 120, 0.5))
    assert background_tint(Image.new("RGB", (10, 10), (0, 0, 0))) == (1.0, 1.0, 1.0)


def test_ambient_tint() -> None:
    img = Image.new("RGBA", (2, 1), (200, 200, 200, 255))
    img.putpixel((1, 0), (200, 200, 200, 0))
    out = np.array(ambient_tint(img, (1.2, 1.0, 0.8)))
    assert tuple(out[0, 0]) == (205, 200, 195, 255)  # 200 * (0.88 + 0.12 * 1.2) = 204.8
    assert out[0, 1, 3] == 0


def test_shadow_alpha_and_apply() -> None:
    a = shadow_alpha((200, 100), (100.0, 50.0), 60.0, 12.0)
    assert a.shape == (100, 200) and a.dtype == np.float32
    assert a[50, 100] > 0.9 and a[50, 199] == 0 and a[0, 100] < 0.01
    assert a[50, 100] == pytest.approx(a[50, 100 - 30], abs=0.1)  # flat inside the long axis
    out = np.array(apply_shadow(Image.new("RGB", (200, 100), (100, 100, 100)), a))
    assert out[50, 100, 0] == round(100 * (1 - 0.35 * float(a[50, 100])))
    assert tuple(out[0, 0]) == (100, 100, 100)


# --- compose.py ----------------------------------------------------------------------------------------------------


def test_shadow_under_the_feet(tmp_path: Path) -> None:
    f = render(shot(tmp_path, GREY, [[150, 500, 250, 550]]), tmp_path / "frames")
    assert f[853, 700, 0] < 115  # just under the soles: darker (rx = 60, ry = 7.2)
    assert tuple(f[875, 700]) == GREY and tuple(f[850, 900]) == GREY  # far below / far to the side: untouched
    assert tuple(f[600, 700]) == WHITE_SHIRT  # grey background: the character keeps its colours


def test_no_feet_no_shadow(tmp_path: Path) -> None:
    f = render(shot(tmp_path, GREY, []), tmp_path / "frames")
    assert tuple(f[853, 700]) == GREY
    f = render(shot(tmp_path / "b", GREY, None), tmp_path / "frames_b")  # no landmarks at all
    assert tuple(f[853, 700]) == GREY


def test_object_in_front_hides_the_shadow(tmp_path: Path) -> None:
    table = np.zeros((1080, 1920), dtype=bool)
    table[851:870, 680:720] = True
    f = render(shot(tmp_path, GREY, [[150, 500, 250, 550]]), tmp_path / "frames", {"sehpa": table})
    assert tuple(f[853, 700]) == GREY  # behind the table
    assert f[853, 660, 0] < 128  # beside the table the shadow shows


def test_ambient_tone_follows_the_room(tmp_path: Path) -> None:
    f = render(shot(tmp_path, (200, 100, 60), None), tmp_path / "frames")
    r, g, b = (int(v) for v in f[600, 700])
    assert r > g > b  # the white shirt takes the warm room light
    assert (r, g, b) == (min(255, round(250 * (0.88 + 0.12 * 200 / 120))), round(250 * (0.88 + 0.12 * 100 / 120)),
                         round(250 * (0.88 + 0.12 * 0.5)))  # fmt: skip
