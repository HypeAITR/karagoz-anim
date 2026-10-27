"""Acceptance tests for WP-06 (compositor). Written by the maintainer; do not edit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from karagoz.engine.compose import Placement, ShotAssets, compose, cutout, frame_layers, place_row, render_frames
from karagoz.engine.patch import Patch, save_patch
from karagoz.engine.xsheet import Exposure
from karagoz.spec import Mouth


def drawing(colour: tuple[int, int, int], arm: bool = False) -> Image.Image:
    """400x600 white canvas: a figure 100..300 x 100..550 with an enclosed white undershirt."""
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 250, 550), fill=colour, outline=(0, 0, 0), width=6)
    d.rectangle((170, 250, 230, 350), fill=(255, 255, 255), outline=(0, 0, 0), width=4)  # enclosed white
    if arm:
        d.rectangle((250, 260, 360, 290), fill=colour, outline=(0, 0, 0), width=4)
    return img


def test_cutout_keeps_enclosed_white() -> None:
    a = np.array(cutout(drawing((200, 120, 80))))
    assert a.shape == (600, 400, 4)
    assert a[10, 10, 3] == 0  # outside
    assert a[300, 200, 3] == 255 and tuple(a[300, 200, :3]) == (255, 255, 255)  # undershirt stays
    assert a[200, 200, 3] == 255


def test_cutout_tolerates_off_white_background() -> None:
    img = drawing((200, 120, 80))
    arr = np.array(img)
    arr[arr.sum(axis=2) == 765] = (247, 249, 246)  # model's not-quite-white
    a = np.array(cutout(Image.fromarray(arr)))
    assert a[10, 10, 3] == 0 and a[300, 200, 3] == 255


def test_place_row() -> None:
    first = [cutout(drawing((200, 120, 80))), cutout(drawing((90, 90, 200)))]
    p = place_row(first)
    # opaque part is 450 px tall (100..549) -> scale = 0.62*1080/450
    s = 0.62 * 1080 / 450
    assert p[0].scale == pytest.approx(s, rel=0.01)
    # bottom-centre of opaque part (x=200, y=550) -> (640, 1015) for i=0 of 2, (1280, 1015) for i=1
    assert p[0].x + 200 * p[0].scale == pytest.approx(640, abs=2)
    assert p[0].y + 550 * p[0].scale == pytest.approx(0.94 * 1080, abs=2)
    assert p[1].x + 200 * p[1].scale == pytest.approx(1280, abs=2)


def test_compose_order_and_clip() -> None:
    bg = Image.new("RGB", (1920, 1080), (30, 30, 30))
    a = cutout(drawing((255, 0, 0)))
    b = cutout(drawing((0, 0, 255)))
    out = compose(bg, [(a, Placement(1.0, 800, 200)), (b, Placement(1.0, 850, 200)), (a, Placement(1.0, 1800, 900))])
    px = np.array(out)
    assert out.size == (1920, 1080) and out.mode == "RGB"
    assert tuple(px[600, 1020]) == (0, 0, 255)  # b over a where they overlap (b fill, a fill underneath)
    assert tuple(px[5, 5]) == (30, 30, 30)


def test_compose_resizes_background_to_fill() -> None:
    bg = Image.new("RGB", (1376, 768), (10, 200, 10))
    assert compose(bg, []).size == (1920, 1080)


@pytest.fixture
def shot_dir(tmp_path: Path) -> Path:
    root = tmp_path / "assets" / "sh01"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (40, 40, 60)).save(root / "background.png")
    base = drawing((200, 120, 80))
    base.save(root / "ali" / "p1.png")
    drawing((200, 120, 80), arm=True).save(root / "ali" / "p2.png")
    mouth = Image.new("RGBA", (40, 20), (120, 20, 20, 255))
    save_patch(Patch(mouth, 180, 150), root / "ali" / "p1.mouth-D")
    eyes = Image.new("RGBA", (60, 10), (0, 0, 0, 255))
    save_patch(Patch(eyes, 170, 120), root / "ali" / "p1.eyes-closed")
    return root


def exposure() -> Exposure:
    body: list[str | None] = ["p1"] * 4 + ["p1-p2.1", "p1-p2.1", "p1-p2.2", "p1-p2.2"] + ["p2"] * 4
    eyes = ["open", "closed", "open"] + ["open"] * 9
    return Exposure(shot="sh01", length=12, body={"ali": body}, eyes={"ali": eyes}, windows=[])


def mouths() -> dict[str, Mouth]:
    return {
        "ali": Mouth.model_validate(
            {
                "version": 1,
                "shot": "sh01",
                "character": "ali",
                "fps": 24,
                "cues": [
                    {"start": 0, "end": 2, "shape": "X", "src": "auto"},
                    {"start": 2, "end": 4, "shape": "D", "src": "auto"},
                    {"start": 4, "end": 12, "shape": "X", "src": "auto"},
                ],
            }
        )
    }


def test_frame_layers_patches_and_fallback(shot_dir: Path) -> None:
    assets = ShotAssets(shot_dir)
    pl = {"ali": Placement(1.0, 0, 0)}
    layers, warn = frame_layers(exposure(), mouths(), assets, ["ali"], pl, 1)
    img = np.array(layers[0][0])
    assert tuple(img[125, 200, :3]) == (0, 0, 0)  # eyes-closed patch applied
    layers, _ = frame_layers(exposure(), mouths(), assets, ["ali"], pl, 2)
    assert tuple(np.array(layers[0][0])[160, 200, :3]) == (120, 20, 20)  # mouth D
    _, warn = frame_layers(exposure(), mouths(), assets, ["ali"], pl, 5)
    assert warn and "p1-p2.1" in warn[0] and "p1" in warn[0]  # missing inbetween 1 of 2 -> p1
    layers, warn = frame_layers(exposure(), mouths(), assets, ["ali"], pl, 6)
    assert np.array_equal(np.array(layers[0][0]), np.array(assets.drawing("ali", "p2")))  # 2 of 2 -> p2


def test_render_frames(shot_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "frames"
    warn = render_frames(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], out)
    files = sorted(out.iterdir())
    assert [f.name for f in files] == [f"{i:05d}.png" for i in range(12)]
    assert Image.open(files[0]).size == (1920, 1080)
    assert files[0].read_bytes() != files[1].read_bytes()  # blink
    assert files[9].read_bytes() == files[10].read_bytes()  # hold
    assert len(warn) == len(set(warn)) and any("p1-p2.2" in w for w in warn)
    out2 = tmp_path / "frames2"
    render_frames(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], out2)
    assert all((out / n.name).read_bytes() == (out2 / n.name).read_bytes() for n in files)
