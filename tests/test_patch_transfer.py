"""Acceptance tests for WP-13 (patches in inbetweens, patches stay inside the character). Maintainer's; do not edit.

Figures as in the WP-06 tests: a 400x600 white canvas with a figure (150..250 x 100..550). The inbetween p1-p2.1 is
the same figure moved 20 px right; p1-p2.2 is moved 20 px right but its face box is far smaller (another head angle).
Only the key p1 has mouth and eye patches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from karagoz.engine.compose import Placement, ShotAssets, cutout, frame_layers, render_frames
from karagoz.engine.patch import Patch, apply_patch, clip_patch, save_patch, transfer_patch
from karagoz.engine.placement import DrawingPlacement
from karagoz.engine.xsheet import Exposure
from karagoz.spec import Mouth

SKIN = (200, 120, 80)
RED = (120, 20, 20)


def drawing(dx: int = 0) -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150 + dx, 100, 250 + dx, 550), fill=SKIN, outline=(0, 0, 0), width=6)
    d.rectangle((185 + dx, 158, 215 + dx, 162), fill=(0, 0, 0))  # closed mouth line (under the mouth patch)
    d.rectangle((175 + dx, 123, 225 + dx, 126), fill=(0, 0, 0))  # eye line (under the eyes patch)
    return img


def landmarks(face: list[int], size: tuple[int, int] = (400, 600)) -> dict[str, object]:
    return {"version": 1, "image": "x.png", "size": list(size), "facing": "right", "head": None,
            "face": {"box": face, "confidence": 0.9}, "feet": [], "feet_source": None, "foot_low": None,
            "pants": None, "hip": None, "hands": [], "cups": []}  # fmt: skip


@pytest.fixture
def shot_dir(tmp_path: Path) -> Path:
    root = tmp_path / "assets" / "sh01"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (40, 40, 60)).save(root / "background.png")
    drawing().save(root / "ali" / "p1.png")
    drawing(20).save(root / "ali" / "p1-p2.1.png")
    drawing(20).save(root / "ali" / "p1-p2.2.png")
    drawing(40).save(root / "ali" / "p2.png")
    save_patch(Patch(Image.new("RGBA", (40, 20), (*RED, 255)), 180, 150), root / "ali" / "p1.mouth-D")
    save_patch(Patch(Image.new("RGBA", (60, 10), (0, 0, 0, 255)), 170, 120), root / "ali" / "p1.eyes-closed")
    faces = {"p1": [150, 100, 250, 200], "p1-p2.1": [170, 100, 270, 200], "p1-p2.2": [200, 150, 230, 180]}
    for name, face in faces.items():
        (root / "ali" / f"{name}.landmarks.json").write_text(json.dumps(landmarks(face)), encoding="utf-8")
    return root


def exposure() -> Exposure:
    body: list[str | None] = ["p1"] * 4 + ["p1-p2.1", "p1-p2.1", "p1-p2.2", "p1-p2.2"] + ["p2"] * 4
    eyes = ["open"] * 5 + ["closed"] + ["open"] * 6
    return Exposure(shot="sh01", length=12, body={"ali": body}, eyes={"ali": eyes}, windows=[])


def mouths() -> dict[str, Mouth]:
    cues = [{"start": 0, "end": 4, "shape": "X", "src": "auto"}, {"start": 4, "end": 8, "shape": "D", "src": "auto"},
            {"start": 8, "end": 12, "shape": "X", "src": "auto"}]  # fmt: skip
    return {"ali": Mouth.model_validate({"version": 1, "shot": "sh01", "character": "ali", "fps": 24, "cues": cues})}


def rgb(img: Image.Image, x: int, y: int) -> tuple[int, ...]:
    return tuple(int(v) for v in np.array(img.convert("RGBA"))[y, x, :3])


# --- patch.py ------------------------------------------------------------------------------------------------------


def test_patch_never_paints_outside_the_character() -> None:
    base = cutout(drawing())  # transparent outside the figure
    white = Patch(Image.new("RGBA", (100, 40), (255, 255, 255, 255)), 100, 300)  # covers x 100..199
    out = np.array(apply_patch(base, white))
    b = np.array(base)
    assert out[310, 120, 3] == 0 and np.array_equal(out[310, 120], b[310, 120])  # background stays as it was
    assert tuple(out[310, 190]) == (255, 255, 255, 255)  # inside the figure the patch is applied
    assert np.array_equal(out[:, :, 3], b[:, :, 3])  # the alpha never changes


def test_opaque_base_unchanged_behaviour() -> None:
    base = drawing().convert("RGBA")
    out = np.array(apply_patch(base, Patch(Image.new("RGBA", (40, 20), (*RED, 128)), 180, 150)))
    assert tuple(out[152, 200, :3]) == tuple(
        round(r * 128 / 255 + s * (1 - 128 / 255)) for r, s in zip(RED, SKIN, strict=True)
    )
    assert out[152, 200, 3] == 255


def test_transfer_patch_moves_and_scales() -> None:
    p = Patch(Image.new("RGBA", (40, 20), (*RED, 255)), 180, 150)
    t = transfer_patch(p, (150, 100, 250, 200), (300, 300, 500, 500))  # centre (200, 150) -> (400, 400), s = 2
    assert (t.x, t.y, t.image.size) == (360, 400, (80, 40))
    same = transfer_patch(p, (150, 100, 250, 200), (170, 100, 270, 200))
    assert (same.x, same.y, same.image.size) == (200, 150, (40, 20))


def test_clip_patch() -> None:
    p = Patch(Image.new("RGBA", (40, 20), (*RED, 255)), 180, 150)
    c = clip_patch(p, (190.0, 100.0, 250.0, 165.5))
    a = np.array(c.image)[:, :, 3]
    assert (c.x, c.y, c.image.size) == (180, 150, (40, 20))
    assert a[0, 9] == 0 and a[0, 10] == 255  # x 189 out, 190 in
    assert a[15, 20] == 255 and a[16, 20] == 0  # y 165 in, 166 out (ceil(165.5) = 166 is the exclusive edge)


# --- compose.py ----------------------------------------------------------------------------------------------------


def test_landmarks_accessor(shot_dir: Path) -> None:
    a = ShotAssets(shot_dir)
    got = a.landmarks("ali", "p1")
    assert got is not None and got["face"]["box"] == [150, 100, 250, 200]
    assert a.landmarks("ali", "p2") is None


def test_inbetween_borrows_the_key_mouth(shot_dir: Path) -> None:
    layers, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 4)
    img = layers[0][0]
    assert rgb(img, 210, 160) == RED  # the p1 mouth, moved 20 px right with the face
    assert rgb(img, 195, 160) == SKIN  # where p1's mouth was, the inbetween has skin
    assert not any("mouth-D" in w for w in warn)


def test_inbetween_borrows_the_key_eyes(shot_dir: Path) -> None:
    layers, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 5)
    assert rgb(layers[0][0], 245, 125) == (0, 0, 0)  # eyes-closed patch moved to 190..250
    assert not any("eyes-closed" in w for w in warn)


def test_nothing_changes_outside_the_face(shot_dir: Path) -> None:
    a = ShotAssets(shot_dir)
    layers, _ = frame_layers(exposure(), mouths(), a, ["ali"], {"ali": Placement(1.0, 0, 0)}, 4)
    out = np.array(layers[0][0].convert("RGBA"))
    base = np.array(a.drawing("ali", "p1-p2.1").convert("RGBA"))
    changed = np.any(out != base, axis=2)
    ys, xs = np.nonzero(changed)
    assert xs.size > 0 and xs.min() >= 170 and xs.max() < 270 and ys.min() >= 100 and ys.max() < 200


def test_other_head_angle_is_not_patched(shot_dir: Path) -> None:
    layers, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 6)
    assert "f=6 ali p1-p2.2 mouth-D patch missing" in warn  # face 30 px against 100: s = 0.3
    assert rgb(layers[0][0], 210, 152) == SKIN


def test_no_landmarks_keeps_the_warning(shot_dir: Path) -> None:
    (shot_dir / "ali" / "p1-p2.1.landmarks.json").unlink()
    _, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 4)
    assert "f=4 ali p1-p2.1 mouth-D patch missing" in warn


def test_landmarks_of_another_size_are_rescaled(shot_dir: Path) -> None:
    lm = landmarks([340, 200, 540, 400], size=(800, 1200))  # the same face box measured on a 2x image
    (shot_dir / "ali" / "p1-p2.1.landmarks.json").write_text(json.dumps(lm), encoding="utf-8")
    layers, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 4)
    assert rgb(layers[0][0], 210, 160) == RED and not any("mouth-D" in w for w in warn)


def test_placement_render_uses_the_borrowed_mouth(shot_dir: Path, tmp_path: Path) -> None:
    pl = DrawingPlacement(scale=1.0, x=0, y=0, mirror=False, anchor="hip", occluders=(), depth=900)
    placements = {"ali": dict.fromkeys(["p1", "p1-p2.1", "p1-p2.2", "p2"], pl)}
    out = tmp_path / "frames"
    warn = render_frames(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], out, placements=placements)
    frame = Image.open(out / "00004.png")
    assert rgb(frame, 210, 160) == RED and rgb(frame, 195, 160) == SKIN
    assert not any("p1-p2.1 mouth-D" in w for w in warn)
    assert any("p1-p2.2 mouth-D patch missing" in w for w in warn)


def test_borrowed_patch_is_aligned_by_the_pixels(shot_dir: Path) -> None:
    # SAM put the inbetween's face box 5 px too far right; the mouth line under the patch pins the real place
    lm = landmarks([175, 100, 275, 200])
    (shot_dir / "ali" / "p1-p2.1.landmarks.json").write_text(json.dumps(lm), encoding="utf-8")
    layers, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 4)
    img = layers[0][0]
    assert rgb(img, 201, 155) == RED and rgb(img, 241, 155) == SKIN  # at x 200..239, not 205..244
    assert not any("mouth-D" in w for w in warn)


def test_drawing_that_does_not_match_the_key_is_not_patched(shot_dir: Path) -> None:
    other = drawing(20)
    ImageDraw.Draw(other).rectangle((170, 140, 270, 180), fill=(0, 0, 255))  # nothing like the key's mouth area
    other.save(shot_dir / "ali" / "p1-p2.1.png")
    layers, warn = frame_layers(exposure(), mouths(), ShotAssets(shot_dir), ["ali"], {"ali": Placement(1.0, 0, 0)}, 4)
    assert "f=4 ali p1-p2.1 mouth-D patch missing" in warn
    assert rgb(layers[0][0], 210, 160) == (0, 0, 255)
