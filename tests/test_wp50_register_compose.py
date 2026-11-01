"""WP-50 extra tests: register + compose sequence handling (items 1-2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from karagoz.engine.compose import Placement, ShotAssets, frame_layers
from karagoz.engine.patch import Patch, save_patch
from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo
from karagoz.engine.register import refine
from karagoz.engine.xsheet import Exposure
from karagoz.spec import Mouth, Regions

SKIN = (200, 120, 80)
RED = (120, 20, 20)


def _fig(lower: tuple[int, int] = (0, 0)) -> Image.Image:
    a = np.zeros((200, 200, 4), np.uint8)
    a[10:120, 60:140] = (200, 120, 80, 255)
    dx, dy = lower
    a[120 + dy : 190 + dy, 50 + dx : 150 + dx] = (40, 40, 120, 255)
    return Image.fromarray(a)


def _lm() -> dict[str, Any]:
    return {"size": [200, 200], "hip": [100, 120], "foot_low": [100, 199], "feet": [], "hands": []}


def _regions() -> Regions:
    return Regions.model_validate(
        {
            "version": 1,
            "location": "salon",
            "image": "background.png",
            "size": [1920, 1080],
            "scale": None,
            "objects": [
                {
                    "name": "sofa",
                    "kind": "sofa",
                    "box": [1, 1, 10, 10],
                    "floor_y": 10,
                    "confidence": 1.0,
                    "mask": "m.png",
                    "front": False,
                }
            ],
            "places": [{"name": "sofa_left", "object": "sofa", "kind": "floor", "floor": [100, 900]}],
        }
    )


def _refine(names: list[str], images: dict[str, Image.Image]) -> dict[str, Any]:
    poses = {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit")}
    seqs = {"s1": ("p1", "p2")} if "s1.1" in names else None
    char = CharacterInput("ali", 120.0, False, "sofa_left", poses, {n: _lm() for n in names}, None, seqs or {})
    placed = {
        "ali": {
            n: DrawingPlacement(scale=1.0, x=100, y=100, mirror=False, anchor="hip", occluders=(), depth=848)
            for n in names
        }
    }
    out, _ = refine([char], placed, _regions(), lambda c, d: images.get(d))
    return out["ali"]


def test_refine_aligns_sequence_drawing_like_an_inbetween() -> None:
    control = _refine(["p1", "p1-p2.1", "p2"], {"p1": _fig(), "p1-p2.1": _fig((10, 0)), "p2": _fig()})
    assert control["p1-p2.1"].x == 90
    out = _refine(["p1", "s1.1", "p2"], {"p1": _fig(), "s1.1": _fig((10, 0)), "p2": _fig()})
    assert out["s1.1"].x == 90


def _drawing(dx: int = 0) -> Image.Image:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((75 + dx, 30, 125 + dx, 180), fill=SKIN, outline=(0, 0, 0), width=4)
    d.rectangle((92 + dx, 59, 108 + dx, 61), fill=(0, 0, 0))
    d.rectangle((87 + dx, 41, 113 + dx, 43), fill=(0, 0, 0))
    return img


def _landmarks(face: list[int]) -> dict[str, object]:
    return {
        "version": 1, "image": "x.png", "size": [200, 200], "facing": "right", "head": None,
        "face": {"box": face, "confidence": 0.9}, "feet": [], "feet_source": None, "foot_low": None,
        "pants": None, "hip": None, "hands": [], "cups": [],
    }  # fmt: skip


def _assets(tmp_path: Path) -> Path:
    root = tmp_path / "assets" / "sh01"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (50, 50, 50)).save(root / "background.png")
    _drawing().save(root / "ali" / "p1.png")
    _drawing(10).save(root / "ali" / "s1.1.png")
    save_patch(Patch(Image.new("RGBA", (20, 10), (*RED, 255)), 90, 55), root / "ali" / "p1.mouth-D")
    save_patch(Patch(Image.new("RGBA", (30, 6), (0, 0, 0, 255)), 85, 40), root / "ali" / "p1.eyes-closed")
    for name, face in (("p1", [75, 30, 125, 80]), ("s1.1", [85, 30, 135, 80])):
        (root / "ali" / f"{name}.landmarks.json").write_text(json.dumps(_landmarks(face)), encoding="utf-8")
    return root


def _exposure(cell: str, eyes: str = "open") -> Exposure:
    return Exposure(shot="sh01", length=4, body={"ali": [cell] * 4}, eyes={"ali": [eyes] * 4}, windows=[])


def _mouths(shape: str = "D") -> dict[str, Mouth]:
    cues = [{"start": 0, "end": 4, "shape": shape, "src": "auto"}]
    return {"ali": Mouth.model_validate({"version": 1, "shot": "sh01", "character": "ali", "fps": 24, "cues": cues})}


def _rgb(img: Image.Image, x: int, y: int) -> tuple[int, ...]:
    return tuple(int(v) for v in np.array(img.convert("RGBA"))[y, x, :3])


def test_sequence_frame_borrows_the_key_mouth_patch(tmp_path: Path) -> None:
    root = _assets(tmp_path)
    layers, warn = frame_layers(
        _exposure("s1.1"), _mouths("D"), ShotAssets(root), ["ali"], {"ali": Placement(1.0, 0, 0)}, 0
    )
    assert "mouth-D" not in " ".join(warn), warn
    assert _rgb(layers[0][0], 105, 60) == RED
    assert _rgb(layers[0][0], 95, 60) == SKIN


def test_sequence_frame_borrows_the_key_eyes_patch(tmp_path: Path) -> None:
    root = _assets(tmp_path)
    layers, warn = frame_layers(
        _exposure("s1.1", "closed"), _mouths("X"), ShotAssets(root), ["ali"], {"ali": Placement(1.0, 0, 0)}, 0
    )
    assert "eyes-closed" not in " ".join(warn), warn
    assert _rgb(layers[0][0], 110, 42) == (0, 0, 0)
