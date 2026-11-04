"""K-0r: the build reports a preset z that contradicts the reference at a parent/child overlap.

Synthetic scene (no SAM, no paid calls): an outlined hip pad drawn in front
of a torso's lower edge. With the pad's z behind the torso, the torso's
completion paints over the drawn pad (the patch the robot hip showed); the
build must say so. With the pad in front, no conflict.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.model import PuppetConfig


def _scene(tmp_path: Path) -> Path:
    img = Image.new("RGB", (240, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 30, 200, 150], fill=(90, 90, 90), outline=(0, 0, 0), width=3)
    draw.rectangle([92, 150, 108, 215], fill=(60, 60, 60), outline=(0, 0, 0), width=3)
    draw.ellipse([40, 110, 160, 175], fill=(120, 40, 50), outline=(0, 0, 0), width=3)
    path = tmp_path / "r.png"
    img.save(path)
    return path


def _cfg(pad_z: int) -> PuppetConfig:
    return PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "torso", "parent": None, "concepts": [], "seed": [[180, 60]], "z": 0},
                {"name": "leg", "parent": "torso", "concepts": [], "seed": [[100, 150], [100, 200]], "z": pad_z},
            ]
        }
    )


def test_pad_behind_torso_reported_as_z_conflict(tmp_path: Path) -> None:
    files = build_puppet(_scene(tmp_path), _cfg(-1), tmp_path / "behind", max_image_calls=0)
    conflicts = files.report["z_conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["upper"] == "torso" and conflicts[0]["lower"] == "leg"
    assert int(conflicts[0]["hidden_drawn_px"]) >= 2000


def test_pad_in_front_has_no_z_conflict(tmp_path: Path) -> None:
    files = build_puppet(_scene(tmp_path), _cfg(1), tmp_path / "front", max_image_calls=0)
    assert files.report["z_conflicts"] == []
