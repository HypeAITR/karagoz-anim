"""K-0: hidden parts are really complete (the rotation test proves it)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.model import PuppetConfig


def make_robot() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 40, 120, 140], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([30, 63, 122, 77], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.ellipse([120 - 16, 70 - 16, 120 + 16, 70 + 16], fill=(120, 40, 40), outline=(0, 0, 0), width=2)
    draw.rectangle([14, 58, 34, 82], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "torso", "parent": None, "concepts": [], "seed": [85, 110], "z": 0},
                {"name": "cap", "parent": "torso", "concepts": [], "seed": [130, 70], "z": 2},
                {"name": "arm", "parent": "cap", "concepts": [], "seed": [60, 70], "z": 1},
                {"name": "hand", "parent": "arm", "concepts": [], "seed": [24, 70], "z": 3},
            ]
        }
    )
    return rgb, cfg


def _write_ref(rgb: NDArray[np.uint8], tmp_path: Path) -> Path:
    ref = tmp_path / "ref.png"
    Image.fromarray(rgb).save(ref)
    return ref


def test_completed_tube_shows_no_paper_in_rotation(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out", completion=True)
    assert files.report["completed"]["arm"] == ["geometric"]
    for joint, info in files.report["joints"].items():
        for angle, paper in info["paper_px"].items():
            assert paper == 0, f"{joint} {angle}: {paper} paper px"


def test_uncompleted_tube_shows_paper_in_rotation(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out", completion=False)
    assert files.report["completed"]["arm"] == []
    assert max(files.report["joints"]["arm"]["paper_px"].values()) > 0
