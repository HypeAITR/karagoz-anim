"""K-0: joint pivots land where the joint is (synthetic robot)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import split_reference
from karagoz.engine.puppet.pivots import compute_pivots

CAP_CENTRE = (120.0, 70.0)


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


def test_arm_pivot_within_2px_of_cap_centre() -> None:
    rgb, cfg = make_robot()
    split = split_reference(rgb, cfg, sam=None)
    pivots = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    x, y = pivots.pivots["arm"]
    assert abs(x - CAP_CENTRE[0]) <= 2.0
    assert abs(y - CAP_CENTRE[1]) <= 2.0
    assert isinstance(x, float) and isinstance(y, float)


def test_root_pivot_is_centre_of_mass() -> None:
    rgb, cfg = make_robot()
    split = split_reference(rgb, cfg, sam=None)
    pivots = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    ys, xs = np.nonzero(split.part_masks["torso"])
    assert pivots.pivots["torso"] == (float(xs.mean()), float(ys.mean()))
    assert pivots.radii["torso"] == 0.0
