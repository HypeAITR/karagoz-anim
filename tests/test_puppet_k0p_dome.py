"""K-0p: a child entering under a covering dome turns about the dome's centre.

Synthetic scenes only (no SAM, no paid calls). The dome scene uses a large
cap (r=60) over a tube end: today's overlap-centre pivot sits at the dome's
rim, the rule seats it at the dome centre. The plain scene has no circular
outline near the joint, so the pivot must stay bit-identical to the overlap
mean (today's fallback, untouched).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import split_reference
from karagoz.engine.puppet.pivots import compute_pivots

DOME_CENTRE = (120.0, 100.0)
DOME_R = 60


def make_dome_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 110, 190], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([20, 90, 122, 110], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    cx, cy = DOME_CENTRE
    draw.ellipse([cx - DOME_R, cy - DOME_R, cx + DOME_R, cy + DOME_R], fill=(120, 40, 40), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "base", "parent": None, "concepts": [], "seed": [40, 40], "z": 0},
                {"name": "dome", "parent": "base", "concepts": [], "seed": [150, 100], "z": 2},
                {"name": "tube", "parent": "dome", "concepts": [], "seed": [40, 100], "z": 1},
            ]
        }
    )
    return rgb, cfg


def make_plain_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 190], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([20, 90, 180, 110], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "base", "parent": None, "concepts": [], "seed": [40, 40], "z": 0},
                {"name": "tube", "parent": "base", "concepts": [], "seed": [100, 100], "z": 1},
            ]
        }
    )
    return rgb, cfg


def test_dome_pivot_within_2px_of_dome_centre() -> None:
    rgb, cfg = make_dome_scene()
    split = split_reference(rgb, cfg, sam=None)
    pivots = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    x, y = pivots.pivots["tube"]
    assert abs(x - DOME_CENTRE[0]) <= 2.0
    assert abs(y - DOME_CENTRE[1]) <= 2.0


def test_plain_overlap_pivot_bit_identical_to_overlap_mean() -> None:
    rgb, cfg = make_plain_scene()
    split = split_reference(rgb, cfg, sam=None)
    pivots = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    mask = np.asarray(split.part_masks["tube"], dtype=np.bool_)
    ys, xs = np.nonzero(mask)
    # The tube sits fully inside its parent, so the overlap is the tube
    # itself and today's fallback pivot is the tube centroid, exactly.
    assert pivots.pivots["tube"] == (float(xs.mean()), float(ys.mean()))
    assert pivots.estimated["tube"] is False
