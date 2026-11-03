"""K-0p round 2: a ring and its child turn about the ring's centre, both limited.

Synthetic scenes only (no SAM, no paid calls). The ring scene uses exact
masks (no split bleed): a wobbly short ring between two tube ends. Today
the ring hangs off its inner edge and the child off its outer edge (~25 px
apart); the rule seats both at the ring centre. The long-link scene has the
same topology with a 70x16 link: the aspect gate must stay shut and pivots
bit-identical. Limits are +/-10 deg on ring and cap joints, nowhere else.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import split_reference
from karagoz.engine.puppet.pivots import compute_pivots

RING_CENTRE = (90.0, 100.0)
RING_R = 30


def _wobble_ring_points() -> list[tuple[float, float]]:
    cx, cy = RING_CENTRE
    pts = []
    for i in range(64):
        theta = 2.0 * math.pi * i / 64
        r = RING_R + 4.0 * math.sin(2.0 * theta + 1.0) + 2.5 * math.sin(3.0 * theta + 0.5)
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pts


def make_ring_scene() -> tuple[dict[str, NDArray[np.bool_]], NDArray[np.bool_], PuppetConfig]:
    """Exact masks (no split bleed): wobbly ring disc between two tube ends."""
    disc_img = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(disc_img).polygon(_wobble_ring_points(), fill=255)
    disc = np.asarray(disc_img, dtype=np.uint8) > 0
    parent_box = np.zeros((200, 200), dtype=np.bool_)
    parent_box[90:111, 10:76] = True
    tube_box = np.zeros((200, 200), dtype=np.bool_)
    tube_box[92:109, 105:191] = True
    masks = {
        "parent": np.asarray(parent_box & ~disc, dtype=np.bool_),
        "ring": np.asarray(disc, dtype=np.bool_),
        "tube": np.asarray(tube_box & ~disc, dtype=np.bool_),
    }
    ink = Image.new("RGB", (200, 200), (255, 255, 255))
    ink_draw = ImageDraw.Draw(ink)
    ink_draw.rectangle([10, 90, 75, 110], outline=(0, 0, 0), width=2)
    ink_draw.rectangle([105, 92, 190, 108], outline=(0, 0, 0), width=2)
    ink_draw.polygon(_wobble_ring_points(), outline=(0, 0, 0), width=2)
    rgb = np.asarray(ink, dtype=np.uint8)
    line = np.asarray((rgb[..., 0] < 25) & (rgb[..., 1] < 25) & (rgb[..., 2] < 25), dtype=np.bool_)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "parent", "parent": None, "concepts": ["synthetic"], "z": 1},
                {"name": "ring", "parent": "parent", "concepts": ["synthetic"], "z": 2},
                {"name": "tube", "parent": "ring", "concepts": ["synthetic"], "z": 1},
            ]
        }
    )
    return masks, line, cfg


def make_long_link_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 90, 110, 110], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([100, 92, 170, 108], fill=(120, 40, 40), outline=(0, 0, 0), width=2)
    draw.rectangle([165, 92, 195, 108], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "parent", "parent": None, "concepts": [], "seed": [60, 100], "z": 1},
                {"name": "link", "parent": "parent", "concepts": [], "seed": [135, 100], "z": 2},
                {"name": "tube", "parent": "link", "concepts": [], "seed": [185, 100], "z": 1},
            ]
        }
    )
    return rgb, cfg


def make_small_dome_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
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


def test_ring_and_child_pivot_within_2px_of_ring_centre() -> None:
    masks, line, cfg = make_ring_scene()
    pivots = compute_pivots(masks, cfg, line, 2.0)
    assert pivots.rings == {"tube": "ring"}
    for name in ("ring", "tube"):
        x, y = pivots.pivots[name]
        assert abs(x - RING_CENTRE[0]) <= 2.0, name
        assert abs(y - RING_CENTRE[1]) <= 2.0, name


def test_long_link_keeps_today_pivot_bit_identical() -> None:
    rgb, cfg = make_long_link_scene()
    split = split_reference(rgb, cfg, sam=None)
    pivots = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    assert pivots.rings == {}
    touch = max(3, int(round(2.0)))
    touch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))
    grown = np.asarray(cv2.dilate(split.part_masks["parent"].astype(np.uint8), touch_k), dtype=np.uint8) > 0
    overlap = np.asarray(split.part_masks["link"] & grown, dtype=np.bool_)
    oys, oxs = np.nonzero(overlap)
    assert pivots.pivots["link"] == (float(oxs.mean()), float(oys.mean()))
    assert pivots.estimated["link"] is False


def test_ring_and_cap_joints_limited_pm10() -> None:
    masks, line, cfg = make_ring_scene()
    pivots = compute_pivots(masks, cfg, line, 2.0)
    assert pivots.limits == {"ring": [-10.0, 10.0]}
    rgb2, cfg2 = make_small_dome_scene()
    split2 = split_reference(rgb2, cfg2, sam=None)
    pivots2 = compute_pivots(split2.part_masks, cfg2, split2.line, 2.0)
    assert pivots2.limits == {"cap": [-10.0, 10.0]}
