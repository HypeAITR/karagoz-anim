"""K-0p round 3: closing ink on completed-then-covered paint (K-0f mouth).

Synthetic only (no SAM, no paid calls). The closed scene is a 240-degree
circular arc plus a straight K-0f-style closing chord across the mouth; the
finished masks close the egg while the chord sits on the tube's hidden
(completed, covered at rest) strip, so the chord is completion ink and must
stay out of the dome fit. The exact-circle ring scene locks precedence: a
part the ring role claims never takes the dome, however circular its loop.
The plain wall keeps today's pivot bit-identically.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import split_reference
from karagoz.engine.puppet.pivots import compute_pivots

EGG_CENTRE = (90.0, 100.0)
EGG_R = 30
RING_CENTRE = (90.0, 100.0)
RING_R = 30


def _arc_points() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    cx, cy = EGG_CENTRE
    arc = []
    for i in range(161):
        theta = math.pi * (1.0 / 3.0 + 4.0 / 3.0 * i / 160.0)
        arc.append((cx + EGG_R * math.cos(theta), cy + EGG_R * math.sin(theta)))
    return arc, [arc[-1], arc[0]]


def make_closed_scene() -> tuple[
    dict[str, NDArray[np.bool_]], dict[str, NDArray[np.bool_]], NDArray[np.bool_], PuppetConfig
]:
    """Closed egg: drawn arc plus straight closing chord over hidden paint."""
    arc, chord = _arc_points()
    disc_img = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(disc_img).ellipse([60, 70, 120, 130], fill=255)
    disc = np.asarray(disc_img, dtype=np.uint8) > 0
    plate_box = np.zeros((200, 200), dtype=np.bool_)
    plate_box[40:160, 10:60] = True
    tube_box = np.zeros((200, 200), dtype=np.bool_)
    tube_box[88:113, 100:191] = True
    finished = {
        "plate": np.asarray(plate_box, dtype=np.bool_),
        "dome": np.asarray(disc, dtype=np.bool_),
        "tube": np.asarray(tube_box, dtype=np.bool_),
    }
    visible = {
        "plate": np.asarray(plate_box, dtype=np.bool_),
        "dome": np.asarray(disc, dtype=np.bool_),
        "tube": np.asarray(tube_box & ~disc, dtype=np.bool_),
    }
    ink = Image.new("RGB", (200, 200), (255, 255, 255))
    ink_draw = ImageDraw.Draw(ink)
    ink_draw.rectangle([10, 40, 59, 159], outline=(0, 0, 0), width=2)
    ink_draw.rectangle([100, 88, 190, 112], outline=(0, 0, 0), width=2)
    ink_draw.line(arc, fill=(0, 0, 0), width=2, joint="curve")
    ink_draw.line(chord, fill=(0, 0, 0), width=2)
    rgb = np.asarray(ink, dtype=np.uint8)
    line = np.asarray((rgb[..., 0] < 25) & (rgb[..., 1] < 25) & (rgb[..., 2] < 25), dtype=np.bool_)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "plate", "parent": None, "concepts": ["synthetic"], "z": 0},
                {"name": "dome", "parent": "plate", "concepts": ["synthetic"], "z": 2},
                {"name": "tube", "parent": "dome", "concepts": ["synthetic"], "z": 1},
            ]
        }
    )
    return finished, visible, line, cfg


def make_exact_ring_scene() -> tuple[dict[str, NDArray[np.bool_]], NDArray[np.bool_], PuppetConfig]:
    """A clean circular loop between two tube ends: role beats circle.

    The tube dives deep under the ring so the joint window reaches the
    whole loop (a shallow touch would starve the window and the old code
    would not fire either). Ink stops where paint goes under: no straight
    tube edge crosses the loop.
    """
    cx, cy = RING_CENTRE
    pts = [
        (cx + RING_R * math.cos(2.0 * math.pi * i / 96.0), cy + RING_R * math.sin(2.0 * math.pi * i / 96.0))
        for i in range(96)
    ]
    disc_img = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(disc_img).polygon(pts, fill=255)
    disc = np.asarray(disc_img, dtype=np.uint8) > 0
    parent_box = np.zeros((200, 200), dtype=np.bool_)
    parent_box[90:111, 10:76] = True
    tube_box = np.zeros((200, 200), dtype=np.bool_)
    tube_box[92:109, 70:191] = True
    masks = {
        "parent": np.asarray(parent_box & ~disc, dtype=np.bool_),
        "ring": np.asarray(disc, dtype=np.bool_),
        "tube": np.asarray(tube_box & ~disc, dtype=np.bool_),
    }
    ink = Image.new("RGB", (200, 200), (255, 255, 255))
    ink_draw = ImageDraw.Draw(ink)
    ink_draw.rectangle([10, 90, 55, 110], outline=(0, 0, 0), width=2)
    ink_draw.rectangle([125, 92, 190, 108], outline=(0, 0, 0), width=2)
    ink_draw.polygon(pts, outline=(0, 0, 0), width=2)
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


def make_plain_wall_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 190], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([20, 90, 180, 110], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "plate", "parent": None, "concepts": [], "seed": [40, 40], "z": 0},
                {"name": "tube", "parent": "plate", "concepts": [], "seed": [100, 100], "z": 1},
            ]
        }
    )
    return rgb, cfg


def test_closed_egg_fires_only_with_finished() -> None:
    finished, visible, line, cfg = make_closed_scene()
    blind = compute_pivots(visible, cfg, line, 2.0)
    assert blind.domes == {}
    seeing = compute_pivots(visible, cfg, line, 2.0, finished)
    assert seeing.domes == {"tube": "dome"}
    x, y = seeing.pivots["tube"]
    assert abs(x - EGG_CENTRE[0]) <= 2.0
    assert abs(y - EGG_CENTRE[1]) <= 2.0


def test_ring_role_beats_circular_outline() -> None:
    masks, line, cfg = make_exact_ring_scene()
    pivots = compute_pivots(masks, cfg, line, 2.0)
    assert pivots.rings == {"tube": "ring"}
    assert pivots.domes == {}


def test_plain_wall_keeps_today_pivot_bit_identical() -> None:
    rgb, cfg = make_plain_wall_scene()
    split = split_reference(rgb, cfg, sam=None)
    pivots = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    assert pivots.domes == {}
    assert pivots.rings == {}
    mask = np.asarray(split.part_masks["tube"], dtype=np.bool_)
    ys, xs = np.nonzero(mask)
    assert pivots.pivots["tube"] == (float(xs.mean()), float(ys.mean()))
    assert pivots.estimated["tube"] is False
