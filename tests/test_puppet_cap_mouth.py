"""K-0r: a cap's side mouth closes clean; a tube under a cap ends closed (general rules only).

Synthetic scenes only (no SAM, no paid calls). The reference pose draws a
tube entering a dome cap from the side; the rig puts the cap above the
tube. Two defects on the K-0q code:

1. the cap's mouth fill takes its tone from the tube's anti-aliased outline
   halo (a darker patch) and the halo stays on the cap; the spanning arc is
   fitted to the whole part and leaves a broken outline with stubs;
2. the tube's extension under the cap is a flat strip with no walls and no
   closed end, and pieces of the cap's contour shared at split stay on the
   tube past its walls (an ink notch when the arm swings).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import estimate_line_width
from karagoz.engine.puppet.complete import complete_parts
from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import split_reference

W, H = 200, 200
CAP_TONE = (60, 30, 36)
HALO = (40, 20, 25)
GREY = (70, 70, 70)
CX, CY, R = 125, 100, 50


def _cfg(tube_parent: str) -> PuppetConfig:
    parts = [
        {"name": "body", "parent": None, "concepts": [], "seed": [179, 179], "z": 0},
        {"name": "cap", "parent": "body", "concepts": [], "seed": [[150, 100], [125, 60]], "z": 5},
        {"name": "tube", "parent": tube_parent, "concepts": [], "seed": [[30, 100]], "z": 2},
    ]
    return PuppetConfig.model_validate({"parts": parts})


def make_mouth_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    """Dome cap with a tube drawn over its left side (the T-pose reference)."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([150, 30, 190, 190], fill=(90, 90, 90), outline=(0, 0, 0), width=3)
    draw.ellipse([CX - R, CY - R, CX + R, CY + R], fill=CAP_TONE, outline=(0, 0, 0), width=5)
    # The tube's outline carries a one-pixel anti-alias halo (darker than the
    # cap, lighter than the line threshold), like a smooth-drawn reference.
    draw.rounded_rectangle([9, 77, 99, 123], radius=19, fill=(0, 0, 0))
    draw.rounded_rectangle([13, 81, 95, 119], radius=15, fill=GREY)
    rgb = np.asarray(img, dtype=np.uint8).copy()
    ring = Image.new("L", (W, H), 0)
    ImageDraw.Draw(ring).rounded_rectangle([8, 76, 100, 124], radius=20, fill=255)
    ImageDraw.Draw(ring).rounded_rectangle([9, 77, 99, 123], radius=19, fill=0)
    yy, xx = np.mgrid[0:H, 0:W]
    inner = np.hypot(xx - CX, yy - CY) <= R - 6
    rgb[(np.asarray(ring) > 0) & inner] = HALO
    return rgb, _cfg("cap")


def _complete(rgb: NDArray[np.uint8], cfg: PuppetConfig):  # type: ignore[no-untyped-def]
    split = split_reference(rgb, cfg, sam=None)
    lw = estimate_line_width(split.line)
    return split, complete_parts(rgb, split.line, split.part_masks, cfg, lw), lw


def _radius() -> NDArray[np.float64]:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.asarray(np.hypot(xx - CX, yy - CY), dtype=np.float64)


def test_cap_mouth_takes_cap_tone_without_halo() -> None:
    """Inside the dome near the old entry, the cap shows its own tone only."""
    rgb, cfg = make_mouth_scene()
    _split, full, lw = _complete(rgb, cfg)
    cap = np.asarray(full.images["cap"])
    opaque = cap[..., 3] > 127
    rr = _radius()
    yy, xx = np.mgrid[0:H, 0:W]
    zone = (rr <= R - 5 - lw) & (xx <= 105) & (np.abs(yy - CY) <= 30)
    assert int((zone & opaque).sum()) > 400
    diff = np.abs(cap[..., :3].astype(int) - np.asarray(CAP_TONE)).max(axis=2)
    off = np.asarray(zone & opaque & (diff > 10), dtype=np.bool_)
    assert int(off.sum()) == 0, f"{int(off.sum())} cap px off the cap tone (patch/halo) inside the dome"


def test_cap_mouth_outline_continuous_without_stubs() -> None:
    """Across the old entry the dome outline is one closed ink band; nothing sticks out."""
    rgb, cfg = make_mouth_scene()
    _split, full, _lw = _complete(rgb, cfg)
    cap = np.asarray(full.images["cap"])
    opaque = cap[..., 3] > 127
    dark = opaque & (cap[..., :3].max(axis=2) < 40)
    gaps = 0
    for deg in np.arange(140.0, 220.0, 1.0):
        t = np.radians(deg)
        x = int(round(CX + (R - 2.0) * np.cos(t)))
        y = int(round(CY - (R - 2.0) * np.sin(t)))
        gaps += int(not bool(dark[y, x]))
    assert gaps == 0, f"{gaps} gaps in the dome outline across the mouth"
    rr = _radius()
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.asarray(opaque & (rr > R + 2.0) & (xx < CX), dtype=np.bool_)
    assert int(out.sum()) == 0, f"{int(out.sum())} cap px stick out past the dome on the entry side"


def make_tube_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    """A straight tube whose end hides under a dome cap drawn on top."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([168, 168, 190, 190], fill=(90, 90, 90), outline=(0, 0, 0), width=3)
    draw.rectangle([8, 86, 110, 114], fill=GREY, outline=(0, 0, 0), width=4)
    draw.ellipse([CX - R, CY - R, CX + R, CY + R], fill=CAP_TONE, outline=(0, 0, 0), width=5)
    return np.asarray(img, dtype=np.uint8), _cfg("cap")


def test_tube_keeps_no_cap_contour_past_its_walls() -> None:
    """Under the cap the tube owns nothing past its own walls (no stub, no bump)."""
    rgb, cfg = make_tube_scene()
    split, full, _lw = _complete(rgb, cfg)
    tube = np.asarray(full.masks["tube"], dtype=np.bool_)
    cap = np.asarray(split.part_masks["cap"], dtype=np.bool_)
    yy, _xx = np.mgrid[0:H, 0:W]
    past = np.asarray(tube & cap & ((yy < 85) | (yy > 115)), dtype=np.bool_)
    assert int(past.sum()) == 0, f"{int(past.sum())} tube px past its walls under the cap"
    alpha = np.asarray(full.images["tube"])[..., 3] > 127
    assert int((alpha & cap & ((yy < 85) | (yy > 115))).sum()) == 0


def test_tube_extension_ends_closed_and_outlined() -> None:
    """The tube's extension under the cap has inked walls and a rounded inked end."""
    rgb, cfg = make_tube_scene()
    split, full, lw = _complete(rgb, cfg)
    tube_img = np.asarray(full.images["tube"])
    on = tube_img[..., 3] > 127
    dark = on & (tube_img[..., :3].max(axis=2) < 40)
    cap = np.asarray(split.part_masks["cap"], dtype=np.bool_)
    drawn = np.asarray(split.part_masks["tube"], dtype=np.bool_)
    ext = np.asarray(on & cap & ~drawn, dtype=np.bool_)
    assert int(ext.sum()) > 200, "no extension under the cap"
    xs = np.nonzero(on[CY])[0]
    tip = int(xs.max())
    assert tip > 100, f"extension tip at x {tip}"
    # Closed end: the last line width on the centre line is ink.
    end_run = dark[CY, tip - int(lw) + 1 : tip + 1]
    assert bool(end_run.all()), "extension end is not outlined"
    # Walls continue: a column inside the extension has ink at both edges.
    col = tip - 3 * int(lw)
    ys = np.nonzero(on[:, col])[0]
    assert len(ys) > 0
    assert bool(dark[ys.min(), col]) and bool(dark[ys.max(), col]), "extension walls not outlined"
    # Rounded: the square corner at the tip is not owned.
    assert not bool(on[87, tip]) and not bool(on[113, tip]), "extension end is square"
    added = np.asarray(on & ~drawn, dtype=np.bool_)
    assert int((added & ~cap).sum()) == 0, "extension leaves the cap (rest would change)"
