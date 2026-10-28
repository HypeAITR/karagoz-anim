"""Acceptance tests for WP-13c (a patch brings its own silhouette). Written by the maintainer; do not edit.

Base: a figure (150..250 x 100..550) cut out of its white paper (transparent outside). The mouth opens and the jaw
drops: the variant's chin (skin with a black chin line) reaches y 569, below the base silhouette's bottom (550).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from karagoz.engine.compose import cutout
from karagoz.engine.patch import Patch, apply_patch, paper_mask

SKIN = (200, 120, 80)
WHITE = (255, 255, 255)


def base() -> Image.Image:
    img = Image.new("RGB", (400, 600), WHITE)
    ImageDraw.Draw(img).rectangle((150, 100, 250, 550), fill=SKIN, outline=(0, 0, 0), width=6)
    return cutout(img)


def jaw_patch() -> Patch:
    """60x60 patch at (170, 520): paper, with a skin jaw (175..225 x 520..564) and its chin line (565..569)."""
    img = Image.new("RGBA", (60, 60), (*WHITE, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((5, 0, 55, 44), fill=(*SKIN, 255))
    d.rectangle((5, 45, 55, 49), fill=(0, 0, 0, 255))
    return Patch(img, 170, 520)


def px(img: Image.Image, x: int, y: int) -> tuple[int, ...]:
    return tuple(int(v) for v in np.array(img.convert("RGBA"))[y, x])


def test_paper_mask() -> None:
    img = Image.new("RGB", (100, 100), (247, 249, 246))  # the model's not-quite-white paper
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 80, 80), fill=(255, 255, 255), outline=(0, 0, 0), width=4)  # enclosed white (teeth)
    d.rectangle((0, 90, 10, 99), fill=(90, 90, 90))
    m = paper_mask(img)
    assert m.dtype == bool and m.shape == (100, 100)
    assert not m[5, 50] and m[50, 50] and m[20, 50] and m[95, 5]


def test_new_chin_line_extends_the_silhouette() -> None:
    out = apply_patch(base(), jaw_patch())
    assert px(out, 200, 560) == (*SKIN, 255)  # jaw below the old silhouette
    assert px(out, 200, 567) == (0, 0, 0, 255)  # the new chin line is drawn
    assert px(out, 200, 540) == (*SKIN, 255)


def test_paper_never_enters() -> None:
    b = base()
    out = apply_patch(b, jaw_patch())
    assert px(out, 172, 575) == px(b, 172, 575) and px(out, 172, 575)[3] == 0  # paper around the jaw: unchanged
    assert px(out, 200, 580) == px(b, 200, 580)  # outside the patch rectangle: unchanged


def test_no_white_on_a_soft_edge() -> None:
    arr = np.array(base())
    arr[300:310, 150, 3] = 128  # an anti-aliased edge pixel column of the base
    b = Image.fromarray(arr)
    paper = Patch(Image.new("RGBA", (20, 20), (*WHITE, 255)), 140, 295)  # all paper
    out = apply_patch(b, paper)
    assert np.array_equal(np.array(out), arr)


def test_enclosed_white_is_kept() -> None:
    img = Image.new("RGBA", (40, 30), (*SKIN, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((5, 5, 35, 25), fill=(*WHITE, 255), outline=(120, 20, 20, 255), width=3)  # teeth inside the lips
    out = apply_patch(base(), Patch(img, 180, 300))
    assert px(out, 200, 315) == (*WHITE, 255)


def test_soft_patch_alpha_over_transparent_base() -> None:
    p = Patch(Image.new("RGBA", (10, 10), (0, 0, 0, 128)), 50, 50)  # outside the figure, half transparent
    out = apply_patch(base(), p)
    assert px(out, 55, 55) == (0, 0, 0, 128)


def test_opaque_base_is_plain_alpha_blending() -> None:
    p = Patch(Image.new("RGBA", (10, 10), (120, 20, 20, 128)), 190, 300)
    out = apply_patch(base(), p)
    w = 128 / 255
    assert px(out, 195, 305) == (*(round(r * w + s * (1 - w)) for r, s in zip((120, 20, 20), SKIN, strict=True)), 255)
