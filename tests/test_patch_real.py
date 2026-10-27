"""Acceptance tests for WP-07b (patches on real model output). Written by the maintainer; do not edit."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from karagoz.engine.patch import apply_patch, make_patch


def face(mouth_open: bool = False, eyes_closed: bool = False, shift: int = 0) -> Image.Image:
    """800x1000 cartoon figure; `shift` moves every outline by that many pixels (the model's jitter)."""
    img = Image.new("RGBA", (800, 1000), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    s = shift
    d.ellipse((200 + s, 100, 600 + s, 520), fill=(240, 200, 160, 255), outline=(0, 0, 0, 255), width=6)
    d.rectangle((230 + s, 560, 570 + s, 980), fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=6)
    d.line((150 + s, 600, 230 + s, 900), fill=(0, 0, 0, 255), width=5)
    for ex in (300, 460):  # two eyes 160 px apart
        if eyes_closed:
            d.rectangle((ex, 250, ex + 50, 275), fill=(0, 0, 0, 255))
        else:
            d.ellipse((ex + 15, 250, ex + 35, 270), fill=(0, 0, 0, 255))
    if mouth_open:
        d.ellipse((350, 380, 450, 450), fill=(120, 30, 30, 255), outline=(0, 0, 0, 255), width=5)
    else:
        d.line((360, 415, 440, 415), fill=(0, 0, 0, 255), width=5)
    return img


def test_outline_jitter_is_ignored() -> None:
    base, variant = face(), face(mouth_open=True, shift=1)
    p = make_patch(base, variant)
    w, h = p.image.size
    assert 330 <= p.x <= 360 and 360 <= p.y <= 390
    assert p.x + w <= 470 and p.y + h <= 470
    out = np.array(apply_patch(base, p))
    assert tuple(out[415, 400, :3]) == (120, 30, 30)


def test_both_eyes_are_kept() -> None:
    p = make_patch(face(), face(eyes_closed=True, shift=1))
    w, _ = p.image.size
    assert p.x <= 300 and p.x + w >= 510  # spans both eyes


def test_region_keeps_everything_inside() -> None:
    p = make_patch(face(), face(eyes_closed=True, shift=1), region=(250, 200, 550, 320), join=0)
    w, _ = p.image.size
    assert p.x <= 300 and p.x + w >= 510


def test_opening_can_be_disabled() -> None:
    import pytest

    from karagoz.engine.patch import PatchError

    with pytest.raises(PatchError, match="too large"):
        make_patch(face(), face(mouth_open=True, shift=2), open_px=1, threshold=24)
