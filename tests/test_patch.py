"""Acceptance tests for WP-07 (face patches). Written by the maintainer; do not edit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from karagoz.engine.patch import Patch, PatchError, apply_patch, load_patch, make_patch, save_patch


def face(mouth_open: bool, noise: bool = False, seed: int = 0) -> Image.Image:
    """400x600 flat cartoon: skin head, black outline, mouth line or open mouth."""
    img = Image.new("RGBA", (400, 600), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((100, 100, 300, 320), fill=(240, 200, 160, 255), outline=(0, 0, 0, 255), width=5)
    d.ellipse((150, 170, 170, 190), fill=(0, 0, 0, 255))
    d.ellipse((230, 170, 250, 190), fill=(0, 0, 0, 255))
    if mouth_open:
        d.ellipse((170, 240, 230, 285), fill=(120, 20, 20, 255), outline=(0, 0, 0, 255), width=4)
    else:
        d.line((175, 260, 225, 260), fill=(0, 0, 0, 255), width=4)
    d.rectangle((120, 340, 280, 560), fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=5)
    if noise:  # the image model nudges far-away pixels a little, and a few a lot
        arr = np.array(img).astype(int)
        rng = np.random.default_rng(seed)
        arr[:, :, :3] += rng.integers(-10, 11, size=arr[:, :, :3].shape)
        arr[580:584, 20:24, :3] = 0  # a speck far from the face
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    return img


def changed(a: Image.Image, b: Image.Image) -> np.ndarray:
    return np.any(np.array(a) != np.array(b), axis=2)


def test_patch_covers_mouth_only() -> None:
    base, variant = face(False), face(True)
    p = make_patch(base, variant)
    assert isinstance(p, Patch) and p.image.mode == "RGBA"
    w, h = p.image.size
    assert 160 <= p.x <= 175 and 230 <= p.y <= 245
    assert p.x + w <= 245 and p.y + h <= 300
    out = apply_patch(base, p)
    diff = changed(base, out)
    ys, xs = np.nonzero(diff)
    assert xs.min() >= p.x and xs.max() < p.x + w and ys.min() >= p.y and ys.max() < p.y + h


def test_outside_patch_is_bit_exact_even_with_model_noise() -> None:
    base, variant = face(False), face(True, noise=True)
    p = make_patch(base, variant)
    out = apply_patch(base, p)
    arr_b, arr_o = np.array(base), np.array(out)
    alpha = np.zeros(arr_b.shape[:2])
    alpha[p.y : p.y + p.image.size[1], p.x : p.x + p.image.size[0]] = np.array(p.image)[:, :, 3]
    outside = alpha == 0
    assert np.array_equal(arr_b[outside], arr_o[outside])
    assert not changed(base, out)[575:590, 15:30].any()  # the far speck was dropped


def test_inside_shows_the_new_mouth() -> None:
    base, variant = face(False), face(True)
    out = np.array(apply_patch(base, make_patch(base, variant)))
    assert tuple(out[262, 200, :3]) == (120, 20, 20)  # open mouth colour at its centre


def test_region_limits_the_search() -> None:
    base, variant = face(False), face(True)
    with pytest.raises(PatchError, match="no change"):
        make_patch(base, variant, region=(0, 0, 400, 200))


def test_identical_images_have_no_change() -> None:
    with pytest.raises(PatchError, match="no change"):
        make_patch(face(False), face(False))


def test_redrawn_image_is_rejected() -> None:
    base = face(False)
    other = Image.new("RGBA", base.size, (10, 200, 10, 255))
    with pytest.raises(PatchError, match="too large"):
        make_patch(base, other)


def test_save_and_load(tmp_path: Path) -> None:
    p = make_patch(face(False), face(True))
    save_patch(p, tmp_path / "mouth-D")
    assert (tmp_path / "mouth-D.png").exists() and (tmp_path / "mouth-D.json").exists()
    q = load_patch(tmp_path / "mouth-D")
    assert (q.x, q.y) == (p.x, p.y) and np.array_equal(np.array(q.image), np.array(p.image))


def test_deterministic() -> None:
    a = make_patch(face(False), face(True, noise=True))
    b = make_patch(face(False), face(True, noise=True))
    assert (a.x, a.y) == (b.x, b.y) and np.array_equal(np.array(a.image), np.array(b.image))
