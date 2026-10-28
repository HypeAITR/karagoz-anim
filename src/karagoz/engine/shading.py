"""Contact shadows and ambient tone for placement-mode renders (WP-21)."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

SHADOW_OPACITY = 0.35
SHADOW_WIDTH = 0.6  # rx = SHADOW_WIDTH * feet width
SHADOW_FLAT = 0.12  # ry = max(3.0, SHADOW_FLAT * rx)
AMBIENT = 0.12


def background_tint(bg: Image.Image) -> tuple[float, float, float]:
    """Per-channel room light of a background: channel means divided by their mean."""
    arr = np.array(bg.convert("RGB"), dtype=np.float64)
    means = arr.mean(axis=(0, 1))
    g = float(means.mean())
    if g == 0:
        return (1.0, 1.0, 1.0)
    return (float(means[0] / g), float(means[1] / g), float(means[2] / g))


def ambient_tint(img: Image.Image, tint: tuple[float, float, float], strength: float = AMBIENT) -> Image.Image:
    """RGBA copy of ``img`` with every channel pulled towards the room ``tint``; alpha unchanged."""
    arr = np.array(img.convert("RGBA"), dtype=np.float64)
    factors = np.array([(1.0 - strength) + strength * t for t in tint], dtype=np.float64)
    result = np.empty_like(arr)
    result[:, :, :3] = np.round(arr[:, :, :3] * factors).clip(0, 255)
    result[:, :, 3] = arr[:, :, 3]
    return Image.fromarray(result.astype(np.uint8))


def shadow_alpha(size: tuple[int, int], center: tuple[float, float], rx: float, ry: float) -> np.ndarray:
    """Blurred ellipse mask of shape ``(h, w)``: 1.0 inside, feathered outside."""
    w, h = size
    if rx <= 0 or ry <= 0:
        return np.zeros((h, w), dtype=np.float32)
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    inside = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    sigma = max(1.0, 0.5 * ry)
    k = 2 * math.ceil(3.0 * sigma) + 1
    blurred = cv2.GaussianBlur(inside.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)
    return np.asarray(blurred, dtype=np.float32)


def apply_shadow(frame: Image.Image, alpha: np.ndarray, opacity: float = SHADOW_OPACITY) -> Image.Image:
    """RGB copy of ``frame`` with every channel darkened where ``alpha`` is opaque."""
    arr = np.array(frame.convert("RGB"), dtype=np.float64)
    factor = (1.0 - opacity * alpha.astype(np.float64))[..., None]
    darkened = np.round(arr * factor).clip(0, 255)
    return Image.fromarray(darkened.astype(np.uint8))
