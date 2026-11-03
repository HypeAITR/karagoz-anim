"""Shared synthetic puppet for the K-1 fit tests (builders only, no tests).

A 160x160 six-part puppet (torso root, an arm chain of three, a leg chain of
two) with asymmetric details (knobs, a claw slit, a one-sided foot) so every
joint angle is observable. Parts are drawn on separate RGBA layers with black
outlines; pivots sit on the visible joints.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.fit import LoadedView
from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.pose import Pose, render_pose

W = H = 160
BLACK = (0, 0, 0, 255)
TRUE_ANGLES = {"arm": 40.0, "fore": -25.0, "hand": 15.0, "leg": 10.0, "foot": -8.0}
TRUE_ROOT = {"tx": 5.0, "ty": -3.0, "rotation": 0.0, "scale": 1.0}


def _layer(fn: Callable[[ImageDraw.ImageDraw], object]) -> NDArray[np.uint8]:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fn(draw)
    return np.asarray(img, dtype=np.uint8)


def build_synth_puppet() -> tuple[Puppet, dict[str, NDArray[np.uint8]]]:
    grey = (95, 95, 95, 255)
    tube = (150, 70, 70, 255)
    dark = (50, 50, 50, 255)
    blue = (80, 110, 150, 255)
    images = {
        "torso": _layer(lambda d: d.rectangle([70, 55, 110, 130], fill=grey, outline=BLACK, width=2)),
        "arm": _layer(
            lambda d: (
                d.rectangle([30, 86, 72, 98], fill=tube, outline=BLACK, width=2),
                d.rectangle([34, 76, 46, 86], fill=tube, outline=BLACK, width=2),
            )
        ),
        "fore": _layer(
            lambda d: (
                d.rectangle([8, 87, 32, 97], fill=tube, outline=BLACK, width=2),
                d.rectangle([18, 97, 30, 105], fill=tube, outline=BLACK, width=2),
            )
        ),
        "hand": _layer(
            lambda d: (
                d.rectangle([0, 82, 12, 102], fill=dark, outline=BLACK, width=2),
                d.rectangle([4, 92, 8, 102], fill=(0, 0, 0, 0), outline=BLACK, width=2),
            )
        ),
        "leg": _layer(lambda d: d.rectangle([86, 128, 98, 152], fill=blue, outline=BLACK, width=2)),
        "foot": _layer(lambda d: d.rectangle([86, 149, 112, 159], fill=dark, outline=BLACK, width=2)),
    }
    parts = {
        "torso": {"parent": None, "pivot": [90.0, 92.0], "z": 0},
        "arm": {"parent": "torso", "pivot": [70.0, 92.0], "z": 1},
        "fore": {"parent": "arm", "pivot": [31.0, 92.0], "z": 1},
        "hand": {"parent": "fore", "pivot": [10.0, 92.0], "z": 2},
        "leg": {"parent": "torso", "pivot": [92.0, 129.0], "z": -1},
        "foot": {"parent": "leg", "pivot": [92.0, 151.0], "z": 2},
    }
    puppet = Puppet.model_validate(
        {
            "version": 1,
            "reference": "synth.png",
            "canvas": [W, H],
            "line_width_px": 2.0,
            "root": "torso",
            "parts": {
                name: {
                    "parent": spec["parent"],
                    "pivot": spec["pivot"],
                    "z": spec["z"],
                    "file": "",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                    "estimated": False,
                }
                for name, spec in parts.items()
            },
        }
    )
    return puppet, images


def white(rgba: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Composite RGBA onto white paper (what the key drawings look like)."""
    out = np.full((rgba.shape[0], rgba.shape[1], 3), 255, np.uint8)
    cover = np.asarray(rgba[..., 3] > 127, dtype=np.bool_)
    out[cover] = np.asarray(rgba[..., :3], dtype=np.uint8)[cover]
    return out


def true_pose() -> Pose:
    return Pose(view="synth", canvas=[W, H], angles=dict(TRUE_ANGLES), **TRUE_ROOT)


def draw_pose(
    puppet: Puppet,
    images: dict[str, NDArray[np.uint8]],
    angles: dict[str, float] | None = None,
    root: dict[str, float] | None = None,
    z_override: dict[str, int] | None = None,
    skip: tuple[str, ...] = (),
) -> NDArray[np.uint8]:
    """Render a drawing: the puppet at ``angles``/``root``, minus ``skip`` parts."""
    pose = Pose(
        view="synth",
        canvas=[W, H],
        angles=dict(angles or {}),
        tx=(root or {}).get("tx", 0.0),
        ty=(root or {}).get("ty", 0.0),
        rotation=(root or {}).get("rotation", 0.0),
        scale=(root or {}).get("scale", 1.0),
        z_override=dict(z_override or {}),
    )
    shown = {name: img for name, img in images.items() if name not in skip}
    return white(render_pose(puppet, pose, shown))


def lengthen(
    images: dict[str, NDArray[np.uint8]], name: str, pivot: tuple[float, float], fx: float
) -> dict[str, NDArray[np.uint8]]:
    """Copy of ``images`` with one part stretched ``fx`` along x about its pivot."""
    out = dict(images)
    img = np.asarray(images[name], dtype=np.uint8)
    height, width = img.shape[:2]
    px, _ = pivot
    mat = np.array([[fx, 0.0, px - fx * px], [0.0, 1.0, 0.0]])
    out[name] = np.asarray(
        cv2.warpAffine(img, mat, (width, height), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0)),
        dtype=np.uint8,
    )
    return out


def shift(rgb: NDArray[np.uint8], dx: int, dy: int) -> NDArray[np.uint8]:
    """Translate a drawing on white paper (the model redrew the lines offset)."""
    height, width = rgb.shape[:2]
    out = np.full_like(rgb, 255)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    dx0, dx1 = sx0 + dx, sx1 + dx
    dy0, dy1 = sy0 + dy, sy1 + dy
    out[dy0:dy1, dx0:dx1] = rgb[sy0:sy1, sx0:sx1]
    return out


def synth_view(puppet: Puppet, images: dict[str, NDArray[np.uint8]]) -> LoadedView:
    return LoadedView(name="synth", puppet=puppet, images=images)
