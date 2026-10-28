"""Face patches: cut out only the region that changed between two drawings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class PatchError(ValueError):
    """Raised when no patch can be cut (nothing changed, or too much changed)."""


@dataclass(frozen=True)
class Patch:
    """A small changed region cut from a variant drawing."""

    image: Image.Image  # RGBA patch pixels; alpha = feathered mask
    x: int  # top-left position in the base drawing
    y: int


def make_patch(
    base: Image.Image,
    variant: Image.Image,
    *,
    region: tuple[int, int, int, int] | None = None,
    threshold: int = 40,
    open_px: int = 5,
    join: int = 120,
    grow: int = 6,
    feather: int = 2,
    max_area: float = 0.06,
) -> Patch:
    """Cut the changed region between ``base`` and ``variant`` as a patch."""
    base_rgba = base.convert("RGBA")
    variant_rgba = variant.convert("RGBA")
    if variant_rgba.size != base_rgba.size:
        variant_rgba = variant_rgba.resize(base_rgba.size, Image.Resampling.LANCZOS)
    width, height = base_rgba.size

    base_arr = np.array(base_rgba).astype(np.int16)
    var_arr = np.array(variant_rgba).astype(np.int16)
    changed = np.abs(base_arr - var_arr).max(axis=2) > threshold

    grow = max(grow, 0)
    feather = max(feather, 0)
    join = max(join, 0)
    if open_px <= 1:
        opened = changed
    else:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        opened = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, open_kernel) != 0

    if not opened.any():
        raise PatchError("no change")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise PatchError("no change")

    def _box(label: int) -> tuple[int, int, int, int]:
        ox0 = int(stats[label, cv2.CC_STAT_LEFT])
        oy0 = int(stats[label, cv2.CC_STAT_TOP])
        ox1 = ox0 + int(stats[label, cv2.CC_STAT_WIDTH])
        oy1 = oy0 + int(stats[label, cv2.CC_STAT_HEIGHT])
        return (ox0, oy0, ox1, oy1)

    kept: list[int] = []
    if region is not None:
        rx0, ry0, rx1, ry1 = region
        for label in range(1, count):
            ox0, oy0, ox1, oy1 = _box(label)
            if ox0 < rx1 and ox1 > rx0 and oy0 < ry1 and oy1 > ry0:
                kept.append(label)
        if not kept:
            raise PatchError("no change")
    else:
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        lx0, ly0, lx1, ly1 = _box(largest)
        kept.append(largest)
        for label in range(1, count):
            if label == largest:
                continue
            ox0, oy0, ox1, oy1 = _box(label)
            if ox0 <= lx1 + join and ox1 >= lx0 - join and oy0 <= ly1 + join and oy1 >= ly0 - join:
                kept.append(label)

    keep_opened = np.isin(labels, np.array(kept)).reshape(height, width)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    if grow > 0:
        search_area = cv2.dilate((keep_opened.astype(np.uint8) * 255), dilate_kernel, iterations=1) != 0
    else:
        search_area = keep_opened
    keep = changed & search_area

    dilated = cv2.dilate((keep.astype(np.uint8) * 255), dilate_kernel, iterations=1)
    if feather > 0:
        size = 2 * feather + 1
        alpha = cv2.GaussianBlur(dilated, (size, size), 0)
        inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        inner = cv2.erode(dilated, inner_kernel, iterations=1)
        alpha[inner != 0] = 255
    else:
        alpha = dilated

    ys, xs = np.nonzero(dilated)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    box_w, box_h = x1 - x0, y1 - y0
    fraction = box_w * box_h / (width * height)
    if fraction > max_area:
        raise PatchError(f"change too large: {box_w}x{box_h} box covers {fraction:.1%} of the image")

    patch_arr = np.empty((box_h, box_w, 4), dtype=np.uint8)
    patch_arr[:, :, :3] = np.array(variant_rgba)[y0:y1, x0:x1, :3]
    patch_arr[:, :, 3] = alpha[y0:y1, x0:x1]
    return Patch(image=Image.fromarray(patch_arr), x=x0, y=y0)


def apply_patch(base: Image.Image, patch: Patch) -> Image.Image:
    """Return a new RGBA image with ``patch`` blended onto ``base``; the base alpha is unchanged."""
    base_rgba = base.convert("RGBA")
    base_w, base_h = base_rgba.size
    out = np.array(base_rgba).astype(np.float32)
    src = np.array(patch.image.convert("RGBA")).astype(np.float32)
    patch_h, patch_w = src.shape[0], src.shape[1]
    dx0, dy0 = max(patch.x, 0), max(patch.y, 0)
    dx1, dy1 = min(patch.x + patch_w, base_w), min(patch.y + patch_h, base_h)
    if dx1 > dx0 and dy1 > dy0:
        tile = src[dy0 - patch.y : dy0 - patch.y + (dy1 - dy0), dx0 - patch.x : dx0 - patch.x + (dx1 - dx0)]
        region = out[dy0:dy1, dx0:dx1]
        weight = tile[:, :, 3:4] / 255.0 * region[:, :, 3:4] / 255.0
        inv = 1.0 - weight
        region[:, :, :3] = np.round(tile[:, :, :3] * weight + region[:, :, :3] * inv)
    return Image.fromarray(out.astype(np.uint8))


def transfer_patch(
    patch: Patch,
    src_face: tuple[float, float, float, float],
    dst_face: tuple[float, float, float, float],
) -> Patch:
    """Move ``patch`` from the source face box onto the destination face box."""
    s = (dst_face[3] - dst_face[1]) / (src_face[3] - src_face[1])
    if s == 1:
        image = patch.image
    else:
        w, h = patch.image.size
        image = patch.image.convert("RGBA").resize(
            (max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS
        )
    sc_x = (src_face[0] + src_face[2]) / 2.0
    sc_y = (src_face[1] + src_face[3]) / 2.0
    dc_x = (dst_face[0] + dst_face[2]) / 2.0
    dc_y = (dst_face[1] + dst_face[3]) / 2.0
    return Patch(image=image, x=round(dc_x + (patch.x - sc_x) * s), y=round(dc_y + (patch.y - sc_y) * s))


def clip_patch(patch: Patch, box: tuple[float, float, float, float]) -> Patch:
    """Return ``patch`` with alpha zeroed outside ``box`` (same image size and position)."""
    x0, y0 = math.floor(box[0]), math.floor(box[1])
    x1, y1 = math.ceil(box[2]), math.ceil(box[3])
    arr = np.array(patch.image.convert("RGBA"))
    height, width = arr.shape[0], arr.shape[1]
    xs = patch.x + np.arange(width)
    ys = patch.y + np.arange(height)
    keep = (ys >= y0)[:, None] & (ys < y1)[:, None] & (xs >= x0)[None, :] & (xs < x1)[None, :]
    arr[~keep, 3] = 0
    return Patch(image=Image.fromarray(arr), x=patch.x, y=patch.y)


def save_patch(patch: Patch, path: Path) -> None:
    """Save ``patch`` as ``<path>.png`` (RGBA) plus ``<path>.json`` (x, y)."""
    patch.image.convert("RGBA").save(Path(str(path) + ".png"))
    Path(str(path) + ".json").write_text(json.dumps({"x": patch.x, "y": patch.y}), encoding="utf-8")


def load_patch(path: Path) -> Patch:
    """Load a patch saved with :func:`save_patch`."""
    data = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    image = Image.open(Path(str(path) + ".png")).convert("RGBA")
    return Patch(image=image, x=int(data["x"]), y=int(data["y"]))
