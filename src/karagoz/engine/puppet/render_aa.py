"""Fast anti-aliased puppet renderer (package K-2b).

Same picture as :func:`pose.render_pose` but cheaper and smoother: every part
image is cropped once to its ink bounding box (:func:`prepare_parts`), warped
with bilinear interpolation only into the warped crop's destination box, and
composited with "over" in premultiplied space in z order (:func:`render_frame`).
:func:`part_masks` renders each part alone for the K-2c measures.

All placement numbers come from ``pose.part_matrix3`` / ``root_matrix3`` (never
recomputed here), so future pose fields (K-1 offsets) flow through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet import pose as _pose_mod
from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.pose import Pose

#: Transparent padding around the ink box, in reference pixels.
PAD_PX = 2


@dataclass
class PreparedPart:
    """One layer cropped to its ink, premultiplied float32 in 0..1."""

    image: NDArray[np.float32]  # (h, w, 4) premultiplied RGBA, 0..1
    offset: tuple[int, int]  # (x0, y0): crop origin in reference pixels
    full_shape: tuple[int, int]  # (h, w) of the original full-canvas image


@dataclass
class PreparedParts:
    """All croppable layers of one puppet view, ready for fast warps."""

    parts: dict[str, PreparedPart] = field(default_factory=dict)
    shadow: PreparedPart | None = None


def _crop_to_ink(img: NDArray[np.uint8]) -> PreparedPart:
    """Crop an RGBA image to its alpha bounding box plus padding, premultiplied."""
    height, width = int(img.shape[0]), int(img.shape[1])
    alpha = img[..., 3]
    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        return PreparedPart(
            image=np.zeros((1, 1, 4), dtype=np.float32),
            offset=(0, 0),
            full_shape=(height, width),
        )
    x0 = max(0, int(xs.min()) - PAD_PX)
    y0 = max(0, int(ys.min()) - PAD_PX)
    x1 = min(width, int(xs.max()) + 1 + PAD_PX)
    y1 = min(height, int(ys.max()) + 1 + PAD_PX)
    crop = np.asarray(img[y0:y1, x0:x1], dtype=np.uint8)
    flat = crop.astype(np.float32) / np.float32(255.0)
    alpha_f = flat[..., 3:4]
    pre = np.concatenate([flat[..., :3] * alpha_f, alpha_f], axis=2)
    return PreparedPart(
        image=np.ascontiguousarray(pre, dtype=np.float32),
        offset=(x0, y0),
        full_shape=(height, width),
    )


def prepare_parts(puppet: Puppet, images: dict[str, NDArray[np.uint8]]) -> PreparedParts:
    """Crop every part image once to its ink box, premultiplied float32.

    ``_shadow`` is kept aside (warped by the root similarity only, drawn
    first, like ``render_pose``). Entries that are not RGBA, hidden parts
    without pixels, and names the puppet does not know are skipped, mirroring
    what ``render_pose`` would draw.
    """
    out = PreparedParts()
    for name, img in images.items():
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 4:
            continue
        if name == "_shadow":
            out.shadow = _crop_to_ink(arr)
        elif name in puppet.parts:
            out.parts[name] = _crop_to_ink(arr)
    return out


def _warp_crop_into(
    crop: NDArray[np.float32],
    mat3: NDArray[np.float64],
    width: int,
    height: int,
) -> tuple[NDArray[np.float32], tuple[int, int, int, int]]:
    """Warp ``crop`` with bilinear filtering into its destination box only.

    Returns the warped ROI and its box ``(x0, y0, w, h)`` on the canvas; the
    box is empty when the part falls fully outside.
    """
    crop_h, crop_w = int(crop.shape[0]), int(crop.shape[1])
    corners = np.array(
        [[0.0, 0.0, 1.0], [float(crop_w), 0.0, 1.0], [float(crop_w), float(crop_h), 1.0], [0.0, float(crop_h), 1.0]],
        dtype=np.float64,
    )
    moved = (np.asarray(mat3, dtype=np.float64) @ corners.T).T
    xs = moved[:, 0] / moved[:, 2]
    ys = moved[:, 1] / moved[:, 2]
    x0 = max(0, int(np.floor(float(xs.min()))) - 1)
    y0 = max(0, int(np.floor(float(ys.min()))) - 1)
    x1 = min(width, int(np.ceil(float(xs.max()))) + 1)
    y1 = min(height, int(np.ceil(float(ys.max()))) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 4), dtype=np.float32), (0, 0, 0, 0)
    roi = np.asarray(mat3, dtype=np.float64).copy()
    roi[0, 2] -= float(x0)
    roi[1, 2] -= float(y0)
    warped = np.asarray(
        cv2.warpAffine(
            crop,
            roi[:2].astype(np.float64),
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0, 0),
        ),
        dtype=np.float32,
    )
    return warped, (x0, y0, x1 - x0, y1 - y0)


def _over_onto(canvas: NDArray[np.float32], warped: NDArray[np.float32], box: tuple[int, int, int, int]) -> None:
    """Composite a warped premultiplied ROI over ``canvas`` ("over", in place)."""
    x0, y0, bw, bh = box
    if bw <= 0 or bh <= 0:
        return
    dst = canvas[y0 : y0 + bh, x0 : x0 + bw]
    src_a = warped[..., 3:4]
    keep = np.float32(1.0) - src_a
    dst[..., :3] = warped[..., :3] + dst[..., :3] * keep
    dst[..., 3:4] = src_a + dst[..., 3:4] * keep


def _composite_part(
    canvas: NDArray[np.float32],
    prep: PreparedPart,
    mat3: NDArray[np.float64],
    width: int,
    height: int,
) -> None:
    """Warp one prepared layer by ``mat3`` (reference -> output) onto ``canvas``."""
    ox, oy = prep.offset
    shift = np.array([[1.0, 0.0, float(ox)], [0.0, 1.0, float(oy)], [0.0, 0.0, 1.0]], dtype=np.float64)
    warped, box = _warp_crop_into(prep.image, np.asarray(mat3, dtype=np.float64) @ shift, width, height)
    _over_onto(canvas, warped, box)


def _unpremultiply(canvas: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Premultiplied float32 0..1 canvas -> straight uint8 RGBA.

    cv2 does the heavy passes (it threads over cores); ``cv2.divide`` yields 0
    where alpha is 0, and there the premultiplied rgb is exactly 0 as well.
    """
    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    alpha = canvas[..., 3]
    # The (..., :3) view is strided, which cv2 refuses as an output: copy it once.
    rgb = np.ascontiguousarray(canvas[..., :3])
    alpha3 = cv2.merge((alpha, alpha, alpha))
    cv2.divide(rgb, alpha3, rgb)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = cv2.convertScaleAbs(rgb, alpha=255.0)
    out[..., 3] = cv2.convertScaleAbs(alpha, alpha=255.0)
    return out


def render_frame(
    puppet: Puppet,
    prepared: PreparedParts,
    pose: Pose,
    z_order: list[str] | None = None,
) -> NDArray[np.uint8]:
    """Draw the puppet at ``pose`` as uint8 RGBA on the pose's output canvas.

    Bilinear warps of the cropped parts, "over" compositing in premultiplied
    space in ``z_order`` (or :func:`pose.z_order` when not given); the shadow
    layer goes first under the root similarity only.
    """
    width, height = _pose_mod.output_size(puppet, pose)
    canvas = np.zeros((height, width, 4), dtype=np.float32)
    if prepared.shadow is not None:
        _composite_part(canvas, prepared.shadow, _pose_mod.root_matrix3(puppet, pose), width, height)
    order = list(z_order) if z_order is not None else _pose_mod.z_order(puppet, pose)
    for name in order:
        part = puppet.parts.get(name)
        if part is None or part.hidden:
            continue
        crop = prepared.parts.get(name)
        if crop is None:
            continue
        _composite_part(canvas, crop, _pose_mod.part_matrix3(puppet, pose, name), width, height)
    return _unpremultiply(canvas)


def part_masks(puppet: Puppet, prepared: PreparedParts, pose: Pose) -> dict[str, NDArray[np.bool_]]:
    """Per visible part, full-canvas bool mask: warped alpha > 0.5 (same warp)."""
    width, height = _pose_mod.output_size(puppet, pose)
    out: dict[str, NDArray[np.bool_]] = {}
    for name, part in puppet.parts.items():
        if part.hidden:
            continue
        crop = prepared.parts.get(name)
        if crop is None:
            continue
        ox, oy = crop.offset
        shift = np.array([[1.0, 0.0, float(ox)], [0.0, 1.0, float(oy)], [0.0, 0.0, 1.0]], dtype=np.float64)
        mat = _pose_mod.part_matrix3(puppet, pose, name) @ shift
        warped, box = _warp_crop_into(crop.image, mat, width, height)
        x0, y0, bw, bh = box
        full = np.zeros((height, width), dtype=np.bool_)
        if bw > 0 and bh > 0:
            full[y0 : y0 + bh, x0 : x0 + bw] = warped[..., 3] > np.float32(0.5)
        out[name] = full
    return out
