"""Shape measures used by the rig (silhouette, ink, distances).

Ported from the experiment's ``deneme/faz1c/olcu.py``: only the functions
``masks.py`` needs. Same maths, English identifiers; each public function notes
its experiment name. Thresholds are module-level ``UPPER_CASE``.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

STUDIO_GREEN = (150, 190, 150)  # experiment: BG
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
INK_MAX_LUMA = 48
INK_MAX_SATURATION = 40
WHITE_MIN_VALUE = 238
WHITE_MAX_SPREAD = 18
STUDIO_GREEN_MAX_DISTANCE = 22.0


# experiment: ink_mask
def ink_mask(rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """True where the pixel is dark neutral ink (black outline)."""
    luma = rgb.astype(np.float32) @ np.array(LUMA_WEIGHTS, np.float32)
    sat = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    return (luma < INK_MAX_LUMA) & (sat < INK_MAX_SATURATION)


# experiment: foreground
def foreground(rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Subject mask: white or studio-green components touching the edge are background."""
    lo = rgb.min(2).astype(np.int16)
    hi = rgb.max(2).astype(np.int16)
    white = (lo > WHITE_MIN_VALUE) & ((hi - lo) < WHITE_MAX_SPREAD)
    green = (
        np.linalg.norm(rgb.astype(np.float32) - np.array(STUDIO_GREEN, np.float32), axis=2) < STUDIO_GREEN_MAX_DISTANCE
    )
    cand = white | green
    _n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    bg = np.zeros(cand.shape, bool)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    for i in edge:
        if i:
            bg |= lab == i
    return ~bg


# experiment: nearest_indices
def nearest_indices(seeds: NDArray[np.bool_]) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    """Distance to the nearest seed pixel, plus that seed's (row, column) per pixel."""
    src = np.where(seeds, 0, 255).astype(np.uint8)
    _dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(seeds)
    iy_map = np.zeros(int(lab.max()) + 1, np.int32)
    ix_map = np.zeros(int(lab.max()) + 1, np.int32)
    iy_map[lab[ys, xs]] = ys
    ix_map[lab[ys, xs]] = xs
    iy: NDArray[np.int32] = iy_map[lab]
    ix: NDArray[np.int32] = ix_map[lab]
    return cast(NDArray[np.float32], _dist), iy, ix


# experiment: disk
def disk(shape: tuple[int, int], x: float, y: float, r: float) -> NDArray[np.bool_]:
    """Filled disk of radius ``r`` centred at (``x``, ``y``) on a ``shape`` canvas."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r
