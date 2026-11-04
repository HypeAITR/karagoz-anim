"""Measured view angles from puppet geometry (package K-0c, section 6).

A turned figure foreshortens lateral distances by cos(yaw). For every mirrored
``_l``/``_r`` pair (shoulder caps, hands, legs, feet) the pivot-to-pivot lateral
distance over the same pair's distance in the front view gives |cos yaw|; the
sign comes from which side's parts are narrower or hidden (the far side) and
from the front-centre feature offset. Near-profile views (|cos| < 0.3), where
lateral pairs collapse, use the near shoulder pivot offset from the torso
centre over the torso half width instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.model import Puppet

# Cosine below which a view counts as near-profile (about 72 degrees).
PROFILE_COS = 0.3
# Declared-vs-measured yaw gap that flags a view, in degrees.
YAW_FLAG_DEG = 8.0
# Median fill-colour distance (0-255 per channel) that flags a palette change.
PALETTE_FLAG_DIST = 40.0


@dataclass
class YawEstimate:
    """A measured turn angle and the per-pair evidence behind it."""

    yaw_deg: float
    pair_angles_deg: dict[str, float] = field(default_factory=dict)
    spread_deg: float = 0.0
    method: str = "none"
    near_side: str | None = None
    note: str = ""


def pair_stems(puppet: Puppet) -> dict[str, tuple[str, str]]:
    """Mirrored ``_l``/``_r`` part name pairs, e.g. arm_l/arm_r -> arm."""
    sides: dict[str, dict[str, str]] = {}
    for name in puppet.parts:
        if name.endswith("_l") or name.endswith("_r"):
            sides.setdefault(name[:-2], {})[name[-1:]] = name
    return {stem: (pair["l"], pair["r"]) for stem, pair in sides.items() if "l" in pair and "r" in pair}


def silhouette_box(masks: dict[str, NDArray[np.bool_]]) -> tuple[int, int, int, int]:
    """Inclusive (x0, y0, x1, y1) bounding box of the union of the masks."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for mask in masks.values():
        ys, xs = np.nonzero(np.asarray(mask, dtype=np.bool_))
        if len(xs) == 0:
            continue
        xs0.append(int(xs.min()))
        ys0.append(int(ys.min()))
        xs1.append(int(xs.max()))
        ys1.append(int(ys.max()))
    if not xs0:
        return (0, 0, 0, 0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def long_axis_len(mask: NDArray[np.bool_]) -> float:
    """Major extent of a mask along its principal axis, in pixels."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.bool_))
    if len(xs) < 8:
        return 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centre = pts.mean(axis=0)
    _, _, vectors = np.linalg.svd(pts - centre, full_matrices=False)
    along = (pts - centre) @ vectors[0]
    return float(along.max() - along.min())


def median_fill(rgb: NDArray[np.uint8], mask: NDArray[np.bool_], line: NDArray[np.bool_]) -> tuple[float, float, float]:
    """Median non-line colour under a mask; black when the mask shows no fill."""
    cells = np.asarray(mask & ~line, dtype=np.bool_)
    if not bool(cells.any()):
        return (0.0, 0.0, 0.0)
    med = np.median(np.asarray(rgb[cells], dtype=np.float64), axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def palette_distance(
    a_rgb: NDArray[np.uint8],
    a_mask: NDArray[np.bool_],
    a_line: NDArray[np.bool_],
    b_rgb: NDArray[np.uint8],
    b_mask: NDArray[np.bool_],
    b_line: NDArray[np.bool_],
) -> float:
    """Euclidean distance of two parts' median fill colours (own line each)."""
    fa = median_fill(a_rgb, a_mask, a_line)
    fb = median_fill(b_rgb, b_mask, b_line)
    return float(math.dist(fa, fb))


def centerline_yaw(seam_offset_px: float, ruler_px: float) -> float:
    """Yaw from a centre-front detail's lateral offset, in degrees.

    A detail painted on the body's front centre (a chest seam) rides the
    body cylinder: offset = ruler * sin(yaw), where the ruler is the body's
    half depth measured from a profile view's torso. The sign follows the
    offset side. Near edge-on the ratio clamps, so a profile reads ±90.
    """

    if ruler_px <= 0:
        raise ValueError(f"ruler must be positive, got {ruler_px}")
    ratio = max(-1.0, min(1.0, seam_offset_px / ruler_px))
    return math.degrees(math.asin(ratio))


def looks_like_back(front_features: dict[str, Any], view_features: dict[str, Any]) -> bool:
    """True when a view shows the body's back.

    Every design detail that faces forward in the front view (its
    ``front: true`` features: face, chest seam, chest grid) is absent from
    this view's preset. Feature presence is preset data read from the
    drawing, not guessed; a view misread this way disagrees with its
    declared yaw and gets flagged for the review tool.
    """

    wanted = [name for name, spec in front_features.items() if bool(getattr(spec, "front", False))]
    if not wanted:
        return False
    return not any(name in view_features for name in wanted)


def yaw_flag(measured: float | None, declared: float) -> bool:
    """True when a measured yaw disagrees with the declared one by over 8 degrees."""
    if measured is None:
        return False
    return abs(measured - declared) > YAW_FLAG_DEG


def _lateral(pivots: dict[str, list[float]], left: str, right: str, height: float) -> float:
    return abs(float(pivots[left][0]) - float(pivots[right][0])) / max(height, 1.0)


def _figure_height(masks: dict[str, NDArray[np.bool_]]) -> float:
    x0, y0, x1, y1 = silhouette_box(masks)
    return float(max(y1 - y0 + 1, 1))


def estimate_yaw(
    puppet: Puppet,
    masks: dict[str, NDArray[np.bool_]],
    front_puppet: Puppet,
    front_masks: dict[str, NDArray[np.bool_]],
) -> YawEstimate:
    """Measure a view's yaw against the front view, in degrees.

    Positive yaw turns toward the viewer's right (the character's left side,
    the ``_l`` parts, go far). Returns the median pair angle; the spread of
    the pair angles is also reported (a large spread means the pose changed,
    not just the viewpoint).
    """
    pairs = pair_stems(puppet)
    front_pairs = pair_stems(front_puppet)
    height = _figure_height(masks)
    front_height = _figure_height(front_masks)
    pivots = {name: part.pivot for name, part in puppet.parts.items() if not part.hidden}
    front_pivots = {name: part.pivot for name, part in front_puppet.parts.items() if not part.hidden}
    angles: dict[str, float] = {}
    for stem, (left, right) in pairs.items():
        if stem not in front_pairs:
            continue
        front_left, front_right = front_pairs[stem]
        if left not in pivots or right not in pivots:
            continue
        if front_left not in front_pivots or front_right not in front_pivots:
            continue
        ratio = _lateral(pivots, left, right, height) / max(
            _lateral(front_pivots, front_left, front_right, front_height), 1e-9
        )
        angles[stem] = math.degrees(math.acos(max(0.0, min(1.0, ratio))))
    if not angles:
        return YawEstimate(yaw_deg=0.0, method="none", note="no shared mirrored pairs")
    area_l = sum(
        int(np.asarray(masks[left], dtype=np.bool_).sum()) for _, (left, _right) in pairs.items() if left in masks
    )
    area_r = sum(
        int(np.asarray(masks[right], dtype=np.bool_).sum()) for _, (_left, right) in pairs.items() if right in masks
    )
    near_side: str | None = None
    if area_l != area_r:
        near_side = "r" if area_l < area_r else "l"
    # An exact tie cannot tell the sides apart; report the magnitude as a
    # positive yaw. Only |measured - declared| feeds the flag, so the tie side
    # never matters downstream.
    sign = 1.0 if near_side != "l" else -1.0
    ordered = sorted(angles.values())
    median = ordered[len(ordered) // 2]
    spread = (ordered[-1] - ordered[0]) if len(ordered) > 1 else 0.0
    cos_med = math.cos(math.radians(median))
    if cos_med < PROFILE_COS:
        profile = _profile_yaw(puppet, masks, front_masks)
        if profile is not None:
            estimate, used = profile
            note = f"near-profile via {used}; pairs median {median:.1f} spread {spread:.1f}"
            return YawEstimate(
                yaw_deg=estimate,
                pair_angles_deg=dict(angles),
                spread_deg=spread,
                method="profile",
                near_side=("l" if estimate > 0 else "r") if estimate != 0 else None,
                note=note,
            )
    return YawEstimate(
        yaw_deg=sign * median,
        pair_angles_deg=dict(angles),
        spread_deg=spread,
        method="pairs",
        near_side=near_side,
        note=f"pairs median {median:.1f} spread {spread:.1f}",
    )


def _profile_yaw(
    puppet: Puppet, masks: dict[str, NDArray[np.bool_]], front_masks: dict[str, NDArray[np.bool_]]
) -> tuple[float, str] | None:
    """Yaw from the visible side's pivot offset over the front torso half width.

    In near-profile the view's own torso is collapsed to its depth, so the
    offset is scaled by the FRONT torso half width (a turn-invariant ruler):
    offset 0 sits at the torso centre (90 degrees). Uses the shoulder pair
    when present, else the first pair with exactly one side visible. The sign
    comes from which side is visible (left visible means a turn toward the
    viewer's right). Returns (yaw_deg, used_stem).
    """
    stems = pair_stems(puppet)
    order = sorted(stems, key=lambda s: (0 if "shoulder" in s else 1, s))
    root = puppet.parts.get(puppet.root)
    if root is None or root.hidden or puppet.root not in masks:
        return None
    front_torso = front_masks.get(puppet.root)
    if front_torso is None:
        return None
    fys, fxs = np.nonzero(np.asarray(front_torso, dtype=np.bool_))
    if len(fxs) == 0:
        return None
    front_hw = max((float(fxs.max()) - float(fxs.min())) / 2.0, 1.0)
    torso = np.asarray(masks[puppet.root], dtype=np.bool_)
    ys, xs = np.nonzero(torso)
    if len(xs) == 0:
        return None
    torso_cx = (float(xs.min()) + float(xs.max())) / 2.0
    for stem in order:
        left, right = stems[stem]
        visible = [n for n in (left, right) if n in masks and bool(masks[n].any())]
        if len(visible) != 1:
            continue
        side = "l" if visible[0] == left else "r"
        pivot = puppet.parts[visible[0]].pivot
        offset = (float(pivot[0]) - torso_cx) / front_hw
        offset = max(-1.0, min(1.0, offset))
        side_sign = 1.0 if side == "l" else -1.0
        yaw = side_sign * (90.0 - math.degrees(math.asin(offset)))
        return (yaw, stem)
    return None
