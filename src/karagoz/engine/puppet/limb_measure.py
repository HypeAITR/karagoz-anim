"""Independent limb-by-limb measurement on target drawings and posed puppet renders.

Separates two distinct sources of discrepancy without hardcoded character rules:
1. ``drawing_diff``: the visual generator drew a limb with a different length,
   width, or local shape than the rigid puppet sprite (substitution candidate).
2. ``motor_error``: limb dimensions match, or the limb barely changed between
   two drawings, yet the fitting engine chose an arbitrary angle or displaced
   joint/leaf orientation.

All part signatures (flat-colour landmarks, tube colours, paired distal lobes,
and posed search windows) are derived automatically from the puppet's own RGBA
sprites and hierarchy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.parts import line_mask
from karagoz.engine.puppet.pose import Pose, chains, part_matrix3

LimbKind = Literal["bone", "leaf"]
MeasureMethod = Literal["landmark", "template"]
PoseLimbStatus = Literal["ok", "drawing_diff", "motor_error"]
TransitionLimbStatus = Literal["ok", "motor_error"]

# Joint pivots closer than this in rest space belong to the same anatomical
# joint (e.g. a shoulder cap or elbow ring sitting on the bone pivot).
COLOCATED_PIVOT_MAX_PX = 20.0

# Pose-exam thresholds.
BONE_LEN_DIFF_RATIO_MAX = 0.038
BONE_WIDTH_DIFF_RATIO_MAX = 0.16
LEAF_DIM_DIFF_RATIO_MAX = 0.16
JOINT_SHIFT_ERR_PX = 12.0
BONE_ANGLE_ERR_DEG = 5.0
LEAF_ANGLE_ERR_DEG = 5.0

# Transition-exam thresholds.
UNCHANGED_MAX_ANGLE_DEG = 3.5
UNCHANGED_MAX_SHIFT_PX = 15.0
TRANSITION_ANGLE_RATIO_MAX = 2.0
TRANSITION_ANGLE_EXCESS_DEG = 4.0
TRANSITION_SHIFT_RATIO_MAX = 2.0
TRANSITION_SHIFT_EXCESS_PX = 12.0


class LimbMeasurement(BaseModel):
    """Single limb or leaf measurement extracted from an RGB image."""

    name: str
    kind: LimbKind
    chain_leaf: str
    parent_bone: str | None = None
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    length_px: float
    width_px: float
    angle_deg: float
    method: MeasureMethod = "landmark"


class PoseLimbExam(BaseModel):
    """Comparison of one limb between target drawing and posed puppet render."""

    name: str
    kind: LimbKind
    drawing: LimbMeasurement
    puppet: LimbMeasurement
    length_diff_px: float
    length_diff_ratio: float
    width_diff_px: float
    width_diff_ratio: float
    joint_shift_px: float
    angle_diff_deg: float
    abs_angle_diff_deg: float
    status: PoseLimbStatus
    reason: str = ""


class PoseExamReport(BaseModel):
    """Full limb-by-limb exam for a single pose."""

    pose_name: str
    limbs: list[PoseLimbExam] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    summary: str = ""


class TransitionLimbExam(BaseModel):
    """Comparison of one limb's change across a transition (pose A -> pose B)."""

    name: str
    kind: LimbKind
    drawing_angle_change_deg: float
    puppet_angle_change_deg: float
    drawing_shift_px: float
    puppet_shift_px: float
    changed_in_drawing: bool
    status: TransitionLimbStatus
    reason: str = ""


class TransitionExamReport(BaseModel):
    """Full limb-by-limb exam across a transition between two poses."""

    transition_name: str
    pose_a: str
    pose_b: str
    limbs: list[TransitionLimbExam] = Field(default_factory=list)
    unchanged_limbs: list[str] = Field(default_factory=list)
    motor_error_limbs: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    summary: str = ""


def _wrap_deg(deg: float) -> float:
    """Wrap an angle difference into (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else float(d)


def _abs_ang_diff(a: float, b: float) -> float:
    return abs(_wrap_deg(a - b))


def _warp_pts(pts: np.ndarray, mat2x3: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts.astype(np.float64), ones])
    out: np.ndarray = hom @ mat2x3.T
    return out


def _alpha_bbox(rgba: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(rgba[..., 3] > 127)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _flat_crop(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    box = _alpha_bbox(rgba)
    if box is None:
        return (
            np.zeros((0, 0, 4), dtype=np.uint8),
            np.zeros((0, 0), dtype=bool),
            (0, 0),
        )
    x0, y0, x1, y1 = box
    crop = rgba[y0:y1, x0:x1]
    alpha = crop[..., 3] > 127
    ink = line_mask(crop[..., :3], 25.0) & alpha
    flat = alpha & ~ink
    return crop, (flat if flat.any() else alpha), (x0, y0)


class _LandmarkDetector:
    """Learned colour-contrast signature for a joint cap or leaf extremity."""

    def __init__(
        self,
        c_ref: np.ndarray,
        c_tube: np.ndarray,
        u_hat: np.ndarray,
        norm: float,
        ref_cent: np.ndarray,
        paired: bool,
        distinct_pts: np.ndarray,
    ) -> None:
        self.c_ref = c_ref
        self.c_tube = c_tube
        self.u_hat = u_hat
        self.norm = norm
        self.ref_cent = ref_cent
        self.paired = paired
        self.distinct_pts = distinct_pts


def _learn_landmark_detector(
    part_rgba: np.ndarray,
    tube_rgba: np.ndarray,
    pivot_xy: Sequence[float],
    *,
    distal_only: bool = False,
) -> _LandmarkDetector | None:
    """Learn a colour landmark detector from a part sprite vs its adjacent bone tube."""
    part_crop, flat, (ox, oy) = _flat_crop(part_rgba)
    tube_crop, tube_flat, _ = _flat_crop(tube_rgba)
    if not flat.any() or not tube_flat.any():
        return None
    c_tube = np.median(tube_crop[tube_flat, :3].astype(np.float64), axis=0)

    n_cc, cc = cv2.connectedComponents(flat.astype(np.uint8), connectivity=8)
    piv = np.array(pivot_xy, dtype=np.float64)
    distinct_comps: list[dict[str, Any]] = []
    for cid in range(1, n_cc):
        m = cc == cid
        area = int(m.sum())
        if area < 100:
            continue
        col = np.median(part_crop[m, :3].astype(np.float64), axis=0)
        diff = float(np.linalg.norm(col - c_tube))
        ys, xs = np.nonzero(m)
        cent = np.array([xs.mean() + ox, ys.mean() + oy], dtype=np.float64)
        dist_piv = float(np.hypot(cent[0] - piv[0], cent[1] - piv[1]))
        if diff >= 14.0:
            distinct_comps.append(
                {
                    "area": area,
                    "col": col,
                    "cent": cent,
                    "dist_piv": dist_piv,
                    "mask": m,
                }
            )

    if not distinct_comps:
        return None
    if distal_only and len(distinct_comps) > 1:
        max_d = max(float(c["dist_piv"]) for c in distinct_comps)
        distinct_comps = [c for c in distinct_comps if float(c["dist_piv"]) >= 0.5 * max_d]
    distinct_mask = np.zeros_like(flat)
    for c in distinct_comps:
        distinct_mask |= c["mask"]
    c_ref = np.median(part_crop[distinct_mask, :3].astype(np.float64), axis=0)
    delta = c_ref - c_tube
    norm = float(np.linalg.norm(delta))
    if norm < 12.0:
        return None
    u_hat = delta / norm
    ys, xs = np.nonzero(distinct_mask)
    pts_full = np.stack([xs + ox, ys + oy], axis=1).astype(np.float64)
    ref_cent = pts_full.mean(axis=0)
    if len(pts_full) > 500:
        pts_full = pts_full[:: max(1, len(pts_full) // 500)]
    total_distinct_area = sum(int(c["area"]) for c in distinct_comps)
    min_lobe_area = max(100, int(0.15 * total_distinct_area))
    large_comps = sorted(
        [c for c in distinct_comps if int(c["area"]) >= min_lobe_area],
        key=lambda x: int(x["area"]),
        reverse=True,
    )
    paired = len(large_comps) >= 2 and (int(large_comps[1]["area"]) >= 0.35 * int(large_comps[0]["area"]))
    return _LandmarkDetector(
        c_ref=c_ref,
        c_tube=c_tube,
        u_hat=u_hat,
        norm=norm,
        ref_cent=ref_cent,
        paired=paired,
        distinct_pts=pts_full,
    )


def _get_part_search_bbox(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose: Pose,
    part_name: str,
    det: _LandmarkDetector | None = None,
    *,
    margin: int = 45,
) -> tuple[int, int, int, int]:
    """Compute a pose-derived search window that neutralizes leaf angle."""
    chains_map = chains(puppet)
    neutral_angles = dict(pose.angles)
    if part_name in chains_map:
        neutral_angles[part_name] = 0.0
    p_neutral = pose.model_copy(update={"angles": neutral_angles})
    if det is not None and len(det.distinct_pts) > 0:
        pts = det.distinct_pts
    else:
        ys, xs = np.nonzero(images[part_name][..., 3] > 127)
        if len(xs) == 0:
            piv = puppet.parts[part_name].pivot
            pts = np.array([[piv[0], piv[1]]], dtype=np.float64)
        else:
            pts = np.stack([xs, ys], axis=1).astype(np.float64)
            if len(pts) > 500:
                pts = pts[:: max(1, len(pts) // 500)]
    w1 = _warp_pts(pts, part_matrix3(puppet, p_neutral, part_name)[:2])
    w2 = _warp_pts(pts, part_matrix3(puppet, pose, part_name)[:2])
    all_pts = np.vstack([w1, w2])
    x0 = int(np.floor(all_pts[:, 0].min())) - margin
    y0 = int(np.floor(all_pts[:, 1].min())) - margin
    x1 = int(np.ceil(all_pts[:, 0].max())) + margin
    y1 = int(np.ceil(all_pts[:, 1].max())) + margin
    return (x0, y0, x1, y1)


def _precompute_warped_part_masks(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose: Pose,
    shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Warp each visible part's eroded alpha mask once per image."""
    h, w = shape_hw
    out: dict[str, np.ndarray] = {}
    kernel = np.ones((7, 7), np.uint8)
    for name, part in puppet.parts.items():
        if part.hidden or name not in images:
            continue
        rgba = images[name]
        box = _alpha_bbox(rgba)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        crop_a = (rgba[y0:y1, x0:x1, 3] > 127).astype(np.uint8)
        crop_eroded = cv2.erode(crop_a, kernel)
        if not crop_eroded.any():
            continue
        m3 = part_matrix3(puppet, pose, name).copy()
        offset_m = np.array(
            [[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        m2x3 = (m3 @ offset_m)[:2]
        warped = cv2.warpAffine(crop_eroded, m2x3, (w, h), flags=cv2.INTER_NEAREST) > 0
        out[name] = warped
    return out


def _build_non_chain_mask(
    warped_masks: dict[str, np.ndarray],
    chain_parts: set[str],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Combine pre-warped eroded masks of parts outside `chain_parts`."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=bool)
    for name, wmask in warped_masks.items():
        if name not in chain_parts:
            mask |= wmask
    return mask


def _detect_clean_landmark(
    img_rgb: np.ndarray,
    det: _LandmarkDetector,
    bbox: tuple[int, int, int, int],
    other_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Detect landmark pixels in `img_rgb` within `bbox`."""
    x0, y0, x1, y1 = bbox
    h, w = img_rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=bool), (x0, y0)
    crop = img_rgb[y0:y1, x0:x1].astype(np.float64)
    val = crop.max(axis=2)
    mn = crop.min(axis=2)
    not_ink_paper = (val > 38.0) & (mn < 225.0)
    proj = (crop - det.c_tube) @ det.u_hat
    dist_ref = np.linalg.norm(crop - det.c_ref, axis=2)
    mask = not_ink_paper & (proj >= 0.40 * det.norm) & (dist_ref <= max(28.0, 1.2 * det.norm))
    if other_mask is not None:
        mask = mask & ~other_mask[y0:y1, x0:x1]
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    clean = np.zeros_like(mask)
    for cid in range(1, n_cc):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= 40:
            clean |= cc == cid
    return clean, (x0, y0)


def _measure_bone_width(
    img_rgb: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    tube_rgba: np.ndarray,
    other_mask: np.ndarray | None = None,
) -> float:
    """Measure the cross-sectional width of a bone between `start_xy` and `end_xy`."""
    vec = end_xy - start_xy
    length = float(np.hypot(vec[0], vec[1]))
    if length < 8.0:
        return 0.0
    ux, uy = vec[0] / length, vec[1] / length
    nx, ny = -uy, ux

    tube_crop, flat, _ = _flat_crop(tube_rgba)
    if not flat.any():
        return 0.0
    c_tube = np.median(tube_crop[flat, :3].astype(np.float64), axis=0)

    max_half_w = max(30.0, min(80.0, 0.75 * float(min(tube_crop.shape[:2]))))

    h, w = img_rgb.shape[:2]
    x0 = max(0, int(min(start_xy[0], end_xy[0]) - max_half_w))
    y0 = max(0, int(min(start_xy[1], end_xy[1]) - max_half_w))
    x1 = min(w, int(max(start_xy[0], end_xy[0]) + max_half_w))
    y1 = min(h, int(max(start_xy[1], end_xy[1]) + max_half_w))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = img_rgb[y0:y1, x0:x1].astype(np.float64)
    dist_col = np.linalg.norm(crop - c_tube, axis=2)
    not_paper = crop.min(axis=2) < 230.0
    fg = not_paper & (dist_col <= 38.0)
    if other_mask is not None:
        fg = fg & ~other_mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(fg)
    if len(xs) < 20:
        return 0.0
    px = xs.astype(np.float64) + x0 - start_xy[0]
    py = ys.astype(np.float64) + y0 - start_xy[1]
    t = (px * ux + py * uy) / length
    s = px * nx + py * ny
    mid = (t >= 0.25) & (t <= 0.75) & (np.abs(s) <= max_half_w)
    if int(mid.sum()) < 15:
        return 0.0
    s_mid = s[mid]
    return float(np.percentile(s_mid, 90) - np.percentile(s_mid, 10))


def _measure_template_part(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose: Pose,
    part_name: str,
    distal_pivot_xy: Sequence[float] | None,
    img_rgb: np.ndarray,
    img_ink: np.ndarray,
    img_ink_dt: np.ndarray,
    other_mask: np.ndarray | None,
    *,
    is_leaf: bool,
) -> tuple[tuple[float, float], tuple[float, float], float, float, float]:
    """Measure a non-landmark bone or leaf directly from image pixels in its window."""
    rgba = images[part_name]
    part_crop, flat_crop, (ox_s, oy_s) = _flat_crop(rgba)
    if part_crop.size == 0:
        piv0 = puppet.parts[part_name].pivot
        return (
            (float(piv0[0]), float(piv0[1])),
            (float(piv0[0]), float(piv0[1])),
            0.0,
            0.0,
            0.0,
        )
    alpha_crop = part_crop[..., 3] > 127
    c_med = (
        np.median(part_crop[flat_crop, :3].astype(np.float64), axis=0)
        if flat_crop.any()
        else np.array([128.0, 128.0, 128.0])
    )
    piv = np.array(puppet.parts[part_name].pivot, dtype=np.float64)
    ys_a, xs_a = np.nonzero(alpha_crop)
    pts_sprite = np.stack([xs_a + ox_s, ys_a + oy_s], axis=1).astype(np.float64)
    cent_sprite = pts_sprite.mean(axis=0)
    end_sprite = np.array(distal_pivot_xy, dtype=np.float64) if distal_pivot_xy is not None else cent_sprite

    rest_vec = end_sprite - piv
    rest_len = float(np.hypot(rest_vec[0], rest_vec[1]))
    if rest_len < 4.0:
        diffs = pts_sprite - piv
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        far_idx = int(np.argmax(dists))
        end_sprite = pts_sprite[far_idx]
        rest_vec = end_sprite - piv
        rest_len = max(1.0, float(np.hypot(rest_vec[0], rest_vec[1])))

    ux_r, uy_r = rest_vec[0] / rest_len, rest_vec[1] / rest_len
    nx_r, ny_r = -uy_r, ux_r
    t_sprite = (pts_sprite[:, 0] - piv[0]) * ux_r + (pts_sprite[:, 1] - piv[1]) * uy_r
    s_sprite = (pts_sprite[:, 0] - piv[0]) * nx_r + (pts_sprite[:, 1] - piv[1]) * ny_r
    sprite_t_span = max(1.0, float(np.percentile(t_sprite, 95) - np.percentile(t_sprite, 5)))
    sprite_s_span = max(1.0, float(np.percentile(s_sprite, 90) - np.percentile(s_sprite, 10)))

    # Neutralize the part's own angle if it is a leaf so the scan does not cheat
    # by reading pose.angles[part_name].
    base_angles = dict(pose.angles)
    if is_leaf:
        base_angles[part_name] = 0.0
    p_base = pose.model_copy(update={"angles": base_angles})

    ink_crop = line_mask(part_crop[..., :3], 25.0) & alpha_crop
    sub_y, sub_x = np.nonzero(ink_crop)
    if len(sub_x) > 350:
        step = max(1, len(sub_x) // 350)
        sub_x, sub_y = sub_x[::step], sub_y[::step]
    ink_pts = (
        np.stack([sub_x + ox_s, sub_y + oy_s], axis=1).astype(np.float64)
        if len(sub_x) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )

    f_y, f_x = np.nonzero(flat_crop)
    if len(f_x) > 350:
        step = max(1, len(f_x) // 350)
        f_x, f_y = f_x[::step], f_y[::step]
    fill_pts = (
        np.stack([f_x + ox_s, f_y + oy_s], axis=1).astype(np.float64)
        if len(f_x) > 0
        else np.zeros((0, 2), dtype=np.float64)
    )

    h, w = img_rgb.shape[:2]
    base_deg = float(p_base.angles.get(part_name, 0.0))
    best_delta = 0.0
    best_score = -1e9

    def _eval_delta(d_deg: float) -> float:
        cand_angles = dict(base_angles)
        cand_angles[part_name] = base_deg + d_deg
        p_cand = p_base.model_copy(update={"angles": cand_angles})
        m2x3 = part_matrix3(puppet, p_cand, part_name)[:2]
        score = 0.0
        if len(ink_pts) > 0:
            wi = _warp_pts(ink_pts, m2x3)
            ix = np.clip(np.round(wi[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(wi[:, 1]).astype(int), 0, h - 1)
            dt_vals = img_ink_dt[iy, ix]
            score += float(np.mean(np.exp(-0.35 * dt_vals)))
        if len(fill_pts) > 0:
            wf = _warp_pts(fill_pts, m2x3)
            fx = np.clip(np.round(wf[:, 0]).astype(int), 0, w - 1)
            fy = np.clip(np.round(wf[:, 1]).astype(int), 0, h - 1)
            cols = img_rgb[fy, fx].astype(np.float64)
            col_dist = np.linalg.norm(cols - c_med, axis=1)
            is_bg = cols.min(axis=1) >= 235.0
            score += 0.5 * float(np.mean((~is_bg) & (col_dist <= 40.0))) - 0.5 * float(np.mean(is_bg))
        return score - 1e-4 * abs(d_deg)

    for d_coarse in np.arange(-36.0, 37.0, 2.0):
        sc = _eval_delta(float(d_coarse))
        if sc > best_score:
            best_score = sc
            best_delta = float(d_coarse)
    for d_fine in np.arange(best_delta - 2.0, best_delta + 2.01, 0.25):
        sc = _eval_delta(float(d_fine))
        if sc > best_score:
            best_score = sc
            best_delta = float(d_fine)

    fitted_angles = dict(base_angles)
    fitted_angles[part_name] = base_deg + best_delta
    p_fitted = p_base.model_copy(update={"angles": fitted_angles})
    m2x3 = part_matrix3(puppet, p_fitted, part_name)[:2]

    w_endpoints = _warp_pts(np.vstack([piv, end_sprite]), m2x3)
    start_w = w_endpoints[0]
    end_w = w_endpoints[1]
    axis_vec = end_w - start_w
    axis_len = max(1e-6, float(np.hypot(axis_vec[0], axis_vec[1])))
    ux_w, uy_w = axis_vec[0] / axis_len, axis_vec[1] / axis_len
    nx_w, ny_w = -uy_w, ux_w
    angle_deg = math.degrees(math.atan2(axis_vec[1], axis_vec[0]))

    w_all = _warp_pts(pts_sprite[:: max(1, len(pts_sprite) // 400)], m2x3)
    margin = 25
    x0 = max(0, int(np.floor(w_all[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(w_all[:, 1].min())) - margin)
    x1 = min(w, int(np.ceil(w_all[:, 0].max())) + margin)
    y1 = min(h, int(np.ceil(w_all[:, 1].max())) + margin)
    if x1 <= x0 or y1 <= y0:
        return (
            (float(start_w[0]), float(start_w[1])),
            (float(end_w[0]), float(end_w[1])),
            float(axis_len),
            float(sprite_s_span),
            float(angle_deg),
        )

    crop = img_rgb[y0:y1, x0:x1].astype(np.float64)
    col_d = np.linalg.norm(crop - c_med, axis=2)
    fg = (crop.min(axis=2) < 232.0) & ((col_d <= 42.0) | img_ink[y0:y1, x0:x1])
    if other_mask is not None:
        fg = fg & ~other_mask[y0:y1, x0:x1]
    f_ys, f_xs = np.nonzero(fg)
    if len(f_xs) < 20:
        return (
            (float(start_w[0]), float(start_w[1])),
            (float(end_w[0]), float(end_w[1])),
            float(axis_len),
            float(sprite_s_span),
            float(angle_deg),
        )

    px = f_xs.astype(np.float64) + x0 - start_w[0]
    py = f_ys.astype(np.float64) + y0 - start_w[1]
    t_img = px * ux_w + py * uy_w
    s_img = px * nx_w + py * ny_w
    img_t_span = float(np.percentile(t_img, 95) - np.percentile(t_img, 5))
    img_s_span = float(np.percentile(s_img, 90) - np.percentile(s_img, 10))
    scale_t = img_t_span / sprite_t_span if sprite_t_span > 1.0 else 1.0
    measured_len = float(rest_len * scale_t)
    end_measured = start_w + np.array([ux_w, uy_w]) * measured_len
    return (
        (float(start_w[0]), float(start_w[1])),
        (float(end_measured[0]), float(end_measured[1])),
        measured_len,
        img_s_span,
        float(angle_deg),
    )


def measure_image_limbs(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose: Pose,
    img_rgb: np.ndarray,
) -> list[LimbMeasurement]:
    """Measure every limb (bones and leaves) in `img_rgb` independently of `pose.angles`."""
    chains_map = chains(puppet)
    results: list[LimbMeasurement] = []
    seen_parts: set[str] = set()

    shape_hw = (int(img_rgb.shape[0]), int(img_rgb.shape[1]))
    warped_masks = _precompute_warped_part_masks(puppet, images, pose, shape_hw)
    img_ink = line_mask(img_rgb, 25.0)
    img_ink_dt = cv2.distanceTransform((~img_ink).astype(np.uint8), cv2.DIST_L2, 3)

    for leaf_name, chain in chains_map.items():
        if not chain:
            continue
        other_mask = _build_non_chain_mask(warped_masks, set(chain), shape_hw)

        # Group chain members into anatomical bones and co-located joint caps
        # (pivot distance < max(20px, 30% of longest segment in chain)), plus leaf.
        n = len(chain)
        max_seg_dist = max(
            (
                math.hypot(
                    puppet.parts[chain[k + 1]].pivot[0] - puppet.parts[chain[k]].pivot[0],
                    puppet.parts[chain[k + 1]].pivot[1] - puppet.parts[chain[k]].pivot[1],
                )
                for k in range(n - 1)
            ),
            default=0.0,
        )
        coloc_max = max(COLOCATED_PIVOT_MAX_PX, 0.30 * max_seg_dist)
        bone_entries: list[tuple[str, str, str]] = []
        i = 0
        while i < n - 1:
            curr_name = chain[i]
            next_name = chain[i + 1]
            p_curr = puppet.parts[curr_name].pivot
            p_next = puppet.parts[next_name].pivot
            dist = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            if dist < coloc_max and i + 2 < n:
                prox_cap = curr_name
                bone_name = next_name
                dist_part = chain[i + 2]
                bone_entries.append((bone_name, prox_cap, dist_part))
                i += 2
            else:
                bone_entries.append((curr_name, curr_name, next_name))
                i += 1

        measured_joints: dict[str, np.ndarray] = {}
        last_bone_name: str | None = None

        for bone_name, prox_part, dist_part in bone_entries:
            if bone_name in seen_parts or bone_name not in images:
                last_bone_name = bone_name
                continue
            seen_parts.add(bone_name)

            det_prox = (
                _learn_landmark_detector(
                    images[prox_part],
                    images[bone_name],
                    puppet.parts[prox_part].pivot,
                    distal_only=False,
                )
                if prox_part != bone_name and prox_part in images
                else None
            )
            det_dist = (
                _learn_landmark_detector(
                    images[dist_part],
                    images[bone_name],
                    puppet.parts[dist_part].pivot,
                    distal_only=(dist_part == leaf_name),
                )
                if dist_part in images
                else None
            )

            c_prox: np.ndarray | None = measured_joints.get(prox_part)
            if c_prox is None and det_prox is not None:
                b_prox = _get_part_search_bbox(puppet, images, pose, prox_part, det_prox, margin=35)
                m_prox, (ox_p, oy_p) = _detect_clean_landmark(img_rgb, det_prox, b_prox, other_mask)
                ys, xs = np.nonzero(m_prox)
                if len(xs) >= 20:
                    c_prox = np.array([xs.mean() + ox_p, ys.mean() + oy_p], dtype=np.float64)
                    measured_joints[prox_part] = c_prox

            c_dist: np.ndarray | None = measured_joints.get(dist_part)
            if c_dist is None and det_dist is not None:
                margin_d = 50 if dist_part == leaf_name else 35
                b_dist = _get_part_search_bbox(puppet, images, pose, dist_part, det_dist, margin=margin_d)
                m_dist, (ox_d, oy_d) = _detect_clean_landmark(img_rgb, det_dist, b_dist, other_mask)
                ys, xs = np.nonzero(m_dist)
                if len(xs) >= 20:
                    c_dist = np.array([xs.mean() + ox_d, ys.mean() + oy_d], dtype=np.float64)
                    measured_joints[dist_part] = c_dist

            if c_prox is not None and c_dist is not None:
                vec = c_dist - c_prox
                length_px = float(np.hypot(vec[0], vec[1]))
                angle_deg = math.degrees(math.atan2(vec[1], vec[0]))
                width_px = _measure_bone_width(img_rgb, c_prox, c_dist, images[bone_name], other_mask)
                results.append(
                    LimbMeasurement(
                        name=bone_name,
                        kind="bone",
                        chain_leaf=leaf_name,
                        parent_bone=last_bone_name,
                        start_xy=(float(c_prox[0]), float(c_prox[1])),
                        end_xy=(float(c_dist[0]), float(c_dist[1])),
                        length_px=length_px,
                        width_px=width_px,
                        angle_deg=angle_deg,
                        method="landmark",
                    )
                )
            else:
                start_xy, end_xy, length_px, width_px, angle_deg = _measure_template_part(
                    puppet,
                    images,
                    pose,
                    bone_name,
                    puppet.parts[dist_part].pivot if dist_part in puppet.parts else None,
                    img_rgb,
                    img_ink,
                    img_ink_dt,
                    other_mask,
                    is_leaf=False,
                )
                measured_joints[prox_part] = np.array(start_xy, dtype=np.float64)
                measured_joints[dist_part] = np.array(end_xy, dtype=np.float64)
                results.append(
                    LimbMeasurement(
                        name=bone_name,
                        kind="bone",
                        chain_leaf=leaf_name,
                        parent_bone=last_bone_name,
                        start_xy=start_xy,
                        end_xy=end_xy,
                        length_px=length_px,
                        width_px=width_px,
                        angle_deg=angle_deg,
                        method="template",
                    )
                )
            last_bone_name = bone_name

        # Now measure the leaf part (`leaf_name`).
        if leaf_name in seen_parts or leaf_name not in images:
            continue
        seen_parts.add(leaf_name)

        det_leaf = (
            _learn_landmark_detector(
                images[leaf_name],
                images[last_bone_name],
                puppet.parts[leaf_name].pivot,
                distal_only=True,
            )
            if last_bone_name is not None and last_bone_name in images
            else None
        )

        if det_leaf is not None and det_leaf.paired and last_bone_name is not None:
            b_leaf = _get_part_search_bbox(puppet, images, pose, leaf_name, det_leaf, margin=50)
            m_leaf, (ox_l, oy_l) = _detect_clean_landmark(img_rgb, det_leaf, b_leaf, other_mask)
            ys, xs = np.nonzero(m_leaf)
            if len(xs) >= 35:
                pts = np.stack([xs + ox_l, ys + oy_l], axis=1).astype(np.float64)
                cent = pts.mean(axis=0)
                parent_meas = next(
                    (r for r in reversed(results) if r.name == last_bone_name),
                    None,
                )
                if parent_meas is not None:
                    # Cross-axis perpendicular to the measured parent bone
                    # direction (`angle_deg - 90`). When the parent bone hangs
                    # within 25 deg of vertical (+90 deg), lock the split axis
                    # to horizontal (1, 0) so small bone-angle jitter never
                    # couples into the leaf orientation.
                    cross_deg = _wrap_deg(parent_meas.angle_deg - 90.0)
                    if abs(cross_deg) <= 25.0:
                        nx, ny = 1.0, 0.0
                    else:
                        rad = math.radians(cross_deg)
                        nx, ny = math.cos(rad), math.sin(rad)
                        if nx < -0.15 or (abs(nx) <= 0.15 and ny < 0):
                            nx, ny = -nx, -ny
                else:
                    nx, ny = 1.0, 0.0

                proj = (pts[:, 0] - cent[0]) * nx + (pts[:, 1] - cent[1]) * ny
                # If the two lobes are separate connected components in `m_leaf`,
                # use their component centroids directly so a tilted lobe tip
                # never crosses the split line into the opposite half.
                n_lcc, lcc, lstats, lcents = cv2.connectedComponentsWithStats(m_leaf.astype(np.uint8), connectivity=8)
                comp_list: list[tuple[int, np.ndarray]] = []
                for cid in range(1, n_lcc):
                    c_area = int(lstats[cid, cv2.CC_STAT_AREA])
                    if c_area >= 25:
                        c_xy = np.array(
                            [lcents[cid, 0] + ox_l, lcents[cid, 1] + oy_l],
                            dtype=np.float64,
                        )
                        comp_list.append((c_area, c_xy))
                comp_list.sort(key=lambda item: item[0], reverse=True)
                if len(comp_list) >= 2 and comp_list[1][0] >= 0.35 * comp_list[0][0]:
                    c1, c2 = comp_list[0][1], comp_list[1][1]
                    p1 = (c1[0] - cent[0]) * nx + (c1[1] - cent[1]) * ny
                    p2 = (c2[0] - cent[0]) * nx + (c2[1] - cent[1]) * ny
                    sol, sag = (c1, c2) if p1 <= p2 else (c2, c1)
                    valid_lobes = True
                else:
                    sol_pts = pts[proj < 0]
                    sag_pts = pts[proj >= 0]
                    valid_lobes = len(sol_pts) >= 15 and len(sag_pts) >= 15
                    if valid_lobes:
                        sol = sol_pts.mean(axis=0)
                        sag = sag_pts.mean(axis=0)

                if valid_lobes:
                    leaf_deg = math.degrees(math.atan2(sag[1] - sol[1], sag[0] - sol[0]))
                    lobe_span = float(np.hypot(sag[0] - sol[0], sag[1] - sol[1]))
                    along = -(pts[:, 0] - cent[0]) * ny + (pts[:, 1] - cent[1]) * nx
                    length_px = float(np.percentile(along, 95) - np.percentile(along, 5))
                    start_pt = parent_meas.end_xy if parent_meas is not None else (float(cent[0]), float(cent[1]))
                    results.append(
                        LimbMeasurement(
                            name=leaf_name,
                            kind="leaf",
                            chain_leaf=leaf_name,
                            parent_bone=last_bone_name,
                            start_xy=start_pt,
                            end_xy=(float(cent[0]), float(cent[1])),
                            length_px=length_px,
                            width_px=lobe_span,
                            angle_deg=_wrap_deg(leaf_deg),
                            method="landmark",
                        )
                    )
                    continue

        start_xy, end_xy, length_px, width_px, angle_deg = _measure_template_part(
            puppet,
            images,
            pose,
            leaf_name,
            None,
            img_rgb,
            img_ink,
            img_ink_dt,
            other_mask,
            is_leaf=True,
        )
        results.append(
            LimbMeasurement(
                name=leaf_name,
                kind="leaf",
                chain_leaf=leaf_name,
                parent_bone=last_bone_name,
                start_xy=start_xy,
                end_xy=end_xy,
                length_px=length_px,
                width_px=width_px,
                angle_deg=_wrap_deg(angle_deg),
                method="template",
            )
        )

    return results


def compare_pose_limbs(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose: Pose,
    drawing_rgb: np.ndarray,
    puppet_rgb: np.ndarray,
    *,
    pose_name: str = "",
) -> PoseExamReport:
    """Compare all limbs between `drawing_rgb` and `puppet_rgb` for one pose."""
    m_draw = measure_image_limbs(puppet, images, pose, drawing_rgb)
    m_pup = m_draw if np.array_equal(drawing_rgb, puppet_rgb) else measure_image_limbs(puppet, images, pose, puppet_rgb)

    pup_by_name = {m.name: m for m in m_pup}
    exams: list[PoseLimbExam] = []
    status_by_name: dict[str, PoseLimbStatus] = {}

    for d in m_draw:
        p = pup_by_name.get(d.name, d)
        len_diff = float(p.length_px - d.length_px)
        len_ratio = float(len_diff / d.length_px) if d.length_px > 1e-3 else 0.0
        wid_diff = float(p.width_px - d.width_px)
        wid_ratio = float(wid_diff / d.width_px) if d.width_px > 1e-3 else 0.0
        shift_px = float(math.hypot(p.end_xy[0] - d.end_xy[0], p.end_xy[1] - d.end_xy[1]))
        ang_diff = _wrap_deg(p.angle_deg - d.angle_deg)
        abs_ang = abs(ang_diff)

        status: PoseLimbStatus = "ok"
        reason = "uyumlu"

        if d.kind == "bone":
            if abs(len_ratio) >= BONE_LEN_DIFF_RATIO_MAX:
                status = "drawing_diff"
                reason = f"boy farki {len_ratio * 100:+.1f}% ({len_diff:+.1f}px)"
            elif abs(wid_ratio) >= BONE_WIDTH_DIFF_RATIO_MAX:
                status = "drawing_diff"
                reason = f"en farki {wid_ratio * 100:+.1f}% ({wid_diff:+.1f}px)"
            elif abs_ang > BONE_ANGLE_ERR_DEG:
                status = "motor_error"
                reason = f"aci sapmasi {ang_diff:+.1f}°"
            elif shift_px >= JOINT_SHIFT_ERR_PX:
                status = "motor_error"
                reason = f"eklem kaymasi {shift_px:.1f}px"
        else:
            parent_status = status_by_name.get(d.parent_bone, "ok") if d.parent_bone is not None else "ok"
            if abs_ang > LEAF_ANGLE_ERR_DEG:
                status = "motor_error"
                reason = f"yaprak yonu sapmasi {ang_diff:+.1f}° (cizim {d.angle_deg:.1f}°, kukla {p.angle_deg:.1f}°)"
            elif abs(len_ratio) >= LEAF_DIM_DIFF_RATIO_MAX or abs(wid_ratio) >= LEAF_DIM_DIFF_RATIO_MAX:
                status = "drawing_diff"
                reason = f"yaprak bicim farki (boy {len_ratio * 100:+.1f}%, en {wid_ratio * 100:+.1f}%)"
            elif shift_px >= JOINT_SHIFT_ERR_PX:
                if parent_status == "drawing_diff" or abs(len_ratio) >= 0.06 or abs(wid_ratio) >= 0.06:
                    status = "drawing_diff"
                    reason = f"cizim kaynakli uc kaymasi {shift_px:.1f}px"
                else:
                    status = "motor_error"
                    reason = f"uc kaymasi {shift_px:.1f}px"

        status_by_name[d.name] = status
        exams.append(
            PoseLimbExam(
                name=d.name,
                kind=d.kind,
                drawing=d,
                puppet=p,
                length_diff_px=round(len_diff, 2),
                length_diff_ratio=round(len_ratio, 4),
                width_diff_px=round(wid_diff, 2),
                width_diff_ratio=round(wid_ratio, 4),
                joint_shift_px=round(shift_px, 2),
                angle_diff_deg=round(ang_diff, 2),
                abs_angle_diff_deg=round(abs_ang, 2),
                status=status,
                reason=reason,
            )
        )

    counts = {
        "ok": sum(1 for e in exams if e.status == "ok"),
        "drawing_diff": sum(1 for e in exams if e.status == "drawing_diff"),
        "motor_error": sum(1 for e in exams if e.status == "motor_error"),
    }
    err_names = [e.name for e in exams if e.status == "motor_error"]
    diff_names = [e.name for e in exams if e.status == "drawing_diff"]
    parts_txt = [f"ok:{counts['ok']}"]
    if counts["drawing_diff"]:
        parts_txt.append(f"cizim_farki:{counts['drawing_diff']}({','.join(diff_names)})")
    else:
        parts_txt.append("cizim_farki:0")
    if counts["motor_error"]:
        parts_txt.append(f"motor_hatasi:{counts['motor_error']}({','.join(err_names)})")
    else:
        parts_txt.append("motor_hatasi:0")

    return PoseExamReport(
        pose_name=pose_name,
        limbs=exams,
        counts=counts,
        summary=" ".join(parts_txt),
    )


def compare_transition_limbs(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose_a: Pose,
    drawing_a_rgb: np.ndarray,
    puppet_a_rgb: np.ndarray,
    pose_b: Pose,
    drawing_b_rgb: np.ndarray,
    puppet_b_rgb: np.ndarray,
    *,
    pose_a_name: str = "pose_a",
    pose_b_name: str = "pose_b",
) -> TransitionExamReport:
    """Compare limb changes from pose A to pose B between drawing and puppet."""
    draw_a = {m.name: m for m in measure_image_limbs(puppet, images, pose_a, drawing_a_rgb)}
    draw_b = {m.name: m for m in measure_image_limbs(puppet, images, pose_b, drawing_b_rgb)}
    pup_a = (
        draw_a
        if np.array_equal(drawing_a_rgb, puppet_a_rgb)
        else {m.name: m for m in measure_image_limbs(puppet, images, pose_a, puppet_a_rgb)}
    )
    pup_b = (
        draw_b
        if np.array_equal(drawing_b_rgb, puppet_b_rgb)
        else {m.name: m for m in measure_image_limbs(puppet, images, pose_b, puppet_b_rgb)}
    )

    exams: list[TransitionLimbExam] = []
    unchanged: list[str] = []
    errors: list[str] = []

    for name, da in draw_a.items():
        if name not in draw_b or name not in pup_a or name not in pup_b:
            continue
        db = draw_b[name]
        pa = pup_a[name]
        pb = pup_b[name]

        d_c = _abs_ang_diff(db.angle_deg, da.angle_deg)
        d_k = _abs_ang_diff(pb.angle_deg, pa.angle_deg)
        s_c = float(math.hypot(db.end_xy[0] - da.end_xy[0], db.end_xy[1] - da.end_xy[1]))
        s_k = float(math.hypot(pb.end_xy[0] - pa.end_xy[0], pb.end_xy[1] - pa.end_xy[1]))

        changed_in_drawing = not (d_c <= UNCHANGED_MAX_ANGLE_DEG and s_c <= UNCHANGED_MAX_SHIFT_PX)
        if not changed_in_drawing:
            unchanged.append(name)

        status: TransitionLimbStatus = "ok"
        reason = "uyumlu"
        if d_k > TRANSITION_ANGLE_RATIO_MAX * max(d_c, 0.25) and (d_k - d_c) > TRANSITION_ANGLE_EXCESS_DEG:
            status = "motor_error"
            reason = f"aci degisimi cizim {d_c:.1f}° iken kukla {d_k:.1f}°"
            errors.append(name)
        elif s_k > TRANSITION_SHIFT_RATIO_MAX * max(s_c, 1.0) and (s_k - s_c) > TRANSITION_SHIFT_EXCESS_PX:
            status = "motor_error"
            reason = f"konum degisimi cizim {s_c:.1f}px iken kukla {s_k:.1f}px"
            errors.append(name)

        exams.append(
            TransitionLimbExam(
                name=name,
                kind=da.kind,
                drawing_angle_change_deg=round(d_c, 2),
                puppet_angle_change_deg=round(d_k, 2),
                drawing_shift_px=round(s_c, 2),
                puppet_shift_px=round(s_k, 2),
                changed_in_drawing=changed_in_drawing,
                status=status,
                reason=reason,
            )
        )

    counts = {
        "ok": sum(1 for e in exams if e.status == "ok"),
        "motor_error": sum(1 for e in exams if e.status == "motor_error"),
        "unchanged": len(unchanged),
    }
    if errors:
        err_detail = ", ".join(
            f"{e.name}(c:{e.drawing_angle_change_deg:.1f}°/k:{e.puppet_angle_change_deg:.1f}°)"
            for e in exams
            if e.status == "motor_error"
        )
        summary = f"motor_hatasi:{len(errors)} [{err_detail}] degismeyen:{len(unchanged)}"
    else:
        summary = f"ok:{counts['ok']} motor_hatasi:0 degismeyen:{len(unchanged)}"

    return TransitionExamReport(
        transition_name=f"{pose_a_name}__{pose_b_name}",
        pose_a=pose_a_name,
        pose_b=pose_b_name,
        limbs=exams,
        unchanged_limbs=unchanged,
        motor_error_limbs=errors,
        counts=counts,
        summary=summary,
    )


def detect_unchanged_chain_members(
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose_x: Pose,
    rgb_x: np.ndarray,
    pose_y: Pose,
    rgb_y: np.ndarray,
) -> set[str]:
    """Return chain part names whose limb did not change in the drawing (`rgb_x` -> `rgb_y`)."""
    draw_x = {m.name: m for m in measure_image_limbs(puppet, images, pose_x, rgb_x)}
    draw_y = {m.name: m for m in measure_image_limbs(puppet, images, pose_y, rgb_y)}
    unchanged_measured: set[str] = set()
    for name, mx in draw_x.items():
        my = draw_y.get(name)
        if my is None:
            continue
        d_c = _abs_ang_diff(my.angle_deg, mx.angle_deg)
        s_c = float(math.hypot(my.end_xy[0] - mx.end_xy[0], my.end_xy[1] - mx.end_xy[1]))
        len_rel = abs(my.length_px - mx.length_px) / max(mx.length_px, 1.0)
        if d_c <= UNCHANGED_MAX_ANGLE_DEG and s_c <= UNCHANGED_MAX_SHIFT_PX and len_rel <= 0.15:
            unchanged_measured.add(name)

    chains_map = chains(puppet)
    locked_members: set[str] = set(unchanged_measured)
    for _leaf, chain in chains_map.items():
        n = len(chain)
        max_seg_dist = max(
            (
                math.hypot(
                    puppet.parts[chain[k + 1]].pivot[0] - puppet.parts[chain[k]].pivot[0],
                    puppet.parts[chain[k + 1]].pivot[1] - puppet.parts[chain[k]].pivot[1],
                )
                for k in range(n - 1)
            ),
            default=0.0,
        )
        coloc_max = max(COLOCATED_PIVOT_MAX_PX, 0.30 * max_seg_dist)
        for i in range(n - 1):
            curr_name = chain[i]
            next_name = chain[i + 1]
            p_curr = puppet.parts[curr_name].pivot
            p_next = puppet.parts[next_name].pivot
            if math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1]) < coloc_max and next_name in unchanged_measured:
                locked_members.add(curr_name)

    return locked_members


def format_pose_limb_table(report: PoseExamReport) -> str:
    """Format a `PoseExamReport` as a Markdown table."""
    lines = [
        f"### Poz Uzuv Sınavı: `{report.pose_name}` ({report.summary})",
        "",
        "| Uzuv | Tür | Çizim boy (px) | Kukla boy (px) | Boy farkı | Çizim açı (°) "
        "| Kukla açı (°) | Açı farkı (°) | Uç kayması (px) | Sınıf | Açıklama |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for e in report.limbs:
        lines.append(
            f"| `{e.name}` | {e.kind} | {e.drawing.length_px:.1f} | {e.puppet.length_px:.1f} "
            f"| {e.length_diff_ratio * 100:+.1f}% | {e.drawing.angle_deg:.1f} | {e.puppet.angle_deg:.1f} "
            f"| {e.angle_diff_deg:+.1f} | {e.joint_shift_px:.1f} | **{e.status}** | {e.reason} |"
        )
    return "\n".join(lines)


def format_transition_limb_table(report: TransitionExamReport) -> str:
    """Format a `TransitionExamReport` as a Markdown table."""
    lines = [
        f"### Geçiş Uzuv Sınavı: `{report.transition_name}` ({report.summary})",
        "",
        "| Uzuv | Tür | Çizim Δaçı (°) | Kukla Δaçı (°) | Çizim Δuç (px) "
        "| Kukla Δuç (px) | Çizimde değişti mi | Sınıf | Açıklama |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for e in report.limbs:
        degisti = "evet" if e.changed_in_drawing else "hayır (sabit)"
        lines.append(
            f"| `{e.name}` | {e.kind} | {e.drawing_angle_change_deg:.1f} | {e.puppet_angle_change_deg:.1f} "
            f"| {e.drawing_shift_px:.1f} | {e.puppet_shift_px:.1f} | {degisti} | **{e.status}** | {e.reason} |"
        )
    return "\n".join(lines)
