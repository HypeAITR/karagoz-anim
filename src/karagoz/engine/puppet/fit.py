"""Fit the layered puppet to a key drawing (package K-1).

Tree order, one free angle per part once its parent is placed: ``snap_part``
searches that angle (coarse 5 degrees, then 0.5 degree refinement) on the
chamfer agreement of the part's lines plus the colour agreement of its flat
fills, minus a paper penalty. Chains (limbs) are fitted jointly from the same
score so a limb never ends up inconsistent; ``fit_pose`` does root similarity
first, then chains, then a second pass, z order and view selection.
``transition`` fits X and Y once each, locks unchanged parts and pulls changed
chains from X to Y, with the rig interval gate deciding ``rig``/``split`` and
the fit score deciding ``ret``.

No paid calls: everything here is local pixels (chamfer + colour).
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from karagoz.engine.puppet.model import Puppet, load_set
from karagoz.engine.puppet.parts import line_mask
from karagoz.engine.puppet.pose import Pose, chains, part_matrix3, render_pose, tree_order, z_order
from karagoz.engine.rig.plan import (
    DEVIATION_THRESHOLD,
    JUMP_THRESHOLD,
    MAX_EXTRA_KEYS,
    SCALE_LIMIT,
)

LINE_LUMA = 25.0
PAPER_LUMA = 235.0
COLOUR_TAU = 60.0
W_LINE = 1.0
W_COLOUR = 0.5
W_PAPER = 1.0
# Agreement kept on drawing pixels other placed parts already explain: a
# chain lying on the face keeps a quarter of those samples' agreement, so
# unexplained drawing pixels decide between parasites and true placements.
EXPLAIN_DECAY = 0.75
MAX_LINE_PTS = 4000
MAX_FLAT_PTS = 4000
MAX_COVER_PTS = 30000
ROOT_MAX_FLAT = 1200
COARSE_STEP = 5.0
FINE_STEP = 0.5
FINE_RANGE = 5.0
# A "separate peak" is a distinctly different configuration, not the shoulder
# of the same optimum: small parts have broad plateaus around the best angle,
# while a hidden or symmetric part scores the same all the way around.
PEAK_SEP_DEG = 45.0
CONF_MIN = 0.04
SCORE_MIN = 0.25
# A stacked part (folded limb, hidden part parked visibly) shares most pixels
# with a sibling yet still matches colours there: overlap past OVERLAP_MAX
# with a score below STACK_SCORE means it explains nothing of its own.
# True placements stay under 0.3 overlap (seams); clean scores clear 1.1.
OVERLAP_MAX = 0.5
STACK_SCORE = 1.0
# A limit range narrower than PEAK_SEP_DEG cannot host a separate peak: the
# ambiguity gap is meaningless there, so a clipped sweep reports
# NO_AMBIGUITY_GAP instead of a coincidental small gap (a ±10° ring never
# "reaches" 45° out, and must never read as ambiguous).
NO_AMBIGUITY_GAP = 1e9
# Slack-joint offset search: prior per reference pixel toward the modelled
# pivot, same scale as the rest prior per degree (a degree moves a 100 px
# limb about as far as a pixel slides its joint).
OFFSET_PRIOR = 0.0005
# Reason recorded in ``Pose.chain_notes`` when the fit keeps a chain
# unfolded because no feasible placement seats it (reviewer acceptance rule:
# the engine must not produce an absurd pose).
NEVER_FIT_REASON = "puppet does not fit this drawing"
SECOND_PASS = True
# From-scratch chain greedy reruns with its first free joint pinned to this
# many candidates (K-0r: a joint decided against a T-pose limb parks at its
# limit while the children compensate). 0 turns the beam off.
GREEDY_BEAM = 3


@dataclass
class SnapResult:
    """One part snapped: best angle, its score and the peak gap (confidence)."""

    angle: float
    score: float
    confidence: float
    scores: dict[float, float] = field(default_factory=dict)


@dataclass
class DrawingFeatures:
    rgb: NDArray[np.uint8]
    line: NDArray[np.bool_]
    dist: NDArray[np.float32]
    paper: NDArray[np.bool_]

    # float64 views for scoring, converted once per drawing (not per score call)
    @cached_property
    def dist_f(self) -> NDArray[np.float64]:
        return np.asarray(self.dist, dtype=np.float64)

    @cached_property
    def paper_f(self) -> NDArray[np.float64]:
        return np.asarray(self.paper.astype(np.float64), dtype=np.float64)

    @cached_property
    def rgb_f(self) -> NDArray[np.float64]:
        return np.asarray(self.rgb.astype(np.float64), dtype=np.float64)


@dataclass
class PartSamples:
    line_xy: NDArray[np.float64]
    flat_xy: NDArray[np.float64]
    flat_rgb: NDArray[np.uint8]
    cover_xy: NDArray[np.float64]
    # Every line pixel of the part (capped at MAX_COVER_PTS) for the
    # explanation cover; scoring warps the sparsely strided line_xy.
    line_cover_xy: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 2), np.float64))


def load_drawing(path: Path) -> NDArray[np.uint8]:
    """Key drawing as RGB (alpha composited onto white paper)."""
    with Image.open(path) as opened:
        rgba = opened.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, rgba).convert("RGB")
    return np.asarray(flat, dtype=np.uint8)


def drawing_features(rgb: NDArray[np.uint8]) -> DrawingFeatures:
    """Line mask, chamfer distance to the lines and paper mask of a drawing."""
    line = np.asarray(line_mask(np.asarray(rgb, dtype=np.uint8), LINE_LUMA), dtype=np.bool_)
    bg = np.asarray((~line).astype(np.uint8), dtype=np.uint8)
    dist = np.asarray(cv2.distanceTransform(bg, cv2.DIST_L2, 3), dtype=np.float32)
    luma = np.asarray(rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    paper = np.asarray((luma >= PAPER_LUMA) & ~line, dtype=np.bool_)
    return DrawingFeatures(rgb=np.asarray(rgb, dtype=np.uint8), line=line, dist=dist, paper=paper)


def _stride(n: int, cap: int) -> int:
    if n <= cap or n == 0:
        return 1
    return int(math.ceil(n / cap))


def extract_part_samples(
    puppet: Puppet,
    images: dict[str, NDArray[np.uint8]],
    *,
    max_line: int = MAX_LINE_PTS,
    max_flat: int = MAX_FLAT_PTS,
) -> dict[str, PartSamples]:
    """Sparse line/flat points per part in reference pixels (+ own colours)."""
    out: dict[str, PartSamples] = {}
    for name, part in puppet.parts.items():
        img = images.get(name)
        if part.hidden or img is None or img.ndim != 3 or img.shape[2] != 4:
            out[name] = PartSamples(
                line_xy=np.zeros((0, 2), np.float64),
                flat_xy=np.zeros((0, 2), np.float64),
                flat_rgb=np.zeros((0, 3), np.uint8),
                cover_xy=np.zeros((0, 2), np.float64),
                line_cover_xy=np.zeros((0, 2), np.float64),
            )
            continue
        # part images are full-canvas; work inside the alpha box (same points, same order)
        full_alpha = np.asarray(img[..., 3] > 127, dtype=np.bool_)
        rows = np.flatnonzero(full_alpha.any(axis=1))
        cols = np.flatnonzero(full_alpha.any(axis=0))
        y0, x0 = (int(rows[0]), int(cols[0])) if len(rows) else (0, 0)
        y1, x1 = (int(rows[-1]) + 1, int(cols[-1]) + 1) if len(rows) else (0, 0)
        alpha = full_alpha[y0:y1, x0:x1]
        rgb = np.asarray(img[y0:y1, x0:x1, :3], dtype=np.uint8)
        ink = np.asarray(line_mask(np.asarray(rgb, dtype=np.uint8), LINE_LUMA) & alpha, dtype=np.bool_)
        flat = np.asarray(alpha & ~ink, dtype=np.bool_)
        lys, lxs = np.nonzero(ink)
        fys, fxs = np.nonzero(flat)
        ays, axs = np.nonzero(alpha)
        lys, lxs, fys, fxs, ays, axs = lys + y0, lxs + x0, fys + y0, fxs + x0, ays + y0, axs + x0
        rgb = np.asarray(img[..., :3], dtype=np.uint8)
        st = _stride(len(lxs), max_line)
        ft = _stride(len(fxs), max_flat)
        ct = _stride(len(axs), MAX_COVER_PTS)
        if len(lxs):
            line_xy = np.stack([lxs[::st].astype(np.float64), lys[::st].astype(np.float64)], axis=1)
        else:
            line_xy = np.zeros((0, 2), np.float64)
        lt = _stride(len(lxs), MAX_COVER_PTS)
        if len(lxs):
            line_cover_xy = np.stack([lxs[::lt].astype(np.float64), lys[::lt].astype(np.float64)], axis=1)
        else:
            line_cover_xy = np.zeros((0, 2), np.float64)
        if len(fxs):
            flat_xy = np.stack([fxs[::ft].astype(np.float64), fys[::ft].astype(np.float64)], axis=1)
            flat_rgb = np.asarray(rgb[fys[::ft], fxs[::ft]], dtype=np.uint8)
        else:
            flat_xy = np.zeros((0, 2), np.float64)
            flat_rgb = np.zeros((0, 3), np.uint8)
        if len(axs):
            cover_xy = np.stack([axs[::ct].astype(np.float64), ays[::ct].astype(np.float64)], axis=1)
        else:
            cover_xy = np.zeros((0, 2), np.float64)
        out[name] = PartSamples(
            line_xy=line_xy, flat_xy=flat_xy, flat_rgb=flat_rgb, cover_xy=cover_xy, line_cover_xy=line_cover_xy
        )
    return out


def _warp_points(xy: NDArray[np.float64], mat2x3: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(xy) == 0:
        return np.zeros((0, 2), np.float64)
    m = np.asarray(mat2x3, dtype=np.float64)
    ones = np.ones((len(xy), 1), np.float64)
    homo = np.concatenate([xy, ones], axis=1)
    return np.asarray(homo @ m.T, dtype=np.float64)


def _bilinear(
    field: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64], outside: float
) -> NDArray[np.float64]:
    """Bilinear samples of a 2D float field at subpixel positions."""
    height, width = field.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    valid = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < width) & (y0 + 1 < height)
    xs = np.clip(x0, 0, width - 1)
    ys = np.clip(y0, 0, height - 1)
    xn = np.minimum(xs + 1, width - 1)
    yn = np.minimum(ys + 1, height - 1)
    flat = field.reshape(-1)  # one flat gather per corner: same values, cheaper than 2D fancy indexing
    r0 = ys * width
    r1 = yn * width
    val = (
        (1.0 - fx) * (1.0 - fy) * flat[r0 + xs]
        + fx * (1.0 - fy) * flat[r0 + xn]
        + (1.0 - fx) * fy * flat[r1 + xs]
        + fx * fy * flat[r1 + xn]
    )
    return np.asarray(np.where(valid, val, outside), dtype=np.float64)


def _bilinear_rgb(
    field: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64], outside: float
) -> NDArray[np.float64]:
    """``_bilinear`` of the three channels of an (H, W, 3) field in one gather: (N, 3), same values."""
    height, width = field.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    valid = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < width) & (y0 + 1 < height)
    xs = np.clip(x0, 0, width - 1)
    ys = np.clip(y0, 0, height - 1)
    xn = np.minimum(xs + 1, width - 1)
    yn = np.minimum(ys + 1, height - 1)
    val = (
        (1.0 - fx) * (1.0 - fy) * field[ys, xs]
        + fx * (1.0 - fy) * field[ys, xn]
        + (1.0 - fx) * fy * field[yn, xs]
        + fx * fy * field[yn, xn]
    )
    return np.asarray(np.where(valid[:, None], val, outside), dtype=np.float64)


def _warp_batch(xy: NDArray[np.float64], mats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Warp points by a stack of 2x3 matrices: (N, 2) and (K, 2, 3) -> (K, N, 2)."""
    homo = np.concatenate([np.asarray(xy, dtype=np.float64), np.ones((len(xy), 1), np.float64)], axis=1)
    return np.asarray(np.einsum("kij,nj->kni", np.asarray(mats, dtype=np.float64), homo), dtype=np.float64)


def batch_scores(
    samp: PartSamples,
    mats: NDArray[np.float64],
    feat: DrawingFeatures,
    tau: float,
    explained: NDArray[np.bool_] | None = None,
) -> list[float]:
    """Score one part under many candidate matrices at once.

    The point transforms and bilinear lookups run stacked, so a 72-angle
    sweep costs about as much as a few single scores.
    """
    mats = np.asarray(mats, dtype=np.float64).reshape(-1, 2, 3)
    total = np.zeros(mats.shape[0], np.float64)
    height, width = feat.line.shape
    dist_f = feat.dist_f
    paper_f = feat.paper_f
    rgb_f = feat.rgb_f
    if len(samp.line_xy):
        coords = _warp_batch(samp.line_xy, mats).reshape(-1, 2)
        dist = _bilinear(dist_f, coords[:, 0], coords[:, 1], 50.0).reshape(mats.shape[0], -1)
        if explained is not None:
            xi = np.round(coords[:, 0]).astype(np.int64).reshape(mats.shape[0], -1)
            yi = np.round(coords[:, 1]).astype(np.int64).reshape(mats.shape[0], -1)
            for k in range(mats.shape[0]):
                valid = (xi[k] >= 0) & (yi[k] >= 0) & (xi[k] < width) & (yi[k] < height)
                hit = np.zeros(xi.shape[1], np.bool_)
                hit[valid] = np.asarray(explained[yi[k][valid], xi[k][valid]], dtype=np.bool_)
                dist[k][hit] = 50.0
        total += W_LINE * np.mean(np.exp(-dist / max(tau, 1e-6)), axis=1)
    if len(samp.flat_xy):
        coords = _warp_batch(samp.flat_xy, mats).reshape(-1, 2)
        x, y = coords[:, 0], coords[:, 1]
        drawn = _bilinear_rgb(rgb_f, x, y, 255.0)
        drawn = drawn.reshape(mats.shape[0], -1, 3)
        own = np.asarray(samp.flat_rgb.astype(np.float64))
        cd = np.mean(np.abs(drawn - own[None, :, :]), axis=2)
        agree = np.exp(-cd / COLOUR_TAU)
        if explained is not None:
            xi = np.round(x).astype(np.int64).reshape(mats.shape[0], -1)
            yi = np.round(y).astype(np.int64).reshape(mats.shape[0], -1)
            for k in range(mats.shape[0]):
                valid = (xi[k] >= 0) & (yi[k] >= 0) & (xi[k] < width) & (yi[k] < height)
                covered = np.zeros(xi.shape[1], np.bool_)
                covered[valid] = np.asarray(explained[yi[k][valid], xi[k][valid]], dtype=np.bool_)
                agree[k][covered] = 0.0
        paper_hit = _bilinear(paper_f, x, y, 1.0).reshape(mats.shape[0], -1)
        total += W_COLOUR * np.mean(agree, axis=1) - W_PAPER * np.mean(paper_hit, axis=1)
    return [float(v) for v in total]


def score_samples(
    samp: PartSamples,
    mat2x3: NDArray[np.float64],
    feat: DrawingFeatures,
    tau: float,
    explained: NDArray[np.bool_] | None = None,
) -> float:
    """Chamfer + colour agreement minus paper penalty (higher is better).

    ``explained`` marks drawing pixels other placed parts already cover: for
    ``fill_last`` samples landing there count as misses (the hole is the
    unexplained rest).
    """
    height, width = feat.line.shape
    dist_f = feat.dist_f
    paper_f = feat.paper_f
    rgb_f = feat.rgb_f
    total = 0.0
    weight = 0.0
    if len(samp.line_xy):
        warped = _warp_points(samp.line_xy, mat2x3)
        x, y = warped[:, 0], warped[:, 1]
        dist = _bilinear(dist_f, x, y, 50.0)
        if explained is not None:
            xi = np.round(x).astype(np.int64)
            yi = np.round(y).astype(np.int64)
            valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
            hit = np.zeros(len(x), np.bool_)
            hit[valid] = np.asarray(explained[yi[valid], xi[valid]], dtype=np.bool_)
            dist[hit] = 50.0
        line_agree = float(np.mean(np.exp(-dist / max(tau, 1e-6)))) if len(dist) else 0.0
        total += W_LINE * line_agree
        weight += W_LINE
    if len(samp.flat_xy):
        warped = _warp_points(samp.flat_xy, mat2x3)
        x, y = warped[:, 0], warped[:, 1]
        drawn = _bilinear_rgb(rgb_f, x, y, 255.0)
        own = np.asarray(samp.flat_rgb.astype(np.float64))
        cd = np.mean(np.abs(drawn - own), axis=1)
        agree = np.exp(-cd / COLOUR_TAU)
        if explained is not None:
            xi = np.round(x).astype(np.int64)
            yi = np.round(y).astype(np.int64)
            valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
            covered = np.zeros(len(x), np.bool_)
            covered[valid] = np.asarray(explained[yi[valid], xi[valid]], dtype=np.bool_)
            agree[covered] = 0.0
        paper_hit = _bilinear(paper_f, x, y, 1.0)
        colour_agree = float(np.mean(agree)) if len(agree) else 0.0
        paper_frac = float(np.mean(paper_hit)) if len(paper_hit) else 0.0
        total += W_COLOUR * colour_agree - W_PAPER * paper_frac
        weight += W_COLOUR + W_PAPER
    if weight == 0:
        return 0.0
    return float(total)


def _tau_for(puppet: Puppet, pose: Pose) -> float:
    return float(max(2.0, float(puppet.line_width_px) * float(pose.scale)))


def _matrix_for(puppet: Puppet, pose: Pose, part: str, angle: float) -> NDArray[np.float64]:
    trial = pose.model_copy(update={"angles": {**pose.angles, part: float(angle)}})
    return np.asarray(part_matrix3(puppet, trial, part)[:2], dtype=np.float64)


def _limits_of(puppet: Puppet, name: str) -> tuple[float, float] | None:
    """Joint rotation limits [lo, hi] of a part, if the puppet declares them.

    Limits are data (`PuppetPart.limits`, degrees relative to rest), never
    part names in code: every search path clips its candidates into range.
    A pinned (``frozen``) member keeps its exact angle either way.
    """
    part = puppet.parts.get(name)
    lim = part.limits if part is not None else None
    if lim is None:
        return None
    return (float(lim[0]), float(lim[1]))


def _clip_cands(cands: list[float], lim: tuple[float, float] | None) -> list[float]:
    """Clip search candidates into joint limits, dropping duplicates."""
    if lim is None:
        return [float(c) for c in cands]
    out: list[float] = []
    seen: set[float] = set()
    for cand in cands:
        value = min(max(float(cand), lim[0]), lim[1])
        key = round(value, 9)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def snap_part(
    puppet: Puppet,
    pose: Pose,
    part: str,
    drawing: NDArray[np.uint8],
    samples: dict[str, PartSamples] | None = None,
    features: DrawingFeatures | None = None,
    *,
    hint_deg: float | None = None,
    window_deg: float = 180.0,
    explained: NDArray[np.bool_] | None = None,
) -> SnapResult:
    """Snap one part: search its angle about its pivot, ancestors fixed.

    Coarse ``COARSE_STEP`` sweep (full circle without a hint, else
    ``hint_deg +/- window_deg/2``), then ``FINE_STEP`` refinement around the
    best few. The score is the part's own chamfer + colour minus paper (see
    :func:`score_samples`); confidence is the gap to the best separate peak.
    When the puppet declares joint limits for the part, every candidate is
    clipped into range.
    """
    feat = features if features is not None else drawing_features(np.asarray(drawing, dtype=np.uint8))
    samp = samples[part] if samples is not None and part in samples else None
    if samp is None:
        raise ValueError(f"no samples for part {part!r}")
    tau = _tau_for(puppet, pose)
    lim = _limits_of(puppet, part)
    coarse = _coarse_angles(hint_deg, window_deg, lim)
    mats = np.stack([_matrix_for(puppet, pose, part, angle) for angle in coarse])
    scores = {float(a): v for a, v in zip(coarse, batch_scores(samp, mats, feat, tau, explained), strict=True)}
    ordered = sorted(scores, key=lambda a: scores[a], reverse=True)
    seeds: list[float] = []
    for cand in ordered:
        if all(_ang_dist(float(cand), float(s)) >= 2 * COARSE_STEP for s in seeds):
            seeds.append(float(cand))
        if len(seeds) >= 3:
            break
    best_angle = float(ordered[0])
    best_score = float(scores[best_angle])
    for seed in seeds:
        lo_c = seed - FINE_RANGE if lim is None else max(seed - FINE_RANGE, lim[0])
        hi_c = seed + FINE_RANGE if lim is None else min(seed + FINE_RANGE, lim[1])
        angle = lo_c
        while angle <= hi_c + 1e-9:
            key = round(float(angle), 1)
            if key not in scores:
                scores[key] = score_samples(samp, _matrix_for(puppet, pose, part, angle), feat, tau, explained)
            if scores[key] > best_score:
                best_score = float(scores[key])
                best_angle = float(key)
            angle += FINE_STEP
    gap = _second_peak_gap(scores, best_angle)
    return SnapResult(angle=best_angle, score=best_score, confidence=gap, scores=scores)


def _explain_line_radius(puppet: Puppet, pose: Pose) -> int:
    """Radius (px) around explained lines that still counts as explained.

    Line agreement rewards samples within chamfer ``tau`` (line width times
    scale) of any drawing line, so a fold lying a few pixels next to dense
    already-explained lines would score almost like a clean placement while
    escaping the exact-pixel decay. Growing the explained lines by the same
    line-width radius closes that gap; the radius derives from the puppet's
    own line width and the pose scale like ``tau``.
    """
    return max(2, int(round(float(puppet.line_width_px) * float(pose.scale))))


def explained_mask(
    puppet: Puppet,
    pose: Pose,
    samples: dict[str, PartSamples],
    shape: tuple[int, int],
    skip: str | Collection[str] | None = None,
    line_radius: int = 0,
) -> NDArray[np.bool_]:
    """Drawing pixels some placed part (except ``skip``) covers.

    Line cover follows every line pixel of the part (dense, capped at
    ``MAX_COVER_PTS``), not the sparsely strided scoring samples, so a
    limb stacked on dense panel lines cannot hide in the stride gaps;
    scoring still uses the sparse samples (speed).

    The flat cover (dilated 5x5) always counts. ``line_radius`` grows the
    explained lines by that many pixels: the soft chain-objective decay
    (see :func:`exclusive_total`) works in the line-width neighbourhood,
    while the hard hole search (:func:`fill_last`) keeps the pixel-exact
    hole (``line_radius=0``) so a last piece still snaps within 1 degree.
    """
    skipped = {skip} if isinstance(skip, str) else set(skip or ())
    height, width = shape
    cover = np.zeros((height, width), np.bool_)
    for name, samp in samples.items():
        if name in skipped or len(samp.flat_xy) == 0:
            continue
        mat = np.asarray(part_matrix3(puppet, pose, name)[:2], dtype=np.float64)
        warped = _warp_points(samp.flat_xy, mat)
        xi = np.round(warped[:, 0]).astype(np.int64)
        yi = np.round(warped[:, 1]).astype(np.int64)
        valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
        cover[yi[valid], xi[valid]] = True
    line_cover = np.zeros((height, width), np.bool_)
    for name, samp in samples.items():
        if name in skipped or len(samp.line_cover_xy) == 0:
            continue
        mat = np.asarray(part_matrix3(puppet, pose, name)[:2], dtype=np.float64)
        warped = _warp_points(samp.line_cover_xy, mat)
        xi = np.round(warped[:, 0]).astype(np.int64)
        yi = np.round(warped[:, 1]).astype(np.int64)
        valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
        line_cover[yi[valid], xi[valid]] = True
    grown = (
        np.asarray(
            cv2.dilate(cover.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
            dtype=np.uint8,
        )
        > 0
    )
    line_radius = max(0, int(line_radius))
    grown_lines = (
        np.asarray(
            cv2.dilate(
                line_cover.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * line_radius + 1, 2 * line_radius + 1)),
            ),
            dtype=np.uint8,
        )
        > 0
    )
    return np.asarray(grown | grown_lines, dtype=np.bool_)


def fill_last(
    puppet: Puppet,
    pose: Pose,
    part: str,
    drawing: NDArray[np.uint8],
    samples: dict[str, PartSamples] | None = None,
    features: DrawingFeatures | None = None,
    *,
    hint_deg: float | None = None,
    window_deg: float = 360.0,
) -> SnapResult:
    """Place the jigsaw's last piece into the hole the other parts leave.

    The score runs on the unexplained pixels only: samples landing on pixels
    already covered by placed parts count as misses (see :func:`score_samples`).
    """
    feat = features if features is not None else drawing_features(np.asarray(drawing, dtype=np.uint8))
    if samples is None:
        raise ValueError("fill_last needs part samples")
    hole = ~explained_mask(puppet, pose, samples, feat.line.shape, skip=part)
    return snap_part(
        puppet, pose, part, feat.rgb, samples, feat, hint_deg=hint_deg, window_deg=window_deg, explained=~hole
    )


REST_PRIOR = 0.0005
STACK_PENALTY = 0.25


def context_blocked(
    puppet: Puppet,
    pose: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Drawing pixels the already-placed parts outside ``chain`` cover.

    The torso and the fitted chains claim their pixels first, so a chain
    candidate cannot parasitize them (a folded limb stacked on the torso
    gains nothing). Unfitted chains sit at rest; their wrong pixels mostly
    land on paper, which penalizes candidates there anyway.
    """
    height, width = shape
    blocked = np.zeros(height * width, np.bool_)
    for name, samp in samples.items():
        if name in chain or len(samp.cover_xy) == 0:
            continue
        mat = np.asarray(part_matrix3(puppet, pose, name)[:2], dtype=np.float64)
        keys = _pixel_keys(_warp_points(samp.cover_xy, mat), height, width)
        blocked[keys[keys >= 0]] = True
    return blocked


def _decay_at(expl: NDArray[np.bool_] | None, xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Keep factor per sample: agreement on explained pixels decays.

    ``xy`` must be in canvas (owned samples always are): each sample keeps
    ``1 - EXPLAIN_DECAY`` of its agreement where another placed part already
    explains the drawing, all of it on unexplained pixels.
    """
    if expl is None or len(xy) == 0:
        return np.ones(len(xy), np.float64)
    xi = np.round(xy[:, 0]).astype(np.int64)
    yi = np.round(xy[:, 1]).astype(np.int64)
    hit = np.asarray(expl[yi, xi], dtype=np.float64)
    return np.asarray(1.0 - EXPLAIN_DECAY * hit, dtype=np.float64)


def _stack_discount(
    keep: NDArray[np.float64], idx_owned: NDArray[np.int64], cover_count: NDArray[np.int16]
) -> NDArray[np.float64]:
    """Discount agreement stacked on a sibling member's pixels.

    ``idx_owned`` are the member's owned in-canvas pixel keys, ``cover_count``
    how many members of the same chain cover each drawing pixel
    (candidate-relative): samples sharing their pixel with another member
    keep ``1 - EXPLAIN_DECAY`` of their agreement, like context-explained
    pixels. A limb folded onto its own tube then scores its exclusive
    pixels only, while a clean placement keeps nearly everything.
    """
    if len(keep) == 0:
        return np.asarray(keep, dtype=np.float64)
    stacked = np.asarray(cover_count[np.asarray(idx_owned, dtype=np.int64)] > 1, dtype=np.float64)
    return np.asarray(keep * (1.0 - EXPLAIN_DECAY * stacked), dtype=np.float64)


def exclusive_total(
    puppet: Puppet,
    pose: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    blocked: NDArray[np.bool_] | None = None,
    explained: NDArray[np.bool_] | None = None,
) -> float:
    """One chain's joint score with exclusive pixel ownership.

    Each drawing pixel is explained once (top part wins, z order): a folded
    limb stacked on the same drawn pixels gains nothing for the extra parts,
    while a clean placement explains every part's own pixels. Parts with
    samples but no owned pixels left score ``-STACK_PENALTY``. Line and
    colour agreement on pixels other placed parts already explain (``explained``)
    decays, so a limb on unexplained drawing pixels beats one lying on the
    face or belly. Agreement stacked on a sibling member's pixels (same
    chain, same candidate) decays the same way, so a limb folded onto its
    own tube scores its exclusive pixels only.
    """
    height, width = feat.line.shape
    dist_f = feat.dist_f
    paper_f = feat.paper_f
    rgb_f = feat.rgb_f
    members = [m for m in reversed(z_order(puppet, pose)) if m in chain and m in samples]
    coords: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for name in members:
        samp = samples[name]
        mat = np.asarray(part_matrix3(puppet, pose, name)[:2], dtype=np.float64)
        coords[name] = (_warp_points(samp.line_xy, mat), _warp_points(samp.flat_xy, mat))
    key_of: dict[str, tuple[NDArray[np.int64], NDArray[np.int64]]] = {}
    # Reused canvas buffers (all zero between calls): only the touched keys are
    # written and reset, instead of allocating and filling two canvases per call.
    cover_count, owner = _scratch_buffers(height * width)
    touched: list[NDArray[np.int64]] = []
    for name in members:
        line_idx0 = _pixel_keys(coords[name][0], height, width)
        flat_idx0 = _pixel_keys(coords[name][1], height, width)
        key_of[name] = (line_idx0, flat_idx0)
        if int((line_idx0 >= 0).sum()):
            cover_count[line_idx0[line_idx0 >= 0]] += 1
            touched.append(line_idx0[line_idx0 >= 0])
        if int((flat_idx0 >= 0).sum()):
            cover_count[flat_idx0[flat_idx0 >= 0]] += 1
            touched.append(flat_idx0[flat_idx0 >= 0])
    blocked_idx = _blocked_keys(blocked)
    if blocked_idx is not None:
        owner[blocked_idx] = 1
    try:
        return _exclusive_sum(
            members, coords, key_of, samples, cover_count, owner, dist_f, paper_f, rgb_f, tau, explained
        )
    finally:
        for keys in touched:
            cover_count[keys] = 0
            owner[keys] = 0
        if blocked_idx is not None:
            owner[blocked_idx] = 0


_SCRATCH: dict[int, tuple[NDArray[np.int16], NDArray[np.int8]]] = {}
_BLOCKED_CACHE: list[tuple[NDArray[np.bool_], NDArray[np.int64]]] = []


def _scratch_buffers(size: int) -> tuple[NDArray[np.int16], NDArray[np.int8]]:
    """Per-process zeroed ``cover_count`` / ``owner`` canvases, reused across calls."""
    if size not in _SCRATCH:
        _SCRATCH[size] = (np.zeros(size, np.int16), np.zeros(size, np.int8))
    return _SCRATCH[size]


def _blocked_keys(blocked: NDArray[np.bool_] | None) -> NDArray[np.int64] | None:
    """Flat keys of a blocked mask; the search passes the same mask object many times."""
    if blocked is None:
        return None
    for mask, keys in _BLOCKED_CACHE:
        if mask is blocked:
            return keys
    keys = np.flatnonzero(np.asarray(blocked, dtype=np.bool_)).astype(np.int64)
    _BLOCKED_CACHE.insert(0, (blocked, keys))
    del _BLOCKED_CACHE[4:]
    return keys


def _exclusive_sum(
    members: list[str],
    coords: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    key_of: dict[str, tuple[NDArray[np.int64], NDArray[np.int64]]],
    samples: dict[str, PartSamples],
    cover_count: NDArray[np.int16],
    owner: NDArray[np.int8],
    dist_f: NDArray[np.float64],
    paper_f: NDArray[np.float64],
    rgb_f: NDArray[np.float64],
    tau: float,
    explained: NDArray[np.bool_] | None,
) -> float:
    """The member loop of :func:`exclusive_total` (owner is written as members claim pixels)."""
    total = 0.0
    for name in members:
        samp = samples[name]
        line_xy, flat_xy = coords[name]
        line_idx, flat_idx = key_of[name]
        owned_line = line_idx >= 0
        owned_flat = flat_idx >= 0
        owned_line[owned_line] = owner[line_idx[owned_line]] == 0
        owned_flat[owned_flat] = owner[flat_idx[owned_flat]] == 0
        owner[line_idx[line_idx >= 0]] = 1
        owner[flat_idx[flat_idx >= 0]] = 1
        if (len(line_xy) + len(flat_xy)) > 0 and int(owned_line.sum()) + int(owned_flat.sum()) == 0:
            total -= STACK_PENALTY
            continue
        if int(owned_line.sum()):
            keep = _decay_at(explained, line_xy[owned_line])
            keep = _stack_discount(keep, line_idx[owned_line], cover_count)
            dist = _bilinear(dist_f, line_xy[owned_line, 0], line_xy[owned_line, 1], 50.0)
            total += W_LINE * float(np.mean(keep * np.exp(-dist / max(tau, 1e-6))))
        if int(owned_flat.sum()):
            own_xy = flat_xy[owned_flat]
            keep = _decay_at(explained, own_xy)
            keep = _stack_discount(keep, flat_idx[owned_flat], cover_count)
            drawn = _bilinear_rgb(rgb_f, own_xy[:, 0], own_xy[:, 1], 255.0)
            own = np.asarray(samp.flat_rgb[owned_flat].astype(np.float64))
            cd = np.mean(np.abs(drawn - own), axis=1)
            paper_hit = _bilinear(paper_f, own_xy[:, 0], own_xy[:, 1], 1.0)
            total += W_COLOUR * float(np.mean(keep * np.exp(-cd / COLOUR_TAU))) - W_PAPER * float(np.mean(paper_hit))
        n_out = int((len(line_xy) - int((line_idx >= 0).sum())) + (len(flat_xy) - int((flat_idx >= 0).sum())))
        if n_out:
            total -= W_PAPER * n_out / max(len(line_xy) + len(flat_xy), 1)
    return total


def _pixel_keys(xy: NDArray[np.float64], height: int, width: int) -> NDArray[np.int64]:
    """Flat drawing-pixel keys for warped points, -1 outside the canvas."""
    if len(xy) == 0:
        return np.zeros(0, np.int64)
    xi = np.round(xy[:, 0]).astype(np.int64)
    yi = np.round(xy[:, 1]).astype(np.int64)
    valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
    keys = np.full(len(xy), -1, np.int64)
    keys[valid] = yi[valid] * width + xi[valid]
    return keys


def chain_objective(
    puppet: Puppet,
    pose: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float] | None = None,
    blocked: NDArray[np.bool_] | None = None,
    explained: NDArray[np.bool_] | None = None,
) -> float:
    """The joint objective a chain fit maximizes: exclusive pixels, ties to rest.

    Near-symmetric configurations can score almost the same; the tiny rest
    prior picks the clean one. With a transition anchor the centre is X's
    angles (continuity) instead of rest. Agreement on ``explained`` pixels
    decays (see :func:`exclusive_total`).
    """
    total = exclusive_total(puppet, pose, chain, samples, feat, tau, blocked, explained)
    for name in chain:
        target = float(center[name]) if center is not None and name in center else 0.0
        total -= REST_PRIOR * abs(_ang_dist(float(pose.angles.get(name, 0.0)), target))
    return total


def _part_confidence(
    puppet: Puppet,
    pose: Pose,
    part: str,
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
) -> tuple[float, float]:
    """Score + second-peak gap for one part, others fixed.

    A limit range narrower than PEAK_SEP_DEG cannot host a separate peak,
    so the sweep is skipped there and the gap reports NO_AMBIGUITY_GAP.
    Wider ranges sweep the full circle exactly as before.
    """
    samp = samples[part]
    lim = puppet.parts[part].limits
    if lim is not None and len(lim) == 2 and 0.0 <= float(lim[1]) - float(lim[0]) < PEAK_SEP_DEG:
        # No second peak fits inside this range: report the current score
        # with a sentinel gap instead of a coincidental small one.
        cur = score_samples(
            samp,
            np.asarray(part_matrix3(puppet, pose, part)[:2], dtype=np.float64),
            feat,
            tau,
            None,
        )
        return cur, NO_AMBIGUITY_GAP
    steps: list[float] = []
    angle = 0.0
    while angle < 360.0:
        steps.append(float(angle))
        angle += 15.0
    mats = np.stack([_matrix_for(puppet, pose, part, a) for a in steps])
    scores = {a: v for a, v in zip(steps, batch_scores(samp, mats, feat, tau, None), strict=True)}
    cur_score = score_samples(samp, np.asarray(part_matrix3(puppet, pose, part)[:2], dtype=np.float64), feat, tau, None)
    ordered = sorted(scores, key=lambda a: scores[a], reverse=True)
    gap = _second_peak_gap(scores, float(ordered[0]))
    return cur_score, gap


def _sweep_cands(current: float, local_deg: float | None, limits: tuple[float, float] | None = None) -> list[float]:
    if local_deg is None:
        cands: list[float] = []
        angle = -180.0
        while angle < 180.0:
            cands.append(float(angle))
            angle += 15.0
        return _clip_cands(cands, limits)
    cands = [float(current)]
    step = 10.0
    k = step
    while k <= local_deg + 1e-9:
        cands += [float(current - k), float(current + k)]
        k += step
    return _clip_cands(cands, limits)


def _greedy_from(
    puppet: Puppet,
    work: Pose,
    chain: list[str],
    from_angles: dict[str, float],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    local_deg: float | None,
    blocked: NDArray[np.bool_] | None,
    pair_grid: bool = False,
    frozen: dict[str, float] | None = None,
    explained: NDArray[np.bool_] | None = None,
) -> tuple[dict[str, float], dict[str, list[float]], float]:
    """Forward greedy: each joint takes the angle that best serves its downstream.

    Candidates compete for unexplained pixels only (exclusive ownership): the
    rest prior (or the transition anchor) breaks near-ties toward the clean
    configuration. With ``pair_grid`` the first two joints (shoulder/hip
    complex, sharing one rotation no single-joint sweep can compose) go first
    on a coarse 2D grid. ``frozen`` members keep their angle throughout (a
    transition pull freezing the locked members of a changed chain).
    ``explained`` decays agreement on pixels other parts already explain.
    A slack joint in a from-scratch search ranks each angle with its own
    disc-best offset (a hang needing its offset is invisible at any fixed
    one); the winning offsets return alongside the angles so the caller
    scores the winner as ranked. Narrow-window pulls keep the fixed-offset
    ranking exactly.
    """
    stay = dict(frozen or {})
    best = dict(from_angles)
    for name, angle in stay.items():
        best[name] = float(angle)
    if pair_grid and len(chain) >= 2:
        first, second = chain[0], chain[1]
        free = [m for m in (first, second) if m not in stay]
        full = [float(v) for v in range(-180, 180, 45)]
        grid_first = _clip_cands(full, _limits_of(puppet, first))
        grid_second = _clip_cands(full, _limits_of(puppet, second))
        if len(free) == 2:
            top = (-math.inf, 0.0, 0.0)
            for a in grid_first:
                for b in grid_second:
                    trial = work.model_copy(update={"angles": {**best, first: a, second: b}})
                    value = exclusive_total(puppet, trial, chain, samples, feat, tau, blocked, explained)
                    value -= REST_PRIOR * (
                        abs(_ang_dist(a, float(center.get(first, 0.0))))
                        + abs(_ang_dist(b, float(center.get(second, 0.0))))
                    )
                    if value > top[0]:
                        top = (value, a, b)
            best[first], best[second] = float(top[1]), float(top[2])
        elif len(free) == 1:
            name = free[0]
            solo = (-math.inf, 0.0)
            for a in grid_first if name == first else grid_second:
                trial = work.model_copy(update={"angles": {**best, name: a}})
                value = exclusive_total(puppet, trial, chain, samples, feat, tau, blocked, explained)
                value -= REST_PRIOR * abs(_ang_dist(a, float(center.get(name, 0.0))))
                if value > solo[0]:
                    solo = (value, a)
            best[name] = float(solo[1])
    off_now: dict[str, list[float]] = {
        k: [float(v[0]), float(v[1])] for k, v in work.offset.items() if isinstance(v, (list, tuple)) and len(v) == 2
    }
    for j, name in enumerate(chain):
        if name in stay:
            continue
        cands = _sweep_cands(float(best.get(name, 0.0)), local_deg, _limits_of(puppet, name))
        target = float(center.get(name, 0.0))
        base = dict(best)
        if local_deg is None:
            # From scratch the coarse downstream guesses are noise (a folded
            # grid pick would poison every upstream ranking); rest is the
            # restart basis, so condition on that instead. Pulls keep today.
            for m in chain[j:]:
                if m not in stay:
                    base[m] = float(from_angles.get(m, 0.0))
        part = puppet.parts.get(name)
        slack = float(part.slack_px) if part is not None else 0.0
        if slack > 0.0 and local_deg is None:
            # Joint angle×offset ranking: a hang needing its offset is
            # invisible at any fixed offset, so each angle competes with its
            # own disc best. The disc holds (0, 0), so no angle ranks below
            # today's fixed-offset value; the winning offset carries to the
            # members below and back to the caller with the angles.
            win_a, win_v = float(cands[0]), -math.inf
            win_o = [0.0, 0.0]
            for a in cands:
                angled = work.model_copy(update={"angles": {**base, name: float(a)}})
                for dx, dy in _offset_disc(slack, tau):
                    value = _offset_value(
                        puppet,
                        _trial_offset(angled, name, dx, dy),
                        chain,
                        samples,
                        feat,
                        tau,
                        center,
                        blocked,
                        explained,
                    )
                    if value > win_v:
                        win_a, win_v, win_o = float(a), value, [float(dx), float(dy)]
            best[name] = win_a
            if win_o[0] * win_o[0] + win_o[1] * win_o[1] < 1e-18:
                off_now.pop(name, None)
            else:
                off_now[name] = win_o
            continue
        vals: list[float] = []
        for a in cands:
            trial = work.model_copy(update={"angles": {**base, name: float(a)}, "offset": dict(off_now)})
            vals.append(
                exclusive_total(puppet, trial, chain, samples, feat, tau, blocked, explained)
                - REST_PRIOR * abs(_ang_dist(float(a), target))
            )
        best[name] = float(cands[int(np.argmax(np.asarray(vals, dtype=np.float64)))])
    trial = work.model_copy(update={"angles": dict(best), "offset": dict(off_now)})
    return (
        best,
        {m: list(v) for m, v in off_now.items()},
        chain_objective(puppet, trial, chain, samples, feat, tau, center, blocked, explained),
    )


def _greedy_beam(
    puppet: Puppet,
    work: Pose,
    chain: list[str],
    from_angles: dict[str, float],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    blocked: NDArray[np.bool_] | None,
    pair_grid: bool,
    frozen: dict[str, float],
    explained: NDArray[np.bool_] | None,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, float], dict[str, list[float]]]:
    """From-scratch greedy with a beam on the chain's first free joint (K-0r).

    The greedy decides each joint with its downstream at rest, so the first
    joint (a cap with a narrow range, a hip) is chosen against a T-pose limb
    and its children then compensate (cap at its limit, arm leaning) in a
    basin no single-joint descent leaves. The beam reruns the whole greedy
    with that joint pinned to each of its sweep candidates (at most
    ``GREEDY_BEAM``) plus the unpinned run, and keeps the feasible candidate
    with the highest joint objective among those whose plain part-score sum
    is not below the unpinned run's. Only a narrow first joint (limit range
    under ``PEAK_SEP_DEG``) takes the beam: a wide one ranks on its own
    sweep. Never worse than the plain greedy: the unpinned run is one of the
    candidates. Returns the winner's angles and offsets, then the unpinned
    run's (the caller polishes both and keeps the better finished fit).
    """
    runs: list[tuple[dict[str, float], dict[str, list[float]]]] = []
    plain, off_plain, _ = _greedy_from(
        puppet,
        work,
        chain,
        from_angles,
        samples,
        feat,
        tau,
        center,
        None,
        blocked,
        pair_grid=pair_grid,
        frozen=frozen,
        explained=explained,
    )
    runs.append((plain, off_plain))
    first = next((m for m in chain if m not in frozen), None)
    lim = _limits_of(puppet, first) if first is not None else None
    # Only a narrow joint (its range cannot host a second peak) is decided
    # blind: its own pixels barely change across the range, so the T-pose
    # downstream decides it. A wide first joint ranks on its own sweep.
    narrow = lim is not None and 0.0 <= float(lim[1]) - float(lim[0]) < PEAK_SEP_DEG
    if first is not None and GREEDY_BEAM > 0 and narrow:
        cands = _sweep_cands(0.0, None, lim)[:GREEDY_BEAM]
        for a in cands:
            if abs(_ang_dist(float(a), float(plain.get(first, 0.0)))) < 1e-9:
                continue
            pinned, off_pin, _ = _greedy_from(
                puppet,
                work,
                chain,
                from_angles,
                samples,
                feat,
                tau,
                center,
                None,
                blocked,
                pair_grid=pair_grid,
                frozen={**frozen, first: float(a)},
                explained=explained,
            )
            runs.append((pinned, off_pin))
    best: tuple[dict[str, float], dict[str, list[float]]] = runs[0]
    best_v = -math.inf
    plain_floor = -math.inf
    for k, (angles, offs) in enumerate(runs):
        trial = work.model_copy(update={"angles": {**work.angles, **angles}, "offset": dict(offs)})
        feas, _ = chain_feasible(puppet, trial, chain, samples, feat.line.shape)
        value = _offset_value(puppet, trial, chain, samples, feat, tau, center, blocked, explained)
        plain_sum = float(
            sum(
                score_samples(samples[m], np.asarray(part_matrix3(puppet, trial, m)[:2], dtype=np.float64), feat, tau)
                for m in chain
                if m in samples
            )
        )
        if k == 0:
            plain_floor = plain_sum
        if not feas:
            value = -math.inf
        # A pinned run must dominate the unpinned one: more joint objective
        # AND no lower own-part agreement (the pose total). Exclusivity alone
        # must not buy a pose that matches its parts worse.
        if k > 0 and plain_sum < plain_floor:
            continue
        if value > best_v:
            best, best_v = (angles, offs), value
    return best[0], best[1], runs[0][0], runs[0][1]


def _offset_disc(slack: float, tau: float) -> list[tuple[float, float]]:
    """Coarse search points inside the slack disc: centre plus rim rings.

    Ring spacing follows the chamfer basin (``3 * tau``) so the grid always
    lands a point inside the optimum's basin; the rim ring is always present
    so a joint slid all the way out is reachable.
    """
    pts = [(0.0, 0.0)]
    if slack <= 0.0:
        return pts
    spacing = max(2.0, min(float(slack) / 3.0, 3.0 * max(float(tau), 1.0)))
    radius = spacing
    while radius < slack - 1e-9:
        count = max(6, int(round(2.0 * math.pi * radius / spacing)))
        for k in range(count):
            ang = 2.0 * math.pi * float(k) / float(count)
            pts.append((radius * math.cos(ang), radius * math.sin(ang)))
        radius += spacing
    count = max(6, int(round(2.0 * math.pi * slack / spacing)))
    for k in range(count):
        ang = 2.0 * math.pi * float(k) / float(count)
        pts.append((slack * math.cos(ang), slack * math.sin(ang)))
    return pts


def _trial_offset(work: Pose, name: str, dx: float, dy: float) -> Pose:
    """Copy of ``work`` with ``name``'s slack offset set (zero entries dropped)."""
    offsets = dict(work.offset)
    if dx * dx + dy * dy < 1e-18:
        offsets.pop(name, None)
    else:
        offsets[name] = [float(dx), float(dy)]
    return work.model_copy(update={"offset": offsets})


def _offset_value(
    puppet: Puppet,
    trial: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    blocked: NDArray[np.bool_] | None,
    explained: NDArray[np.bool_] | None,
) -> float:
    """Chain objective with the small prior toward zero offset.

    The angle prior terms are constant while offsets move, so offset
    candidates compete on their own agreement minus the pull to the modelled
    pivot; a joint that explains nothing extra stays exactly on its pivot.
    """
    total = chain_objective(puppet, trial, chain, samples, feat, tau, center, blocked, explained)
    for name in chain:
        off = trial.offset.get(name)
        if isinstance(off, (list, tuple)) and len(off) == 2:
            total -= OFFSET_PRIOR * math.hypot(float(off[0]), float(off[1]))
    return total


def _resnap_leaf(
    puppet: Puppet,
    work: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    blocked: NDArray[np.bool_] | None,
    explained: NDArray[np.bool_] | None,
    frozen: dict[str, float],
) -> None:
    """Re-snap the chain's leaf member with exclusive scoring, in place.

    Full-circle coarse sweep plus fine refinement, scored with the joint
    exclusive objective (stolen pixels discounted, seams excluded from
    feasibility) at the current upstream angles and offsets; the feasible
    gated value keeps the re-snap only on improvement. Catches what the
    stacking-blind baseline folds (a claw parked on its own tube) and what
    the local descent cannot walk out of. No-op without a slack member in
    the chain, so slack-0 fits keep today's path exactly.
    """
    if not chain or chain[-1] in frozen:
        return
    slack_here = False
    for m in chain:
        part = puppet.parts.get(m)
        if part is not None and float(part.slack_px) > 0.0:
            slack_here = True
            break
    if not slack_here:
        return
    leaf = chain[-1]
    lim = _limits_of(puppet, leaf)
    cur = float(work.angles.get(leaf, 0.0))

    def trial_value(angle: float) -> tuple[bool, float]:
        trial = work.model_copy(update={"angles": {**work.angles, leaf: float(angle)}})
        feas, _ = chain_feasible(puppet, trial, chain, samples, feat.line.shape)
        if not feas:
            return False, -math.inf
        return True, _offset_value(puppet, trial, chain, samples, feat, tau, center, blocked, explained)

    _, best_total = trial_value(cur)
    best_angle = cur
    angle = 0.0
    while angle < 360.0:
        for cand in _clip_cands([angle], lim):
            feas, value = trial_value(cand)
            if feas and value > best_total:
                best_total, best_angle = value, float(cand)
        angle += COARSE_STEP
    angle = best_angle - FINE_RANGE
    while angle <= best_angle + FINE_RANGE + 1e-9:
        for cand in _clip_cands([angle], lim):
            feas, value = trial_value(cand)
            if feas and value > best_total:
                best_total, best_angle = value, float(cand)
        angle += FINE_STEP
    if best_angle != cur:
        work.angles[leaf] = float(best_angle)


def _fit_offsets(
    puppet: Puppet,
    work: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    blocked: NDArray[np.bool_] | None,
    explained: NDArray[np.bool_] | None,
    frozen: dict[str, float],
    *,
    coarse: bool,
    refine: bool,
) -> None:
    """Search slack-joint offsets in place: coarse disc grid, then descent.

    Parent-first, one member at a time (a parent's slide moves everything
    below it, so it settles first). Members without slack and frozen members
    take today's code path untouched: only a non-zero offset that beats the
    objective is recorded, so a slack-0 fit leaves ``pose.offset`` empty
    exactly.
    """
    for name in chain:
        if name in frozen:
            continue
        part = puppet.parts.get(name)
        slack = float(part.slack_px) if part is not None else 0.0
        if slack <= 0.0:
            continue
        cur = work.offset.get(name)
        if isinstance(cur, (list, tuple)) and len(cur) == 2:
            best_dx, best_dy = float(cur[0]), float(cur[1])
        else:
            best_dx, best_dy = 0.0, 0.0
        best = _offset_value(
            puppet,
            _trial_offset(work, name, best_dx, best_dy),
            chain,
            samples,
            feat,
            tau,
            center,
            blocked,
            explained,
        )
        if coarse:
            for dx, dy in _offset_disc(slack, tau):
                if dx * dx + dy * dy > slack * slack + 1e-9:
                    continue
                value = _offset_value(
                    puppet,
                    _trial_offset(work, name, dx, dy),
                    chain,
                    samples,
                    feat,
                    tau,
                    center,
                    blocked,
                    explained,
                )
                if value > best:
                    best, best_dx, best_dy = value, float(dx), float(dy)
        if refine:
            for step in (2.0, 1.0, 0.5):
                moved = True
                sweeps = 0
                while moved and sweeps < 6:
                    moved = False
                    sweeps += 1
                    neighbours = (
                        (best_dx + step, best_dy),
                        (best_dx - step, best_dy),
                        (best_dx, best_dy + step),
                        (best_dx, best_dy - step),
                        (0.0, 0.0),
                    )
                    for dx, dy in neighbours:
                        if dx * dx + dy * dy > slack * slack + 1e-9:
                            continue
                        value = _offset_value(
                            puppet,
                            _trial_offset(work, name, dx, dy),
                            chain,
                            samples,
                            feat,
                            tau,
                            center,
                            blocked,
                            explained,
                        )
                        if value > best:
                            best, best_dx, best_dy = value, float(dx), float(dy)
                            moved = True
        if best_dx * best_dx + best_dy * best_dy < 1e-18:
            work.offset.pop(name, None)
        else:
            work.offset[name] = [float(best_dx), float(best_dy)]


def _polish_chain(
    puppet: Puppet,
    base: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    blocked: NDArray[np.bool_] | None,
    explained: NDArray[np.bool_] | None,
    stay: dict[str, float],
    best_total: float,
    best_angles: dict[str, float],
    best_offsets: dict[str, list[float]],
) -> Pose:
    """Refine one chain candidate: coordinate descent, then offsets (on a copy).

    The descent maximizes plain part sums and is kept only when the joint
    objective does not drop below ``best_total`` (the candidate's ranked
    value); offsets then get the coarse grid and descent.
    """
    work = base.model_copy(deep=True)
    best_angles = dict(best_angles)
    # Restore the winner into work for the descent below.
    for m in chain:
        work.angles[m] = float(best_angles[m])
    for m in chain:
        if m in best_offsets:
            work.offset[m] = list(best_offsets[m])
        else:
            work.offset.pop(m, None)
    # Coordinate descent on the joint objective. Turning one joint only moves
    # its downstream members, so candidates re-score just those (cached plain
    # sums for the rest); only the turned joint's prior term changes.
    part_score: dict[str, float] = {}
    settled = work.model_copy(update={"angles": dict(best_angles)})
    for member in chain:
        mat = np.asarray(part_matrix3(puppet, settled, member)[:2], dtype=np.float64)
        part_score[member] = score_samples(samples[member], mat, feat, tau, None)
    best_plain = float(sum(part_score.values()))
    best_prior = REST_PRIOR * sum(
        abs(_ang_dist(float(best_angles.get(m, 0.0)), float(center.get(m, 0.0)))) for m in chain
    )
    pre_descent = dict(best_angles)
    for step in (5.0, 2.0, 0.5, 0.25):
        improved = True
        sweeps = 0
        while improved and sweeps < 4:
            improved = False
            sweeps += 1
            for j, name in enumerate(chain):
                if name in stay:
                    continue
                affected = chain[j:]
                old_term = REST_PRIOR * abs(_ang_dist(float(best_angles[name]), float(center.get(name, 0.0))))
                lim = _limits_of(puppet, name)
                cands = _clip_cands([best_angles[name] - step, best_angles[name] + step], lim)
                for cand in cands:
                    trial = work.model_copy(update={"angles": {**best_angles, name: float(cand)}})
                    new = 0.0
                    fresh: dict[str, float] = {}
                    for member in affected:
                        mat = np.asarray(part_matrix3(puppet, trial, member)[:2], dtype=np.float64)
                        fresh[member] = score_samples(samples[member], mat, feat, tau, None)
                        new += fresh[member]
                    old = sum(part_score[m] for m in affected)
                    new_term = REST_PRIOR * abs(_ang_dist(float(cand), float(center.get(name, 0.0))))
                    if best_plain - old + new - (best_prior - old_term + new_term) > best_plain - best_prior:
                        best_plain = best_plain - old + new
                        best_prior = best_prior - old_term + new_term
                        best_angles[name] = float(cand)
                        part_score.update(fresh)
                        improved = True
    # The descent maximizes plain part sums; keep it only when the exclusive
    # joint objective agrees it did not wander back into a fold.
    descended = work.model_copy(update={"angles": dict(best_angles)})
    if chain_objective(puppet, descended, chain, samples, feat, tau, center, blocked, explained) < best_total:
        best_angles = dict(pre_descent)
    for name in chain:
        work.angles[name] = float(best_angles[name])
    _fit_offsets(puppet, work, chain, samples, feat, tau, center, blocked, explained, stay, coarse=True, refine=True)
    return work


def _dominates(
    puppet: Puppet,
    first: Pose,
    second: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    center: dict[str, float],
    blocked: NDArray[np.bool_] | None,
    explained: NDArray[np.bool_] | None,
) -> bool:
    """True when ``first`` beats ``second`` on the joint objective without a lower own-part sum."""
    values = []
    for pose in (first, second):
        feas, _ = chain_feasible(puppet, pose, chain, samples, feat.line.shape)
        value = _offset_value(puppet, pose, chain, samples, feat, tau, center, blocked, explained)
        plain = float(
            sum(
                score_samples(samples[m], np.asarray(part_matrix3(puppet, pose, m)[:2], dtype=np.float64), feat, tau)
                for m in chain
                if m in samples
            )
        )
        values.append((value if feas else -math.inf, plain))
    return values[0][0] > values[1][0] and values[0][1] >= values[1][1]


def fit_chain(
    puppet: Puppet,
    pose: Pose,
    chain_name: str,
    drawing: NDArray[np.uint8],
    samples: dict[str, PartSamples] | None = None,
    features: DrawingFeatures | None = None,
    *,
    anchor: dict[str, float] | None = None,
    local_deg: float | None = None,
    frozen: dict[str, float] | None = None,
    beam: bool = True,
) -> Pose:
    """Fit a whole limb jointly (2-5 angles).

    Part-by-part snaps first (the baseline), then forward greedy runs from the
    baseline and — for a fit from scratch — from rest zeros, then coordinate
    descent; the best joint objective wins, so the chain fit is never worse
    than part-by-part on the same drawing. An unobservable carrier joint (an
    elbow ring) can poison the baseline with a wild angle its descendants then
    compensate; the from-rest greedy does not inherit that mistake. Without a
    keypoint model the limb end comes from the chain's own chamfer search;
    single-part snapping stays for the human's Shift-snap and low-confidence
    parts. A transition pull passes ``anchor`` (X's angles) and a narrow
    ``local_deg`` window so the chain stays near X. ``frozen`` pins members to
    fixed angles (the locked members of a changed chain): they keep their
    angle through the baseline, the greedy and the descent, so the pull only
    moves what the difference map unlocked. Declared joint limits
    (``PuppetPart.limits``) clip every other candidate on every search path;
    ``frozen`` still pins exactly. Slack joints (``PuppetPart.slack_px``
    above 0) search an offset inside the slack disc together with the angles
    (coarse grid then descent, small prior to 0); only a non-zero offset
    that beats the objective is recorded. A winner that folds a member onto
    its own chain (see :func:`chain_feasible`) loses to the best feasible
    candidate (the straight continuation always runs); a chain kept unfolded
    without a seating fit is flagged in ``pose.chain_notes`` with the
    never-fit reason. ``beam`` runs the from-scratch greedy with the
    narrow-first-joint beam (:func:`_greedy_beam`); the view fit uses it
    only in its last pass, where the other chains' context is final.
    """
    feat = features if features is not None else drawing_features(np.asarray(drawing, dtype=np.uint8))
    if samples is None:
        raise ValueError("fit_chain needs part samples")
    names = chains(puppet)
    if chain_name not in names:
        raise ValueError(f"unknown chain {chain_name!r} (known: {sorted(names)})")
    chain = names[chain_name]
    stay = {m: float(v) for m, v in (frozen or {}).items() if m in chain}
    center = {m: float(anchor[m]) if anchor is not None and m in anchor else 0.0 for m in chain}
    tau = _tau_for(puppet, pose)
    work = pose.model_copy(deep=True)
    for name, angle in stay.items():
        work.angles[name] = float(angle)
    # Candidate offsets from the input pose. A straight input already sits
    # in the offset basin; a raw input far from truth (T-pose for a raised
    # arm) may drag them onto dense lines instead, so they only ever seed
    # baseline B below, never the fit directly.
    _fit_offsets(puppet, work, chain, samples, feat, tau, center, None, None, stay, coarse=True, refine=True)
    offsets_b = {m: list(work.offset[m]) for m in chain if m in work.offset}
    for m in chain:
        work.offset.pop(m, None)

    def snap_baseline() -> None:
        for name in chain:
            if name in stay:
                continue
            if local_deg is None:
                res = snap_part(puppet, work, name, feat.rgb, samples, feat)
            else:
                res = snap_part(
                    puppet,
                    work,
                    name,
                    feat.rgb,
                    samples,
                    feat,
                    hint_deg=float(work.angles.get(name, 0.0)),
                    window_deg=float(local_deg),
                )
            work.angles[name] = float(res.angle)
            work.scores[name] = float(res.score)
            work.confidences[name] = float(res.confidence)

    # Baseline A: part by part in tree order, zero offsets (today's path).
    snap_baseline()
    angles_a = {m: float(work.angles.get(m, 0.0)) for m in chain}
    angles_b: dict[str, float] = {}
    if offsets_b:
        # Baseline B: same snaps with the input-pose offsets in place. Its
        # angles only seed the B offset hypothesis below; the hypotheses meet
        # at the greedy level, where each gets a full restart.
        for m in chain:
            if m in offsets_b:
                work.offset[m] = list(offsets_b[m])
        snap_baseline()
        angles_b = {m: float(work.angles.get(m, 0.0)) for m in chain}
    # A is today's default: reset the work pose to it for the context fields.
    for m in chain:
        work.angles[m] = float(angles_a[m])
        work.offset.pop(m, None)
    blocked = context_blocked(puppet, work, chain, samples, feat.line.shape)
    # Pixels other placed parts explain (dilated cover, robust to the sparse
    # striding of large parts): agreement there decays, so the chain prefers
    # unexplained drawing pixels. Other parts sit still during this fit, so
    # one field serves every candidate below.
    explained = explained_mask(
        puppet, work, samples, feat.line.shape, skip=set(chain), line_radius=_explain_line_radius(puppet, work)
    )
    # HA: offsets at the A baseline (coarse grid then descent, like angles).
    _fit_offsets(puppet, work, chain, samples, feat, tau, center, blocked, explained, stay, coarse=True, refine=True)
    offsets_a = {m: list(work.offset[m]) for m in chain if m in work.offset}
    if anchor is None:
        # The partwise baseline is stacking-blind (own score only): a claw
        # snapped onto its own tube's dense lines beats the open claw, and
        # everything downstream then keeps the fold. Re-snapping the leaf
        # with the joint exclusive objective at settled upstream angles and
        # offsets finds the open claw when it seats (a full-circle sweep, not
        # a local walk). Slack arm chains only; skipped under a transition
        # pull (locality) and with frozen leaves.
        _resnap_leaf(puppet, work, chain, samples, feat, tau, center, blocked, explained, stay)
    baseline_angles = dict(work.angles)
    # The re-snap may have moved the leaf: refresh the baseline snapshot so
    # the floor below evaluates the re-snapped placement, not the blind snap.
    angles_a = {m: float(work.angles.get(m, 0.0)) for m in chain}
    floor_a = work.model_copy(update={"angles": {**work.angles, **angles_a}, "offset": dict(offsets_a)})
    feas_floor, _ = chain_feasible(puppet, floor_a, chain, samples, feat.line.shape)
    best_total = _offset_value(puppet, floor_a, chain, samples, feat, tau, center, blocked, explained)
    if not feas_floor:
        best_total = -math.inf
    best_angles = dict(angles_a)
    best_offsets = dict(offsets_a)
    if anchor is None:
        # From scratch the partwise baseline can inherit one wild carrier
        # joint (an elbow ring) that its descendants then compensate, so the
        # global greedy restarts from rest zeros with the pair grid on.
        from_angles = {m: 0.0 for m in chain}
        pair_grid = True
    else:
        # A transition pull stays near X: greedy restarts from the baseline
        # with the narrow local window.
        from_angles = dict(best_angles)
        pair_grid = False
    # Greedy under the A offsets (today's restart).
    for m in chain:
        if m in offsets_a:
            work.offset[m] = list(offsets_a[m])
        else:
            work.offset.pop(m, None)
    beam_plain: tuple[dict[str, float], dict[str, list[float]]] | None = None
    if beam and anchor is None and local_deg is None:
        greedy_a, off_ga, plain_a, off_pa = _greedy_beam(
            puppet, work, chain, from_angles, samples, feat, tau, center, blocked, pair_grid, stay, explained
        )
        if any(abs(_ang_dist(float(greedy_a.get(m, 0.0)), float(plain_a.get(m, 0.0)))) > 1e-9 for m in chain):
            beam_plain = (plain_a, off_pa)
    else:
        greedy_a, off_ga, _ = _greedy_from(
            puppet,
            work,
            chain,
            from_angles,
            samples,
            feat,
            tau,
            center,
            local_deg,
            blocked,
            pair_grid=pair_grid,
            frozen=stay,
            explained=explained,
        )
    # The candidate set without the beam's pinned winner (today's path):
    # polished alongside the beam's set below when the two differ.
    alt_total, alt_angles, alt_offsets = best_total, dict(best_angles), dict(best_offsets)
    if beam_plain is not None:
        trial_pa = work.model_copy(update={"angles": {**work.angles, **beam_plain[0]}, "offset": dict(beam_plain[1])})
        feas_pa, _ = chain_feasible(puppet, trial_pa, chain, samples, feat.line.shape)
        value_pa = _offset_value(puppet, trial_pa, chain, samples, feat, tau, center, blocked, explained)
        if feas_pa and value_pa > alt_total:
            alt_total, alt_angles, alt_offsets = value_pa, dict(beam_plain[0]), dict(beam_plain[1])
    trial_ga = work.model_copy(update={"angles": {**work.angles, **greedy_a}, "offset": dict(off_ga)})
    feas_ga, _ = chain_feasible(puppet, trial_ga, chain, samples, feat.line.shape)
    value_ga = _offset_value(puppet, trial_ga, chain, samples, feat, tau, center, blocked, explained)
    if feas_ga and value_ga > best_total:
        best_total, best_angles, best_offsets = value_ga, dict(greedy_a), dict(off_ga)
    if offsets_b:
        # HB: the input-pose offsets refined at the B baseline, then the same
        # restart under them; the feasibility-gated value picks the hypothesis.
        for m in chain:
            work.angles[m] = float(angles_b[m])
            if m in offsets_b:
                work.offset[m] = list(offsets_b[m])
            else:
                work.offset.pop(m, None)
        _fit_offsets(
            puppet, work, chain, samples, feat, tau, center, blocked, explained, stay, coarse=False, refine=True
        )
        offsets_hb = {m: list(work.offset[m]) for m in chain if m in work.offset}
        for m in chain:
            if m in offsets_hb:
                work.offset[m] = list(offsets_hb[m])
            else:
                work.offset.pop(m, None)
        greedy_b, off_gb, _ = _greedy_from(
            puppet,
            work,
            chain,
            from_angles,
            samples,
            feat,
            tau,
            center,
            local_deg,
            blocked,
            pair_grid=pair_grid,
            frozen=stay,
            explained=explained,
        )
        trial_gb = work.model_copy(update={"angles": {**work.angles, **greedy_b}, "offset": dict(off_gb)})
        feas_gb, _ = chain_feasible(puppet, trial_gb, chain, samples, feat.line.shape)
        value_gb = _offset_value(puppet, trial_gb, chain, samples, feat, tau, center, blocked, explained)
        if feas_gb and value_gb > best_total:
            best_total, best_angles, best_offsets = value_gb, dict(greedy_b), dict(off_gb)
        if feas_gb and value_gb > alt_total:
            alt_total, alt_angles, alt_offsets = value_gb, dict(greedy_b), dict(off_gb)
    base_work = work.model_copy(deep=True)
    work = _polish_chain(
        puppet,
        base_work,
        chain,
        samples,
        feat,
        tau,
        center,
        blocked,
        explained,
        stay,
        best_total,
        best_angles,
        best_offsets,
    )
    if beam_plain is not None and alt_angles != best_angles:
        # The beam's pick is ranked before refinement; the plain path may
        # refine past it. Keep the beam only when its finished fit wins on
        # the joint objective without lowering the chain's own part scores.
        other = _polish_chain(
            puppet,
            base_work,
            chain,
            samples,
            feat,
            tau,
            center,
            blocked,
            explained,
            stay,
            alt_total,
            alt_angles,
            alt_offsets,
        )
        if not _dominates(puppet, work, other, chain, samples, feat, tau, center, blocked, explained):
            work = other
    # Never fold: an infeasible winner (a member stacked on its own chain
    # past OVERLAP_MAX, seams excluded) loses to the best feasible
    # candidate. The straight continuation (rest, or X under a transition
    # pull) always runs, so a chain the puppet cannot seat stays unfolded
    # and flagged instead of folding onto dense lines.
    shape = feat.line.shape
    feasible_win, _ = chain_feasible(puppet, work, chain, samples, shape)
    if not feasible_win:
        options = [
            ("fit", {m: float(work.angles.get(m, 0.0)) for m in chain}),
            ("baseline", {m: float(baseline_angles.get(m, 0.0)) for m in chain}),
            ("continuation", {m: float(center.get(m, 0.0)) for m in chain}),
        ]
        scored = []
        for _label, ang in options:
            trial = work.model_copy(update={"angles": {**work.angles, **ang}})
            feas, _ = chain_feasible(puppet, trial, chain, samples, shape)
            total = chain_objective(puppet, trial, chain, samples, feat, tau, center, blocked, explained)
            scored.append((ang, feas, total))
        feasible_opts = [s for s in scored if s[1]]
        kept = max(feasible_opts, key=lambda s: s[2]) if feasible_opts else max(scored, key=lambda s: s[2])
        for name in chain:
            work.angles[name] = float(kept[0][name])
        seated = True
        for member in chain:
            if member not in samples:
                continue
            mat = np.asarray(part_matrix3(puppet, work, member)[:2], dtype=np.float64)
            if score_samples(samples[member], mat, feat, tau, None) < SCORE_MIN:
                seated = False
                break
        if not feasible_opts or not seated:
            work.chain_notes[chain_name] = NEVER_FIT_REASON
    for name in chain:
        score, gap = _part_confidence(puppet, work, name, samples, feat, tau)
        work.scores[name] = float(score)
        work.confidences[name] = float(gap)
    return work


def fit_root(
    puppet: Puppet,
    pose: Pose,
    drawing: NDArray[np.uint8],
    samples: dict[str, PartSamples] | None = None,
    features: DrawingFeatures | None = None,
) -> Pose:
    """Root similarity from the torso: multi-start chamfer search, then refine."""
    feat = features if features is not None else drawing_features(np.asarray(drawing, dtype=np.uint8))
    if samples is None:
        raise ValueError("fit_root needs part samples")
    root = puppet.root
    full = samples[root]
    # The root search evaluates thousands of candidates: thin the torso flats.
    stride = _stride(len(full.flat_xy), ROOT_MAX_FLAT)
    samp = PartSamples(
        line_xy=full.line_xy,
        flat_xy=full.flat_xy[::stride],
        flat_rgb=full.flat_rgb[::stride],
        cover_xy=full.cover_xy,
    )
    pw, ph = int(puppet.canvas[0]), int(puppet.canvas[1])
    dh, dw = feat.line.shape
    scale0 = min(dw / max(pw, 1), dh / max(ph, 1))
    pivot = puppet.parts[root].pivot
    cx, cy = float(pivot[0]), float(pivot[1])
    dcx, dcy = dw / 2.0, dh / 2.0
    pcx, pcy = pw / 2.0, ph / 2.0
    t0x = dcx - (scale0 * (pcx - cx) + cx)
    t0y = dcy - (scale0 * (pcy - cy) + cy)
    span = max(24.0, 0.08 * max(dw, dh))
    work = pose.model_copy(deep=True)

    def root_score(scale: float, rot: float, tx: float, ty: float) -> float:
        trial = work.model_copy(
            update={"scale": float(scale), "rotation": float(rot), "tx": float(tx), "ty": float(ty)}
        )
        tau = _tau_for(puppet, trial)
        mat = np.asarray(part_matrix3(puppet, trial, root)[:2], dtype=np.float64)
        return score_samples(samp, mat, feat, tau, None)

    best = (float(scale0), 0.0, float(t0x), float(t0y))
    best_s = root_score(*best)
    grid: list[tuple[float, float, float, float, float]] = []
    scales = [scale0 + k * 0.03 for k in range(-4, 5) if scale0 + k * 0.03 > 0.2]
    params: list[tuple[float, float, float, float]] = []
    for scale in scales:
        for rot in (-10.0, -5.0, 0.0, 5.0, 10.0):
            for tx in (t0x - span, t0x - span / 2, t0x, t0x + span / 2, t0x + span):
                for ty in (t0y - span, t0y - span / 2, t0y, t0y + span / 2, t0y + span):
                    params.append((float(scale), float(rot), float(tx), float(ty)))
    by_scale: dict[float, list[int]] = {}
    for i, (scale, _rot, _tx, _ty) in enumerate(params):
        by_scale.setdefault(scale, []).append(i)
    values = [0.0] * len(params)
    for scale, idx in by_scale.items():
        trial0 = work.model_copy(update={"scale": float(scale), "rotation": 0.0, "tx": 0.0, "ty": 0.0})
        tau_s = _tau_for(puppet, trial0)
        for chunk in range(0, len(idx), 256):
            sel = idx[chunk : chunk + 256]
            mats = np.stack(
                [
                    np.asarray(
                        part_matrix3(
                            puppet,
                            work.model_copy(
                                update={
                                    "scale": params[j][0],
                                    "rotation": params[j][1],
                                    "tx": params[j][2],
                                    "ty": params[j][3],
                                }
                            ),
                            root,
                        )[:2],
                        dtype=np.float64,
                    )
                    for j in sel
                ]
            )
            for j, value in zip(sel, batch_scores(samp, mats, feat, tau_s, None), strict=True):
                values[j] = value
    for (scale, rot, tx, ty), value in zip(params, values, strict=True):
        grid.append((value, scale, rot, tx, ty))
        if value > best_s:
            best_s = value
            best = (scale, rot, tx, ty)
    # Multi-start polish: the coarse grid can sit in the wrong basin (a small
    # translation step overshoots while scale compensates), so the best few
    # coarse cells each get a fine hill-climb.
    grid.sort(key=lambda cell: cell[0], reverse=True)
    starts = [(cell[1], cell[2], cell[3], cell[4]) for cell in grid[:6]]
    if best not in starts:
        starts.append(best)
    for start in starts:
        cand_best = (float(start[0]), float(start[1]), float(start[2]), float(start[3]))
        cand_s = root_score(*cand_best)
        for ds, dr, dt in ((0.01, 1.0, 2.0), (0.005, 0.5, 1.0)):
            improved = True
            rounds = 0
            while improved and rounds < 25:
                improved = False
                rounds += 1
                scale, rot, tx, ty = cand_best
                for cand in (
                    (scale + ds, rot, tx, ty),
                    (scale - ds, rot, tx, ty),
                    (scale, rot + dr, tx, ty),
                    (scale, rot - dr, tx, ty),
                    (scale, rot, tx + dt, ty),
                    (scale, rot, tx - dt, ty),
                    (scale, rot, tx, ty + dt),
                    (scale, rot, tx, ty - dt),
                ):
                    if cand[0] <= 0.2:
                        continue
                    value = root_score(*cand)
                    if value > cand_s:
                        cand_s = value
                        cand_best = (float(cand[0]), float(cand[1]), float(cand[2]), float(cand[3]))
                        improved = True
                        break
        if cand_s > best_s:
            best_s = cand_s
            best = cand_best
    work.scale = float(best[0])
    work.rotation = float(best[1])
    work.tx = float(best[2])
    work.ty = float(best[3])
    return work


@dataclass
class LoadedView:
    name: str
    puppet: Puppet
    images: dict[str, NDArray[np.uint8]]
    yaw_deg: float = 0.0


def _rgba_from(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as opened:
        rgba = opened.convert("RGBA")
    return np.asarray(rgba, dtype=np.uint8)


def load_view(set_dir: Path, view: str) -> LoadedView:
    """One built view puppet plus its part images (and ``_shadow`` when built)."""
    pset = load_set(Path(set_dir))
    if view in pset.views:
        entry = pset.views[view]
    elif view in pset.head_views:
        entry = pset.head_views[view]
    else:
        raise ValueError(f"unknown view {view!r} (known: {sorted([*pset.views, *pset.head_views])})")
    base = Path(set_dir) / entry.dir
    puppet = Puppet.model_validate_json((base / "puppet.json").read_text(encoding="utf-8"))
    images: dict[str, NDArray[np.uint8]] = {}
    for name, part in puppet.parts.items():
        if part.hidden or not part.file:
            continue
        png = base / part.file
        if png.is_file():
            images[name] = _rgba_from(png)
    shadow = base / "parts" / "_shadow.png"
    if shadow.is_file():
        images["_shadow"] = _rgba_from(shadow)
    return LoadedView(name=view, puppet=puppet, images=images, yaw_deg=float(entry.yaw_deg))


def all_views(set_dir: Path) -> dict[str, LoadedView]:
    """Every body view of the set (head views render through ``head_for``; K-2)."""
    pset = load_set(Path(set_dir))
    return {name: load_view(set_dir, name) for name in pset.views}


def chain_order(puppet: Puppet) -> list[str]:
    """Chain names parents-first (by each chain's top part in tree order)."""
    order = tree_order(puppet)
    pos = {name: i for i, name in enumerate(order)}
    names = chains(puppet)

    def key(leaf: str) -> tuple[int, str]:
        top = names[leaf][0] if names[leaf] else leaf
        return (pos.get(top, len(order)), leaf)

    return sorted(names, key=key)


def resolve_z(
    puppet: Puppet,
    pose: Pose,
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
) -> Pose:
    """Top-most wins: where two parts overlap, the one agreeing with the drawing stays up."""
    work = pose.model_copy(deep=True)
    names = [n for n in z_order(puppet, work) if n in samples and len(samples[n].flat_xy)]
    warped: dict[str, NDArray[np.float64]] = {}
    for name in names:
        mat = np.asarray(part_matrix3(puppet, work, name)[:2], dtype=np.float64)
        warped[name] = _warp_points(samples[name].flat_xy, mat)
    height, width = feat.line.shape
    drew = feat.rgb_f

    def agree_idx(name: str, idx: NDArray[np.int64], xs: NDArray[np.int64], ys: NDArray[np.int64]) -> float:
        own = np.asarray(samples[name].flat_rgb[idx].astype(np.float64))
        cd = np.mean(np.abs(drew[ys, xs] - own), axis=1)
        return float(np.mean(np.exp(-cd / COLOUR_TAU)))

    rounded: dict[str, tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]] = {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for name in names:
        w = warped[name]
        xs = np.round(w[:, 0]).astype(np.int64)
        ys = np.round(w[:, 1]).astype(np.int64)
        ok = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        rounded[name] = (xs, ys, ok)
        if bool(ok.any()):
            boxes[name] = (int(xs[ok].min()), int(ys[ok].min()), int(xs[ok].max()), int(ys[ok].max()))
        else:
            boxes[name] = (0, 0, -1, -1)
    for i, upper in enumerate(names):
        xu, yu, ok_u = rounded[upper]
        ax0, ay0, ax1, ay1 = boxes[upper]
        for lower in names[:i]:
            bx0, by0, bx1, by1 = boxes[lower]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            xl, yl, ok_l = rounded[lower]
            cells_l = set(zip(xl[ok_l].tolist(), yl[ok_l].tolist(), strict=True))
            hit_u = np.zeros(len(xu), np.bool_)
            for k in range(len(xu)):
                hit_u[k] = bool(ok_u[k]) and (int(xu[k]), int(yu[k])) in cells_l
            idx_u = np.nonzero(hit_u)[0]
            if len(idx_u) < 50:
                continue
            cells_hit = set(zip(xu[idx_u].tolist(), yu[idx_u].tolist(), strict=True))
            hit_l = np.zeros(len(xl), np.bool_)
            for k in range(len(xl)):
                hit_l[k] = bool(ok_l[k]) and (int(xl[k]), int(yl[k])) in cells_hit
            idx_l = np.nonzero(hit_l)[0]
            if len(idx_l) < 50:
                continue
            au = agree_idx(upper, idx_u, xu[idx_u], yu[idx_u])
            al = agree_idx(lower, idx_l, xl[idx_l], yl[idx_l])
            if al > au + 0.15:
                zu = int(work.z_override.get(upper, puppet.parts[upper].z))
                work.z_override[lower] = zu + 1
    return work


def overlap_fraction(
    puppet: Puppet,
    pose: Pose,
    samples: dict[str, PartSamples],
    shape: tuple[int, int],
) -> dict[str, float]:
    """Per part, the fraction of its in-bounds pixels another part also covers.

    Order-free (whoever is on top still shares them): a part stacked onto a
    sibling's pixels — a folded limb, a hidden part parked somewhere visible —
    stands out whatever the z order says.
    """
    height, width = shape
    count = np.zeros(height * width, np.int16)
    keys: dict[str, NDArray[np.int64]] = {}
    for name, samp in samples.items():
        mat = np.asarray(part_matrix3(puppet, pose, name)[:2], dtype=np.float64)
        both: list[NDArray[np.int64]] = []
        for pts in (samp.line_xy, samp.flat_xy):
            if len(pts):
                keys_m = _pixel_keys(_warp_points(pts, mat), height, width)
                both.append(keys_m[keys_m >= 0])
        merged = np.concatenate(both) if both else np.zeros(0, np.int64)
        keys[name] = merged
        if len(merged):
            count[merged] += 1
    out: dict[str, float] = {}
    for name, merged in keys.items():
        out[name] = float(np.mean(count[merged] > 1)) if len(merged) else 0.0
    return out


def chain_feasible(
    puppet: Puppet,
    pose: Pose,
    chain: list[str],
    samples: dict[str, PartSamples],
    shape: tuple[int, int],
) -> tuple[bool, dict[str, float]]:
    """Whether a chain placement folds a member onto its own chain.

    Per member, the fraction of its in-bounds pixels another member of the
    same chain also covers; past ``OVERLAP_MAX`` the placement is infeasible,
    whatever its score. Pixels near a shared joint (the seam where two tubes
    legitimately meet) count for neither side: each in-chain joint
    contributes a disc around the pivot warped by both frames (they disagree
    under slack), sized from the puppet's own line width and the pose scale.
    Members with no pixels outside the seams count 0 (a ring is nearly all
    seam).
    """
    height, width = shape
    members = [m for m in chain if m in samples]
    mats = {name: np.asarray(part_matrix3(puppet, pose, name), dtype=np.float64) for name in members}
    keys: dict[str, NDArray[np.int64]] = {}
    for name in members:
        samp = samples[name]
        merged: list[NDArray[np.int64]] = []
        flat = np.asarray(mats[name][:2], dtype=np.float64)
        for pts in (samp.line_xy, samp.flat_xy):
            if len(pts):
                keyed = _pixel_keys(_warp_points(pts, flat), height, width)
                merged.append(keyed[keyed >= 0])
        keys[name] = np.concatenate(merged) if merged else np.zeros(0, np.int64)
    scale = float(pose.scale)
    radius = max(4.0, 3.0 * float(puppet.line_width_px) * (scale if scale > 0.0 else 1.0))
    seam = np.zeros(height * width, np.bool_)
    for name in members:
        parent = puppet.parts[name].parent
        if parent is None or parent not in mats:
            continue
        pivot = np.array(
            [float(puppet.parts[name].pivot[0]), float(puppet.parts[name].pivot[1]), 1.0],
            dtype=np.float64,
        )
        for mat in (mats[name], mats[parent]):
            cx, cy = float(mat[0] @ pivot), float(mat[1] @ pivot)
            x0 = max(0, int(math.floor(cx - radius)))
            x1 = min(width, int(math.ceil(cx + radius)) + 1)
            y0 = max(0, int(math.floor(cy - radius)))
            y1 = min(height, int(math.ceil(cy + radius)) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
            seam[(ys * width + xs)[inside]] = True
    count = np.zeros(height * width, np.int16)
    for name in members:
        if len(keys[name]):
            count[keys[name]] += 1
    fracs: dict[str, float] = {}
    for name in members:
        keyed = keys[name]
        use = keyed[~seam[keyed]] if len(keyed) else keyed
        fracs[name] = float(np.mean(count[use] > 1)) if len(use) else 0.0
    return all(f <= OVERLAP_MAX for f in fracs.values()), fracs


def needs_human(pose: Pose, puppet: Puppet | None = None) -> list[str]:
    """Parts with low confidence, low score or heavy overlap: the review asks a human.

    Hidden parts (no file, no pixels) are never listed.
    """
    out: list[str] = []
    for name in pose.scores:
        if puppet is not None and puppet.parts.get(name) is not None and puppet.parts[name].hidden:
            continue
        if float(pose.confidences.get(name, 0.0)) < CONF_MIN or float(pose.scores.get(name, 0.0)) < SCORE_MIN:
            out.append(name)
            continue
        if float(pose.overlap.get(name, 0.0)) > OVERLAP_MAX and float(pose.scores.get(name, 0.0)) < STACK_SCORE:
            out.append(name)
    if puppet is not None:
        leaves = chains(puppet)
        for leaf, note in (pose.chain_notes or {}).items():
            if note == NEVER_FIT_REASON and leaf in leaves:
                for member in leaves[leaf]:
                    part = puppet.parts.get(member)
                    if part is not None and part.hidden:
                        continue
                    out.append(member)
    return sorted(set(out))


def pose_total(pose: Pose) -> float:
    """Total fit score: the sum of the per-part scores."""
    return float(sum(float(v) for v in pose.scores.values()))


def fit_view_pose(
    loaded: LoadedView,
    drawing: NDArray[np.uint8],
    samples: dict[str, PartSamples] | None = None,
    features: DrawingFeatures | None = None,
    *,
    second_pass: bool = SECOND_PASS,
) -> Pose:
    """Full fit of one view: root, chains in tree order, second pass, z order."""
    feat = features if features is not None else drawing_features(np.asarray(drawing, dtype=np.uint8))
    own_samples = samples if samples is not None else extract_part_samples(loaded.puppet, loaded.images)
    dh, dw = feat.line.shape
    pose = Pose(view=loaded.name, canvas=[int(dw), int(dh)])
    pose = fit_root(loaded.puppet, pose, feat.rgb, own_samples, feat)
    order = chain_order(loaded.puppet)
    # The narrow-joint beam runs once, in the last pass: there every other
    # chain is placed, so the blocked/explained context it ranks in is final.
    for leaf in order:
        pose = fit_chain(loaded.puppet, pose, leaf, feat.rgb, own_samples, feat, beam=not second_pass)
    if second_pass:
        for leaf in order:
            pose = fit_chain(loaded.puppet, pose, leaf, feat.rgb, own_samples, feat, beam=True)
    pose = resolve_z(loaded.puppet, pose, own_samples, feat)
    tau = _tau_for(loaded.puppet, pose)
    for name in loaded.puppet.parts:
        if name not in pose.scores and name in own_samples:
            mat = np.asarray(part_matrix3(loaded.puppet, pose, name)[:2], dtype=np.float64)
            score = float(score_samples(own_samples[name], mat, feat, tau, None))
            pose.scores[name] = score
            if name == loaded.puppet.root:
                pose.confidences[name] = 1.0 if score >= SCORE_MIN else 0.0
            else:
                pose.confidences.setdefault(name, 0.0)
    pose.overlap = overlap_fraction(loaded.puppet, pose, own_samples, feat.line.shape)
    return pose


def _get_view(set_dir: Path, views: dict[str, LoadedView] | None, name: str) -> LoadedView:
    if views is not None and name in views:
        return views[name]
    return load_view(Path(set_dir), name)


def fit_pose(
    set_dir: Path,
    drawing: NDArray[np.uint8],
    views: dict[str, LoadedView] | None = None,
) -> tuple[Pose, dict[str, LoadedView], dict[str, float]]:
    """Fit every view of the set, keep the best total score.

    Views load one at a time (a built view is hundreds of MB); only the best
    two root fits go on to full chain fits. Returns the winning pose, the
    loaded winning views and each fully-fitted view's total score.
    """
    rgb = np.asarray(drawing, dtype=np.uint8)
    feat = drawing_features(rgb)
    if views is not None:
        view_names = list(views)
    else:
        pset = load_set(Path(set_dir))
        view_names = list(pset.views)
    if not view_names:
        raise ValueError("the set has no views to fit")
    # Cheap root-only round first: full chain fits run on the best two views.
    root_scores: dict[str, float] = {}
    samples_of: dict[str, dict[str, PartSamples]] = {}  # small; reused by the full fits below
    for name in view_names:
        loaded = _get_view(Path(set_dir), views, name)
        own = extract_part_samples(loaded.puppet, loaded.images)
        samples_of[name] = own
        dh, dw = feat.line.shape
        root_pose = fit_root(loaded.puppet, Pose(view=name, canvas=[int(dw), int(dh)]), rgb, own, feat)
        tau = _tau_for(loaded.puppet, root_pose)
        mat = np.asarray(part_matrix3(loaded.puppet, root_pose, loaded.puppet.root)[:2], dtype=np.float64)
        root_scores[name] = float(score_samples(own[loaded.puppet.root], mat, feat, tau, None))
        del loaded
        del own
    ranked = sorted(root_scores, key=lambda n: root_scores[n], reverse=True)[:2]
    best_pose: Pose | None = None
    best_total = -math.inf
    totals: dict[str, float] = {}
    kept: dict[str, LoadedView] = {}
    for name in ranked:
        loaded = _get_view(Path(set_dir), views, name)
        own = samples_of[name]
        pose = fit_view_pose(loaded, rgb, own, feat)
        total = pose_total(pose)
        totals[name] = total
        kept[name] = loaded
        if total > best_total:
            best_total = total
            best_pose = pose
    assert best_pose is not None
    return best_pose, kept, totals


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=px)


def _label_cell(img: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(img)
    font = _font(20)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, right - left + 12, bottom - top + 8], fill=(255, 255, 255))
    draw.text((6, 4), text, font=font, fill=(0, 0, 0))


def explained_alpha(
    puppet: Puppet,
    pose: Pose,
    images: dict[str, NDArray[np.uint8]],
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """Output pixels the posed puppet covers (render alpha, no resampling loss)."""
    height, width = shape
    rendered = render_pose(puppet, pose, images)
    if rendered.shape[1] != width or rendered.shape[0] != height:
        alpha = np.asarray(rendered[..., 3] > 127, dtype=np.uint8)
        grown = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        return np.asarray(grown, dtype=np.bool_)
    return np.asarray(rendered[..., 3] > 127, dtype=np.bool_)


def residual_mask(
    feat: DrawingFeatures,
    covered: NDArray[np.bool_],
    line_width: float,
) -> NDArray[np.bool_]:
    """Drawing foreground no part explains: where the model changed the design."""
    radius = max(2, int(round(float(line_width))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grown = np.asarray(cv2.dilate(covered.astype(np.uint8), kernel), dtype=np.uint8) > 0
    foreground = np.asarray(~feat.paper, dtype=np.bool_)
    return np.asarray(foreground & ~grown, dtype=np.bool_)


def fit_report_dict(
    loaded: LoadedView,
    pose: Pose,
    drawing_name: str,
    totals: dict[str, float],
    residual_px: int,
) -> dict[str, Any]:
    """JSON-able fit report: root, per part and per chain, residual, needs_human."""
    names = chains(loaded.puppet)
    part_of = {part: leaf for leaf, members in names.items() for part in members}
    chain_scores: dict[str, float] = {}
    for leaf, members in names.items():
        chain_scores[leaf] = float(sum(float(pose.scores.get(m, 0.0)) for m in members))
    return {
        "drawing": drawing_name,
        "view": pose.view,
        "canvas": list(pose.canvas),
        "root": {"tx": pose.tx, "ty": pose.ty, "rotation": pose.rotation, "scale": pose.scale},
        "parts": {
            name: {
                "angle": float(pose.angles.get(name, 0.0)),
                "score": float(pose.scores.get(name, 0.0)),
                "confidence": float(pose.confidences.get(name, 0.0)),
                "overlap": float(pose.overlap.get(name, 0.0)),
                "offset": [float(x) for x in (pose.offset.get(name) or [0.0, 0.0])],
                "z": int(pose.z_override.get(name, part.z)),
                "chain": part_of.get(name),
            }
            for name, part in loaded.puppet.parts.items()
        },
        "chains": {leaf: {"parts": list(members), "score": chain_scores[leaf]} for leaf, members in names.items()},
        "chain_notes": dict(pose.chain_notes or {}),
        "view_totals": {name: float(value) for name, value in totals.items()},
        "total": float(pose_total(pose)),
        "needs_human": needs_human(pose, loaded.puppet),
        "residual_px": int(residual_px),
    }


def write_fit_outputs(
    loaded: LoadedView,
    pose: Pose,
    drawing: NDArray[np.uint8],
    out_dir: Path,
    totals: dict[str, float],
    samples: dict[str, PartSamples] | None = None,
    features: DrawingFeatures | None = None,
    drawing_name: str = "",
) -> dict[str, Any]:
    """Write ``fit_report.json`` + ``fit_sheet.png``; return the report dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(drawing, dtype=np.uint8)
    feat = features if features is not None else drawing_features(rgb)
    height, width = feat.line.shape
    rendered = render_pose(loaded.puppet, pose, loaded.images)
    if rendered.shape[1] != width or rendered.shape[0] != height:
        fixed = np.zeros((height, width, 4), np.uint8)
        fixed[..., :3] = np.asarray(
            cv2.resize(rendered[..., :3], (width, height), interpolation=cv2.INTER_NEAREST), dtype=np.uint8
        )
        fixed[..., 3] = np.asarray(
            cv2.resize(rendered[..., 3], (width, height), interpolation=cv2.INTER_NEAREST), dtype=np.uint8
        )
        rendered = fixed
    own = samples if samples is not None else extract_part_samples(loaded.puppet, loaded.images)
    covered = explained_alpha(loaded.puppet, pose, loaded.images, (height, width))
    residual = residual_mask(feat, covered, float(loaded.puppet.line_width_px) * float(pose.scale))
    residual_px = int(residual.sum())
    report = fit_report_dict(loaded, pose, drawing_name or out.name, totals, residual_px)
    (out / "fit_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "pose.json").write_text(pose.model_dump_json(indent=1) + "\n", encoding="utf-8")
    Image.fromarray(np.asarray(rendered, dtype=np.uint8)).save(out / "render.png")
    overlay = np.asarray(rgb.copy(), dtype=np.uint8)
    for name, samp in own.items():
        if len(samp.line_xy) == 0:
            continue
        mat = np.asarray(part_matrix3(loaded.puppet, pose, name)[:2], dtype=np.float64)
        warped = _warp_points(samp.line_xy, mat)
        xi = np.round(warped[:, 0]).astype(np.int64)
        yi = np.round(warped[:, 1]).astype(np.int64)
        valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
        overlay[yi[valid], xi[valid]] = (255, 0, 0)
    residual_img = np.full((height, width, 3), 255, np.uint8)
    residual_img[residual] = rgb[residual]
    panels = [
        (Image.fromarray(rgb), "drawing"),
        (Image.fromarray(np.asarray(_on_white(rendered), dtype=np.uint8)), f"puppet {pose.view}"),
        (Image.fromarray(overlay), "overlay (puppet lines red)"),
        (Image.fromarray(residual_img), f"residual {residual_px} px"),
    ]
    sheet = _grid_sheet(panels, columns=2)
    sheet.save(out / "fit_sheet.png")
    return report


def _on_white(rgba: NDArray[np.uint8]) -> NDArray[np.uint8]:
    out = np.full((rgba.shape[0], rgba.shape[1], 3), 255, np.uint8)
    cover = np.asarray(rgba[..., 3] > 127, dtype=np.bool_)
    out[cover] = np.asarray(rgba[..., :3], dtype=np.uint8)[cover]
    return out


def _grid_sheet(panels: list[tuple[Image.Image, str]], columns: int = 2) -> Image.Image:
    labelled: list[Image.Image] = []
    for img, text in panels:
        cell = img.convert("RGB")
        _label_cell(cell, text)
        labelled.append(cell)
    col_w = max(c.width for c in labelled)
    row_h = max(c.height for c in labelled)
    rows = (len(labelled) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * col_w, rows * row_h), (200, 200, 200))
    for i, cell in enumerate(labelled):
        sheet.paste(cell, ((i % columns) * col_w, (i // columns) * row_h))
    return sheet


def _wrap_delta(delta: float) -> float:
    """Angle difference wrapped to [-180, 180]."""
    return float((float(delta) + 180.0) % 360.0 - 180.0)


def difference_locked(
    puppet: Puppet,
    pose_x: Pose,
    rgb_x: NDArray[np.uint8],
    rgb_y: NDArray[np.uint8],
    samples: dict[str, PartSamples],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Per-part X->Y difference map: which parts still match within tolerance.

    The region is the part's X-fit mask grown by the line width (a
    ``ctx``-style threshold from the puppet's own line width, never a bare
    constant). Locked parts keep ``pose_X`` angles exactly, so they render
    pixel-identical in every frame (no flicker by construction).
    """
    line_width = max(1.0, float(puppet.line_width_px))
    per_part: dict[str, dict[str, float]] = {}
    locked: list[str] = []
    y = np.asarray(rgb_y, dtype=np.int16)
    x = np.asarray(rgb_x, dtype=np.int16)
    hx, wx = rgb_x.shape[:2]
    hy, wy = rgb_y.shape[:2]
    same_size = (hx == hy) and (wx == wy)
    if not same_size:
        pil_y = Image.fromarray(np.asarray(rgb_y, dtype=np.uint8)).resize((wx, hx), Image.Resampling.LANCZOS)
        y = np.asarray(pil_y, dtype=np.int16)
    y_for_lines: NDArray[np.uint8] = (
        np.asarray(rgb_y, dtype=np.uint8) if same_size else np.asarray(pil_y, dtype=np.uint8)
    )
    line_y = np.asarray(line_mask(y_for_lines, LINE_LUMA), dtype=np.bool_)
    if not same_size:
        shrunk = np.asarray(Image.fromarray(line_y).resize((wx, hx), Image.Resampling.NEAREST))
        line_y = np.asarray(shrunk, dtype=np.bool_)
    dist_y = np.asarray(cv2.distanceTransform((~line_y).astype(np.uint8), cv2.DIST_L2, 3), dtype=np.float32)
    for name, samp in samples.items():
        if len(samp.flat_xy) == 0 and len(samp.line_xy) == 0:
            continue
        mat = np.asarray(part_matrix3(puppet, pose_x, name)[:2], dtype=np.float64)
        pts = (
            np.concatenate([samp.flat_xy, samp.line_xy])
            if len(samp.flat_xy) and len(samp.line_xy)
            else (samp.flat_xy if len(samp.flat_xy) else samp.line_xy)
        )
        warped = _warp_points(np.asarray(pts, dtype=np.float64), mat)
        xi = np.round(warped[:, 0]).astype(np.int64)
        yi = np.round(warped[:, 1]).astype(np.int64)
        valid = (xi >= 0) & (yi >= 0) & (xi < wx) & (yi < hx)
        if int(valid.sum()) < 20:
            continue
        xv, yv = xi[valid], yi[valid]
        cd = float(np.mean(np.abs(y[yv, xv] - x[yv, xv])))
        line_hit = 0.0
        if len(samp.line_xy):
            wl = _warp_points(samp.line_xy, mat)
            xl = np.round(wl[:, 0]).astype(np.int64)
            yl = np.round(wl[:, 1]).astype(np.int64)
            ok = (xl >= 0) & (yl >= 0) & (xl < wx) & (yl < hx)
            if int(ok.sum()):
                line_hit = float(np.mean(np.asarray(dist_y[yl[ok], xl[ok]], dtype=np.float64)))
        per_part[name] = {"colour_dist": cd, "line_shift": line_hit, "pixels": int(valid.sum())}
        if cd < 6.0 + line_width and line_hit < line_width:
            locked.append(name)
    return sorted(locked), per_part


def gate_split_needed(max_delta_deg: float) -> tuple[bool, int]:
    """The rig interval gate's verdict on one X->Y angle span.

    Uses the gate's own constants (imported from ``engine.rig.plan``, never
    copied): a segment longer than ``DEVIATION_THRESHOLD`` of a half-turn
    needs splitting (T-89: a large angle means a new intermediate pose); the
    jump bar and scale limit pass for equal rigid segments; at most
    ``MAX_EXTRA_KEYS`` intermediates are proposed.
    """
    _ = (JUMP_THRESHOLD, SCALE_LIMIT)
    span_bar = float(DEVIATION_THRESHOLD) * 180.0
    if max_delta_deg <= span_bar:
        return False, 0
    segments = int(math.ceil(max_delta_deg / span_bar))
    need = max(1, min(segments - 1, int(MAX_EXTRA_KEYS)))
    return True, need


def _pick_chain_winner(
    puppet: Puppet,
    chain: list[str],
    candidates: list[Pose],
    samples: dict[str, PartSamples],
    feat: DrawingFeatures,
    tau: float,
    anchor: dict[str, float],
    blocked: NDArray[np.bool_] | None,
) -> tuple[Pose, float]:
    """The candidate that best explains Y under the transition objective.

    Infeasible candidates (a member stacked on its own chain past overlap)
    are disqualified whatever their total: a fold never wins over a feasible
    chain. When every candidate folds, the first one (the X-anchored pull) is
    returned so the reach rule below still fires on it.
    """
    shape = feat.line.shape
    best = candidates[0]
    best_total = float("-inf")
    feasible_any = False
    for cand in candidates:
        feas, _why = chain_feasible(puppet, cand, chain, samples, shape)
        if not feas:
            continue
        total = chain_objective(puppet, cand, chain, samples, feat, tau, anchor, blocked)
        if not feasible_any or total > best_total:
            best, best_total, feasible_any = cand, total, True
    return best, best_total


@dataclass
class TransitionResult:
    pose_x: Pose
    pose_y: Pose
    locked: list[str]
    changed_chains: dict[str, dict[str, Any]]
    decision: str
    reason: str
    scores: dict[str, float]
    needs_human: list[str] = field(default_factory=list)


def transition_from_poses(
    loaded: LoadedView,
    pose_x: Pose,
    pose_y: Pose,
    rgb_x: NDArray[np.uint8],
    rgb_y: NDArray[np.uint8],
    samples: dict[str, PartSamples] | None = None,
) -> TransitionResult:
    """X->Y without refitting from scratch: lock the still parts, pull the rest.

    Changed chains start from their ``pose_X`` angles and are refined on Y's
    lines and colours ("the hand is at Z in X and at ZX in Y: pull X's hand to
    ZX"). The winner's score or feasibility decides first (``ret`` naming the chain
    and why when a changed chain cannot reach Y — never forced; low confidence only
    lists the part in ``needs_human``); the interval gate then
    decides ``split`` for large-but-clean motions, else ``rig`` (K-2
    interpolates X->Y).
    """
    own = samples if samples is not None else extract_part_samples(loaded.puppet, loaded.images)
    feat_y = drawing_features(np.asarray(rgb_y, dtype=np.uint8))
    arr_x = np.asarray(rgb_x, dtype=np.uint8)
    arr_y = np.asarray(rgb_y, dtype=np.uint8)
    locked, _metrics = difference_locked(loaded.puppet, pose_x, arr_x, arr_y, own)
    locked_set = set(locked)
    names = chains(loaded.puppet)
    # Rule 2(b): Lock chain members whose limb did not change in the drawing
    # between X and Y (provided both poses seat the chain above SCORE_MIN and
    # no ancestor in the chain made a large rotation).
    # Imported here: only transitions use the limb measures, so pose fits do not depend on that module.
    from karagoz.engine.puppet.limb_measure import detect_unchanged_chain_members

    unchanged_draw = detect_unchanged_chain_members(loaded.puppet, loaded.images, pose_x, arr_x, pose_y, arr_y)
    tau_y0 = _tau_for(loaded.puppet, pose_y)
    for _leaf, members in names.items():
        chain_seated = True
        for m in members:
            if loaded.puppet.parts[m].hidden or m not in own:
                continue
            sx = float(pose_x.scores[m]) if m in pose_x.scores else 1.0
            if m in pose_y.scores:
                sy = float(pose_y.scores[m])
            else:
                mat_m = np.asarray(part_matrix3(loaded.puppet, pose_y, m)[:2], dtype=np.float64)
                sy = float(score_samples(own[m], mat_m, feat_y, tau_y0, None))
            if sx < SCORE_MIN or sy < SCORE_MIN:
                chain_seated = False
                break
        if not chain_seated:
            continue
        for idx, member in enumerate(members):
            if member in locked_set or member not in unchanged_draw:
                continue
            ancestors_still = all(
                abs(_wrap_delta(float(pose_y.angles.get(anc, 0.0)) - float(pose_x.angles.get(anc, 0.0)))) <= 18.0
                for anc in members[:idx]
            )
            if ancestors_still:
                locked_set.add(member)
                locked.append(member)
    changed = sorted(leaf for leaf, members in names.items() if any(m not in locked_set for m in members))
    base = pose_y.model_copy(deep=True)
    for name in locked:
        if name in pose_x.angles:
            base.angles[name] = float(pose_x.angles[name])
        elif name in base.angles:
            del base.angles[name]
        if name in pose_x.offset:
            base.offset[name] = [float(v) for v in pose_x.offset[name]]
        elif name in base.offset:
            del base.offset[name]
    for leaf in changed:
        start = base.model_copy(deep=True)
        anchor = {}
        frozen = {}
        for member in names[leaf]:
            if member in pose_x.angles:
                start.angles[member] = float(pose_x.angles[member])
                anchor[member] = float(pose_x.angles[member])
            if member in locked_set and member in pose_x.angles:
                frozen[member] = float(pose_x.angles[member])
        pulled = fit_chain(
            loaded.puppet,
            start,
            leaf,
            np.asarray(rgb_y, dtype=np.uint8),
            own,
            feat_y,
            anchor=anchor,
            local_deg=90.0,
            frozen=frozen,
        )
        # The X-anchored pull cannot cross a large move (its window is local):
        # also try the chain's from-scratch fit on Y and keep whichever scores
        # better under the transition (X-anchor) objective. Locked members stay
        # frozen in both starts. The winner is picked below by feasible joint
        # total over three candidates (pull, fresh, pose_Y's own seating): a
        # folded candidate never wins over a feasible one.
        fresh = fit_chain(
            loaded.puppet,
            start,
            leaf,
            np.asarray(rgb_y, dtype=np.uint8),
            own,
            feat_y,
            frozen=frozen,
        )
        tau_pick = _tau_for(loaded.puppet, base)
        blocked_pick = context_blocked(loaded.puppet, start, names[leaf], own, feat_y.line.shape)
        # pose_Y's own seating of this chain, with the locked (frozen) members
        # kept at X: the cold from-scratch fit can hang in a folded basin the Y
        # fit already left, so the Y fit competes as a third candidate.
        yside = pose_y.model_copy(deep=True)
        yside.angles.update({m: float(a) for m, a in frozen.items()})
        yside.offset.update({m: list(pose_x.offset[m]) for m in frozen if m in pose_x.offset})
        best, _ = _pick_chain_winner(
            loaded.puppet,
            names[leaf],
            [pulled, fresh, yside],
            own,
            feat_y,
            tau_pick,
            anchor,
            blocked_pick,
        )
        for member in names[leaf]:
            base.angles[member] = float(best.angles.get(member, 0.0))
            if member in best.offset:
                base.offset[member] = [float(v) for v in best.offset[member]]
            base.scores[member] = float(best.scores.get(member, 0.0))
            base.confidences[member] = float(best.confidences.get(member, 0.0))
    tau = _tau_for(loaded.puppet, base)
    for name in locked:
        if name in own:
            score, gap = _part_confidence(loaded.puppet, base, name, own, feat_y, tau)
            base.scores[name] = float(score)
            base.confidences[name] = float(gap)
    deltas: dict[str, dict[str, Any]] = {}
    worst = 0.0
    for leaf in changed:
        parts = [m for m in names[leaf] if m in base.angles or m in pose_x.angles]
        dd = {m: _wrap_delta(float(base.angles.get(m, 0.0)) - float(pose_x.angles.get(m, 0.0))) for m in parts}
        top = max((abs(v) for v in dd.values()), default=0.0)
        worst = max(worst, float(top))
        deltas[leaf] = {"parts": parts, "delta": {m: float(v) for m, v in dd.items()}, "max_delta": float(top)}
    split, need = gate_split_needed(float(worst))
    scores = {name: float(base.scores.get(name, 0.0)) for name in loaded.puppet.parts}
    bar = float(DEVIATION_THRESHOLD) * 180.0
    # The winner's score or feasibility decides first: a chain that cannot
    # reach Y (a new shape, foreshortening, a turn, a fold) is ret no matter
    # how far it wandered trying. Low confidence never rets: it only lists
    # the part in needs_human. The gate's split only applies to
    # large-but-clean motions.
    bad: list[str] = []
    for leaf in changed:
        feas, _why = chain_feasible(loaded.puppet, base, names[leaf], own, feat_y.line.shape)
        if not feas:
            bad.append(f"{leaf}/folded")
        for member in names[leaf]:
            part = loaded.puppet.parts[member]
            if part.hidden:
                continue
            if float(base.scores.get(member, 0.0)) < SCORE_MIN:
                bad.append(f"{leaf}/{member}")
    if bad:
        return TransitionResult(
            pose_x=pose_x,
            pose_y=base,
            locked=locked,
            changed_chains=deltas,
            decision="ret",
            reason=(
                f"chain(s) cannot reach Y ({'; '.join(sorted(set(bad)))}): "
                "needs a substitution drawing (K-3) or the drawn in-between path"
            ),
            scores=scores,
            needs_human=needs_human(base, loaded.puppet),
        )
    if split:
        where = [f"{(k + 1) / (need + 1):.2f}" for k in range(need)]
        return TransitionResult(
            pose_x=pose_x,
            pose_y=base,
            locked=locked,
            changed_chains=deltas,
            decision="split",
            reason=(
                f"interval gate: {worst:.1f}deg over {bar:.1f}deg per segment, "
                f"{need} intermediate pose(s) at {', '.join(where)}"
            ),
            scores=scores,
            needs_human=needs_human(base, loaded.puppet),
        )
    return TransitionResult(
        pose_x=pose_x,
        pose_y=base,
        locked=locked,
        changed_chains=deltas,
        decision="rig",
        reason=f"every changed chain reaches Y (worst {worst:.1f}deg); K-2 interpolates X->Y",
        scores=scores,
        needs_human=needs_human(base, loaded.puppet),
    )


def transition_dict(result: TransitionResult) -> dict[str, Any]:
    """JSON-able ``transition.json``: poses, locks, deltas, decision, scores."""
    return {
        "pose_X": result.pose_x.model_dump(mode="json"),
        "pose_Y": result.pose_y.model_dump(mode="json"),
        "locked": list(result.locked),
        "changed_chains": result.changed_chains,
        "decision": result.decision,
        "reason": result.reason,
        "scores": {k: float(v) for k, v in result.scores.items()},
        "confidences": {k: float(v) for k, v in result.pose_y.confidences.items()},
        "needs_human": list(result.needs_human),
    }


def _thumb(img: Image.Image, height: int = 720) -> Image.Image:
    if img.height <= height:
        return img
    scale = height / img.height
    return img.resize((max(1, round(img.width * scale)), height), Image.Resampling.LANCZOS)


def write_transition_outputs(
    loaded: LoadedView,
    result: TransitionResult,
    rgb_x: NDArray[np.uint8],
    rgb_y: NDArray[np.uint8],
    out_dir: Path,
    name_x: str = "X",
    name_y: str = "Y",
) -> dict[str, Any]:
    """Write ``transition.json`` + ``transition_sheet.png``; return the dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = transition_dict(result)
    (out / "transition.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (out / "pose_X.json").write_text(result.pose_x.model_dump_json(indent=1) + "\n", encoding="utf-8")
    (out / "pose_Y.json").write_text(result.pose_y.model_dump_json(indent=1) + "\n", encoding="utf-8")
    rx = np.asarray(rgb_x, dtype=np.uint8)
    ry = np.asarray(rgb_y, dtype=np.uint8)
    if rx.shape != ry.shape:
        small = Image.fromarray(ry).resize((rx.shape[1], rx.shape[0]), Image.Resampling.LANCZOS)
        ry = np.asarray(small, dtype=np.uint8)
    px = _on_white(render_pose(loaded.puppet, result.pose_x, loaded.images))
    py = _on_white(render_pose(loaded.puppet, result.pose_y, loaded.images))
    if px.shape != rx.shape:
        px = np.asarray(
            Image.fromarray(px).resize((rx.shape[1], rx.shape[0]), Image.Resampling.LANCZOS), dtype=np.uint8
        )
        py = np.asarray(
            Image.fromarray(py).resize((rx.shape[1], rx.shape[0]), Image.Resampling.LANCZOS), dtype=np.uint8
        )
    diff = np.abs(rx.astype(np.int16) - ry.astype(np.int16)).mean(axis=2)
    changed_map = np.full(rx.shape, 255, np.uint8)
    region = diff > (6.0 + float(loaded.puppet.line_width_px))
    changed_map[region] = (255, 120, 120)
    overlay = np.asarray(ry.copy(), dtype=np.uint8)
    own = extract_part_samples(loaded.puppet, loaded.images)
    height, width = ry.shape[:2]
    for _leaf, info in result.changed_chains.items():
        for member in info.get("parts", []):
            samp = own.get(member)
            if samp is None or len(samp.line_xy) == 0:
                continue
            mat = np.asarray(part_matrix3(loaded.puppet, result.pose_y, member)[:2], dtype=np.float64)
            warped = _warp_points(samp.line_xy, mat)
            xi = np.round(warped[:, 0]).astype(np.int64)
            yi = np.round(warped[:, 1]).astype(np.int64)
            valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
            overlay[yi[valid], xi[valid]] = (255, 0, 0)
    panels = [
        (_thumb(Image.fromarray(rx)), name_x),
        (_thumb(Image.fromarray(ry)), name_y),
        (_thumb(Image.fromarray(px)), "puppet at X"),
        (_thumb(Image.fromarray(py)), "puppet at Y"),
        (_thumb(Image.fromarray(np.asarray(changed_map, dtype=np.uint8))), "X->Y changed regions"),
        (_thumb(Image.fromarray(overlay)), f"Y overlay ({result.decision})"),
    ]
    sheet = _grid_sheet(panels, columns=3)
    sheet.save(out / "transition_sheet.png")
    return doc


def _fit_view_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool job: fit one view to one drawing; plain data in and out."""
    loaded = load_view(Path(str(payload["set_dir"])), str(payload["view"]))
    rgb = load_drawing(Path(str(payload["image"])))
    pose = fit_view_pose(loaded, rgb)
    return {
        "image": str(payload["image"]),
        "view": str(payload["view"]),
        "pose": pose.model_dump(mode="json"),
        "total": pose_total(pose),
    }


def _transition_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool job: one X->Y interval from already-fitted poses."""
    loaded = load_view(Path(str(payload["set_dir"])), str(payload["view"]))
    rgb_x = load_drawing(Path(str(payload["x"])))
    rgb_y = load_drawing(Path(str(payload["y"])))
    pose_x = Pose.model_validate(payload["pose_x"])
    pose_y = Pose.model_validate(payload["pose_y"])
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)
    out = Path(str(payload["out_dir"]))
    x_name = Path(str(payload["x"])).name
    y_name = Path(str(payload["y"])).name
    doc = write_transition_outputs(loaded, result, rgb_x, rgb_y, out, x_name, y_name)
    doc["from"] = str(payload["x"])
    doc["to"] = str(payload["y"])
    return doc


def carry_locked(pose_x: dict[str, Any], prev_doc: dict[str, Any]) -> dict[str, Any]:
    """The next interval's X pose with the previous interval's locked members held.

    ``pose_x`` is a pose dump (the shared key's own fit), ``prev_doc`` the
    previous interval's transition document (``locked`` and ``pose_Y``). Every
    locked member takes ``pose_Y``'s angle and offset, so a limb held still
    across X->Y stays at the same value when Y->Z starts.
    """
    out = dict(pose_x)
    angles = dict(out.get("angles") or {})
    offset = dict(out.get("offset") or {})
    held = prev_doc.get("pose_Y") or {}
    held_angles = dict(held.get("angles") or {})
    held_offset = dict(held.get("offset") or {})
    for name in prev_doc.get("locked") or []:
        if name in held_angles:
            angles[name] = float(held_angles[name])
        else:
            angles.pop(name, None)
        if name in held_offset:
            offset[name] = [float(v) for v in held_offset[name]]
        else:
            offset.pop(name, None)
    out["angles"] = angles
    out["offset"] = offset
    return out


def default_workers() -> int:
    """Half the CPUs, at least one, at most four (one view is hundreds of MB)."""
    import os

    return max(1, min(4, max(1, (os.cpu_count() or 2) // 2)))


def fit_chain_poses(
    set_dir: Path,
    images: list[Path],
    out_dir: Path,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    """Fit every pose once, then every consecutive interval, in parallel.

    A pose shared by two intervals is fitted once and reused; one pose's view
    fits run as parallel jobs. Returns the chain summary dict and writes per
    pose fit outputs plus per interval transition outputs under ``out_dir``.
    """
    base = Path(set_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pset = load_set(base)
    view_names = list(pset.views)
    jobs: list[dict[str, Any]] = []
    for img in images:
        for view in view_names:
            jobs.append({"set_dir": str(base), "view": view, "image": str(img)})
    won: dict[str, dict[str, Any]] = {}
    n_workers = workers or default_workers()
    with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
        for got in pool.map(_fit_view_job, jobs):
            key = str(got["image"])
            if key not in won or float(got["total"]) > float(won[key]["total"]):
                won[key] = got
    poses: dict[str, Pose] = {}
    pose_views: dict[str, str] = {}
    for img in images:
        key = str(img)
        poses[key] = Pose.model_validate(won[key]["pose"])
        pose_views[key] = str(won[key]["view"])
    for img in images:
        loaded = load_view(base, pose_views[str(img)])
        rgb = load_drawing(Path(str(img)))
        pose = poses[str(img)]
        own = extract_part_samples(loaded.puppet, loaded.images)
        feat = drawing_features(rgb)
        totals = {pose_views[str(img)]: pose_total(pose)}
        write_fit_outputs(loaded, pose, rgb, out / Path(str(img)).stem, totals, own, feat, str(img))
        del loaded
        del own
    intervals: list[dict[str, Any]] = []
    for left, right in zip(images, images[1:], strict=False):
        key = f"{Path(str(left)).stem}__{Path(str(right)).stem}"
        view = pose_views[str(left)] if pose_views[str(left)] == pose_views[str(right)] else pose_views[str(right)]
        intervals.append(
            {
                "set_dir": str(base),
                "view": view,
                "x": str(left),
                "y": str(right),
                "pose_x": poses[str(left)].model_dump(mode="json"),
                "pose_y": poses[str(right)].model_dump(mode="json"),
                "out_dir": str(out / key),
            }
        )
    # Intervals run in order: a member an interval locks keeps X's value in its
    # Y pose, so the next interval must start from that held value, not from
    # Y's own fit (otherwise the held limb jumps at the interval boundary).
    results: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for job in intervals:
        if prev is not None and prev.get("view") == job["view"]:
            job["pose_x"] = carry_locked(job["pose_x"], prev)
        doc = _transition_job(job)
        doc["view"] = job["view"]
        results.append(doc)
        prev = doc
    summary = {
        "poses": {Path(k).name: {"view": pose_views[k], "pose": poses[k].model_dump(mode="json")} for k in poses},
        "intervals": [
            {"from": doc.get("from"), "to": doc.get("to"), "decision": doc.get("decision"), "reason": doc.get("reason")}
            for doc in results
        ],
    }
    (out / "chain_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def _coarse_angles(hint: float | None, window_deg: float, limits: tuple[float, float] | None = None) -> list[float]:
    if hint is None:
        start = -180.0
        cands = [start + i * COARSE_STEP for i in range(int(360.0 / COARSE_STEP))]
        return _clip_cands(cands, limits)
    half = float(window_deg) / 2.0
    lo = float(hint) - half
    n = max(1, int(round((2.0 * half) / COARSE_STEP)) + 1)
    return _clip_cands([lo + i * COARSE_STEP for i in range(n)], limits)


def _ang_dist(a: float, b: float) -> float:
    """Circular distance of two angles in degrees."""
    d = abs(float(a) - float(b)) % 360.0
    return float(min(d, 360.0 - d))


def _second_peak_gap(scores: dict[float, float], best: float) -> float:
    if not scores:
        return 0.0
    best_s = scores[best]
    runner = None
    for angle, value in scores.items():
        if _ang_dist(float(angle), float(best)) < PEAK_SEP_DEG:
            continue
        if runner is None or value > runner:
            runner = value
    if runner is None:
        return 0.0
    return float(best_s - runner)
