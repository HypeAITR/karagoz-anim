"""Head as a rigid layer: per-key head image, headless ground, rigid head motion.

Ported from the experiment's ``deneme/r2/plan/kafa.py``: the head mask (head + hair + face +
mustache with its own contour), the neck point, the headless ground (nearest fill plus the drawn
neck edge), the rigid A->B measurement, the per-frame rigid matrix and the premultiplied head
layer drawn once per frame. Same maths; English identifiers, each public function noting its
experiment name. Thresholds are module-level ``UPPER_CASE`` with the experiment's base values;
lengths and areas scale with the interval's line width through ``interval.IntervalContext``.
Returned dicts keep the experiment's keys so outputs compare directly.

Everything flows through the interval context plus explicit arguments: no environment variables,
no module-level paths, no printing. Masks come from the SAM cache only. Chains (a head layer
inherited from an earlier interval) are out of scope: the chain root is always None here.

The experiment's ``kafa.py`` imports ``kanonik`` lazily inside the functions that need the
canonical helpers; this module does the same (``canonical`` imports this module at top level),
so the module graph stays one-way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig import masks, measure
from karagoz.engine.rig.interval import IntervalContext

CONCEPTS = ("head", "face", "hair", "mustache")  # experiment: KAVRAMLAR
ANGLE_RANGE = 12.0  # experiment: ACI_ARALIK; degrees; the head tilts, never turns
ANGLE_STEP = 0.25  # experiment: ACI_ADIM
SEARCH_RADIUS = 90  # experiment: ARAMA_R base
LINE_COLOR = (22, 22, 25)  # experiment: CIZGI_RENK
THICKNESS = 6  # experiment: KALINLIK base
STRAY_RADIUS = 90  # experiment: KIL_R base
STRAY_AREA = 600  # experiment: KIL_ALAN base
IDENTITY_TOLERANCE = 1e-6


# experiment: _maskeler
def _masks(path: Path | str, shape: tuple[int, int], cache_dir: Path | str) -> dict[str, list[NDArray[np.bool_]]]:
    return {k.replace(" ", "_"): v for k, v in masks.load_masks(masks.sam_cache_key(path), shape, cache_dir).items()}


# experiment: _en_buyuk
def _largest(found: list[NDArray[np.bool_]], shape: tuple[int, int]) -> NDArray[np.bool_]:
    return max(found, key=lambda m: int(m.sum())) if found else np.zeros(shape, bool)


# experiment: kafa_maskesi
def head_mask(
    ctx: IntervalContext, rgb: NDArray[np.uint8], found: dict[str, list[NDArray[np.bool_]]]
) -> NDArray[np.bool_]:
    """Head + hair + face + mustache, holes closed, with its own line. The neck line's short start falls to opening."""
    from karagoz.engine.rig import canonical

    shape = rgb.shape[:2]
    seen = np.zeros(shape, bool)
    for concept in CONCEPTS:
        seen |= _largest(found.get(concept, []), shape)
    grown = canonical.expand_contour(ctx, masks.fill_holes(seen), rgb)
    # Every drawing near the head but unconnected to the body is the head's (left on the ground it would hang in
    # the air once the head is gone, H6)
    paper = canonical.background(rgb, canonical.PAPER_LIMIT)
    near = masks.dilate(grown, ctx.px(STRAY_RADIUS))
    parts = masks.components(~paper & ~grown, min_area=1)
    for part in parts[1:]:
        if (part & near).any() and part.sum() < ctx.area(STRAY_AREA):
            grown = grown | masks.dilate(part, ctx.px(2))
    return grown


# experiment: boyun_noktasi
def neck_point(
    ctx: IntervalContext, rgb: NDArray[np.uint8], head: NDArray[np.bool_], found: dict[str, list[NDArray[np.bool_]]]
) -> NDArray[np.float64]:
    """Middle of the neck mask's contact band with the head: where the head joins the body."""
    neck = _largest(found.get("neck", []), rgb.shape[:2])
    contact = neck & masks.dilate(head, ctx.px(6))
    if not contact.any():
        contact = neck
    ys, xs = np.nonzero(contact)
    if not ys.size:
        return np.array([np.nan, np.nan])
    return np.array([float(xs.mean()), float(ys.mean())])


# experiment: kafasiz_zemin
def headless_ground(ctx: IntervalContext, rgb: NDArray[np.uint8], head: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Under the head: the nearest lineless fill; where skin meets paper inside the fill an edge is drawn."""
    from karagoz.engine.rig import canonical

    hidden = masks.dilate(head, ctx.px(2))
    ink = masks.dilate(measure.ink_mask(rgb), ctx.px(2))
    _, iy, ix = measure.nearest_indices(~hidden & ~ink)
    ground = rgb.copy()
    ground[hidden] = rgb[iy, ix][hidden]
    # Inner edge: between near-paper and not-near-paper
    paper = canonical.background(rgb)
    # Paper grain counts as paper too: off-white under the background threshold must not draw a fake edge
    colour = rgb[iy, ix]
    paper_id = (
        paper[iy, ix]
        | (colour.min(axis=2) >= 215)
        | ((colour.max(axis=2).astype(np.int32) - colour.min(axis=2) < 20) & (colour.min(axis=2) >= 170))
    )
    # Inside the fill only the body-connected part is body (the neck); the rest is paper, never carried as the
    # nearest colour ray or hair remnant (H6)
    body_d = hidden & ~paper_id
    connected = masks.components(~canonical.background(rgb, canonical.PAPER_LIMIT) & ~hidden, min_area=1)[0]
    for part in masks.components(body_d, min_area=1):
        if not (masks.dilate(part, ctx.px(3)) & connected).any():
            body_d = body_d & ~part
    paper_d = hidden & ~body_d
    paper_colour = np.median(
        rgb[canonical.background(rgb, canonical.PAPER_LIMIT) & masks.dilate(hidden, ctx.px(STRAY_RADIUS))], axis=0
    )
    ground[paper_d] = paper_colour.astype(np.uint8)
    edge = hidden & masks.dilate(paper_d, ctx.px(THICKNESS) // 2) & masks.dilate(body_d, ctx.px(THICKNESS) // 2)
    ground[edge] = LINE_COLOR
    return ground


# experiment: _katman
def _cleaned(
    ctx: IntervalContext, rgb: NDArray[np.uint8], head: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    from karagoz.engine.rig import canonical

    return canonical.clean_edge(ctx, rgb, head)


# experiment: rigid_olc
def measure_rigid(
    ctx: IntervalContext,
    a: NDArray[np.uint8],
    mask_a: NDArray[np.bool_],
    b: NDArray[np.uint8],
    mask_b: NDArray[np.bool_],
    neck_a: NDArray[np.float64],
) -> dict[str, Any]:
    """Turn A's head around the neck point and search it in B; the best angle and shift. Pure image matching."""
    grey_a, grey_b = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    best: dict[str, Any] = {"skor": -2.0}
    h, w = grey_a.shape
    search = ctx.px(SEARCH_RADIUS)
    for d in np.arange(-ANGLE_RANGE, ANGLE_RANGE + 1e-6, ANGLE_STEP):
        rot = cv2.getRotationMatrix2D((float(neck_a[0]), float(neck_a[1])), -float(d), 1.0)
        turned = cv2.warpAffine(grey_a, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        moved = cv2.warpAffine(mask_a.astype(np.uint8), rot, (w, h), flags=cv2.INTER_NEAREST)
        ys2, xs2 = np.nonzero(moved)
        xa, xb, ya, yb = int(xs2.min()), int(xs2.max()) + 1, int(ys2.min()), int(ys2.max()) + 1
        patch = turned[ya:yb, xa:xb].astype(np.float32)
        inside = moved[ya:yb, xa:xb].astype(np.float32)
        ax0, ay0 = max(xa - search, 0), max(ya - search, 0)
        area = grey_b[ay0 : min(yb + search, h), ax0 : min(xb + search, w)].astype(np.float32)
        score = np.nan_to_num(
            cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED, mask=inside), posinf=-1.0, neginf=-1.0
        )
        _, mx, _, loc = cv2.minMaxLoc(score)
        if mx > best["skor"]:
            best = {
                "skor": float(mx),
                "derece": float(d),
                "oteleme": [float(ax0 + loc[0] - xa), float(ay0 + loc[1] - ya)],
            }
    return {k: (round(v, 3) if isinstance(v, float) else [round(c, 2) for c in v]) for k, v in best.items()}


# experiment: matris
def matrix(neck: NDArray[np.float64], shift: NDArray[np.float64], degrees: float) -> NDArray[np.float64]:
    """2x3: turn ``degrees`` around the neck point, then shift (image coordinates, + is clockwise)."""
    r = np.radians(degrees)
    c, s = np.cos(r), np.sin(r)
    rot = np.array([[c, -s], [s, c]])
    return np.column_stack([rot, neck + shift - rot @ neck])


# experiment: kare_matrisi
def frame_matrix(data: dict[str, Any], t: float, from_b: bool) -> NDArray[np.float64]:
    """The frame's rigid transform: A's head by t; B's head back by (1-t). t=0 is A, t=1 is full B identity."""
    neck_a = np.array(data["boyun_a"], float)
    shift = np.array(data["oteleme"], float)
    if not from_b:
        return matrix(neck_a, t * shift, t * float(data["derece"]))
    return matrix(neck_a + shift, (t - 1.0) * shift, (t - 1.0) * float(data["derece"]))


# experiment: ara_matrisi
def mid_matrix(data: dict[str, Any], t: float, t_key: float) -> NDArray[np.float64]:
    """A middle key's head is A's rigid head at t_key: the frame transform is A(t) composed with A(t_key)^-1."""
    if abs(t - t_key) < 1e-9:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    bottom = np.array([[0.0, 0.0, 1.0]])
    return (
        np.vstack([frame_matrix(data, t, False), bottom])
        @ np.linalg.inv(np.vstack([frame_matrix(data, t_key, False), bottom]))
    )[:2]


# experiment: kur
def setup(
    ctx: IntervalContext,
    a: NDArray[np.uint8],
    b: NDArray[np.uint8],
    path_a: Path | str,
    path_b: Path | str,
    key_b: str,
    cache_dir: Path | str,
    canonical_dir: Path | str,
    head_path: Path | str,
    inherited_neck: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Write the head layers and the rigid measurement; return the headless grounds and masks.

    ``inherited_neck`` is the neck point inherited from the earlier interval (the experiment's
    joint-inheritance branch); None measures it here.
    """
    out = Path(canonical_dir)
    out.mkdir(parents=True, exist_ok=True)
    shape = a.shape[:2]
    masks_a, masks_b = _masks(path_a, shape, cache_dir), _masks(path_b, shape, cache_dir)
    head_a, head_b = head_mask(ctx, a, masks_a), head_mask(ctx, b, masks_b)
    neck_a, neck_b = neck_point(ctx, a, head_a, masks_a), neck_point(ctx, b, head_b, masks_b)
    if inherited_neck is not None:
        neck_a = np.array(inherited_neck, float)
    rigid = measure_rigid(ctx, a, head_a, b, head_b, neck_a)
    # When a cup or hand covers the neck SAM returns empty: the neck follows the head's measured motion
    if not np.isfinite(neck_b).all():
        neck_b = neck_a + np.array(rigid["oteleme"], float)
    for name, rgb, head in ((ctx.key_a, a, head_a), (key_b, b, head_b)):
        cleaned, alpha = _cleaned(ctx, rgb, head)
        Image.fromarray(cleaned).save(out / f"{name}_kafa.png")
        Image.fromarray((alpha * head * 255 + 0.5).astype(np.uint8)).save(out / f"{name}_kafa_alfa.png")
    data = {
        "kaynak_ozetleri": ctx.source_digests(),
        "boyun_a": [round(float(c), 1) for c in neck_a],
        "boyun_b_olcum": [round(float(c), 1) for c in neck_b],
        "derece": rigid["derece"],
        "oteleme": rigid["oteleme"],
        "skor": rigid["skor"],
        "kafa_px": {"a": int(head_a.sum()), "b": int(head_b.sum())},
    }
    target = Path(head_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return {
        "veri": data,
        "kafasiz_a": headless_ground(ctx, a, head_a),
        "kafasiz_b": headless_ground(ctx, b, head_b),
        "kafa_a": head_a,
        "kafa_b": head_b,
    }


# experiment: zincir_matrisi
def chain_matrix(
    data: dict[str, Any], t: float, root: tuple[Path, str, NDArray[np.float64]] | None
) -> NDArray[np.float64]:
    """The frame's head transform: from the chain root when chained, else from the interval's A."""
    m = frame_matrix(data, t, False)
    return m if root is None else (np.vstack([m, [0.0, 0.0, 1.0]]) @ root[2])[:2]


class Layer:
    """One source's head layer: premultiplied RGBA, rigidly resampled once per frame."""

    # experiment: Katman
    def __init__(
        self,
        canonical_dir: Path | str,
        name: str,
        data: dict[str, Any],
        root: tuple[Path, str, NDArray[np.float64]] | None = None,
    ):
        folder, key = (root[0], root[1]) if root else (Path(canonical_dir), name)
        rgb = np.array(Image.open(folder / f"{key}_kafa.png").convert("RGB")).astype(np.float32)
        alpha = np.array(Image.open(folder / f"{key}_kafa_alfa.png").convert("L")).astype(np.float32) / 255.0
        self.pm = np.dstack([rgb * alpha[..., None], alpha])
        self.data = data

    # experiment: ciz
    def draw(self, ground: NDArray[np.uint8], m: NDArray[np.float64]) -> NDArray[np.uint8]:
        """The rigid head over the ground. At identity there is no resampling: the drawing arrives bit by bit."""
        h, w = ground.shape[:2]
        if np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=IDENTITY_TOLERANCE):
            bent = self.pm
        else:
            bent = cv2.warpAffine(
                self.pm, m.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
        return (bent[..., :3] + (1.0 - bent[..., 3:]) * ground + 0.5).astype(np.uint8)
