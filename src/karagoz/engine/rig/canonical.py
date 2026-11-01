"""Canonical sources: same pixels outside the arm root in every source, gap fill.

Ported from the experiment's ``deneme/r2/plan/kanonik.py``: every function reachable from its
setup, with explicit inputs and an output directory instead of module paths and environment
variables. Same maths and same numeric results; English identifiers, each public function noting
its experiment name. Thresholds are module-level ``UPPER_CASE`` with the experiment's base
values; lengths and areas scale with the interval's line width through
``interval.IntervalContext``. Returned dicts keep the experiment's keys so outputs compare directly.

This module also hosts the canonical-source loading cluster the experiment keeps in
``cok_kaynak.py`` (``load_source`` and its helpers): the ground-ownership step needs it, and the
product keeps the dependency one-way (this module must not import ``frames``).

Everything flows through the interval context plus explicit arguments: no environment variables,
no module-level paths, no printing. SAM is never called: masks come from the cache only. The
single image-model call (armless fill under the arm) arrives as an injected ``fill.Fill`` and
hits its file cache (``fill.armless_fill``); without a cache hit and without a client the setup
raises instead of reaching the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast, overload

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig import bend, changed, head, masks, measure
from karagoz.engine.rig import fill as fill_mod
from karagoz.engine.rig.interval import IntervalContext

GAP_MARGIN = 15  # experiment: BOSLUK_PAY base
CONTOUR_MARGIN = 7  # experiment: KONTUR_PAY base; line weight (R2 drawings ~6-8 px)
OWNER_MARGIN = 3  # experiment: SAHIP_PAY base
HEAD_OWNER = 4  # experiment: SAHIP_KAFA base
CRUMB_AREA = 400  # experiment: KIRINTI_ALAN base
OWNER_STRAND = 400  # experiment: SAHIP_KIL base
SHADOW_SPILL = 60  # experiment: GOLGE_TASMA base
STRAND_SPECK = 100  # experiment: KIL_BENEK base
SHADOW_THIN = 4  # experiment: GOLGE_INCE base
SHADOW_NEAR = 6  # experiment: GOLGE_KOMSU base
CASING_CLOSE = 61  # experiment: ZARF_KAPANIS base
CONTOUR_LONG = 14  # experiment: KONTUR_UZUN base
ROOT_SIDE = 40  # experiment: KOK_YAN base
PATCH_PAPER = 50  # experiment: YAMA_KAGIT base
SKIN_NEAR = 70  # experiment: TEN_YAKIN base
SKIN_MIN = 30  # experiment: TEN_MIN base
HALO_LIMIT = 249  # paper walked from the paper at this limit is an edge halo; the top stays 240-248
PAPER_LIMIT = 251  # light garment is 240-248: paper parts from it
CUP_CONCEPTS = ("glass", "cup", "tea glass", "tea_glass")  # experiment: BARDAK_KAVRAMLARI
LINE_COLOR = (22, 22, 25)
WING_MARGIN = 8.0  # experiment (cok_kaynak): KANAT_PAY base
WING_THIN = 8  # experiment (cok_kaynak): KANAT_INCE base
WING_ROUND = 14  # experiment (cok_kaynak): KANAT_YUVARLA base
CASING_FACTOR = 1.3  # experiment (cok_kaynak): KAPSUL_PAY
CUT_MIN_LENGTH = 40  # experiment (cok_kaynak): KESIK_EN_KISA base
LIGHT_MIN = 205  # experiment (cok_kaynak): ACIK_MIN; smallest channel
PROMPT_ARMLESS_A = (
    "Redraw this exact image of this exact character with ONE change: the character's near arm "
    "(on the viewer's left side, from the shoulder down, including the hand) and the tea glass it holds "
    "are gone, as if that arm were hidden behind the body. In the area they covered, draw what is behind "
    "them: the white tank top's side continuing down with its black outline, and the dark gray sweatpants "
    "with the waistband and hip continuing, with their black outlines. The tank top keeps its strap and "
    "armhole edge with a clean black outline. Everything else stays pixel-for-pixel identical: head, face, "
    "the far arm on the viewer's right, legs, slippers, same position, same size, same line weight. Flat 2D "
    "cel-animation style, solid flat colors, bold clean black outlines, pure solid white background, full "
    "body head to slippers. No colored guide marks. No cast shadow from the removed arm or glass: the tank "
    "top and sweatpants under them are clean, with no gray smudge."
)  # experiment: ISTEM_KOLSUZ
PROMPT_ARMLESS_B = (
    "Redraw this exact image of this exact character with ONE change: the character's near arm "
    "(from the shoulder on the viewer's left, crossing in front of the chest, including the hand) and the "
    "tea glass it holds are gone, as if that arm were hidden behind the body. In the area they covered, "
    "draw what is behind them: the white tank top's chest and belly continuing, its armhole edge and strap "
    "on the viewer's left with a clean black outline, and the tank top's side edge continuing down with its "
    "black outline. Everything else stays pixel-for-pixel identical: head, face, the far arm on the viewer's "
    "right, legs, slippers, same position, same size, same line weight. Flat 2D cel-animation style, solid "
    "flat colors, bold clean black outlines, pure solid white background, full body head to slippers. No "
    "colored guide marks. No cast shadow from the removed arm or glass: the tank top under them is clean, "
    "with no gray smudge."
)  # experiment: ISTEM_KOLSUZ_B


def _write_mask(m: NDArray[np.bool_], path: Path) -> None:
    Image.fromarray((m.astype(np.uint8)) * 255).save(path)


def _read_mask(path: Path) -> NDArray[np.bool_]:
    return np.array(Image.open(path).convert("L")) > 127


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    return np.array(Image.open(path).convert("RGB"))


def _luma(rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Perceived brightness. experiment: olcu.luma_of."""
    return rgb.astype(np.float32) @ np.array(measure.LUMA_WEIGHTS, np.float32)


def _grey(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


# experiment: arka_plan
def background(rgb: NDArray[np.uint8], limit: int = 236) -> NDArray[np.bool_]:
    """Background reached by walking from the image edge through light pixels."""
    light = (rgb.min(axis=2) >= limit).astype(np.uint8)
    n, lab = cv2.connectedComponents(light, connectivity=4)
    if n <= 1:
        return np.zeros(rgb.shape[:2], bool)
    edge = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0])


# experiment: kol_ten_kapsa
def enclose_skin(ctx: IntervalContext, arm: NDArray[np.bool_], rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Skin the mask missed, joined to the arm without a line and ending beside it, is the arm's: left on the
    ground it would hang in the air once the arm is gone (R13, thumb root).

    Neck and chest skin part with a contour, a separate component; a component carrying past the arm's side
    (neighbouring skin) is not taken.
    """
    red = rgb.astype(np.int32)
    skin = (red[..., 0] - red[..., 2] > 30) & (red[..., 0] > 120) & ~masks.dilate(measure.ink_mask(rgb), ctx.px(1))
    near = masks.dilate(arm, ctx.px(SKIN_NEAR))
    result = arm.copy()
    for part in masks.components(skin & ~arm, min_area=ctx.area(SKIN_MIN)):
        if (masks.dilate(part, ctx.px(2)) & arm).any() and (part & near).sum() >= 0.98 * part.sum():
            result |= part
    return result


# experiment: kol_ve_bardak_olc
def measure_arm_cup(
    ctx: IntervalContext, name: str, drawing: Path | str, cache_dir: Path | str, chained: bool
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """The near arm (visible ``upper_L|fore_L|hand_L``) and cup masks of the original drawing."""
    reconciled = masks.reconcile_and_split(name, drawing, cache_dir)
    arm = reconciled["katmanlar"]["upper_L"] | reconciled["katmanlar"]["fore_L"] | reconciled["katmanlar"]["hand_L"]
    shape = reconciled["rgb"].shape[:2]
    if chained:
        arm = enclose_skin(ctx, arm, reconciled["rgb"])
    cached = masks.load_masks(masks.sam_cache_key(drawing), shape, cache_dir)
    union = np.zeros(shape, bool)
    for concept in CUP_CONCEPTS:
        for m in cached.get(concept, []):
            union |= m
    hand = reconciled["katmanlar"]["hand_L"]
    hand_wide = masks.dilate(hand, ctx.px(5))
    picked = next((b for b in masks.components(union, min_area=1) if (b & hand_wide).any()), None)
    if picked is None:
        picked = np.zeros(shape, bool)
    return arm, picked


# experiment: bardak_saf
def pure_cup(ctx: IntervalContext, cup: NDArray[np.bool_], drawing: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Drop the outside's share from the cup mask: the place reached by walking from outside the mask through
    light pixels, and the part it cuts off from the cup.

    The SAM mask can cross the rim to the neighbouring line (the collar); that line would hang as a loose arc
    once the hand is gone (B1). The cup's see-through inside is closed by the rim line, unreachable.
    """
    light = drawing.min(axis=2) >= 225
    area = (light & masks.dilate(cup, ctx.px(3))).astype(np.uint8)
    _count, lab = cv2.connectedComponents(area, connectivity=4)
    reached = np.unique(lab[masks.dilate(cup, ctx.px(3)) & ~cup & light])
    rest = cup & ~np.isin(lab, reached[reached > 0])
    return masks.components(masks.opening(rest, ctx.px(2)), min_area=1)[0] & rest


# experiment: kontur_genislet
def expand_contour(ctx: IntervalContext, mask: NDArray[np.bool_], rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Take the part mask out by the line weight (the SAM mask passes inside the line).

    Only the black line and edge softening are taken: facing the background the part carries its own outer
    contour whole, facing a neighbour it never steals the neighbour's fill.
    """
    # Line ownership: a line reached from the mask by walking uninterrupted inside the line is the part's. A
    # parallel line with fill between (the strap, the armhole) is the neighbour's line, however near in distance.
    line = masks.dilate(measure.ink_mask(rgb), ctx.px(1))
    grown = mask.copy()
    for _ in range(ctx.px(CONTOUR_MARGIN)):
        grown = grown | (masks.dilate(grown, ctx.px(1)) & line)
    # A thinner-than-line protrusion is the neighbour's line, not the part's (the touching top edge on the arm)
    return masks.opening(grown, ctx.px(4)) | mask


# experiment: kontur_kapsa
def enclose_contour(
    ctx: IntervalContext, m: NDArray[np.bool_], own_fill: NDArray[np.bool_], rgb: NDArray[np.uint8]
) -> NDArray[np.bool_]:
    """The limb's mask holds its own contour: a line within line weight of the limb's fill is the limb's even
    where the opening thinned and dropped it.

    Where the mask misses the line the thin contour counts as a protrusion and the opening clips it; from there
    the lower layer's line shows (the hook, C3). A short line piece is the neighbour's touching end, not taken.
    """
    line = masks.dilate(measure.ink_mask(rgb), ctx.px(1))
    inner = own_fill & ~line
    candidates = line & masks.dilate(inner, ctx.px(CONTOUR_MARGIN)) & ~m
    result = m.copy()
    for part in masks.components(candidates, min_area=1):
        ys, xs = np.nonzero(part)
        if np.hypot(np.ptp(xs), np.ptp(ys)) >= ctx.px(CONTOUR_LONG):
            result |= part
    return result


# experiment: yan_kutle_at
def drop_side_mass(
    ctx: IntervalContext, m: NDArray[np.bool_], rgb: NDArray[np.uint8], neck: int | None = None
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Cut the side mass (shoulder fold) parted from the main mass by an inner line and a narrow neck, and its lines.

    The main mass is the largest component of the lineless mask opened by ``neck`` px, grown back without crossing
    the line. Unreached small areas (behind the hairline, the largest measured 367 px) stay in the piece; 500 px
    and above are dropped. Returns: the kept mask and the cut region (the new edge's contour is synthesised there).
    """
    if neck is None:
        neck = ctx.px(12)
    line = masks.dilate(measure.ink_mask(rgb), ctx.px(1))
    own_fill = m & ~line
    empty = np.zeros(m.shape, bool)
    pieces = masks.components(masks.opening(own_fill, neck), min_area=ctx.area(64))
    if not pieces:
        return m, empty
    mass = pieces[0]
    for _ in range(neck + 2):
        mass = masks.dilate(mass, ctx.px(1)) & own_fill
    dropped = np.zeros(m.shape, bool)
    for part in masks.components(own_fill & ~mass, min_area=ctx.area(500)):
        dropped |= part
    if not dropped.any():
        return m, empty
    kept = m & ~dropped & ~(line & masks.dilate(dropped, ctx.px(9)) & ~masks.dilate(mass, ctx.px(4)))
    cut = masks.dilate(dropped, ctx.px(40))
    # Fringe around the cut is dropped, the notch the cut leaves is closed
    cleaned = (
        cv2.morphologyEx(
            masks.opening(kept, ctx.px(8)).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(61), ctx.odd(61))),
        )
        > 0
    )
    return (kept & ~cut) | (cleaned & cut), cut


# experiment: kenar_arindir
def clean_edge(
    ctx: IntervalContext,
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    line_colour: tuple[int, int, int] = (22, 22, 25),
    light_luma: float = 245.0,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Clean the line-softening pixels on the piece's outer edge: colour becomes the line colour, the mix the alpha.

    In the source these pixels mix the line with the light ground outside; carried as coloured they would halo on
    another ground. Returns: the new rgb and 0-1 alpha (1 outside the edge).
    """
    line = measure.ink_mask(rgb)
    # The compositor opaquely fills 2 px inside the silhouette; a wider edge would thicken the line
    edge = mask & ~masks.erode(mask, ctx.px(2)) & masks.dilate(line, ctx.px(2)) & ~line
    dark = float(np.dot(line_colour, (0.299, 0.587, 0.114)))
    mix = np.clip((light_luma - _luma(rgb)) / (light_luma - dark), 0.0, 1.0)
    alpha = np.ones(mask.shape, np.float32)
    alpha[edge] = mix[edge]
    # A light speck inside the line is no hole: alpha falls only outwards
    closed = cv2.morphologyEx(
        alpha * mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(7), ctx.odd(7)))
    )
    alpha = np.maximum(alpha * mask, closed) + (~mask).astype(np.float32)
    cleaned = rgb.copy()
    cleaned[edge] = line_colour
    return cleaned, alpha


# experiment: kol_temizle
def clean_arm(
    ctx: IntervalContext,
    arm: NDArray[np.bool_],
    cup: NDArray[np.bool_],
    own: NDArray[np.uint8],
    ground: NDArray[np.uint8],
    garment_colour: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Cut the neighbour's fill from the arm mask: pixels the colour of the ground or garment, neither skin nor line."""
    skin = np.median(own[arm & ~masks.dilate(measure.ink_mask(own), ctx.px(3))], axis=0)
    skin_like = np.abs(own.astype(np.int32) - skin).sum(axis=2) < 75
    same = (np.abs(own.astype(np.int32) - ground.astype(np.int32)).sum(axis=2) < 30) | (
        np.abs(own.astype(np.int32) - garment_colour).sum(axis=2) < 45
    )
    cleaned = masks.opening(arm & ~(same & ~skin_like & ~measure.ink_mask(own) & ~cup), ctx.px(3))
    return masks.components(cleaned | cup, min_area=ctx.area(64))[0] & arm


# experiment: bardak_tamamla
def complete_cup(
    ctx: IntervalContext,
    cup: NDArray[np.bool_],
    arm: NDArray[np.bool_],
    own: NDArray[np.uint8],
    ground: NDArray[np.uint8],
) -> NDArray[np.bool_]:
    """Add the cup pieces SAM missed: components touching the cup or hand, apart from the ground, neither skin nor line.

    Where a finger splits the cup its lower piece falls out of the SAM mask; in the canonical drawing the ground
    would crush it.
    """
    skin = np.median(own[arm & ~masks.dilate(measure.ink_mask(own), ctx.px(3))], axis=0)
    candidates = (
        (np.abs(own.astype(np.int32) - ground.astype(np.int32)).sum(axis=2) > 60)
        & (np.abs(own.astype(np.int32) - skin).sum(axis=2) > 75)
        & ~measure.ink_mask(own)
        & ~arm
        & ~cup
    )
    # The cup is convex: a component reaching outside its cover is the neighbour's line (the far arm behind)
    ys_c, xs_c = np.nonzero(cup)
    cover = np.zeros(cup.shape, np.uint8)
    cv2.fillConvexPoly(cover, cv2.convexHull(np.column_stack([xs_c, ys_c])), 1)
    near = masks.dilate(cover > 0, ctx.px(6))
    result = cup.copy()
    for part in masks.components(candidates & near, min_area=8):
        # A thin line (the navel contour) is no cup
        if masks.opening(part, ctx.px(2)).any() and (masks.dilate(part, ctx.px(8)) & (cup | arm)).any():
            result |= part
    return result


# experiment: kolsuz_A_kur
def build_armless(
    ctx: IntervalContext, a: NDArray[np.uint8], muse: NDArray[np.uint8], arm_cup: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Armless A: the gap from Muse, every other pixel from A. Painted pure white in the gap becomes 254."""
    gap = masks.dilate(arm_cup, ctx.px(GAP_MARGIN))
    paper = background(muse)
    ground = a.copy()
    ground[gap] = muse[gap]
    painted_white = gap & (~paper) & (ground.min(axis=2) == 255)
    ground[painted_white] = 254
    return ground, gap


def _shoulder_root(
    ctx: IntervalContext,
    ground: NDArray[np.uint8],
    arm_rgb: NDArray[np.uint8],
    arm: NDArray[np.bool_],
    gap: NDArray[np.bool_],
    shoulder: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """In the armless body, the arm-skin-coloured component touching the shoulder (the shoulder root Muse left)."""
    # The region closed by the line (shaded lower end included); where the line is open, fall to the skin colour
    for part in masks.components(~masks.dilate(measure.ink_mask(ground), ctx.px(1)) & gap, min_area=ctx.area(64)):
        if part[int(shoulder[1]), int(shoulder[0])] and not (part & background(ground)).any():
            return part
    skin = np.median(arm_rgb[arm & ~measure.ink_mask(arm_rgb)], axis=0)
    near = (np.abs(ground.astype(np.int32) - skin).sum(axis=2) < 75) & gap
    pieces = masks.components(masks.dilate(masks.erode(near, 2), ctx.px(2)), min_area=ctx.area(64))
    return pieces[0] if pieces else np.zeros(ground.shape[:2], bool)


# experiment: giysi_maskesi
def garment_mask(ctx: IntervalContext, ground: NDArray[np.uint8], skin: NDArray[np.float64]) -> NDArray[np.bool_]:
    """On the ground, the fill that is neither skin, background nor line."""
    return (
        ~(np.abs(ground.astype(np.int32) - skin).sum(axis=2) < 75)
        & ~background(ground)
        & ~masks.dilate(measure.ink_mask(ground), ctx.px(2))
    )


# experiment: dolgu_modu
def fill_mode(p: NDArray[np.uint8]) -> NDArray[np.float64]:
    """The flat fill's tone: the median of the most frequent colour box (shade and cloth fold are the minority)."""
    q = (p // 8).astype(np.int32)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    distinct, counts = np.unique(key, return_counts=True)
    out: NDArray[np.float64] = np.median(p[key == distinct[int(np.argmax(counts))]], axis=0)
    return out


# experiment: deltoid_diski
def deltoid_disk(
    ctx: IntervalContext, ground: NDArray[np.uint8], pivot: NDArray[np.float64], r: float, skin: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """The disk around the pivot outside the garment: the shoulder opening as the arm turns. The contour comes
    from the arm piece."""
    yy, xx = np.indices(ground.shape[:2])
    disk: NDArray[np.bool_] = ((xx - pivot[0]) ** 2 + (yy - pivot[1]) ** 2 <= r**2) & ~masks.dilate(
        garment_mask(ctx, ground, skin), ctx.px(8)
    )
    return disk


# experiment: kubbe_ic_yaricap
def dome_inner_radius(
    ctx: IntervalContext, own: NDArray[np.uint8], arm: NDArray[np.bool_], pivot: NDArray[np.float64], dome_r: float
) -> float:
    """The arm's background-facing edge distance to the pivot (10th percentile around the dome)."""
    edge = arm & masks.dilate(background(own), ctx.px(CONTOUR_MARGIN) + 2)
    ys, xs = np.nonzero(edge)
    d = np.hypot(xs - pivot[0], ys - pivot[1])
    return float(np.percentile(d[d <= 1.3 * dome_r], 10)) - ctx.px(CONTOUR_MARGIN)


@overload
def _shift(m: NDArray[np.bool_], d: list[int]) -> NDArray[np.bool_]: ...


@overload
def _shift(m: NDArray[np.uint8], d: list[int]) -> NDArray[np.uint8]: ...


def _shift(m: NDArray[np.bool_] | NDArray[np.uint8], d: list[int]) -> NDArray[np.bool_] | NDArray[np.uint8]:
    """Shift an image or mask by whole pixels; the opened place is background."""
    t = np.array([[1, 0, d[0]], [0, 1, d[1]]], dtype=np.float32)
    if m.ndim == 2:
        moved: NDArray[np.bool_] = (
            cv2.warpAffine(m.astype(np.uint8), t, (int(m.shape[1]), int(m.shape[0])), flags=cv2.INTER_NEAREST) > 0
        )
        return moved
    rgb = cast(NDArray[np.uint8], m)
    moved_rgb = cv2.warpAffine(
        rgb, t, (int(m.shape[1]), int(m.shape[0])), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255)
    )
    return cast(NDArray[np.uint8], moved_rgb)


# experiment: kanonik_kaynak_kur
def build_canonical_source(
    ctx: IntervalContext,
    own: NDArray[np.uint8],
    arm_own: NDArray[np.bool_],
    cup_own: NDArray[np.bool_],
    ground_a: NDArray[np.uint8],
) -> dict[str, Any]:
    """The source's canonical drawing: the arm from the source, every other pixel from armless A."""
    arm_full = expand_contour(ctx, arm_own | cup_own, own)
    cup_full = expand_contour(ctx, cup_own, own)
    # The source's side mass and lines (its own body) never enter the canonical: a line the rig never carries
    # is not in the source either
    arm_main, cut = drop_side_mass(ctx, arm_full & ~cup_full, own)
    # The source body's leftover line crumb (a short line touching the arm) spikes when turned
    arm_main = masks.opening(arm_main, ctx.px(6))
    region = enclose_contour(ctx, arm_main, arm_own & ~cup_own, own) | (arm_full & cup_full)
    drawing = ground_a.copy()
    drawing[region] = own[region]
    drawing[arm_main & ~masks.erode(arm_main, ctx.px(6)) & cut] = (22, 22, 25)
    return {"cizim": drawing, "kolsuz": ground_a, "kol": arm_own & region, "bardak": cup_own, "bolge": region}


# experiment: kur
def build(
    ctx: IntervalContext,
    *,
    plan: dict[str, Any],
    match_points: dict[str, dict[str, Any]],
    fill: fill_mod.Fill | None,
    cache_dir: Path | str,
    out_dir: Path | str,
    r9_dir: Path | str,
    dome_radius: float,
    images: dict[str, Path | str],
    chained: bool = False,
    ready_armless: Path | str | None = None,
    mid_dir: Path | str | None = None,
    mid: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical sources under ``out_dir`` from scratch (a cached Muse drawing is never regenerated).

    ``plan`` is the frame plan, ``match_points`` the matched points, ``images`` the key drawings by source
    name, ``dome_radius`` the deltoid dome radius. ``chained`` carries the experiment's ``"zaman" in hareket``
    branches (chain-only refinements); ``ready_armless`` is an earlier interval's armless drawing reused as is
    (the experiment's ``kolsuz_hazir``); ``mid_dir``/``mid`` feed middle-key sources. Returns the summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r9 = Path(r9_dir)
    r9.mkdir(parents=True, exist_ok=True)
    ctx.verify(plan, "frame plan")
    sources = plan["kaynaklar"]
    key_a, key_b = ctx.key_a, ctx.key_b
    a = _read_rgb(Path(images[key_a]))
    h, w = a.shape[:2]
    # A middle key's arm comes from the rig, so its mask is ready; it never enters the chain's measurements
    mids = [name for name in sources if name not in (key_a, key_b)]
    main = {name: b for name, b in sources.items() if name not in mids}
    measured = {name: measure_arm_cup(ctx, name, Path(images[name]), cache_dir, chained) for name in main}
    source_rgb = {name: _read_rgb(Path(images[name])) for name in main}
    for name, b in main.items():
        if "hizalama" in b:
            source_rgb[name] = _shift(source_rgb[name], b["hizalama"])
            arm_m, cup_m = measured[name]
            measured[name] = (_shift(arm_m, b["hizalama"]), _shift(cup_m, b["hizalama"]))

    # When an earlier chain interval already built this drawing's armless form the same one is reused
    arm_a, cup_a = measured[key_a]
    if ready_armless is not None:
        muse = _read_rgb(Path(ready_armless))
    else:
        muse = fill_mod.armless_fill(
            images[key_a],
            (w, h),
            PROMPT_ARMLESS_A,
            fill_mod.cache_path(out, "A", images[key_a], PROMPT_ARMLESS_A),
            fill,
        )
    ground_a, gap = build_armless(ctx, a, muse, arm_a | cup_a)
    head_data = head.setup(
        ctx, a, source_rgb[key_b], images[key_a], images[key_b], key_b, cache_dir, out, r9 / "kafa.json"
    )
    ground_a = head.headless_ground(ctx, ground_a, head_data["kafa_a"])
    muse_b = fill_mod.armless_fill(
        images[key_b],
        (w, h),
        PROMPT_ARMLESS_B,
        fill_mod.cache_path(out, "B", images[key_b], PROMPT_ARMLESS_B),
        fill,
    )
    ground_b, _ = build_armless(ctx, source_rgb[key_b], muse_b, measured[key_b][0] | measured[key_b][1])
    # Shadow rule (WP-R5): the arm's shadow on the body owns no ground; greys missing from the fill drawing
    # are taken from there
    grey_b = (
        (ground_b.max(axis=2).astype(np.int32) - ground_b.min(axis=2) < 25)
        & (ground_b.mean(axis=2) < 225)
        & (ground_b.mean(axis=2) > 130)
    )
    lighter_m = muse_b.astype(np.float32).mean(axis=2) > ground_b.astype(np.float32).mean(axis=2) + 15
    shadow = (
        grey_b
        & lighter_m
        & ~masks.dilate(measure.ink_mask(ground_b) | measure.ink_mask(muse_b), ctx.px(2))
        & (muse_b.mean(axis=2) > 225)
    )
    shadow = masks.dilate(masks.opening(shadow, 2), ctx.px(3)) & ~masks.dilate(measure.ink_mask(ground_b), ctx.px(1))
    ground_b[shadow] = muse_b[shadow]
    headless_b = head.headless_ground(ctx, ground_b, head_data["kafa_b"])
    # Item 1: the changed region from the A-B displacement field
    arm_all = measured[key_a][0] | measured[key_a][1] | measured[key_b][0] | measured[key_b][1]
    head_all = head_data["kafa_a"] | head_data["kafa_b"]
    gray_a = _grey(a)
    gray_b = _grey(source_rgb[key_b])
    field = changed.measure_field(ctx, gray_a, gray_b, masks.dilate(arm_all, ctx.px(10)) | head_all)
    built = changed.build_region(
        ctx, field, a.shape[:2], arm_all, head_all, ~background(a) | ~background(source_rgb[key_b])
    )
    _write_mask(built["bolge"], out / "degisen.png")
    still = changed.still_body(ctx, built["bolge"], head_all, ~background(a) | ~background(source_rgb[key_b]))
    _write_mask(still, out / "degisen_sabit.png")
    # Where an arc point falls under the arm the collar is unmeasurable: extra points from the uncovered place (T-79)
    ruled = [d["a"] for d in match_points.values() if d["tur"] not in ("denetimsiz", "gorunmuyor")]
    hidden_n = masks.dilate(arm_all | head_all, ctx.px(8))
    extra = changed.area_points(
        ctx,
        field,
        gray_a,
        built["bolge"],
        masks.dilate(arm_all, ctx.px(10)),
        head_all,
        ruled + [head_data["veri"]["boyun_a"]],
    )
    extra.update(changed.verify_rules(ctx, gray_a, gray_b, match_points, hidden_n))
    (r9 / "alan_noktalari.json").write_text(json.dumps(extra, ensure_ascii=False, indent=1), encoding="utf-8")
    (r9 / "degisen.json").write_text(
        json.dumps(
            {
                "kaynak_ozetleri": ctx.source_digests(),
                "bolge_px": int(built["bolge"].sum()),
                "kola_bagli_olmayan_adalar": built["adalar"],
                "bant_kayma": changed.band_summary(
                    field,
                    {
                        "kafa_alti_450_700": (450, 700),
                        "gogus_700_950": (700, 950),
                        "gobek_950_1150": (950, 1150),
                        "kalca_1150_1400": (1150, 1400),
                        "uyluk_1400_1650": (1400, 1650),
                        "ayak_1650_1880": (1650, 1880),
                    },
                ),
                "esik_px": changed.STILL_THRESHOLD * ctx.scale,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    changed.draw_region(ctx, a, source_rgb[key_b], built["bolge"], r9 / "degisen.png")
    Image.fromarray(ground_a).save(out / "govde_kolsuz_A.png")
    _write_mask(gap, out / "bosluk_A.png")

    shoulder_a = np.array(sources[key_a]["eklemler"]["shoulder_L"], float)
    root = _shoulder_root(ctx, ground_a, a, arm_a, gap, shoulder_a)
    # The garment around the arm; the garment strip entering the arm mask (the neighbouring strap) parts this way
    skin_a = np.median(a[arm_a & ~masks.dilate(measure.ink_mask(a), ctx.px(3))], axis=0)
    around = masks.dilate(arm_a, ctx.px(60)) & ~arm_a & ~background(a) & ~masks.dilate(measure.ink_mask(a), ctx.px(3))
    garment_r = np.median(a[around & (np.abs(a.astype(np.int32) - skin_a).sum(axis=2) >= 75)], axis=0)
    measured = {
        name: (
            # Compare against the ground at the source's own place: an aligned source's body stands shifted
            clean_arm(ctx, k, c, source_rgb[name], _shift(ground_a, sources[name].get("hizalama", [0, 0])), garment_r),
            complete_cup(ctx, c, k, source_rgb[name], _shift(ground_a, sources[name].get("hizalama", [0, 0]))),
        )
        for name, (k, c) in measured.items()
    }
    arm_a, cup_a = measured[key_a]
    _write_mask(root, out / "omuz_koku_A.png")

    skin = np.median(a[arm_a & ~masks.dilate(measure.ink_mask(a), ctx.px(3))], axis=0)
    # The disk stays inside every source's dome: spilling it would double-arc outside the arm's contour
    disk_r = min(
        dome_inner_radius(
            ctx,
            source_rgb[name],
            measured[name][0],
            np.array(sources[name]["eklemler"]["shoulder_L"], float),
            dome_radius,
        )
        for name in main
    )
    # The arm root's inside (fold shade included) and the pivot's arm-side contour own no ground; there the contour
    # comes from the joined silhouette. The upper contour behind the pivot is the shoulder's true silhouette, stays.
    direction = masks.unit(np.array(sources[key_a]["eklemler"]["elbow_L"], float) - shoulder_a)
    yy, xx = np.indices(ground_a.shape[:2])
    arm_side = (xx - shoulder_a[0]) * direction[0] + (yy - shoulder_a[1]) * direction[1] > 0
    lower_contour = (
        masks.dilate(root, ctx.px(10))
        & arm_side
        & ~masks.dilate(garment_mask(ctx, ground_a, skin) & ~root, ctx.px(8))
        & ~background(ground_a)
    )
    garment = garment_mask(ctx, ground_a, skin)
    # Line ownership: the armhole line between skin and garment is the arm's, the ground never carries it. Once the
    # arm moves the garment opens under it; the hole's inside and the disk take the garment colour (no double arc
    # of the ground line with the arm's)
    mouth = (
        measure.ink_mask(ground_a)
        & masks.dilate(root, ctx.px(CONTOUR_MARGIN) + 3)
        & masks.dilate(garment & ~root, ctx.px(CONTOUR_MARGIN) + 3)
    )
    # A background-facing line is a silhouette (the outer contour down from the strap), no mouth
    mouth = masks.dilate(mouth, ctx.px(2)) & ~masks.dilate(background(ground_a), ctx.px(CONTOUR_MARGIN) + 5)
    disk = deltoid_disk(ctx, ground_a, shoulder_a, disk_r, skin) | root | lower_contour | mouth
    # The disk is a single mass: an island inside (the armless drawing's armhole line, a light speck) would show
    # as a cut line once the arm is gone
    disk = masks.fill_holes(
        cv2.morphologyEx(
            disk.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(25), ctx.odd(25))),
        )
        > 0
    )
    if ready_armless is not None:
        # A ready armless drawing has its armhole drawn: no shoulder root to paint
        disk = np.zeros(ground_a.shape[:2], bool)
    # The opening shoulder root's colour comes from the drawing's visible garment: not from the shade the armless
    # drawing put under the arm (H1)
    a_i = a.astype(np.int32)
    not_skin = a_i[..., 0] - a_i[..., 2] <= 30
    # `garment` may count light fill as paper (the contour gap): here paper parts with a separate limit
    garment_colour = (
        fill_mode(
            a[
                not_skin
                & ~background(a, PAPER_LIMIT)
                & ~gap
                & ~masks.dilate(measure.ink_mask(a), ctx.px(3))
                & masks.dilate(root, ctx.px(200))
            ]
        )
        if ready_armless is None
        else np.zeros(3)
    )
    paper_before = background(ground_a)
    ground_a[disk] = garment_colour.astype(np.uint8)
    if ready_armless is None:
        # The armless fill's grey on the light garment is the arm's shadow, owns no ground (H3): it goes to a
        # separate layer at the source
        ground_mean = ground_a.mean(axis=2)
        grey_t = (
            (ground_a.max(axis=2).astype(np.int32) - ground_a.min(axis=2) < 25)
            & (ground_mean > 150)
            & (ground_mean < 236)
            & ~masks.dilate(measure.ink_mask(ground_a), ctx.px(2))
        )
        # The shadow never ends with the gap: grey beside the gap that the armless drawing shows clean is the
        # arm's too, or a band stays on the body (B7). At the source it goes to the shadow layer
        grey_d = grey_t & (
            gap | (masks.dilate(gap, ctx.px(SHADOW_SPILL)) & (muse.astype(np.float32).mean(axis=2) > ground_mean + 15))
        )
        ground_a[grey_d] = garment_colour.astype(np.uint8)
    # The kept line's disk-facing edge mixes with the old skin colour: turn it into a line-garment mix
    ring = (
        masks.dilate(disk, ctx.px(4))
        & ~disk
        & ~measure.ink_mask(ground_a)
        & ~paper_before
        & masks.dilate(measure.ink_mask(ground_a), ctx.px(3))
    )
    mix_h = np.clip((245.0 - _luma(ground_a)) / (245.0 - 22.0), 0.0, 1.0)[..., None]
    blended = mix_h * np.array((22.0, 22.0, 25.0)) + (1.0 - mix_h) * garment_colour
    ground_a[ring] = blended[ring].astype(np.uint8)
    # The disk opens to the background: the garment is unreadable from colour later, store it openly
    _write_mask(garment & ~disk, out / "giysi_A.png")
    gap |= disk
    Image.fromarray(ground_a).save(out / "govde_kolsuz_A.png")
    _write_mask(gap, out / "bosluk_A.png")
    _write_mask(disk, out / "omuz_disk.png")

    summary: dict[str, Any] = {"kaynak_ozetleri": ctx.source_digests()}
    garment_final = garment & ~disk
    for name in mids:
        build_mid_source(
            ctx,
            plan,
            name,
            ground_a,
            garment_final,
            disk,
            built["bolge"],
            skin,
            out,
            mid_dir,
            mid,
            match_points,
            head_data["veri"],
            extra,
            still,
            head_all,
            gap,
        )
    for name, b in main.items():
        own = source_rgb[name]
        arm_own, cup_own = measured[name]
        if name != key_a:
            # B in its own place: pixels in the changed region uncovered by B's arm are B's patch, the rest bent A
            bent = bend.Bend(
                ctx,
                plan,
                ground_a,
                garment_final,
                disk,
                built["bolge"],
                still=still,
                head=head_all,
                gap_gray=(gap * 255).astype(np.uint8),
                matched_points=match_points,
                neck=head_data["veri"],
                blocks=extra,
            ).frame(int(b["kare"]), False)["zemin"]
            # The earlier head's place is patch too: headless fill there, stays facing the paper
            patch = bend.patch_mask(
                ctx,
                masks.erode(built["bolge"] | masks.dilate(head_data["kafa_a"], ctx.px(6)), ctx.px(bend.PATCH_INSET)),
                masks.dilate(head_data["kafa_b"], ctx.px(2)),
            )
            if chained:
                # Facing the paper the patch never melts: the last source saying paper keeps the first source's
                # arm root from spilling outside the new arm (double line, skin band; R13)
                arm_b_side = masks.dilate(measured[key_b][0], ctx.px(PATCH_PAPER))
                # The armless drawing may draw the arm root wide: paper reads from the source itself too
                # (`ground_ownership` turns that place to paper)
                patch = patch | (
                    masks.dilate(patch, ctx.px(PATCH_PAPER))
                    & arm_b_side
                    & (background(headless_b, PAPER_LIMIT) | background(own, PAPER_LIMIT))
                    & ~masks.dilate(head_data["kafa_b"], ctx.px(2))
                )
            patch_garment = garment_mask(ctx, headless_b, skin)
            seam = bend.seam_constraints(
                ctx,
                patch,
                masks.dilate(head_data["kafa_b"], ctx.px(8)),
                measure.ink_mask(headless_b),
                measure.ink_mask(bent),
            )
            patch_rgb = headless_b
            _write_mask(patch, out / f"{name}_yama.png")
            _write_mask(patch_garment, out / f"{name}_yama_giysi.png")
            Image.fromarray(patch_rgb).save(out / f"{name}_yama_rgb.png")
            (out / f"{name}_yama_dikis.json").write_text(json.dumps(seam), encoding="utf-8")
            alpha = cast(
                NDArray[np.float32],
                cv2.GaussianBlur(patch.astype(np.float32), (0, 0), bend.PATCH_SOFTEN * ctx.scale),
            )
            patched = bend.Bend(
                ctx,
                plan,
                ground_a,
                garment_final,
                disk,
                built["bolge"],
                patch_rgb=patch_rgb,
                patch_alpha=alpha,
                patch_garment=patch_garment,
                seam=seam,
                still=still,
                head=head_all,
                gap_gray=(gap * 255).astype(np.uint8),
                matched_points=match_points,
                neck=head_data["veri"],
                blocks=extra,
            ).frame(int(b["kare"]), True)
            # Item 1: the canonical drawing is the source itself (the arm, its shadow included); identity inside
            # the patch, armless A plus a border band outside
            full = bend.Bend(
                ctx,
                plan,
                ground_a,
                garment_final,
                disk,
                built["bolge"],
                patch_rgb=own,
                patch_alpha=alpha,
                patch_garment=patch_garment,
                seam=seam,
                still=still,
                head=head_all,
                gap_gray=(gap * 255).astype(np.uint8),
                matched_points=match_points,
                neck=head_data["veri"],
                blocks=extra,
            ).frame(int(b["kare"]), True)["zemin"]
            # H12: B' head is the first source's rigidly aligned form; the arm sits over the head
            head_b_grown = masks.dilate(head_data["kafa_b"], ctx.px(2))
            full[head_b_grown] = patched["zemin"][head_b_grown]
            full = head.Layer(out, key_a, head_data["veri"], None).draw(
                full, head.chain_matrix(head_data["veri"], 1.0, None)
            )
            full[arm_own | cup_own] = own[arm_own | cup_own]
            full = clean_crumbs(ctx, full, patch_garment)
            canon = {
                **build_canonical_source(ctx, own, arm_own, cup_own, full),
                "cizim": full,
                "kolsuz": patched["zemin"],
            }
        if name == key_a:
            canon = {
                "cizim": a,
                "kolsuz": ground_a,
                "kol": arm_a,
                "bardak": cup_a,
                "bolge": gap,
            }
        Image.fromarray(canon["cizim"]).save(out / f"{name}.png")
        Image.fromarray(canon["kolsuz"]).save(out / f"{name}_kolsuz.png")
        _write_mask(canon["kol"], out / f"{name}_kol.png")
        _write_mask(canon["bardak"], out / f"{name}_bardak.png")
        _write_mask(canon["bolge"], out / f"{name}_bolge.png")
        summary[name] = {
            "kol_px": int(canon["kol"].sum()),
            "bardak_px": int(canon["bardak"].sum()),
            "bolge_px": int(canon["bolge"].sum()),
        }

    summary["sahiplik"] = ground_ownership(ctx, plan, out, images, chained, dome_radius)
    Image.fromarray(_read_rgb(out / "govde_kolsuz_A.png")).save(out / f"{key_a}_kolsuz.png")
    diff = np.abs(a.astype(np.int32) - muse.astype(np.int32)).sum(axis=2)
    band = masks.dilate(gap, ctx.px(2)) & ~gap
    summary["bosluk"] = {
        "bosluk_px": int(gap.sum()),
        "serit_ortalama_fark": float(diff[band].mean()),
        "serit_fark_gt60_px": int((diff[band] > 60).sum()),
        "serit_px": int(band.sum()),
    }
    (out / "ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# experiment: kirinti_temizle
def clean_crumbs(ctx: IntervalContext, rgb: NDArray[np.uint8], garment: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """A small dark stain broken off the body returns to paper. Only the line colour: where the light top mixes
    into the paper through the contour gap no skin is erased."""
    paper = background(rgb)
    line = measure.ink_mask(rgb)
    # Brokenness is measured on the line web: a line joined to the body's contour is no crumb (erased for lying
    # far from the fill, the edge frays, H11)
    main = masks.components(masks.dilate(line, ctx.px(1)), min_area=1)[0]
    for part in masks.components(~paper & ~masks.dilate(garment, ctx.px(6)), min_area=1):
        if part.sum() < ctx.area(CRUMB_AREA) and (part & line).sum() >= 0.5 * part.sum() and not (part & main).any():
            rgb[masks.dilate(part, 1)] = 255
    return rgb


# experiment: ara_kaynak_kur
def build_mid_source(
    ctx: IntervalContext,
    plan: dict[str, Any],
    name: str,
    ground_a: NDArray[np.uint8],
    garment_final: NDArray[np.bool_],
    disk: NDArray[np.bool_],
    region: NDArray[np.bool_],
    skin: NDArray[np.float64],
    out_dir: Path | str,
    mid_dir: Path | str | None,
    mid: dict[str, Any] | None,
    match_points: dict[str, dict[str, Any]],
    neck: dict[str, Any],
    blocks: dict[str, dict[str, Any]],
    still: NDArray[np.bool_],
    head_all: NDArray[np.bool_],
    gap: NDArray[np.bool_],
) -> None:
    """A middle key's canonical files: the patch (armless + outside the hole), the canonical drawing. Identity
    inside the patch; bent A shows through the hole."""
    out = Path(out_dir)
    folder = Path(mid_dir) if mid_dir is not None else out
    b = plan["kaynaklar"][name]
    own = _read_rgb(folder / f"{name}.png")
    ground = _read_rgb(folder / f"{name}_kolsuz.png")
    arm = _read_mask(folder / f"{name}_kol.png")
    cup = _read_mask(folder / f"{name}_bardak.png")
    hole = _read_mask(folder / f"{name}_delik.png")
    head_m = _read_mask(out / f"{name}_kafa_alfa.png")
    # Patch: the measured region + its own arm; the head and the hole stay outside
    patch = bend.patch_mask(
        ctx,
        masks.erode(region, ctx.px(bend.PATCH_INSET)) | masks.dilate(arm, ctx.px(8)),
        masks.dilate(head_m, ctx.px(2)) | hole,
    )
    patch_garment = garment_mask(ctx, ground, skin)
    obj = bend.Bend(
        ctx,
        plan,
        ground_a,
        garment_final,
        disk,
        region,
        mid=mid,
        still=still,
        head=head_all,
        gap_gray=(gap * 255).astype(np.uint8),
        matched_points=match_points,
        neck=neck,
        blocks=blocks,
    )
    bent = obj.frame(int(b["kare"]), False)["zemin"]
    seam = bend.seam_constraints(
        ctx, patch, masks.dilate(head_m, ctx.px(8)), measure.ink_mask(ground), measure.ink_mask(bent)
    )
    _write_mask(patch, out / f"{name}_yama.png")
    _write_mask(patch_garment, out / f"{name}_yama_giysi.png")
    Image.fromarray(ground).save(out / f"{name}_yama_rgb.png")
    (out / f"{name}_yama_dikis.json").write_text(json.dumps(seam), encoding="utf-8")
    alpha = cast(NDArray[np.float32], cv2.GaussianBlur(patch.astype(np.float32), (0, 0), bend.PATCH_SOFTEN * ctx.scale))

    def ground_at(r: NDArray[np.uint8]) -> NDArray[np.uint8]:
        assert mid is not None
        obj.add_patch(name, r, alpha, patch_garment, seam, None, float(mid["t"]), "k")
        return cast(NDArray[np.uint8], obj.frame(int(b["kare"]), name)["zemin"])

    armless = ground_at(ground)
    # Armless and drawing part only at the arm and head: the band bend must not move the arm, they sit bit by bit
    full = armless.copy()
    seen = masks.dilate(head_m, ctx.px(2)) | masks.dilate(arm | cup, ctx.px(2))
    full[seen] = own[seen]
    # An A line left outside the patch at the region border, broken off the body, returns to paper
    for part in masks.components(~background(full) & region, min_area=1):
        if part.sum() < 400:
            full[masks.dilate(part, ctx.px(1))] = 255
    Image.fromarray(full).save(out / f"{name}.png")
    Image.fromarray(armless).save(out / f"{name}_kolsuz.png")
    _write_mask(arm, out / f"{name}_kol.png")
    _write_mask(cup, out / f"{name}_bardak.png")
    _write_mask(masks.dilate(arm | cup, ctx.px(8)), out / f"{name}_bolge.png")


# experiment: zemin_sahipligi
def ground_ownership(
    ctx: IntervalContext,
    plan: dict[str, Any],
    out_dir: Path | str,
    images: dict[str, Path | str],
    chained: bool,
    dome_radius: float,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, int]]:
    """Items 1-2: every pixel without the source's arm owns the ground. The armless drawing stays only where the
    arm pieces cover.

    Outside the arm pieces the source-ground difference parts in two: the arm's shadow (light fill gone grey)
    to a separate layer, the rest (armpit skin, the arm's cut wing) written to the ground; once the arm turns
    it stays in place, bending with the body.
    """
    out = Path(out_dir)
    store = cache if cache is not None else {}
    result = {}
    key_a = ctx.key_a
    for name, b in plan["kaynaklar"].items():
        store.pop(name, None)
        loaded = load_source(ctx, name, b["eklemler"], out, dome_radius, store)
        # The patch meets the source in its own coordinates (the border band not yet bent)
        own = _read_rgb(Path(images[name]))
        arm = np.zeros(own.shape[:2], bool)
        for piece in loaded["parcalar"].values():
            arm |= piece["mask"]
        head_m = _read_mask(out / f"{name}_kafa_alfa.png")
        ground_path = out / ("govde_kolsuz_A.png" if name == key_a else f"{name}_yama_rgb.png")
        ground = _read_rgb(ground_path)
        owned = masks.dilate(arm, ctx.px(OWNER_MARGIN)) | masks.dilate(head_m, ctx.px(HEAD_OWNER))
        if name != key_a:
            owned |= ~_read_mask(out / f"{name}_yama.png")
        else:
            # The shoulder root is under the arm: the source's line there never returns to the ground, it would
            # ghost once the arm is gone (H2, H3)
            owned |= _read_mask(out / "omuz_disk.png")
        diff = (np.abs(own.astype(np.int32) - ground.astype(np.int32)).sum(axis=2) > 0) & ~owned
        grey = (
            (own.max(axis=2).astype(np.int32) - own.min(axis=2) < 25)
            & (own.mean(axis=2) > 130)
            & (own.mean(axis=2) < 225)
        )
        shadow = (
            diff
            & grey
            & (ground.astype(np.float32).mean(axis=2) > own.astype(np.float32).mean(axis=2) + 15)
            & ~masks.dilate(measure.ink_mask(own), ctx.px(1))
        )
        if name == key_a and len(plan["kaynaklar"]) == 2:
            # Grey on the top found only at this source (the skirt, the side) is tone, not line: it dies as the
            # last source's grey arrives, never jumps at the cut (C2)
            other = _read_rgb(Path(images[ctx.key_b]))
            grey_other = (
                (other.max(axis=2).astype(np.int32) - other.min(axis=2) < 25)
                & (other.mean(axis=2) > 130)
                & (other.mean(axis=2) < 225)
            )
            light_a = _read_mask(out / "giysi_A.png")
            near_other = masks.dilate(grey_other, ctx.px(SHADOW_NEAR))
            far = masks.opening(
                grey
                & (own.mean(axis=2) > 150)
                & light_a
                & ~owned
                & ~masks.dilate(measure.ink_mask(own), ctx.px(1))
                & ~near_other,
                ctx.px(SHADOW_THIN),
            )
            far_grown = masks.dilate(far, ctx.px(2)) & ~masks.dilate(measure.ink_mask(own), ctx.px(1)) & ~owned
            Image.fromarray(np.dstack([own, (far_grown * 255).astype(np.uint8)]), "RGBA").save(
                out / f"{name}_golge_uzak.png"
            )
            ground[far_grown] = fill_mode(own[light_a & (own.min(axis=2) >= 225)]).astype(np.uint8)
            diff = diff & ~far_grown
        if name != key_a and len(plan["kaynaklar"]) == 2:
            # This source's arm casing (armpit fold included) never reaches the other source's opening region:
            # that arm's drawing, no arm there (C1)
            casing = (
                cv2.morphologyEx(
                    arm.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(CASING_CLOSE), ctx.odd(CASING_CLOSE))),
                )
                > 0
            )
            _write_mask(masks.dilate(casing, ctx.px(10)), out / f"{name}_kol_zarf.png")
        if name != key_a:
            # The arm shadow may pass into the armless fill on the drawing too: grey on the top where the first
            # source's ground has no grey is the limb's shadow, goes to the layer (C2)
            # Drawing, not ground: the first source's own arm shadow left the ground, that place stays white too
            ground_a_now = _read_rgb(out / "govde_kolsuz_A.png")
            grey_a = (
                (ground_a_now.max(axis=2).astype(np.int32) - ground_a_now.min(axis=2) < 25)
                & (ground_a_now.mean(axis=2) > 130)
                & (ground_a_now.mean(axis=2) < 225)
            )
            base = (
                grey
                & (own.mean(axis=2) > 150)
                & _read_mask(out / f"{name}_yama_giysi.png")
                & ~owned
                & ~masks.dilate(measure.ink_mask(own), ctx.px(1))
            )
            near_a = masks.dilate(grey_a, ctx.px(SHADOW_NEAR))
            # At the skirt the two sources overlap: the difference's thin remnant never goes to the layer, it would
            # show as a cut line while arriving gradually
            grey_t = masks.opening(base & ~near_a, ctx.px(SHADOW_THIN))
            # The whole cleaned band goes to the layer: left edge softening would ring the shadow white
            shadow = shadow | (
                masks.dilate(grey_t, ctx.px(2)) & ~masks.dilate(measure.ink_mask(own), ctx.px(1)) & ~owned
            )
            # Toneless tone: the top's most frequent tone; edge softening goes too
            light = _read_mask(out / f"{name}_yama_giysi.png") & (own.min(axis=2) >= 225)
            ground[masks.dilate(grey_t, ctx.px(2)) & ~masks.dilate(measure.ink_mask(own), ctx.px(1)) & ~owned] = (
                fill_mode(own[light]).astype(np.uint8)
            )
            diff = diff & ~masks.dilate(grey_t, ctx.px(2))
        hard = diff & ~shadow
        # The arm's hair ends: a small line surrounded by paper goes with the arm, never hangs in the air on
        # the ground
        paper = background(own)
        for part in masks.components(measure.ink_mask(own) & hard, min_area=1):
            ring = masks.dilate(part, ctx.px(3)) & ~part
            if part.sum() < ctx.area(OWNER_STRAND) and (ring & paper).sum() >= 0.3 * ring.sum():
                hard &= ~masks.dilate(part, ctx.px(2))
        before = ground.copy()
        ground[hard] = own[hard]
        if chained:
            # The armless drawing's arm root may cross the source's silhouette: where the source says paper the
            # ground beside the arm is paper too (skin band on the outer edge, double line; R13)
            paper_side = (
                masks.dilate(arm, ctx.px(ROOT_SIDE))
                & masks.dilate(background(own, PAPER_LIMIT), ctx.px(1))
                & ~masks.dilate(arm, ctx.px(1))
                & ~masks.dilate(head_m, ctx.px(HEAD_OWNER))
            )
            ground[paper_side] = 255
        # A leftover writing broken off the body (a fingertip SAM missed) never passes to the ground: it would
        # hang in the air once the arm is gone
        for part in masks.components(~background(ground), min_area=1)[1:]:
            ground[part] = before[part]
        # Beside the arm, on light ground (paper or light garment), ink broken off the line web is the arm's hair:
        # left on the ground it specks once the arm is gone (B4). Broken line over skin (chest hair) is the
        # drawing itself
        ground_line = measure.ink_mask(ground)
        main_ground = masks.components(masks.dilate(ground_line, ctx.px(1)), min_area=1)[0]
        arm_side = masks.dilate(arm, ctx.px(30))
        for part in masks.components(
            ground_line & ~main_ground & arm_side & ~masks.dilate(head_m, ctx.px(HEAD_OWNER)), min_area=1
        ):
            if part.sum() < ctx.area(STRAND_SPECK):
                ring = masks.dilate(part, ctx.px(3)) & ~masks.dilate(part, ctx.px(1))
                colour = np.median(ground[ring], axis=0)
                if colour.min() >= 225:
                    ground[masks.dilate(part, ctx.px(1))] = colour.astype(np.uint8)
        if name == key_a:
            # Where the first source's arm covered: not in the drawing, fills with the other source's content once
            # the arm is gone (C1). A line touching the arm's edge and staying on the ground comes out from under
            # the arm too; a halo (near-white walked to paper) is paper
            disk_m = _read_mask(out / "omuz_disk.png")
            gap_m = _read_mask(out / "bosluk_A.png")
            opening = arm.copy()
            arm_edge = masks.dilate(arm, ctx.px(14))
            for part in masks.components((ground.mean(axis=2) < 200) & gap_m, min_area=1):
                if (part & arm_edge).any():
                    opening |= masks.dilate(part, ctx.px(2))
            _write_mask(opening & ~disk_m, out / f"{name}_acilan.png")
            # Paper squeezed between the arm and a hairline is unreachable from the edge: where the source is
            # lighter than the top it is paper (not the odd light speck)
            halo = (
                (background(ground, HALO_LIMIT) | masks.opening(own.min(axis=2) >= HALO_LIMIT, 1))
                & gap_m
                & ~disk_m
                & ~opening
            )
            ground[halo] = 255
        Image.fromarray(ground).save(ground_path)
        Image.fromarray(np.dstack([own, (shadow * 255).astype(np.uint8)]), "RGBA").save(out / f"{name}_golge.png")
        result[name] = {"sert_px": int(hard.sum()), "golge_px": int(shadow.sum())}
        store.pop(name, None)
    return result


# experiment: kaynak_oku
def read_source(out_dir: Path | str, name: str) -> dict[str, Any]:
    """The canonical source the rig uses, read from disk."""
    out = Path(out_dir)
    return {
        "ad": name,
        "cizim": _read_rgb(out / f"{name}.png"),
        "kolsuz": _read_rgb(out / f"{name}_kolsuz.png"),
        "kol": _read_mask(out / f"{name}_kol.png"),
        "bardak": _read_mask(out / f"{name}_bardak.png"),
        "bolge": _read_mask(out / f"{name}_bolge.png"),
        "omuz_koku": _read_mask(out / "omuz_koku_A.png"),
        "omuz_disk": _read_mask(out / "omuz_disk.png"),
        "giysi": _read_mask(out / "giysi_A.png"),
    }


# experiment (cok_kaynak): _segment_uzaklik
def segment_distance(
    xx: NDArray[np.intp], yy: NDArray[np.intp], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-pixel distance to the segment a-b."""
    v = b - a
    t = np.clip(((xx - a[0]) * v[0] + (yy - a[1]) * v[1]) / max(float(v @ v), 1e-9), 0.0, 1.0)
    out: NDArray[np.float64] = np.hypot(xx - (a[0] + t * v[0]), yy - (a[1] + t * v[1]))
    return out


# experiment (cok_kaynak): _kiris
def chord(
    ctx: IntervalContext,
    arm: NDArray[np.bool_],
    joint: NDArray[np.float64],
    u1: NDArray[np.float64],
    u2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """From the joint, the chord across the arm along the two bones' bisector: its middle and half length."""
    _ = ctx
    n = masks.unit(masks.unit(u1) + masks.unit(u2))
    d = np.array([-n[1], n[0]])
    h, w = arm.shape
    ts = np.arange(-260.0, 261.0)
    xs = np.clip((joint[0] + ts * d[0]).astype(int), 0, w - 1)
    ys = np.clip((joint[1] + ts * d[1]).astype(int), 0, h - 1)
    inside = arm[ys, xs]
    if not inside.any():
        return joint.copy(), 12.0
    # The uninterrupted run holding the joint (or nearest to it)
    runs = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))).reshape(-1, 2)
    gaps = [0.0 if ts[a] <= 0 <= ts[b - 1] else min(abs(ts[a]), abs(ts[b - 1])) for a, b in runs]
    a, b = runs[int(np.argmin(gaps))]
    t0, t1 = ts[a], ts[b - 1]
    return joint + 0.5 * (t0 + t1) * d, max(0.5 * float(t1 - t0), 6.0)


# experiment (cok_kaynak): kol_disi_ayir
def split_shoulder_wing(
    ctx: IntervalContext,
    m: NDArray[np.bool_],
    shoulder: NDArray[np.float64],
    elbow: NDArray[np.float64],
    paper: NDArray[np.bool_],
    dome_r: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Cut the shoulder side off the arm where, facing the background, it exceeds the arm's thickness (from the
    bone's far half).

    Armpit skin draws joined to the arm but owns the body; kept on the arm it wings outside the silhouette once
    turned. The garment-facing side never parts (the dome spills there). Returns: the kept arm and the cut piece;
    the cut piece is written to the ground.
    """
    axis = masks.unit(elbow - shoulder)
    length = float(np.hypot(*(elbow - shoulder)))
    yy, xx = np.indices(m.shape)
    u = (xx - shoulder[0]) * axis[0] + (yy - shoulder[1]) * axis[1]
    v = -(xx - shoulder[0]) * axis[1] + (yy - shoulder[1]) * axis[0]
    ring = masks.dilate(m, ctx.px(6)) & ~m & paper
    side = 1.0 if (v[ring] > 0).sum() >= (v[ring] < 0).sum() else -1.0
    tip = m & (u >= 0.5 * length) & (u <= 0.9 * length)
    if not tip.any():
        return m, np.zeros(m.shape, bool)
    limit = max(float(np.percentile(side * v[tip], 95)) + ctx.px(WING_MARGIN), dome_r)
    # Behind the pivot counts too, or the cut corners at the dome; the dome's top along the bone (|v| small)
    # stays on the arm
    wing = m & (u < 0.5 * length) & (side * v > limit)
    # A line-weight band is the arm's own contour, no body piece
    wing = masks.opening(wing, ctx.px(WING_THIN))
    if not wing.any():
        return m, wing
    # The corner where the cut joins the dome rounds: never sharp on the silhouette once turned
    kept = m & ~wing
    near = masks.dilate(wing, 2 * ctx.px(WING_ROUND))
    kept = (kept & ~near) | (masks.opening(kept, ctx.px(WING_ROUND)) & near)
    return kept, m & ~kept


# experiment (cok_kaynak): kubbe_olcu
def dome_measures(
    ctx: IntervalContext, mask: NDArray[np.bool_], pivot: NDArray[np.float64], axis: NDArray[np.float64], length: float
) -> NDArray[np.float64]:
    """The upper arm's shoulder-side extents in the bone frame: behind the pivot, both sides of the axis (to the
    bone's half)."""
    _ = ctx
    ys, xs = np.nonzero(mask)
    u = (xs - pivot[0]) * axis[0] + (ys - pivot[1]) * axis[1]
    v = -(xs - pivot[0]) * axis[1] + (ys - pivot[1]) * axis[0]
    near = u <= 0.5 * length
    out: NDArray[np.float64] = np.array([-u.min(), v[near].max(), -v[near].min()], float)
    return out


# experiment (cok_kaynak): kaynak_yukle
def load_source(
    ctx: IntervalContext,
    name: str,
    joints: dict[str, Any],
    out_dir: Path | str,
    dome_radius: float,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read the canonical source; cut the near arm from the plan's joints into upper arm, forearm and hand + cup.

    Border: every pixel goes to the nearest bone (between neighbouring bones the border is the elbow/wrist
    bisector). ``cache`` memoises by source name; pass a fresh dict per setup. (The experiment's ``hizalama``
    parameter is dropped: its body never reads it.)
    """
    store = cache if cache is not None else {}
    if name in store:
        return store[name]
    out = Path(out_dir)
    k = read_source(out, name)
    rgb, arm = k["cizim"], k["kol"]
    cup = pure_cup(ctx, k["bardak"], rgb)
    # Outside the region the ground's line is the arm's never, even touching the arm (turned, it spikes)
    # The missed edge skin (fold shade) is the arm's: only then is the contour behind it reached by the line
    # walk (H2)
    as_int = rgb.astype(np.int32)
    skin_family = (as_int[..., 0] - as_int[..., 2] > 30) & (as_int[..., 0] > 120)
    arm_skin = arm | (masks.dilate(arm, ctx.px(12)) & skin_family & ~cup)
    full = enclose_contour(ctx, expand_contour(ctx, arm_skin | cup, rgb), arm_skin & ~cup, rgb) & k["bolge"]
    yy, xx = np.indices(arm.shape)
    pt = {j: np.array(v, float) for j, v in joints.items()}
    bone_ends = {
        "upper_L": ("shoulder_L", "elbow_L"),
        "fore_L": ("elbow_L", "wrist_L"),
        "hand_L": ("wrist_L", "hand_center_L"),
    }
    nearest = np.stack(
        [segment_distance(xx, yy, pt[bone_ends[b][0]], pt[bone_ends[b][1]]) for b in ("upper_L", "fore_L", "hand_L")]
    ).argmin(axis=0)
    cup_full = expand_contour(ctx, cup, rgb)
    pieces: dict[str, dict[str, Any]] = {}
    leftover = np.zeros(arm.shape, bool)
    # Forearm first: the mass parted from the forearm by the fold line is the upper arm's, never ownerless with
    # fill showing under (B3)
    for i, b in ((1, "fore_L"), (0, "upper_L"), (2, "hand_L")):
        m = full & (nearest == i) & ~cup_full
        if b == "upper_L":
            m = m | leftover
        if b == "hand_L":
            # Hand + cup is one piece. The cup is convex: where its open part empties in the mask the rim line
            # would hang broken
            # The cover is from the SAM cup: the line walk reaches the collar line the cup touches too, and being
            # thin it falls to the opening
            ys_c, xs_c = np.nonzero(cup)
            cover = np.zeros(m.shape, np.uint8)
            cv2.fillConvexPoly(cover, cv2.convexHull(np.column_stack([xs_c, ys_c])), 1)
            # The cover grows by the cup line: more would carry the neighbour's line (the collar) and paper by
            # hand (H9, the cup halo)
            m = masks.opening(m & masks.dilate(arm, ctx.px(10)), ctx.px(6)) | masks.dilate(cover > 0, ctx.px(3))
            cut = np.zeros(m.shape, bool)
        else:
            raw_m = m
            m, cut = drop_side_mass(ctx, m, rgb)
            if b == "fore_L":
                leftover = raw_m & ~m
            if b == "upper_L":
                # A mass parted by the fold but inside the arm casing and facing the garment is the arm's skin:
                # goes with the arm, identity never breaks at the source frame (H2, H13)
                casing = segment_distance(xx, yy, pt["shoulder_L"], pt["elbow_L"]) <= CASING_FACTOR * dome_radius
                back = raw_m & ~m & casing & ~masks.dilate(background(rgb, PAPER_LIMIT), ctx.px(40))
                m = m | back
                cut = cut & ~masks.dilate(back, ctx.px(6))
                # Outside the bone capsule is no arm (the dome top belongs to the capsule); only on the
                # paper-facing side: the garment-facing side is the arm's own contour, left on the ground it
                # never goes with the arm (H2)
                m = m & ~(
                    (segment_distance(xx, yy, pt["shoulder_L"], pt["elbow_L"]) > CASING_FACTOR * dome_radius)
                    & masks.dilate(background(rgb, PAPER_LIMIT), ctx.px(40))
                )
                # Item 2: the side exceeding the arm's thickness owns the ground (never cut and dropped,
                # `ground_ownership` writes it to the ground)
                m, wing = split_shoulder_wing(
                    ctx, m, pt["shoulder_L"], pt["elbow_L"], background(rgb, PAPER_LIMIT), dome_radius
                )
                cut = cut | masks.dilate(wing, ctx.px(8))
            # The light slit between line and fill (left from the neighbouring garment) is no hole: takes the
            # fill or the line colour
            m = (
                cv2.morphologyEx(
                    m.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(19), ctx.odd(19))),
                )
                > 0
            )
        piece_colour = m & ~(rgb.min(axis=2) >= LIGHT_MIN) if b != "hand_L" else m
        _, iy, ix = measure.nearest_indices(piece_colour)
        piece_rgb = rgb[iy, ix]
        # The cut edge's line is never written to the colour: over skin (its own hide) there is no line; at the
        # frame look underneath
        # Where the edge carries the source's own line no synthesised line is pressed: doubled they sit thick
        # and jagged (C3)
        band = m & ~masks.erode(m, ctx.px(6)) & cut & ~masks.dilate(measure.ink_mask(rgb), ctx.px(6))
        for part in masks.components(band, min_area=1):
            ys_b, xs_b = np.nonzero(part)
            if np.hypot(np.ptp(xs_b), np.ptp(ys_b)) < ctx.px(CUT_MIN_LENGTH):
                band = band & ~part
        pieces[b] = {"ad": b, "mask": m, "rgb": piece_rgb, "kesik_bant": band}
    pieces = {b: pieces[b] for b in ("upper_L", "fore_L", "hand_L")}
    # A cut arm line kept by no piece goes to the nearest bone's piece: no line gap at the elbow fold (C3)
    ownerless = full & masks.dilate(measure.ink_mask(rgb), ctx.px(1)) & ~cup_full
    for b in ("upper_L", "fore_L", "hand_L"):
        ownerless = ownerless & ~pieces[b]["mask"]
    for i, b in enumerate(("upper_L", "fore_L")):
        extra = ownerless & (nearest == i)
        pieces[b]["mask"] = pieces[b]["mask"] | extra
        pieces[b]["rgb"] = pieces[b]["rgb"].copy()
        pieces[b]["rgb"][extra] = rgb[extra]
    skin_inside = arm & ~masks.dilate(measure.ink_mask(rgb), ctx.px(3))
    # A garment island the mask cut out holes once the arm turns: the closed void takes skin
    for piece in pieces.values():
        # A piece is a single mass: a broken island the line walk brought (the collar line) never turns with
        # the piece
        filled = masks.fill_holes(masks.components(piece["mask"], min_area=1)[0])
        piece["rgb"] = piece["rgb"].copy()
        piece["rgb"][filled & ~piece["mask"]] = np.median(rgb[skin_inside], axis=0).astype(np.uint8)
        piece["mask"] = filled
        piece["rgb"], piece["alfa"] = clean_edge(ctx, piece["rgb"], piece["mask"])
    # The inner fold open at the source too (the armpit gap): the cap fills the place the turn opens, not this (C3)
    union_k = np.zeros(arm.shape, bool)
    for piece in pieces.values():
        union_k |= piece["mask"]
    fold = (
        cv2.morphologyEx(
            union_k.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(41), ctx.odd(41))),
        )
        > 0
    ) & ~union_k
    caps = {}
    for joint, (b_top, b_low) in {"elbow_L": ("upper_L", "fore_L"), "wrist_L": ("fore_L", "hand_L")}.items():
        u1 = pt[joint] - pt[bone_ends[b_top][0]]
        u2 = pt[bone_ends[b_low][1]] - pt[joint]
        centre, r = chord(ctx, full, pt[joint], u1, u2)
        disk = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= (1.3 * r) ** 2
        colour = np.median(rgb[disk & skin_inside if (disk & skin_inside).any() else skin_inside], axis=0)
        caps[joint] = {
            "merkez": (float(centre[0]), float(centre[1])),
            "r": 1.03 * r,
            "renk": tuple(int(c) for c in colour),
        }
    skin = np.median(rgb[skin_inside], axis=0)
    k["ten"] = skin
    ground = k["kolsuz"]
    ground_line = measure.ink_mask(ground)
    garment = k["giysi"]
    # The arm passes over the shoulder root, never over the garment fill; the line between is the arm's
    allowed = ~garment
    root_ring = masks.dilate(k["omuz_koku"], ctx.px(10)) & (k["omuz_koku"] | masks.dilate(ground_line, ctx.px(1)))
    k["kubbe_eksen"] = masks.unit(pt["elbow_L"] - pt["shoulder_L"])
    k["kubbe_boy"] = float(np.hypot(*(pt["elbow_L"] - pt["shoulder_L"])))
    k["kubbe_olcu"] = dome_measures(ctx, pieces["upper_L"]["mask"], pt["shoulder_L"], k["kubbe_eksen"], k["kubbe_boy"])
    k.update(
        {
            "tam": full,
            "parcalar": pieces,
            "kapaklar": caps,
            "kivrim": fold,
            "izinli": allowed,
            "eklemler": pt,
            "giysi": garment,
            "kok_halka": root_ring,
            "z_arka": background(ground),
        }
    )
    store[name] = k
    return k
