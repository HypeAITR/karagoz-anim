"""Body bending between two keys and the single cut.

Ported from the experiment's ``deneme/r2/plan/bukulme.py``: the ``Bend``
class (ground layers per plan and drawing per frame), the control points, the
support and inner maps, the patch mask, the seam weights and constraints, the
shadow strengths, the MLS ground/patch sources and the frame progress. Same
maths, English identifiers, each public function noting its experiment name.
Thresholds are module-level ``UPPER_CASE`` with the experiment's base values;
lengths and areas scale with the interval's line width through
``interval.IntervalContext``. Returned dicts keep the experiment's keys so
outputs compare directly.

The ground is A's mesh in every frame, moving points travel towards B with
the upper-arm progress ``t``, still points stay put. The ramp stands only on
the edge facing the unmoving body: the neck and straps travel as far as the
points say. After the cut a B patch lies on top: the measured changed region
outside B's arm cover, bent from its own place towards the same targets.
Vertices outside the region displace zero: the still region is pixel-identical.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.rig import deform, interval, match

PATCH_MIN_AREA = 200  # experiment: YAMA_ALAN; area-scaled; smaller difference islands never patch
BORDER_TRIANGLE = 24.0  # experiment: SINIR_UCGEN; dense-mesh triangle diagonal + margin
PATCH_INSET = 12  # experiment: YAMA_ICERI; the blurred patch alpha must not spill out of the region
RAMP_WIDTH = 60.0  # experiment: GECIS_PX; the bend rises from zero to full this far inside the border
BAND_HALF = 60  # experiment: BANT_PX; half of the band spilling out of the region
SEAM_RADIUS = 60.0  # experiment: DIKIS_R; the strip where the arm sticks to the garment
SEAM_LENGTH_FRAC = 0.35  # experiment: DIKIS_BOY; the stickiness reach along the bone, times bone length
AREA_BUDGET = 1.3  # experiment: ALAN_BUTCE
STRENGTH_STEP = 0.05  # experiment: GUC_ADIM
SEAM_MIN_AREA = 12  # experiment: DIKIS_MIN_ALAN; area-scaled
SEAM_MAX_SHIFT = 14.0  # experiment: DIKIS_MAX_KAYMA
SHADOW_FADE = 0.15  # experiment: GOLGE_YOL
HALO_MIN = 249  # experiment: HALE_MIN
PAPER_MIN = 251  # experiment: KAGIT_MIN
PATCH_SOFTEN = 3.0  # experiment: YAMA_YUMUSAK; times the line-width scale
DENSE_DILATE = 121  # experiment: the inline tek(121)
SEAM_KERNEL = 7  # experiment: the inline tek(7)
SEAM_ARM_KERNEL = 17  # experiment: the inline tek(17)
HEAD_ALPHA_THRESHOLD = 127


# experiment: ara_oku
def read_midpoints(plan: dict[str, Any], path: Path | str) -> dict[str, Any] | None:
    """The measured points when the plan holds a middle key."""
    if len(plan["kaynaklar"]) <= 2:
        return None
    out: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return out


# experiment: nokta_sozlugu
def collect_points(
    matched: dict[str, dict[str, Any]],
    neck: dict[str, Any] | None = None,
    blocks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """All points: the ruled matches plus the neck (from the head measurement) plus extra blocks from uncovered
    ground. ``neck``/``blocks`` are None when that measurement is missing (only the ruled ones, before setup)."""
    points = dict(matched)
    if neck is not None:
        a, b = np.array(neck["boyun_a"], float), np.array(neck["boyun_b_olcum"], float)
        points["boyun"] = {
            "a": a.tolist(),
            "b": b.tolist(),
            "kayma_px": round(float(np.hypot(*(b - a))), 1),
            "tur": "sabit" if np.hypot(*(b - a)) <= match.STILL_THRESHOLD else "hareketli",
        }
    if blocks is not None:
        points.update(blocks)
    return points


# experiment: noktalar
def control_points(collected: dict[str, dict[str, Any]], mid: dict[str, Any] | None = None) -> dict[str, Any]:
    """The points bending the ground: rest, target, moving flag; the pivot apart.

    Unseen points are unused. The deltoid is not a drawn point but the dome
    circle's centre: it gives the arm's path, never bends the ground (it shifts
    40 px past the neighbouring sleeve, and would fold triangles at 60 px; its
    underside is a flat garment-colour disc anyway).
    """
    items = [
        (name, d) for name, d in collected.items() if d["tur"] not in ("gorunmuyor", "denetimsiz") and name != "deltoid"
    ]
    names = [name for name, _ in items]
    extra: dict[str, Any] = {}
    if mid is not None:
        # Middle key: the measured point; the unmeasured one on the A-B line.
        t_mid = float(mid["t"])
        extra = {
            "t_k": t_mid,
            "k": np.array(
                [
                    mid["noktalar"].get(name)
                    or [
                        collected[name]["a"][j] + t_mid * (collected[name]["b"][j] - collected[name]["a"][j])
                        for j in (0, 1)
                    ]
                    for name in names
                ],
                float,
            ),
        }
    return {
        **extra,
        "a": np.array([d["a"] for _, d in items], float),
        "b": np.array([d["b"] for _, d in items], float),
        "hareketli": np.array([d["tur"] == "hareketli" for _, d in items]),
        "pivot_a": np.array(collected["deltoid"]["a"], float),
        "pivot_b": np.array(collected["deltoid"]["b"], float),
    }


# experiment: hedef_noktalar
def target_points(n: dict[str, Any], t: float) -> NDArray[np.float64]:
    """Targets at the frame: a moving point travels t of the way from A to B, a still point stays at A's place."""
    q: NDArray[np.float64] = n["a"].copy()
    if "k" not in n:
        q[n["hareketli"]] += t * (n["b"] - n["a"])[n["hareketli"]]
        return q
    t_mid = n["t_k"]
    h = (
        n["a"] + (t / t_mid) * (n["k"] - n["a"])
        if t <= t_mid
        else n["k"] + ((t - t_mid) / (1.0 - t_mid)) * (n["b"] - n["k"])
    )
    q[n["hareketli"]] = h[n["hareketli"]]
    return q


# experiment: kafa_maskesi (bukulme)
def head_mask(shape: tuple[int, int], alphas: list[NDArray[np.uint8]]) -> NDArray[np.bool_]:
    """The head layer of both poses: apart from the body, unseen under the ground; the neck adjoins its edge."""
    m = np.zeros(shape, bool)
    for alpha in alphas:
        m |= alpha > HEAD_ALPHA_THRESHOLD
    return m


# experiment: destek_haritasi
def support_map(
    ctx: interval.IntervalContext,
    region: NDArray[np.bool_],
    still: NDArray[np.bool_] | None = None,
    head: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """0-1 weight: 0 on the edge facing the unmoving body, 1 ``RAMP_WIDTH`` inside. That body is exactly zero:
    pixel-identical with A.

    Without ``still`` (the unmoving body mass) every edge of the region is a
    border. The head void and the paper count as borders and the neck and
    collar could never bend. ``head`` is None when the head alphas are missing.
    """
    # Linear inside a triangle: the ramp starts one triangle inside so a weighted vertex's triangle never crosses over.
    if still is None:
        free = region
    else:
        head_m = head if head is not None else np.zeros(region.shape, bool)
        band = ctx.px(BAND_HALF)
        free = (
            cv2.dilate(
                (region | head_m).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1)),
            )
            > 0
        ) & ~still
    dist = cast(NDArray[np.float32], cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5))
    s = np.clip((dist - ctx.px(BORDER_TRIANGLE)) / ctx.px(RAMP_WIDTH), 0.0, 1.0)
    return cast(NDArray[np.float32], s * s * (3.0 - 2.0 * s))


# experiment: ic_haritasi
def inner_map(ctx: interval.IntervalContext, alpha: NDArray[np.float32]) -> NDArray[np.float32]:
    """0-1 weight inside the patch: 0 on the border band, 1 past ``BORDER_TRIANGLE`` + ``RAMP_WIDTH`` inside."""
    dist = cast(NDArray[np.float32], cv2.distanceTransform((alpha >= 0.5).astype(np.uint8), cv2.DIST_L2, 5))
    s = np.clip((dist - ctx.px(BORDER_TRIANGLE)) / ctx.px(RAMP_WIDTH), 0.0, 1.0)
    return cast(NDArray[np.float32], s * s * (3.0 - 2.0 * s))


# experiment: golge_gucu
def shadow_strength(t: float, t_source: float, t_neighbour: float) -> float:
    """Strength of the source's arm shadow at the frame: 1 at its source, 0 past ``SHADOW_FADE`` of the way to the
    neighbour source.

    The shadow belongs to the arm: when the arm leaves, no grey ghost stays (H3). At the source frame it is as drawn.
    """
    return float(np.clip(1.0 - abs(t - t_source) / max(SHADOW_FADE * abs(t_neighbour - t_source), 1e-6), 0.0, 1.0))


# experiment: golge_gelis
def shadow_arrival(t: float) -> float:
    """The last source's arm shadow: appears after the first source's shadow dies, full before the cut. The only
    transition in time is in this layer, never in the line (C2)."""
    return float(np.clip((t - SHADOW_FADE) / max(1.0 - 2.0 * SHADOW_FADE, 1e-6), 0.0, 1.0))


# experiment: destek
def support_at(
    vertices: NDArray[np.float64], weights: NDArray[np.float32] | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Support weight at the vertices. MLS holds the still point inside."""
    xi = np.clip(np.round(vertices[:, 0]).astype(int), 0, weights.shape[1] - 1)
    yi = np.clip(np.round(vertices[:, 1]).astype(int), 0, weights.shape[0] - 1)
    out: NDArray[np.float64] = weights[yi, xi].astype(np.float64)
    return out


# experiment: dikis_agirligi
def seam_weights(
    ctx: interval.IntervalContext,
    vertices: NDArray[np.float64],
    garment: NDArray[np.bool_],
    pivot: NDArray[np.float64],
    axis: NDArray[np.float64],
    length: float,
) -> NDArray[np.float64]:
    """Between the upper arm's sewn part (1) and its free arm (0): distance to the garment and the shoulder side along
    the bone.

    The sleeve line is sewn to the body, never turns with the arm; only so do A's and B's sleeves fall on the same spot.
    """
    dist = cast(NDArray[np.float32], cv2.distanceTransform((~garment).astype(np.uint8), cv2.DIST_L2, 5))
    xi = np.clip(np.round(vertices[:, 0]).astype(int), 0, garment.shape[1] - 1)
    yi = np.clip(np.round(vertices[:, 1]).astype(int), 0, garment.shape[0] - 1)
    s1 = np.clip(1.0 - dist[yi, xi] / ctx.px(SEAM_RADIUS), 0.0, 1.0)
    u = (vertices - pivot) @ axis
    s2 = np.clip(1.0 - u / (SEAM_LENGTH_FRAC * length), 0.0, 1.0)
    out: NDArray[np.float64] = s1 * s1 * (3.0 - 2.0 * s1) * s2 * s2 * (3.0 - 2.0 * s2)
    return out


# experiment: yapisik_kaynak
def blended_source(
    rigid: deform.Source,
    body: deform.Source,
    weight: NDArray[np.float64],
    visible: NDArray[np.bool_],
    mesh: deform.Mesh,
) -> tuple[deform.Source, dict[int, float]]:
    """Per-vertex blend of the arm source with the body source; the blend strength is capped by the area budget.

    The largest strength whose visible triangles stay within ``AREA_BUDGET``
    against the rigid arm is picked. When full strength never fits, the drawing
    cannot reach that angle: an early-warning signal. The returned dict keeps
    the strength per frame.
    """
    strength: dict[int, float] = {}

    def source(frame: int, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        r = rigid(frame, vertices)
        g = body(frame, vertices)
        base = deform.area_ratio(mesh, r)[visible]
        for s in np.arange(1.0, -0.001, -STRENGTH_STEP):
            v = r + s * weight[:, None] * (g - r)
            o = deform.area_ratio(mesh, v)[visible] / base
            if o.min() >= 1.0 / AREA_BUDGET and o.max() <= AREA_BUDGET:
                strength[frame] = float(s)
                return v
        strength[frame] = 0.0
        return r

    return source, strength


# experiment: mls_a_kaynak
def mls_ground_source(
    n: dict[str, Any], progress: list[float], weights: NDArray[np.float32] | NDArray[np.float64]
) -> deform.Source:
    """The A ground: rest points at A, MLS inside the support; exactly zero outside the support."""

    def source(frame: int, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        q = target_points(n, progress[frame])
        if not (q != n["a"]).any():
            return vertices.copy()
        return vertices + support_at(vertices, weights)[:, None] * (deform.mls_rigid(vertices, n["a"], q) - vertices)

    return source


# experiment: mls_b_kaynak
def mls_patch_source(
    n: dict[str, Any],
    progress: list[float],
    seam: list[list[float]] | None = None,
    inner: NDArray[np.float32] | NDArray[np.float64] | None = None,
    t_own: float = 1.0,
    key: str = "b",
) -> deform.Source:
    """The B patch: rest points at B, towards the same targets. At t=1 a moving point is home, a still point sits at
    A's place.

    ``seam``: [bx, by, hx, hy] pairs of the lines on the patch border; they sit
    home too. ``inner``: the patch's inner map (0 at the border, 1 past the
    ramp inside). Its own frame's displacement is subtracted: at the source
    frame the inside is identity (the drawing arrives bit by bit), only the
    border band sits on the neighbour lines.
    """
    d = np.array(seam or [], float).reshape(-1, 4)
    p0 = np.vstack([n[key], d[:, :2]])

    def source(frame: int, vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        h = deform.mls_rigid(vertices, p0, np.vstack([target_points(n, progress[frame]), d[:, 2:]]))
        if inner is None:
            return h
        own = deform.mls_rigid(vertices, p0, np.vstack([target_points(n, t_own), d[:, 2:]]))
        return h - support_at(vertices, inner)[:, None] * (own - vertices)

    return source


# experiment: ilerleme
def progress(plan: dict[str, Any]) -> list[float]:
    """The planned upper-arm screen angle's A->B progress (0-1). In a chain the plan's shared ``t_kare``: with the upper
    arm barely turning the angle is noise."""
    if "t_kare" in plan:
        return [float(x) for x in plan["t_kare"]]

    def angle(k: dict[str, Any]) -> float:
        e = k["hedef_eklemler"]
        return math.atan2(e["elbow_L"][1] - e["shoulder_L"][1], e["elbow_L"][0] - e["shoulder_L"][0])

    def wrapped(x: float, first: float) -> float:
        return (x - first + math.pi) % (2 * math.pi) - math.pi

    a = [angle(k) for k in plan["kareler"]]
    last = wrapped(a[-1], a[0])
    return [float(np.clip(wrapped(x, a[0]) / last, 0.0, 1.0)) if last else 0.0 for x in a]


# experiment: yama_maskesi
def patch_mask(
    ctx: interval.IntervalContext, region: NDArray[np.bool_], hidden: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """B's patch: the measured changed region outside the head (under the arm comes from armless B).

    Small islands fall.
    """
    m = region & ~hidden
    count, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return np.isin(lab, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= ctx.area(PATCH_MIN_AREA)])


# experiment: dikis_kisitlari
def seam_constraints(
    ctx: interval.IntervalContext,
    patch: NDArray[np.bool_],
    hidden: NDArray[np.bool_],
    b_lines: NDArray[np.bool_],
    ground_lines: NDArray[np.bool_],
) -> list[list[float]]:
    """Per B line crossing the patch border, [bx, by, hx, hy]: its place on the border and the nearest ground line.

    The patch MLS holds these pairs too: lines meet at the border, for no single character or pose.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(SEAM_KERNEL), ctx.odd(SEAM_KERNEL)))
    border = (cv2.dilate(patch.astype(np.uint8), kernel) > 0) & ~(cv2.erode(patch.astype(np.uint8), kernel) > 0)
    arm_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(SEAM_ARM_KERNEL), ctx.odd(SEAM_ARM_KERNEL)))
    far_from_arm = ~(cv2.dilate(hidden.astype(np.uint8), arm_kernel) > 0)
    count, _, stats, centres = cv2.connectedComponentsWithStats(
        (border & far_from_arm & b_lines).astype(np.uint8), connectivity=8
    )
    d, labels = cv2.distanceTransformWithLabels(
        (~ground_lines).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(ground_lines)
    place = np.zeros((int(labels.max()) + 1, 2), int)
    place[labels[ys, xs]] = np.column_stack([xs, ys])
    result = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < ctx.area(SEAM_MIN_AREA):
            continue
        x, y = int(round(centres[i, 0])), int(round(centres[i, 1]))
        if d[y, x] <= ctx.px(SEAM_MAX_SHIFT):
            hx, hy = place[labels[y, x]]
            result.append([float(x), float(y), float(hx), float(hy)])
    return result


class Bend:
    """One plan's ground layers and drawing per frame. Meshes are built once at setup. A is the ground, every other
    source a patch over it."""

    def __init__(
        self,
        ctx: interval.IntervalContext,
        plan: dict[str, Any],
        ground_a: NDArray[np.uint8],
        garment_a: NDArray[np.bool_],
        disk: NDArray[np.bool_],
        region: NDArray[np.bool_],
        patch_rgb: NDArray[np.uint8] | None = None,
        patch_alpha: NDArray[np.float32] | None = None,
        patch_garment: NDArray[np.bool_] | None = None,
        seam: list[list[float]] | None = None,
        shadow_a: NDArray[np.float32] | None = None,
        shadow_b: NDArray[np.float32] | None = None,
        mid: dict[str, Any] | None = None,
        still: NDArray[np.bool_] | None = None,
        head: NDArray[np.bool_] | None = None,
        far_shadow_rgba: NDArray[np.uint8] | None = None,
        arm_casing_gray: NDArray[np.uint8] | None = None,
        gap_gray: NDArray[np.uint8] | None = None,
        opening_gray: NDArray[np.uint8] | None = None,
        matched_points: dict[str, dict[str, Any]] | None = None,
        neck: dict[str, Any] | None = None,
        blocks: dict[str, dict[str, Any]] | None = None,
    ):
        self.ctx = ctx
        self.n = control_points(collect_points(matched_points or {}, neck, blocks), mid)
        self.t = progress(plan)
        self.shadow_a = shadow_a
        two_sources = len(plan["kaynaklar"]) == 2
        soften = PATCH_SOFTEN * ctx.scale
        self.far_shadow: NDArray[np.float32] | None
        if far_shadow_rgba is not None and two_sources:
            u = far_shadow_rgba.astype(np.float32)
            blurred = cast(NDArray[np.float32], cv2.GaussianBlur(u[..., 3] / 255.0, (0, 0), soften))
            self.far_shadow = deform.premultiplied(u[..., :3], blurred)
        else:
            self.far_shadow = None
        self.t_stops = sorted(float(self.t[int(b["kare"])]) for b in plan["kaynaklar"].values())
        self.region = region
        # Outside the region the paper band bends freely but its texture is never resampled: paper on both sides stays
        # pixel-identical with A (the line is never cut).
        self.outside = (~region & (ground_a.min(axis=2) >= PAPER_MIN)) if still is not None else None
        self.ground_a = ground_a
        self.ground_source = mls_ground_source(self.n, self.t, support_map(ctx, region, still, head))
        dilate = ctx.odd(DENSE_DILATE)
        self.ground_mesh = deform.build_mesh(
            np.ones(ground_a.shape[:2], bool),
            dense=cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
            > 0,
        )
        self.pm_a = deform.premultiplied(ground_a)
        self.garment_raw_a = garment_a
        self.garment_a = garment_a.astype(np.float32)
        self.disk = disk.astype(np.float32)
        self.patches: dict[str, dict[str, Any]] = {}
        self.arm_casing = (
            cv2.GaussianBlur((arm_casing_gray > HEAD_ALPHA_THRESHOLD).astype(np.float32), (0, 0), soften)
            if arm_casing_gray is not None
            else None
        )
        self.gap_a = (gap_gray > HEAD_ALPHA_THRESHOLD).astype(np.float32) if gap_gray is not None else None
        self.opening = (
            cv2.GaussianBlur((opening_gray > HEAD_ALPHA_THRESHOLD).astype(np.float32), (0, 0), soften)
            if opening_gray is not None and two_sources
            else None
        )
        self.b_name = max(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"]))
        self.a_name = min(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"]))
        if patch_rgb is not None:
            assert patch_alpha is not None and patch_garment is not None
            self.add_patch(self.b_name, patch_rgb, patch_alpha, patch_garment, seam, shadow_b, 1.0, "b")

    def add_patch(
        self,
        name: str,
        rgb: NDArray[np.uint8],
        alpha: NDArray[np.float32],
        garment: NDArray[np.bool_],
        seam: list[list[float]] | None,
        shadow: NDArray[np.float32] | None,
        t_own: float,
        key: str,
    ) -> None:
        """Add a source patch over the ground."""
        self.patches[name] = {
            "ag": deform.build_mesh(alpha > 0.0, dense=np.ones(alpha.shape, bool)),
            "pm": deform.premultiplied(rgb, alpha),
            "giysi": garment.astype(np.float32),
            "giysi_ham": garment,
            "golge": shadow,
            "t": t_own,
            "kaynak": mls_patch_source(self.n, self.t, seam, inner_map(self.ctx, alpha), t_own, key),
        }

    def source_of(self, name: str) -> deform.Source:
        """The ground source for the first source, the patch source otherwise."""
        return self.ground_source if name == self.a_name else self.patches[name]["kaynak"]

    def mesh_of(self, name: str) -> deform.Mesh:
        """The ground mesh for the first source, the patch mesh otherwise."""
        return self.ground_mesh if name == self.a_name else self.patches[name]["ag"]

    def garment_of(self, name: str) -> NDArray[np.bool_]:
        """The raw garment mask of the named source."""
        return self.garment_raw_a if name == self.a_name else self.patches[name]["giysi_ham"]

    def _strength(self, t: float, t_source: float) -> float:
        """Shadow strength: against the neighbour source on the frame's side."""
        stops = self.t_stops
        i = stops.index(min(stops, key=lambda x: abs(x - t_source)))
        neighbour = stops[i + 1] if (t >= t_source and i + 1 < len(stops)) or i == 0 else stops[i - 1]
        return shadow_strength(t, t_source, neighbour)

    def frame(self, i: int, patched: bool | str) -> dict[str, Any]:
        """The frame's ground (rgb), garment and disk masks, pivot and per-triangle area ratios. ``patched``: the name
        of the source to lay on top (True: the last source)."""
        target = self.ground_source(i, self.ground_mesh.rest)
        mx, my = deform.flow_map(self.ground_mesh, target, self.pm_a.shape[:2])
        pm = cv2.remap(self.pm_a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        garment = cv2.remap(self.garment_a, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5
        disk = cv2.remap(self.disk, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5
        z = pm[..., :3]
        if self.outside is not None:
            z = z.copy()
            m = self.outside & (z.min(axis=2) >= PAPER_MIN)
            z[m] = self.ground_a[m]
        ratio = {"a": deform.area_ratio(self.ground_mesh, target)}
        # Shadow layer (premultiplied): on the same mesh as the ground, fading with distance from the source.
        if self.shadow_a is not None and not patched:
            g = cv2.remap(self.shadow_a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * self._strength(
                self.t[i], 0.0
            )
            z = g[..., :3] + (1.0 - g[..., 3:]) * z
        if not patched and self.far_shadow is not None:
            g = cv2.remap(self.far_shadow, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * (
                1.0 - shadow_arrival(self.t[i])
            )
            z = g[..., :3] + (1.0 - g[..., 3:]) * z
        if not patched and self.opening is not None and self.b_name in self.patches:
            # The opening region (under the arm at the first source) with the other source's content, in the frame's
            # bend: the edge and contour come from the drawing, never jump at the cut (C1).
            patch = self.patches[self.b_name]
            opening = cv2.remap(self.opening, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
            mxb, myb = deform.flow_map(patch["ag"], patch["kaynak"](i, patch["ag"].rest), self.pm_a.shape[:2])
            if self.arm_casing is not None:
                opening = opening * (
                    1.0
                    - cv2.remap(self.arm_casing, mxb, myb, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
                )
            y = cv2.remap(patch["pm"], mxb, myb, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            yg = cv2.remap(patch["giysi"], mxb, myb, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5
            z = opening * y[..., :3] + (1.0 - opening * y[..., 3:]) * z
            inside = (opening[..., 0] * y[..., 3]) >= 0.5
            # The other drawing's paper grain is paper: no fake edge or closed pocket beside pure white (G1, G2).
            z[(opening[..., 0] > 0.02) & (z.min(axis=2) >= HALO_MIN)] = 255.0
            if self.t[i] > 0.0:
                # In the gap around the arm where the other drawing says paper, the first drawing's pale remnant (halo,
                # hair trace) returns to paper; line and skin stay (G1).
                assert self.gap_a is not None
                gap = cv2.remap(self.gap_a, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5
                y_paper = (y[..., 3] >= 0.5) & (y[..., :3].min(axis=2) >= HALO_MIN * y[..., 3])
                z[gap & y_paper & (z.mean(axis=2) > 100) & (z[..., 0] - z[..., 2] < 30)] = 255.0
            garment = (garment & ~inside) | (yg & inside)
            disk = disk & ~inside
            if patch["golge"] is not None:
                # The shadow falls only on the garment; never on paper or line.
                g = cv2.remap(
                    patch["golge"], mxb, myb, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                ) * shadow_arrival(self.t[i])
                g = g * ((z.min(axis=2) < PAPER_MIN - 2) & (z.mean(axis=2) > 100))[..., None]
                z = g[..., :3] + (1.0 - g[..., 3:]) * z
        patch_name: str | None = self.b_name if patched is True else (patched if isinstance(patched, str) else None)
        if patch_name is not None and patch_name in self.patches:
            patch = self.patches[patch_name]
            hb = patch["kaynak"](i, patch["ag"].rest)
            mx, my = deform.flow_map(patch["ag"], hb, self.pm_a.shape[:2])
            y = cv2.remap(patch["pm"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            yg = cv2.remap(patch["giysi"], mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5
            inside = y[..., 3] >= 0.5
            if patch["golge"] is not None:
                # In a two-source interval the last source's shadow arrives gradually, never jumps at the cut.
                strength = (
                    shadow_arrival(self.t[i])
                    if self.opening is not None and patch_name == self.b_name
                    else self._strength(self.t[i], patch["t"])
                )
                g = cv2.remap(patch["golge"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * strength
                y = g + (1.0 - g[..., 3:]) * y
            z = y[..., :3] + (1.0 - y[..., 3:]) * z
            garment = (garment & ~inside) | (yg & inside)
            disk = disk & ~inside
            ratio["b"] = deform.area_ratio(patch["ag"], hb)
        return {
            "zemin": (np.clip(z, 0, 255) + 0.5).astype(np.uint8),
            "giysi": garment,
            "disk": disk,
            "pivot": self.n["pivot_a"] + self.t[i] * (self.n["pivot_b"] - self.n["pivot_a"]),
            "oran": ratio,
        }
