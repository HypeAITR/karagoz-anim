"""Per-frame layer compositing with z-order: the arm chain over the armless ground.

Ported from the experiment's ``deneme/r2/rig_kare.py``: only what the frame step reaches
(``affine_matrix``, ``is_identity_tf``, ``composite_arm_frame``, ``part_alpha`` and their helpers
``_resolve_tf`` and ``canonical_paper``). Same maths; English identifiers, each public function
noting its experiment name. Lengths scale with the interval's line width through
``interval.IntervalContext``.

Not ported: ``kare_birlestir`` (the old R2 path), ``mask_donustur``, ``katman_donustur_rgba`` and
``rot2d_deg`` — the frame step never reaches them.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.rig import masks, measure
from karagoz.engine.rig.interval import IntervalContext

LINE_COLOR = (22, 22, 25)  # experiment: kontur_rengi default
CONTOUR_THICKNESS = 6  # experiment: kontur_kalinlik default base


# experiment: afin_matris
def affine_matrix(
    p0_xy: tuple[float, float],
    pk_xy: tuple[float, float],
    phi_deg: float,
    scale: float = 1.0,
    axis: tuple[float, float] = (1.0, 0.0),
) -> NDArray[np.float64]:
    """T(pk) · R(phi) · S_axis(scale, axis) · T(-p0) 2x3 affine matrix.

    ``axis`` is the bone unit direction in the source image; ``scale`` scales the layer along that axis
    around p0 (the perpendicular stays), then turns and carries it to pk. At ``scale=1.0`` bit by bit the
    plain turn matrix.
    """
    r_rad = math.radians(phi_deg)
    c, s = math.cos(r_rad), math.sin(r_rad)
    ux, uy = float(axis[0]), float(axis[1])
    n = math.hypot(ux, uy)
    if n > 1e-12:
        ux, uy = ux / n, uy / n
    else:
        ux, uy = 1.0, 0.0
    k = float(scale) - 1.0
    # S = I + (scale - 1) u u^T, then A = R S
    s00 = 1.0 + k * ux * ux
    s01 = k * ux * uy
    s11 = 1.0 + k * uy * uy
    a00 = c * s00 - s * s01
    a01 = c * s01 - s * s11
    a10 = s * s00 + c * s01
    a11 = s * s01 + c * s11
    tx = pk_xy[0] - (a00 * p0_xy[0] + a01 * p0_xy[1])
    ty = pk_xy[1] - (a10 * p0_xy[0] + a11 * p0_xy[1])
    return np.array([[a00, a01, tx], [a10, a11, ty]], dtype=np.float64)


def _resolve_tf(
    tf: tuple[Any, ...],
) -> tuple[tuple[float, float], tuple[float, float], float, float, tuple[float, float]]:
    """Resolve a transform tuple: (p0, pk, phi) or (p0, pk, phi, scale, axis). experiment: _tf_coz."""
    if len(tf) == 5:
        p0_xy, pk_xy, phi_deg, scale, axis = tf
        return p0_xy, pk_xy, phi_deg, scale, axis
    p0_xy, pk_xy, phi_deg = tf
    return p0_xy, pk_xy, phi_deg, 1.0, (1.0, 0.0)


# experiment: kanonik_kagit
def canonical_paper(ctx: IntervalContext, rgb: NDArray[np.uint8], limit: int = 251) -> NDArray[np.bool_]:
    """Paper walked from the edge through light pixels (the light garment, 240-248, parts from it)."""
    _ = ctx
    _count, lab = cv2.connectedComponents((rgb.min(axis=2) >= limit).astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0])


# experiment: kimlik_tf_mi
def is_identity_tf(tf: tuple[Any, ...]) -> bool:
    """Whether the transform leaves the piece at the source place (axis scale ±0.5%, angle 0.05°, shift 0.5 px)."""
    p0, pk, phi, scale, _ = _resolve_tf(tf)
    return abs(phi) < 0.05 and abs(scale - 1.0) < 0.005 and math.hypot(pk[0] - p0[0], pk[1] - p0[1]) < 0.5


# experiment: parca_alfasi
def part_alpha(p: dict[str, Any]) -> NDArray[np.float32]:
    """The piece's soft alpha (in source coordinates)."""
    if "alfa" in p:
        out: NDArray[np.float32] = cv2.GaussianBlur(p["mask"].astype(np.float32), (0, 0), 0.7) * p["alfa"]
        return out
    soft: NDArray[np.float32] = cv2.GaussianBlur(p["mask"].astype(np.float32), (0, 0), 1.0)
    return soft


# experiment: kol_kare_birlestir
def composite_arm_frame(
    ctx: IntervalContext,
    ground: NDArray[np.uint8],
    pieces: list[dict[str, Any]],
    caps: list[dict[str, Any]],
    shoulder_clip: dict[str, Any] | None = None,
    leftover: dict[str, Any] | None = None,
    shoulder_disk: dict[str, Any] | None = None,
    line_colour: tuple[int, int, int] = LINE_COLOR,
    contour_thickness: int | None = None,
    skin: NDArray[Any] | None = None,
    joint_round: dict[str, Any] | None = None,
    cap_except: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Draw the arm chain over the armless ground; every frame is source pixels + joint caps.

    pieces: bottom to top {"ad", "rgb", "mask", "tf"}; rgb is nearest-filled outside the mask.
    caps: {"tf", "merkez", "r", "renk"} joint disks; drawn in the gap the pieces leave, edged with contour
    where they meet the chain's outer border, unseen where they pass under the pieces.
    shoulder_clip: {"parca", "pivot", "yon", "izinli", optional "kok"}: piece pixels falling on the pivot's
    parent side only over ``izinli``.
    leftover: {"maske", "giysi", "arka_plan"}: a shoulder-root piece the arm never covers is erased where it
    touches the background 3x more than the garment (the arm's outer side).
    shoulder_disk: {"maske", "giysi"}: the contourless shoulder disk on the ground; joins the united
    silhouette, edges with contour facing the background.
    joint_round: {"merkez", "r", "yonler": {piece: unit direction from the joint into the piece}}: when the
    joint angle changes a lot the piece's cut end past the joint clips, the disk draws whole.
    skin: for the pieces' ``kesik`` (cut edge band) drawing: a band over skin takes no line (ends lineless on
    its own hide).
    """
    thick = contour_thickness if contour_thickness is not None else ctx.px(CONTOUR_THICKNESS)
    h, w = ground.shape[:2]
    yy, xx = np.indices((h, w))
    render = ground.copy()
    placed: list[tuple[str, NDArray[np.uint8], NDArray[np.bool_], NDArray[np.float32], NDArray[np.bool_]]] = []
    union = np.zeros((h, w), bool)
    for p in pieces:
        p0, pk, phi, scale, axis = _resolve_tf(p["tf"])
        m_aff = affine_matrix(p0, pk, phi, scale, axis)
        if "bas" in p:
            # The mesh deformer draws: a premultiplied layer, its colour meaningful where its alpha is
            w_rgb, w_alfa = p["bas"]()
            coloured = w_alfa > 0.02
        else:
            # Bicubic overshoots beside the line (a light speck over skin); bilinear never does
            w_rgb = cv2.warpAffine(p["rgb"], m_aff, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # Soft edge: a turned binary mask leaves the contour staircased
            # A cleaned piece carries the edge's true alpha; light softening suffices for the binary mask's stair
            w_alfa = cv2.warpAffine(
                part_alpha(p), m_aff, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            coloured = np.ones((h, w), bool)
        w_mask = w_alfa >= 0.5
        if shoulder_clip is not None and p["ad"] == shoulder_clip["parca"]:
            u = (xx - shoulder_clip["pivot"][0]) * shoulder_clip["yon"][0] + (
                yy - shoulder_clip["pivot"][1]
            ) * shoulder_clip["yon"][1]
            forbidden = u < -1.0
            # A source arm pixel kept inside the armhole never fronts the garment (only at a source whose body
            # stands where the ground's does)
            if shoulder_clip.get("kok") is not None:
                forbidden = forbidden | (
                    cv2.warpAffine(shoulder_clip["kok"].astype(np.uint8), m_aff, (w, h), flags=cv2.INTER_NEAREST) > 0
                )
            forbidden = forbidden & ~shoulder_clip["izinli"]
            w_mask = w_mask & ~forbidden
            w_alfa = w_alfa * (~forbidden)
        if joint_round is not None and p["ad"] in joint_round["yonler"]:
            c_e, dir_e = joint_round["merkez"], joint_round["yonler"][p["ad"]]
            past = ((xx - c_e[0]) * dir_e[0] + (yy - c_e[1]) * dir_e[1] < 0) & (
                (xx - c_e[0]) ** 2 + (yy - c_e[1]) ** 2 > joint_round["r"] ** 2
            )
            w_mask = w_mask & ~past
            w_alfa = w_alfa * (~past)
        placed.append((p["ad"], w_rgb, w_mask, w_alfa, coloured))
        union |= w_mask

    erased = np.zeros((h, w), bool)
    if leftover is not None:
        for part in masks.components(leftover["maske"] & ~union, min_area=1):
            border = masks.dilate(part, ctx.px(3)) & ~part
            if 3 * (border & leftover["giysi"]).sum() < (border & leftover["arka_plan"]).sum():
                render[part] = 255
                erased |= part

    cap_union = np.zeros((h, w), bool)
    cap_rgb = np.zeros((h, w, 3), np.uint8)
    for cap in caps:
        c = affine_matrix(*_resolve_tf(cap["tf"])) @ np.array([cap["merkez"][0], cap["merkez"][1], 1.0])
        disk = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= float(cap["r"]) ** 2
        cap_rgb[disk] = cap["renk"]
        cap_union |= disk
    # A cap only fills the pieces' concave wedge, never humps out
    filled = (
        cv2.morphologyEx(
            union.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(41), ctx.odd(41))),
        )
        > 0
    )
    cap_open = cap_union & ~union & filled
    if cap_except is not None:
        # A fold open at the source too is no wedge, takes neither skin fill nor contour
        cap_open = cap_open & ~cap_except
    if joint_round is not None:
        c_e = joint_round["merkez"]
        cap_open = cap_open | (((xx - c_e[0]) ** 2 + (yy - c_e[1]) ** 2 <= joint_round["r"] ** 2) & ~union)
        cap_rgb[cap_open & ~cap_union] = caps[0]["renk"]
        cap_union = cap_union | cap_open
    if cap_open.any():
        render[cap_open] = cap_rgb[cap_open]

    # Pieces join sharply with each other, the soft edge only on the arm's united silhouette
    arm_rgb = render.copy()
    for _, w_rgb, w_mask, _, coloured in placed:
        spilling = masks.dilate(w_mask, ctx.px(2)) & coloured
        arm_rgb[spilling] = w_rgb[spilling]
    alfa = np.zeros((h, w), np.float32)
    inner = masks.erode(union | cap_open, ctx.px(2))
    for _, w_rgb, w_mask, w_alfa, _ in placed:
        arm_rgb[w_mask] = w_rgb[w_mask]
        alfa = np.maximum(alfa, w_alfa)
    alfa[inner] = 1.0
    mix = alfa[..., None]
    render = (mix * arm_rgb + (1.0 - mix) * render + 0.5).astype(np.uint8)

    # Around the joint the united silhouette's outer band is an unbroken line: the two pieces' contour ends meet
    silhouette = union | cap_open
    contour_where = masks.dilate(cap_union, ctx.px(12))
    if shoulder_disk is not None:
        silhouette = silhouette | (shoulder_disk["maske"] & ~erased)
        # The disk's garment-neighbouring edge carries the garment's own line; contour only facing the background
        # No second line on an edge already lined on the ground
        contour_where = contour_where | (
            masks.dilate(shoulder_disk["maske"], ctx.px(12))
            & ~masks.dilate(shoulder_disk["giysi"], ctx.px(14))
            & ~masks.dilate(measure.ink_mask(ground), ctx.px(8))
            # Where the disk wears the garment colour its garment-facing edge is an inner border: line only
            # facing the paper
            & (masks.dilate(shoulder_disk["kagit"], ctx.px(6)) if "kagit" in shoulder_disk else True)
        )
    band = silhouette & ~masks.erode(silhouette, thick - 1) & contour_where
    skin_below = np.zeros((h, w), bool)
    if skin is not None:
        # Skin family: the armpit fold's dark skin is hide too; hide over hide sits lineless (H8)
        ground_i = ground.astype(np.int32)
        skin_below = (np.abs(ground_i - np.asarray(skin, np.int32)).sum(axis=2) < 75) | (
            (ground_i[..., 0] - ground_i[..., 2] > 30) & (ground_i[..., 0] > 120)
        )
        band = band & ~masks.dilate(skin_below & ~silhouette, thick)
    render[band & ~measure.ink_mask(render)] = line_colour
    if skin is not None:
        # Continuity on the seen border (H2): where the arm's skin touches a non-hide fill lineless the border
        # takes a line; cut or source no matter
        r_i = render.astype(np.int32)
        r_skin = (r_i[..., 0] - r_i[..., 2] > 30) & (r_i[..., 0] > 120)
        ink = measure.ink_mask(render)
        outside = masks.dilate(union, ctx.px(2)) & ~union & ~r_skin & ~ink & (render.min(axis=2) >= 150)
        missing = outside & ~masks.dilate(ink, thick // 2 + 1)
        stroked = masks.dilate(missing, thick) & union & ~masks.erode(union, thick // 2 + 1) & r_skin
        render[stroked] = line_colour
    if shoulder_disk is not None and "kagit" in shoulder_disk:
        # In the opening region the body never touches the paper lineless: the side contour from under the arm
        # comes from the silhouette (H2, the body's left contour)
        paper = canonical_paper(ctx, render)
        near = masks.dilate(shoulder_disk["maske"], ctx.px(90)) & ~union
        ink = measure.ink_mask(render)
        border = masks.dilate(paper, ctx.px(2)) & ~paper & ~ink & near
        # Only the long gap: edge softening beside the line and lone pixels are no gap, drawn they speckle the
        # silhouette
        missing = border & ~masks.dilate(ink, thick + 2)
        for part in masks.components(missing, min_area=ctx.area(40)):
            render[masks.dilate(part, thick // 2) & ~paper & near] = line_colour
        for p in pieces:
            if p.get("kesik") is not None:
                render[p["kesik"]() & ~skin_below] = line_colour
    return {
        "render": render,
        "kol_mask": silhouette,
        "parcalar": {name: m for name, _, m, _, _ in placed},
        "kapak_acik": cap_open,
        "silinen": erased,
    }
