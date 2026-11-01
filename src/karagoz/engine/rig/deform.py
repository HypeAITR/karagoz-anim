"""Mesh and MLS deformers: displacement source -> mesh -> render.

Ported from the experiment's ``deneme/r2/plan/ag_deformer.py``: the triangle
mesh over a mask, the pixel map, premultiplied compositing helpers, the area
ratio, rigid MLS and the identity/LBS sources. Same maths, English
identifiers, each public function noting its experiment name. Thresholds are
module-level ``UPPER_CASE`` with the experiment's values.

A layer is premultiplied RGBA, a mesh (rest vertices + UV + triangles) and a
z-order. The source interface ``Source(frame, rest) -> bent`` maps rest
vertices to bent ones; adding a source never touches the render. Neighbour
triangles share vertices, so an edge pixel gets the same value from either
triangle and no crack forms. A triangle with zero displacement renders its
source pixels as-is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

DENSE_STEP = 15  # experiment: YOGUN_ADIM; triangle edge in the bent region
SPARSE_STEP = 60  # experiment: SEYREK_ADIM
EDGE_STEP = 12  # experiment: KENAR_ADIM
MESH_MARGIN = 4  # experiment: AG_PAY; the mesh border passes this far outside the alpha
OUTSIDE_MAP_VALUE = -10.0

Source = Callable[[int, NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class Mesh:
    """Triangle mesh: rest vertices, pixel texture coordinates, triangles."""

    rest: NDArray[np.float64]  # experiment: kose (N, 2)
    uv: NDArray[np.float64]  # experiment: uv, in pixels (N, 2)
    tri: NDArray[np.int32]  # experiment: ucgen (M, 3)


@dataclass
class Layer:
    """One drawable layer: premultiplied pixels, its mesh, z-order."""

    name: str  # experiment: ad
    pm: NDArray[np.float32]  # (H, W, 4) float32
    mesh: Mesh  # experiment: ag
    z: int = 0


# experiment: pm_kur
def premultiplied(
    rgb: NDArray[np.uint8] | NDArray[np.float32], alpha: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Premultiplied RGBA from RGB + alpha; opaque when no alpha is given. RGB arrives uint8 or float32."""
    a = np.ones(rgb.shape[:2], np.float32) if alpha is None else alpha.astype(np.float32)
    return np.dstack([rgb.astype(np.float32) * a[..., None], a])


def _grid(mask: NDArray[np.bool_], step: int) -> NDArray[np.int64]:
    ys, xs = np.mgrid[0 : mask.shape[0] : step, 0 : mask.shape[1] : step]
    inside = mask[ys, xs]
    return np.column_stack([xs[inside], ys[inside]])


# experiment: ag_kur
def build_mesh(
    mask: NDArray[np.bool_],
    dense: NDArray[np.bool_] | None = None,
    dense_step: int = DENSE_STEP,
    sparse_step: int = SPARSE_STEP,
    extra: NDArray[np.float64] | None = None,
) -> Mesh:
    """Triangle mesh covering all of ``mask``: dense inside ``dense``, sparse elsewhere.

    The border follows the mask with a ``MESH_MARGIN`` margin. ``extra`` are
    points that must be vertices (the source's break lines: a linear triangle
    follows a break only through a vertex).
    """
    h, w = mask.shape
    # A mask cut by the image edge: edge-touching vertices sit on the image corner.
    margin = cv2.dilate(
        mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MESH_MARGIN + 1, 2 * MESH_MARGIN + 1))
    )
    contours, _ = cv2.findContours(margin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points: list[NDArray[Any]] = [_grid(margin > 0, sparse_step)]
    if dense is not None:
        points.append(_grid((margin > 0) & dense, dense_step))
    for contour in contours:
        contour = contour.reshape(-1, 2)
        points.append(contour[::EDGE_STEP])
        hull = cv2.convexHull(contour).reshape(-1, 2)
        points.append(hull)
    if extra is not None and len(extra):
        e = np.round(extra).astype(int)
        e = e[(e[:, 0] >= 0) & (e[:, 0] < w) & (e[:, 1] >= 0) & (e[:, 1] < h)]
        points.append(e[margin[e[:, 1], e[:, 0]] > 0])
    vertices = np.unique(np.concatenate(points), axis=0)
    order = {(int(x), int(y)): i for i, (x, y) in enumerate(vertices)}
    sub = cv2.Subdiv2D((-2, -2, w + 4, h + 4))
    sub.insert([(float(x), float(y)) for x, y in vertices])
    tri = []
    for t in sub.getTriangleList():
        a = (int(round(t[0])), int(round(t[1])))
        b = (int(round(t[2])), int(round(t[3])))
        c = (int(round(t[4])), int(round(t[5])))
        if a not in order or b not in order or c not in order:
            continue
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if area == 0:
            continue
        # A triangle untouched by the mask (a concave part) is dropped.
        x0, x1 = min(a[0], b[0], c[0]), max(a[0], b[0], c[0]) + 1
        y0, y1 = min(a[1], b[1], c[1]), max(a[1], b[1], c[1]) + 1
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(
            local, np.array([[a[0] - x0, a[1] - y0], [b[0] - x0, b[1] - y0], [c[0] - x0, c[1] - y0]], np.int32), 1
        )
        if not (local.astype(bool) & (margin[y0:y1, x0:x1] > 0)).any():
            continue
        tri.append((order[a], order[b], order[c]) if area > 0 else (order[a], order[c], order[b]))
    rest = vertices.astype(np.float64)
    return Mesh(rest=rest, uv=rest.copy(), tri=np.array(tri, np.int32))


# experiment: harita
def flow_map(
    mesh: Mesh, target: NDArray[np.float64], shape: tuple[int, int]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Target pixel -> texture coordinate from the bent vertices. Outside the mesh is -10 (transparent in remap)."""
    h, w = shape
    mx = np.full((h, w), OUTSIDE_MAP_VALUE, np.float32)
    my = np.full((h, w), OUTSIDE_MAP_VALUE, np.float32)
    for i, j, k in mesh.tri:
        ax, ay = target[i]
        bx, by = target[j]
        cx, cy = target[k]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-9:
            continue
        x0, x1 = max(int(np.floor(min(ax, bx, cx))), 0), min(int(np.ceil(max(ax, bx, cx))) + 1, w)
        y0, y1 = max(int(np.floor(min(ay, by, cy))), 0), min(int(np.ceil(max(ay, by, cy))) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
        w0 = ((bx - xx) * (cy - yy) - (by - yy) * (cx - xx)) / area
        w1 = ((cx - xx) * (ay - yy) - (cy - yy) * (ax - xx)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        u = w0 * mesh.uv[i, 0] + w1 * mesh.uv[j, 0] + w2 * mesh.uv[k, 0]
        v = w0 * mesh.uv[i, 1] + w1 * mesh.uv[j, 1] + w2 * mesh.uv[k, 1]
        mx[y0:y1, x0:x1][inside] = u[inside]
        my[y0:y1, x0:x1][inside] = v[inside]
    return mx, my


# experiment: bas
def warp(pm: NDArray[np.float32], mesh: Mesh, target: NDArray[np.float64]) -> NDArray[np.float32]:
    """Draw the layer with the bent mesh; the result is premultiplied RGBA."""
    mx, my = flow_map(mesh, target, pm.shape[:2])
    out = cv2.remap(pm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return cast(NDArray[np.float32], out)


# experiment: coz
def unpremultiply(pm: NDArray[np.float32]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Premultiplied RGBA -> (uint8 rgb, alpha). A near-zero-alpha pixel's colour is meaningless, stays 0."""
    a = pm[..., 3]
    rgb = np.zeros(pm.shape[:2] + (3,), np.float32)
    filled = a > 1e-3
    rgb[filled] = pm[..., :3][filled] / a[filled][:, None]
    return (np.clip(rgb, 0, 255) + 0.5).astype(np.uint8), a


# experiment: alan_orani
def area_ratio(mesh: Mesh, target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-triangle signed area ratio (bent / rest): the Jacobian determinant. Negative means folding."""

    def area(p: NDArray[np.float64]) -> NDArray[np.float64]:
        a, b, c = p[mesh.tri[:, 0]], p[mesh.tri[:, 1]], p[mesh.tri[:, 2]]
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    return area(target) / area(mesh.rest)


# experiment: mls_sert
def mls_rigid(vertices: NDArray[np.float64], p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rigid moving least squares: each vertex moves with the proximity-weighted best rotation of the controls.

    Rotates locally instead of bending the line; a control point lands exactly on its target.
    """
    d2 = ((vertices[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    w = 1.0 / np.maximum(d2, 1e-6)
    total = w.sum(axis=1, keepdims=True)
    p_mean = (w[..., None] * p[None]).sum(axis=1) / total
    q_mean = (w[..., None] * q[None]).sum(axis=1) / total
    ph = p[None] - p_mean[:, None]
    qh = q[None] - q_mean[:, None]
    c = (w * (ph * qh).sum(axis=2)).sum(axis=1)
    s = (w * (-ph[..., 1] * qh[..., 0] + ph[..., 0] * qh[..., 1])).sum(axis=1)
    n = np.maximum(np.hypot(c, s), 1e-12)
    c, s = c / n, s / n
    v = vertices - p_mean
    return np.column_stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]]) + q_mean


# experiment: kimlik_kaynak
def identity_source(frame: int, rest: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero displacement."""
    return rest.copy()


# experiment: lbs_kaynak
def lbs_source(
    matrix: Callable[[int], NDArray[np.float64]],
    pre: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> Source:
    """Single-bone skinning: all weights 1, the bone's 2x3 affine at the frame. ``pre`` shapes the rest first."""

    def source(frame: int, rest: NDArray[np.float64]) -> NDArray[np.float64]:
        p = rest if pre is None else pre(rest)
        return p @ matrix(frame)[:, :2].T + matrix(frame)[:, 2]

    return source
