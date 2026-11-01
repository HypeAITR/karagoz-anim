"""Bend from a single source with ARAP. Portable: only numpy, scipy and `arap`; no paths, no characters.

The source drawing is split into layers; each layer reaches the frame on its own mesh or on a rigid
matrix. The source never changes: no cutting and no stitching.

API
    build_mesh(mask, contour_step=8, inner_step=16) -> Mesh
    Layer(name, rgb, alpha, points=None, weights=None, contour_step=8, inner_step=16)
        rgb [H,W,3] uint8, alpha [H,W] 0..1; points [N,2] pins on the source (x, y),
        weights [N] multiplier (low confidence < 1). points None: rigid layer, given a 2x3 matrix.
    Rig(layers, w=1000.0)  list order is BOTTOM to TOP draw order.
    Rig.frame(targets, paper=255) -> Frame
        targets {layer name: [N,2] pin target (points order) | 2x3 matrix}.
        Frame: rgb, vertices {name: [V,2]}, ratio {name: [T] area ratio}.
    Rig.carry(name, vertices, points) -> [N,2]  source points carried to the frame (to attach a layer).
    revealed_texture(rgb, mesh, vertices, target_rgb, region) -> rgb
        Pixels missing from the source: taken from the target drawing where the mesh bent to that
        drawing lands the source pixel, only inside `region`.
    area_ratio(mesh, vertices), pixel_map(src_vertices, dst_vertices, triangles, shape)

A pin outside the mesh is attached to the nearest vertex (with its shift, so source = target keeps
the identity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ni  # type: ignore[import-untyped]
from scipy.spatial import Delaunay  # type: ignore[import-untyped]

from karagoz.engine.arap import ARAP

SAMPLES = np.array([[1, 1, 1], [4, 1, 1], [1, 4, 1], [1, 1, 4], [2, 2, 1], [2, 1, 2], [1, 2, 2]], float)
SAMPLES = SAMPLES / SAMPLES.sum(axis=1, keepdims=True)


@dataclass
class Mesh:
    """Triangle mesh over one layer."""

    vertices: np.ndarray  # [V,2] x, y
    triangles: np.ndarray  # [T,3]


def _thin_points(points: np.ndarray, step: float) -> np.ndarray:
    """Keep points at least `step` apart; with a grid, every dropped point is within `step` of a kept one."""
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    kept: list[int] = []
    step2 = step * step
    for i, (x, y) in enumerate(points.tolist()):
        gx, gy = int(x // step), int(y // step)
        nearby = [q for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in grid.get((gx + dx, gy + dy), ())]
        if any((qx - x) ** 2 + (qy - y) ** 2 < step2 for qx, qy in nearby):
            continue
        grid.setdefault((gx, gy), []).append((x, y))
        kept.append(i)
    return points[kept]


def build_mesh(mask: np.ndarray, contour_step: float = 8, inner_step: float = 16) -> Mesh:
    """Dilated-by-1 boundary (so the outline is not cut), inner grid, Delaunay; drop off-mask triangles, no gap."""
    grown = ni.binary_dilation(mask > 0, iterations=1)
    border = grown & ~ni.binary_erosion(grown)
    ys, xs = np.nonzero(border)
    # No border pixel stays further than `contour_step` from the nearest point: thin with a smaller step.
    outer = _thin_points(np.column_stack([xs, ys]).astype(float), 0.75 * contour_step)
    dist = ni.distance_transform_edt(grown)
    grid_y, grid_x = np.mgrid[0 : grown.shape[0] : inner_step, 0 : grown.shape[1] : inner_step]
    grid_y, grid_x = grid_y.astype(int).ravel(), grid_x.astype(int).ravel()
    inner_mask = dist[grid_y, grid_x] > 0.6 * contour_step
    points = np.vstack([outer, np.column_stack([grid_x[inner_mask], grid_y[inner_mask]]).astype(float)])
    found = Delaunay(points)
    tris = found.simplices
    sampled = np.einsum("ij,tjk->tik", SAMPLES, points[tris])  # [T,7,2]
    idx = np.clip(sampled.round().astype(int), 0, [grown.shape[1] - 1, grown.shape[0] - 1])
    keep = grown[idx[..., 1], idx[..., 0]].mean(axis=1) >= 0.5
    my, mx = np.nonzero(mask > 0)
    simplex = found.find_simplex(np.column_stack([mx, my]).astype(float))
    keep[np.unique(simplex[simplex >= 0])] = True
    e0, e1 = points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]]
    keep &= np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) > 1e-3
    tris = tris[keep]
    used = np.unique(tris)
    remap = -np.ones(len(points), int)
    remap[used] = np.arange(len(used))
    return Mesh(points[used], remap[tris])


def area_ratio(mesh: Mesh, vertices: np.ndarray) -> np.ndarray:
    """Per-triangle deformed area over source area."""

    def area(q: np.ndarray) -> np.ndarray:
        a, b = q[mesh.triangles[:, 1]] - q[mesh.triangles[:, 0]], q[mesh.triangles[:, 2]] - q[mesh.triangles[:, 0]]
        values: np.ndarray = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        return values

    ratio: np.ndarray = area(vertices) / area(mesh.vertices)
    return ratio


def pixel_map(
    src_vertices: np.ndarray, dst_vertices: np.ndarray, triangles: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Source position for every target pixel: (u, v) [H,W]; NaN where no triangle lands."""
    h, w = shape
    u, v = np.full(shape, np.nan), np.full(shape, np.nan)
    for t in triangles:
        a, b, c = dst_vertices[t]
        x0, x1 = int(np.floor(min(a[0], b[0], c[0]))), int(np.ceil(max(a[0], b[0], c[0])))
        y0, y1 = int(np.floor(min(a[1], b[1], c[1]))), int(np.ceil(max(a[1], b[1], c[1])))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        e1, e2 = b - a, c - a
        d = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(d) < 1e-9:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        qx, qy = xx - a[0], yy - a[1]
        l1 = (qx * e2[1] - qy * e2[0]) / d
        l2 = (e1[0] * qy - e1[1] * qx) / d
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)
        k = src_vertices[t]
        u[y0 : y1 + 1, x0 : x1 + 1][inside] = (k[0, 0] + l1 * (k[1, 0] - k[0, 0]) + l2 * (k[2, 0] - k[0, 0]))[inside]
        v[y0 : y1 + 1, x0 : x1 + 1][inside] = (k[0, 1] + l1 * (k[1, 1] - k[0, 1]) + l2 * (k[2, 1] - k[0, 1]))[inside]
    return u, v


def _sample(splines: list[np.ndarray], u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Prefiltered cubic: one sample keeps the line weight."""
    h, w = u.shape
    output = np.zeros((h, w, len(splines)), np.float32)
    ys, xs = np.nonzero(np.isfinite(u))
    if not len(ys):
        return output
    coords = np.vstack([v[ys, xs], u[ys, xs]])
    for i, s in enumerate(splines):
        output[ys, xs, i] = ni.map_coordinates(s, coords, order=3, prefilter=False, mode="nearest")
    return output


class Layer:
    """One source drawing with its mesh (or a rigid matrix)."""

    def __init__(
        self,
        name: str,
        rgb: np.ndarray,
        alpha: np.ndarray,
        points: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        contour_step: float = 8,
        inner_step: float = 16,
    ) -> None:
        """``rgb`` [H,W,3] uint8, ``alpha`` [H,W] 0..1; ``points`` [N,2] source pins (x, y).

        ``points`` None means a rigid layer, driven by a 2x3 matrix in the frame.
        """
        self.name = name
        self.alpha = np.asarray(alpha, np.float32)
        premultiplied = np.dstack([np.asarray(rgb, np.float32) * self.alpha[..., None], self.alpha * 255.0])
        self.splines = [ni.spline_filter(premultiplied[..., i], order=3, mode="nearest") for i in range(4)]
        self.shape: tuple[int, int] = (self.alpha.shape[0], self.alpha.shape[1])
        self.rigid = points is None
        self.mesh: Mesh | None = None
        self.weights: np.ndarray | None = None
        self.pins: np.ndarray | None = None
        self.shift: np.ndarray | None = None
        self.outside_count = 0
        self.solver: ARAP | None = None
        if self.rigid:
            return
        self.mesh = build_mesh(self.alpha > 0, contour_step, inner_step)
        target = np.asarray(points, float)
        self.weights = np.ones(len(target)) if weights is None else np.asarray(weights, float)
        _, inside = ARAP.barycentric(target, self.mesh.vertices, self.mesh.triangles)
        # An off-mesh pin goes to the nearest vertex; the shift is added to the target too.
        self.pins = target.copy()
        assert self.pins is not None
        for i in np.flatnonzero(~inside):
            self.pins[i] = self.mesh.vertices[np.argmin(((self.mesh.vertices - target[i]) ** 2).sum(axis=1))]
        self.shift = self.pins - target
        self.outside_count = int((~inside).sum())

    def replace_texture(self, rgb: np.ndarray) -> None:
        """Same mesh, new texture (once a revealed region is filled)."""
        premultiplied = np.asarray(rgb, np.float32) * self.alpha[..., None]
        self.splines[:3] = [ni.spline_filter(premultiplied[..., i], order=3, mode="nearest") for i in range(3)]

    def setup(self, w: float) -> None:
        """Build the ARAP solver with base pin weight ``w``."""
        assert self.mesh is not None and self.pins is not None and self.weights is not None
        self.solver = ARAP(self.pins, self.mesh.triangles, self.mesh.vertices, w=w, weights=self.weights)
        if not self.solver.pin_mask.all():
            raise ValueError(f"{self.name}: pin left outside the mesh")

    def solve(self, target: np.ndarray) -> np.ndarray:
        """Pin target [N,2] -> mesh vertices [V,2]."""
        assert self.solver is not None and self.shift is not None
        return self.solver.solve(np.asarray(target, float) + self.shift)

    def render(self, vertices: np.ndarray) -> np.ndarray:
        """Premultiplied layer image [H,W,4] with the mesh at ``vertices``."""
        assert self.mesh is not None
        u, v = pixel_map(self.mesh.vertices, vertices, self.mesh.triangles, self.shape)
        return _sample(self.splines, u, v)

    def render_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Rigid: the source position of a target pixel comes from the inverse matrix."""
        inverse = np.linalg.inv(np.vstack([np.asarray(matrix, float), [0, 0, 1]]))
        yy, xx = np.indices(self.shape)
        u = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
        v = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
        return _sample(self.splines, u, v)


@dataclass
class Frame:
    """One composited frame."""

    rgb: np.ndarray
    vertices: dict[str, np.ndarray] = field(default_factory=dict)
    ratio: dict[str, np.ndarray] = field(default_factory=dict)


class Rig:
    """Layers composited bottom to top."""

    def __init__(self, layers: list[Layer], w: float = 1000.0) -> None:
        """``layers`` list order is the draw order, bottom to top."""
        self.layers = layers
        self.by_name = {layer.name: layer for layer in layers}
        for layer in layers:
            if not layer.rigid:
                layer.setup(w)

    def carry(self, name: str, vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Where source ``points`` land when the ``name`` mesh is at ``vertices``."""
        mesh = self.by_name[name].mesh
        assert mesh is not None
        bary, inside = ARAP.barycentric(np.asarray(points, float), mesh.vertices, mesh.triangles)
        output = np.asarray(points, float).copy()
        j = 0
        for i, known in enumerate(inside):
            if known:
                output[i] = sum(w * vertices[v] for v, w in bary[j])
                j += 1
            else:
                nearest = np.argmin(((mesh.vertices - output[i]) ** 2).sum(axis=1))
                output[i] = output[i] + vertices[nearest] - mesh.vertices[nearest]
        return output

    def solve(self, name: str, target: np.ndarray) -> np.ndarray:
        """Pin target for layer ``name`` -> mesh vertices."""
        return self.by_name[name].solve(target)

    def frame(self, targets: dict[str, np.ndarray], paper: int = 255) -> Frame:
        """Composite a frame from per-layer pin targets or 2x3 matrices."""
        h, w = self.layers[0].shape
        canvas = np.full((h, w, 3), float(paper), np.float32)
        result = Frame(canvas)
        for layer in self.layers:
            if layer.rigid:
                image = layer.render_matrix(targets[layer.name])
            else:
                vertices = layer.solve(targets[layer.name])
                assert layer.mesh is not None
                result.vertices[layer.name] = vertices
                result.ratio[layer.name] = area_ratio(layer.mesh, vertices)
                image = layer.render(vertices)
            a = np.clip(image[..., 3:] / 255.0, 0.0, 1.0)
            canvas = np.clip(image[..., :3], 0, 255 * a) + (1.0 - a) * canvas
        result.rgb = (np.clip(canvas, 0, 255) + 0.5).astype(np.uint8)
        return result


def revealed_texture(
    rgb: np.ndarray, mesh: Mesh, vertices: np.ndarray, target_rgb: np.ndarray, region: np.ndarray
) -> np.ndarray:
    """Fill `region` source pixels from the target drawing; outside the region the source is unchanged."""
    u, v = pixel_map(vertices, mesh.vertices, mesh.triangles, region.shape)
    ys, xs = np.nonzero(np.isfinite(u) & region)
    output = rgb.copy()
    for c in range(3):
        output[ys, xs, c] = np.clip(
            ni.map_coordinates(
                target_rgb[..., c].astype(np.float32), np.vstack([v[ys, xs], u[ys, xs]]), order=1, mode="nearest"
            ),
            0,
            255,
        ).astype(np.uint8)
    return output
