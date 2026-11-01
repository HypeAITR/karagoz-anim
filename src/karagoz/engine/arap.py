# Source: Meta AnimatedDrawings, animated_drawings/model/arap.py (https://github.com/facebookresearch/AnimatedDrawings)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# WP-R14 changes (same maths, Igarashi & Igarashi 2009 two-stage solve):
# - Sparse setup instead of dense A1/A2/G (gigabytes at thousands of vertices, `det` overflow); the normal
#   equations are factored once with `splu`, singularity prevented with a tiny diagonal instead of a `det` loop.
# - `w` can be given per pin (`weights`): a low-confidence point pulls less. Default is the original behaviour.
# - float64 (original was float32); `np.bool8`, `sp._csr`, `print` and `plot_mesh` removed. A pin outside the
#   mesh is dropped (as in the original).
"""ARAP: vertex positions from new pin positions, keeping edges as rigid as possible."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]


class ARAP:
    """As-rigid-as-possible deformer over a fixed triangle mesh."""

    def __init__(
        self,
        pins_xy: np.ndarray,
        triangles: np.ndarray,
        vertices: np.ndarray,
        w: float = 1000.0,
        weights: np.ndarray | None = None,
    ) -> None:
        """Pin setup: ``pins_xy`` [P,2] initial pin positions; ``triangles`` [T,3]; ``vertices`` [V,2].

        ``w`` is the base pin weight; ``weights`` [P] is an optional per-pin multiplier.
        """
        self.vertices = np.asarray(vertices, np.float64).copy()
        triangles = np.asarray(triangles, np.int64)
        pins_xy = np.asarray(pins_xy, np.float64)
        per_pin = np.ones(len(pins_xy)) if weights is None else np.asarray(weights, np.float64)

        edge_set: set[tuple[int, int]] = set()
        neighbours: dict[int, set[int]] = defaultdict(set)
        for v0, v1, v2 in triangles.tolist():
            edge_set |= {tuple(sorted((v0, v1))), tuple(sorted((v1, v2))), tuple(sorted((v2, v0)))}
            neighbours[v0] |= {v1, v2}
            neighbours[v1] |= {v2, v0}
            neighbours[v2] |= {v0, v1}
        self.edge_vertex_indices = sorted(edge_set)
        self.edge_count, self.vertex_count = len(self.edge_vertex_indices), len(self.vertices)
        self.edge_vectors = np.array([self.vertices[j] - self.vertices[i] for i, j in self.edge_vertex_indices])

        bary, self.pin_mask = self.barycentric(pins_xy, self.vertices, triangles)
        self.pin_count = int(self.pin_mask.sum())
        self.pin_weights = w * per_pin[self.pin_mask]

        # A1 and G have 2 rows per edge; A2 has 1 row per edge. COO duplicates sum (original `-=`).
        rows1: list[int] = []
        cols1: list[int] = []
        data1: list[float] = []
        rows_g: list[int] = []
        cols_g: list[int] = []
        data_g: list[float] = []

        def add_block(rows: list[int], cols: list[int], data: list[float], row: int, col: int, mat: np.ndarray) -> None:
            """Append a nonzero 2x2 block."""
            for a in range(2):
                for b in range(2):
                    if mat[a, b] != 0.0:
                        rows.append(row + a)
                        cols.append(col + b)
                        data.append(float(mat[a, b]))

        identity = np.identity(2)
        for k, (vi, vj) in enumerate(self.edge_vertex_indices):
            add_block(rows1, cols1, data1, 2 * k, 2 * vi, -identity)
            add_block(rows1, cols1, data1, 2 * k, 2 * vj, identity)
            ring = [vi, vj] + sorted(neighbours[vi] & neighbours[vj])
            g_rows: list[tuple[float, float]] = []
            for v in ring[1:]:
                vx, vy = self.vertices[v] - self.vertices[vi]
                g_rows.extend(((vx, vy), (vy, -vx)))
            g_k = np.array(g_rows)
            g_star = np.linalg.pinv(g_k.T @ g_k) @ g_k.T
            ex, ey = self.edge_vectors[k]
            edge = np.array([[ex, ey], [ey, -ex]])
            n = len(ring) - 1
            g = g_star @ np.hstack([np.tile(-identity, (n, 1)), np.identity(2 * n)])
            h = edge @ g
            for order, v in enumerate(ring):
                add_block(rows1, cols1, data1, 2 * k, 2 * v, -h[:, 2 * order : 2 * order + 2])
                add_block(rows_g, cols_g, data_g, 2 * k, 2 * v, g[:, 2 * order : 2 * order + 2])
        rows2: list[int] = []
        cols2: list[int] = []
        data2: list[float] = []
        for k, (vi, vj) in enumerate(self.edge_vertex_indices):
            rows2 += [k, k]
            cols2 += [vi, vj]
            data2 += [-1.0, 1.0]
        for p, b in enumerate(bary):
            for v, vw in b:
                rows1 += [2 * self.edge_count + 2 * p, 2 * self.edge_count + 2 * p + 1]
                cols1 += [2 * v, 2 * v + 1]
                data1 += [self.pin_weights[p] * vw] * 2
                rows2.append(self.edge_count + p)
                cols2.append(v)
                data2.append(self.pin_weights[p] * vw)
        a1 = sp.csr_matrix(
            (data1, (rows1, cols1)), shape=(2 * (self.edge_count + self.pin_count), 2 * self.vertex_count)
        )
        a2 = sp.csr_matrix((data2, (rows2, cols2)), shape=(self.edge_count + self.pin_count, self.vertex_count))
        self.g_matrix = sp.csr_matrix((data_g, (rows_g, cols_g)), shape=(2 * self.edge_count, 2 * self.vertex_count))
        self.transposed_a1 = a1.T.tocsr()
        self.transposed_a2 = a2.T.tocsr()
        self.lu1 = spla.splu((self.transposed_a1 @ a1 + 1e-8 * sp.identity(2 * self.vertex_count)).tocsc())
        self.lu2 = spla.splu((self.transposed_a2 @ a2 + 1e-8 * sp.identity(self.vertex_count)).tocsc())

    def solve(self, pins_xy: np.ndarray) -> np.ndarray:
        """New pin positions (setup order, [P,2]) -> vertices [V,2]."""
        pins = np.asarray(pins_xy, np.float64)[self.pin_mask]
        rhs1 = np.hstack([np.zeros(2 * self.edge_count), (self.pin_weights[:, None] * pins).ravel()])
        v1 = self.lu1.solve(self.transposed_a1 @ rhs1)
        fitted = (self.g_matrix @ v1).reshape(-1, 2)
        norm = np.hypot(fitted[:, 0], fitted[:, 1])
        norm[norm == 0] = 1.0
        c, s = fitted[:, 0] / norm, fitted[:, 1] / norm
        edges = self.edge_vectors
        top = np.column_stack([c * edges[:, 0] + s * edges[:, 1], -s * edges[:, 0] + c * edges[:, 1]])
        rhs2 = np.vstack([top, self.pin_weights[:, None] * pins])
        return np.column_stack(
            [self.lu2.solve(self.transposed_a2 @ rhs2[:, 0]), self.lu2.solve(self.transposed_a2 @ rhs2[:, 1])]
        )

    @staticmethod
    def barycentric(
        points: np.ndarray, vertices: np.ndarray, triangles: np.ndarray
    ) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
        """Barycentric weights of the first triangle containing (or edging) each point; off-mesh is False."""
        corner = vertices[triangles[:, 0]]
        edge0, edge1 = vertices[triangles[:, 1]] - corner, vertices[triangles[:, 2]] - corner
        det = edge0[:, 0] * edge1[:, 1] - edge0[:, 1] * edge1[:, 0]
        det = np.where(det == 0, np.inf, det)
        output: list[list[tuple[int, float]]] = []
        mask: list[bool] = []
        for p in points:
            q = p - corner
            u = (q[:, 0] * edge1[:, 1] - q[:, 1] * edge1[:, 0]) / det
            v = (edge0[:, 0] * q[:, 1] - edge0[:, 1] * q[:, 0]) / det
            inside = np.flatnonzero((u >= 0) & (v >= 0) & (u + v <= 1) & np.isfinite(det))
            if not len(inside):
                mask.append(False)
                continue
            t = inside[0]
            output.append(list(zip(triangles[t].tolist(), (1.0 - u[t] - v[t], u[t], v[t]), strict=True)))
            mask.append(True)
        return output, np.array(mask, bool)
