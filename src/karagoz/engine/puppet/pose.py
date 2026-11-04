"""Poses for the layered puppet (package K-1).

A :class:`Pose` puts names to numbers: per-part joint angles (degrees,
relative to the parent, rest = 0), a root similarity transform and optional
z overrides. :func:`render_pose` draws the puppet in z order including the
floor-shadow layer from K-0d; K-2 reuses the same function for inbetweens.

Angle convention matches ``build.rotate_rgba`` (``cv2.getRotationMatrix2D``):
positive angles turn counter-clockwise on screen (image grid, x right,
y down). Part names are data from the puppet, never hard-coded here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from karagoz.engine.puppet.model import Puppet


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pose(_Strict):
    """One fitted placement of a view puppet.

    ``angles`` maps part name -> degrees relative to the parent (rest = 0;
    the root has no angle, its placement is the root similarity). The root
    maps reference pixels to output pixels as: rotate ``rotation`` degrees
    about the root pivot, scale by ``scale`` about the same pivot, then shift
    by ``(tx, ty)`` output pixels. ``canvas`` is the output size ``[w, h]``;
    ``[0, 0]`` means the puppet's own canvas. ``scores`` / ``confidences``
    are per-part fit quality, filled by ``fit``; ``overlap`` is the fraction
    of each part's pixels another part also covers (order-free); ``view``
    names the set view. ``offset`` maps part name -> [dx, dy] joint slack
    in reference pixels (default empty = every joint on its modelled
    pivot; K-2 interpolates it like the angles). ``chain_notes`` maps
    chain leaf name -> reviewer note (e.g. why an unseated chain was
    kept unfolded).
    """

    view: str = ""
    angles: dict[str, float] = Field(default_factory=dict)
    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    z_override: dict[str, int] = Field(default_factory=dict)
    offset: dict[str, list[float]] = Field(default_factory=dict)
    chain_notes: dict[str, str] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    confidences: dict[str, float] = Field(default_factory=dict)
    overlap: dict[str, float] = Field(default_factory=dict)
    canvas: list[int] = Field(default_factory=lambda: [0, 0])


def children_of(puppet: Puppet) -> dict[str, list[str]]:
    """Parent name -> child names in puppet.json order."""
    kids: dict[str, list[str]] = {name: [] for name in puppet.parts}
    for name, part in puppet.parts.items():
        if part.parent is not None and part.parent in kids:
            kids[part.parent].append(name)
    return kids


def tree_order(puppet: Puppet) -> list[str]:
    """Part names with parents before children (root first)."""
    kids = children_of(puppet)
    order: list[str] = []
    stack: list[str] = [puppet.root]
    while stack:
        node = stack.pop(0)
        if node in order:
            continue
        order.append(node)
        stack.extend(kids.get(node, []))
    for name in puppet.parts:
        if name not in order:
            order.append(name)
    return order


def ancestors_of(puppet: Puppet, part: str) -> list[str]:
    """Names from the root's child down to ``part``'s parent (top first)."""
    chain: list[str] = []
    node: str | None = puppet.parts[part].parent
    while node is not None and node in puppet.parts:
        chain.append(node)
        node = puppet.parts[node].parent
    chain.reverse()
    return chain


def chains(puppet: Puppet) -> dict[str, list[str]]:
    """Joint chains derived from the tree.

    A chain is a path from a child of the root to a leaf, named after the
    leaf part (``head``, ``hand_r``, ``foot_l``, ...). Single-part puppets
    yield one chain with the root alone. Hidden parts stay in their chain;
    rendering and scoring skip them (they have no pixels).
    """
    kids = children_of(puppet)
    leaves = [n for n in puppet.parts if not kids.get(n)]
    if puppet.root in leaves:
        return {puppet.root: [puppet.root]}
    out: dict[str, list[str]] = {}
    for leaf in sorted(leaves):
        path: list[str] = [leaf]
        node: str | None = puppet.parts[leaf].parent
        while node is not None and node in puppet.parts:
            path.append(node)
            node = puppet.parts[node].parent
        path.reverse()
        if path and path[0] == puppet.root:
            path = path[1:]
        if path:
            out[leaf] = path
    return out


def _rot3(cx: float, cy: float, angle_deg: float, scale: float = 1.0) -> NDArray[np.float64]:
    """3x3 rotation (plus uniform scale) about ``(cx, cy)``.

    Same sign convention as ``cv2.getRotationMatrix2D``: the 2x3 top rows
    equal its output for the same inputs.
    """
    rad = math.radians(float(angle_deg))
    c = math.cos(rad) * float(scale)
    s = math.sin(rad) * float(scale)
    return np.array(
        [
            [c, s, (1.0 - c) * cx - s * cy],
            [-s, c, s * cx + (1.0 - c) * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _trans3(dx: float, dy: float) -> NDArray[np.float64]:
    return np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], dtype=np.float64)


def root_matrix3(puppet: Puppet, pose: Pose) -> NDArray[np.float64]:
    """Root similarity as a 3x3 matrix (reference -> output pixels)."""
    pivot = puppet.parts[puppet.root].pivot
    cx, cy = float(pivot[0]), float(pivot[1])
    if not math.isfinite(pose.scale) or pose.scale <= 0:
        raise ValueError(f"pose scale must be positive, got {pose.scale}")
    inner = _rot3(cx, cy, pose.rotation, pose.scale)
    return _trans3(pose.tx, pose.ty) @ inner


def _offset_of(pose: Pose, part: str) -> tuple[float, float]:
    """Joint slack offset of ``part`` in reference pixels (default origin).

    The offset slides the joint: the part and everything below it move
    with it, ancestors are unaffected.
    """
    value = pose.offset.get(part)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return (0.0, 0.0)


def part_matrix3(puppet: Puppet, pose: Pose, part: str) -> NDArray[np.float64]:
    """Total 3x3 matrix for ``part`` (reference -> output pixels).

    The part's own rotation applies first, then each ancestor's rotation
    from the parent up to the root's child, then the root similarity.
    A part's slack offset translates its joint in the parent frame, so
    the part and everything below it shift with it. Empty offsets take
    the exact old code path.
    """
    mat = _rot3(
        float(puppet.parts[part].pivot[0]),
        float(puppet.parts[part].pivot[1]),
        float(pose.angles.get(part, 0.0)),
    )
    ox, oy = _offset_of(pose, part)
    if ox or oy:
        mat = _trans3(ox, oy) @ mat
    for anc in reversed(ancestors_of(puppet, part)):
        piv = puppet.parts[anc].pivot
        mat = _rot3(float(piv[0]), float(piv[1]), float(pose.angles.get(anc, 0.0))) @ mat
        ox, oy = _offset_of(pose, anc)
        if ox or oy:
            mat = _trans3(ox, oy) @ mat
    return root_matrix3(puppet, pose) @ mat


def z_order(puppet: Puppet, pose: Pose) -> list[str]:
    """Visible part names bottom-first (low z first, puppet order breaks ties)."""
    names = list(puppet.parts)
    pos = {name: i for i, name in enumerate(names)}

    def key(name: str) -> tuple[int, int]:
        part = puppet.parts[name]
        return (int(pose.z_override.get(name, part.z)), pos[name])

    return sorted(names, key=key)


def output_size(puppet: Puppet, pose: Pose) -> tuple[int, int]:
    """Output canvas ``(w, h)``: the pose canvas, else the puppet canvas."""
    if len(pose.canvas) == 2 and pose.canvas[0] > 0 and pose.canvas[1] > 0:
        return (int(pose.canvas[0]), int(pose.canvas[1]))
    return (int(puppet.canvas[0]), int(puppet.canvas[1]))


def render_pose(
    puppet: Puppet,
    pose: Pose,
    images: dict[str, NDArray[np.uint8]] | None = None,
) -> NDArray[np.uint8]:
    """Draw the puppet in z order as RGBA (transparent paper).

    ``images`` maps part name -> RGBA in reference pixels (plus optional
    ``"_shadow"`` floor layer, warped by the root similarity only and drawn
    first). Parts missing from ``images`` (hidden parts) are skipped.
    Per-part slack offsets (``Pose.offset``) shift joints through
    :func:`part_matrix3`, so rendering follows fitted offsets exactly.
    """
    if images is None:
        raise ValueError("render_pose needs part images (name -> RGBA); the puppet.json has no pixels")
    width, height = output_size(puppet, pose)
    canvas = np.zeros((height, width, 4), np.uint8)
    root = root_matrix3(puppet, pose)
    shadow = images.get("_shadow")
    if shadow is not None and shadow.ndim == 3 and shadow.shape[2] == 4:
        warped = np.asarray(
            cv2.warpAffine(
                np.asarray(shadow, dtype=np.uint8),
                root[:2].astype(np.float64),
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderValue=(0, 0, 0, 0),
            ),
            dtype=np.uint8,
        )
        cover = warped[..., 3] > 127
        canvas[cover] = warped[cover]
    for name in z_order(puppet, pose):
        part = puppet.parts[name]
        if part.hidden:
            continue
        img = images.get(name)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        mat = part_matrix3(puppet, pose, name)
        warped = np.asarray(
            cv2.warpAffine(
                np.asarray(img, dtype=np.uint8),
                mat[:2].astype(np.float64),
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderValue=(0, 0, 0, 0),
            ),
            dtype=np.uint8,
        )
        cover = warped[..., 3] > 127
        canvas[cover] = warped[cover]
    return canvas
