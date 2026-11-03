"""Joint timing and angle interpolation between two fitted poses (package K-2a).

Agents never write coordinates: they name poses, this module turns two poses
into the frames between them. Timing (:func:`ease`, :func:`frame_times`) is
separate from placement (:func:`interpolate` takes an already-eased ``t``);
:func:`interpolate_sequence` joins them and staggers changed chains
(follow-through / overlapping action) so child joints break after their parent
but still land on the last frame.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.pose import Pose, chains

_EASE_KINDS = ("inout", "linear", "in", "out")


def ease(t: float, kind: str = "inout") -> float:
    """Eased time for ``t`` in [0, 1] (clamped).

    ``"linear"`` is identity, ``"in"`` is quadratic (zero slope at the start),
    ``"out"`` is mirrored quadratic (zero slope at the end), ``"inout"`` is
    smoothstep ``3t^2 - 2t^3`` (zero slope at both ends). Every kind maps 0 to
    exactly 0.0 and 1 to exactly 1.0 and is monotone on [0, 1].
    """
    if kind not in _EASE_KINDS:
        raise ValueError(f"unknown ease kind {kind!r}; want one of {_EASE_KINDS}")
    x = min(max(float(t), 0.0), 1.0)
    if kind == "linear":
        return x
    if kind == "in":
        return x * x
    if kind == "out":
        return 1.0 - (1.0 - x) * (1.0 - x)
    return x * x * (3.0 - 2.0 * x)


def frame_times(n: int, kind: str = "inout") -> list[float]:
    """Eased times of ``n`` frames including both ends: ``[0.0, ..., 1.0]``."""
    if n < 2:
        raise ValueError(f"need at least 2 frames, got {n}")
    return [ease(i / (n - 1), kind) for i in range(n)]


def _shortest_delta(ax: float, ay: float) -> float:
    """Signed Y-minus-X angle in [-180, 180): 170 -> -170 is +20 via 180."""
    return ((float(ay) - float(ax) + 180.0) % 360.0) - 180.0


def _offsets_of(pose: Pose) -> dict[str, tuple[float, float]]:
    """Per-part joint offsets, tolerant of the field missing (pre-K-1 Pose).

    Reads ``pose.offsets`` when K-1 has added it; missing parts mean (0, 0).
    """
    raw = getattr(pose, "offsets", None) or {}
    return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _with_offsets(pose: Pose, offsets: dict[str, tuple[float, float]]) -> Pose:
    """Return ``pose`` carrying ``offsets``; empty offsets leave it untouched.

    Works before K-1 adds ``Pose.offsets`` (plain attribute, invisible to
    ``model_dump`` and ``==``) and after (a real field value).
    """
    if not offsets:
        return pose
    if "offsets" in type(pose).model_fields:
        return pose.model_copy(update={"offsets": dict(offsets)})
    out = pose.model_copy()
    object.__setattr__(out, "offsets", dict(offsets))
    return out


def _angle_keys(pose_x: Pose, pose_y: Pose) -> list[str]:
    """Union of both angle dicts (X order, then Y-only keys).

    Locked parts missing from both dicts (e.g. the root) stay missing: their
    value is implicitly 0.0 everywhere, which is exactly X's value, so the end
    frames keep bit-identical ``angles`` dicts.
    """
    return list(pose_x.angles) + [k for k in pose_y.angles if k not in pose_x.angles]


def _carrier_update(pose: Pose) -> dict[str, Any]:
    """Fresh copies of the carrier fields (view, canvas, z-order, fit scores).

    Carrier fields always travel together: from X for ``t < 1``, from Y at the end.
    """
    return {
        "view": pose.view,
        "z_override": dict(pose.z_override),
        "scores": dict(pose.scores),
        "confidences": dict(pose.confidences),
        "overlap": dict(pose.overlap),
        "canvas": list(pose.canvas),
    }


def interpolate(pose_x: Pose, pose_y: Pose, t: float, *, locked: Collection[str] = ()) -> Pose:
    """The inbetween pose at time ``t`` in [0, 1] (clamped).

    Per-part angles travel the shortest angular path; root tx, ty and rotation
    move linearly, scale geometrically; per-part ``offsets`` move linearly
    (missing parts count as (0, 0)). ``locked`` parts keep ``pose_x``'s exact
    value in every frame. The carrier fields (view, canvas, z_override and the
    fit scores) come from ``pose_x`` for ``t < 1`` and from ``pose_y`` at
    ``t == 1``; the end frames are bit-identical copies of X / Y (with locked
    applied), so a 170 -> -170 wrap still lands exactly on Y.
    """
    tt = min(max(float(t), 0.0), 1.0)
    frozen = set(locked)
    off_x = _offsets_of(pose_x)
    off_y = _offsets_of(pose_y)

    if tt <= 0.0:
        first_angles = {k: pose_x.angles.get(k, 0.0) for k in _angle_keys(pose_x, pose_y)}
        out = pose_x.model_copy(update=_carrier_update(pose_x) | {"angles": first_angles})
        return _with_offsets(out, {k: off_x.get(k, (0.0, 0.0)) for k in off_x})
    if tt >= 1.0:
        last_angles = {
            k: (pose_x.angles.get(k, 0.0) if k in frozen else pose_y.angles.get(k, 0.0))
            for k in _angle_keys(pose_x, pose_y)
        }
        out = pose_y.model_copy(
            update=_carrier_update(pose_y)
            | {
                "angles": last_angles,
                "tx": pose_y.tx,
                "ty": pose_y.ty,
                "rotation": pose_y.rotation,
                "scale": pose_y.scale,
            }
        )
        last_merged = {k: off_y.get(k, (0.0, 0.0)) for k in set(off_x) | set(off_y)}
        for k in frozen:
            if k in off_x:
                last_merged[k] = off_x[k]
            elif k in off_y:
                last_merged[k] = (0.0, 0.0)
        return _with_offsets(out, last_merged)

    angles: dict[str, float] = {}
    for k in _angle_keys(pose_x, pose_y):
        if k in frozen:
            angles[k] = pose_x.angles.get(k, 0.0)
            continue
        ax = pose_x.angles.get(k, 0.0)
        angles[k] = ax + _shortest_delta(ax, pose_y.angles.get(k, 0.0)) * tt
    sx, sy = float(pose_x.scale), float(pose_y.scale)
    # Degenerate fitted scale stays linear rather than failing.
    scale = sx * (sy / sx) ** tt if sx > 0.0 and sy > 0.0 else sx + (sy - sx) * tt
    out = pose_x.model_copy(
        update=_carrier_update(pose_x)
        | {
            "angles": angles,
            "tx": pose_x.tx + (pose_y.tx - pose_x.tx) * tt,
            "ty": pose_x.ty + (pose_y.ty - pose_x.ty) * tt,
            "rotation": pose_x.rotation + (pose_y.rotation - pose_x.rotation) * tt,
            "scale": scale,
        }
    )
    mid_merged: dict[str, tuple[float, float]] = {}
    for k in set(off_x) | set(off_y):
        if k in frozen:
            mid_merged[k] = off_x.get(k, (0.0, 0.0))
            continue
        ox0, ox1 = off_x.get(k, (0.0, 0.0))
        oy0, oy1 = off_y.get(k, (0.0, 0.0))
        mid_merged[k] = (ox0 + (oy0 - ox0) * tt, ox1 + (oy1 - ox1) * tt)
    return _with_offsets(out, mid_merged)


def _part_delays(puppet: Puppet, pose_x: Pose, pose_y: Pose) -> dict[str, int]:
    """Frames each part lags: depth below its chain's first moving joint.

    A chain is a root-to-leaf path from :func:`pose.chains`; the first moving
    joint is the first part whose angle or offset differs between X and Y. The
    root itself is never delayed. A part in several chains takes the max depth.
    """
    off_x = _offsets_of(pose_x)
    off_y = _offsets_of(pose_y)

    def moved(part: str) -> bool:
        if pose_x.angles.get(part, 0.0) != pose_y.angles.get(part, 0.0):
            return True
        return off_x.get(part, (0.0, 0.0)) != off_y.get(part, (0.0, 0.0))

    delays: dict[str, int] = {}
    for path in chains(puppet).values():
        first = next((j for j, part in enumerate(path) if moved(part)), None)
        if first is None:
            continue
        for j in range(first + 1, len(path)):
            part = path[j]
            if part == puppet.root:
                continue
            depth = j - first
            if depth > delays.get(part, 0):
                delays[part] = depth
    return delays


def interpolate_sequence(
    puppet: Puppet,
    pose_x: Pose,
    pose_y: Pose,
    n: int,
    *,
    kind: str = "inout",
    follow_through: int = 0,
    locked: Collection[str] = (),
) -> list[Pose]:
    """``n`` poses from X to Y; first equals X, last equals Y (locked: X).

    ``follow_through=k`` staggers each changed chain: a joint at depth ``d``
    below the chain's first moving joint starts ``d * k`` frames late, on a
    stretched curve of the same ``kind`` so it still lands exactly on Y in the
    last frame (frame ``i`` uses local ``(i - d*k) / (n-1 - d*k)``). The root
    similarity always follows the shared clock. Every value stays between X
    and Y: no overshoot.
    """
    if n < 2:
        raise ValueError(f"need at least 2 frames, got {n}")
    if follow_through < 0:
        raise ValueError(f"follow_through must be >= 0, got {follow_through}")
    frozen = set(locked)
    delays = _part_delays(puppet, pose_x, pose_y) if follow_through else {}
    parts = set(pose_x.angles) | set(pose_y.angles) | set(_offsets_of(pose_x)) | set(_offsets_of(pose_y))
    lag = {part: delays.get(part, 0) * follow_through for part in parts}
    times = frame_times(n, kind)
    off_x = _offsets_of(pose_x)
    off_y = _offsets_of(pose_y)

    out: list[Pose] = []
    for i in range(n):
        if i == 0:
            out.append(interpolate(pose_x, pose_y, 0.0, locked=frozen))
            continue
        if i == n - 1:
            out.append(interpolate(pose_x, pose_y, 1.0, locked=frozen))
            continue
        frame = interpolate(pose_x, pose_y, times[i], locked=frozen)
        angles = dict(frame.angles)
        off = dict(_offsets_of(frame))
        for part, steps in lag.items():
            if steps <= 0 or part in frozen:
                continue
            if part not in angles and part not in off:
                continue
            span = (n - 1) - steps
            local = 0.0 if span <= 0 else min(max((i - steps) / span, 0.0), 1.0)
            tp = ease(local, kind)
            if part in angles:
                ax = pose_x.angles.get(part, 0.0)
                angles[part] = ax + _shortest_delta(ax, pose_y.angles.get(part, 0.0)) * tp
            if part in off:
                ox0, ox1 = off_x.get(part, (0.0, 0.0))
                oy0, oy1 = off_y.get(part, (0.0, 0.0))
                off[part] = (ox0 + (oy0 - ox0) * tp, ox1 + (oy1 - ox1) * tp)
        frame = frame.model_copy(update={"angles": angles})
        out.append(_with_offsets(frame, off))
    return out
