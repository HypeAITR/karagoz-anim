"""Recompute the pivots of an already-built puppet set (K-0p).

``repivot_set`` copies a built set directory and refreshes every view's
``puppet.json`` pivots plus the ``joints`` section of its ``report.json``
with the current pivot rule. Part images are untouched: no SAM, no image
calls, no rebuild.

Pivots describe the visible joint, so the rule is measured from the visible
geometry, reconstructed from the shipped art: a part's visible mask is its
finished mask minus every higher part's finished mask (higher paint covers
it at rest) minus the seam-win zones where a plate draws over it (the
report's ``seam_cover``). The line art is the dark neutral ink of the rest
recomposition with those same seam wins, which is the reference art the
build measured (recomposition outside the band is 0 on the graded view).
Joints the rule does not fire on keep their built pivots bit-identically.

``paper_px`` is recomputed with the new pivots from the shipped finished
art, mirroring the build (rest with seam wins, rotations without);
``own_px`` is preserved from the build because it needs the completion
intermediates (continued outlines) that a built set does not ship.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.puppet.build import PAPER_MIN, ROTATION_ANGLES, recompose, rotate_rgba, rotation_zone, sweep_radius
from karagoz.engine.puppet.build import subtree_names as build_subtree_names
from karagoz.engine.puppet.model import PartSpec, Puppet, PuppetConfig, PuppetSet
from karagoz.engine.puppet.parts import line_mask
from karagoz.engine.puppet.pivots import compute_pivots


def _load_view(view_dir: Path) -> tuple[Puppet, dict[str, Any], dict[str, NDArray[np.uint8]], dict[str, Any]]:
    """Read a built view: puppet, report, part images and finished masks."""
    puppet = Puppet.model_validate_json((view_dir / "puppet.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = json.loads((view_dir / "report.json").read_text(encoding="utf-8"))
    width, height = int(puppet.canvas[0]), int(puppet.canvas[1])
    images: dict[str, NDArray[np.uint8]] = {}
    masks: dict[str, NDArray[np.bool_]] = {}
    for name, part in puppet.parts.items():
        if part.hidden or not part.file:
            masks[name] = np.zeros((height, width), dtype=np.bool_)
            continue
        rgba: NDArray[np.uint8] = np.asarray(Image.open(view_dir / part.file).convert("RGBA"), dtype=np.uint8)
        images[name] = rgba
        masks[name] = np.asarray(rgba[..., 3] > 127, dtype=np.bool_)
    return puppet, report, images, masks


def _load_wins(view_dir: Path, report: dict[str, Any]) -> dict[str, NDArray[np.bool_]]:
    """Seam-win zones the build saved (plate draws over a dome disc at rest)."""
    wins: dict[str, NDArray[np.bool_]] = {}
    for plate, info in report.get("seam_wins", {}).items():
        path = view_dir / str(info.get("file", ""))
        if not path.is_file():
            continue
        wins[plate] = np.asarray(np.asarray(Image.open(path).convert("L")) > 127, dtype=np.bool_)
    return wins


def _visible_masks(
    puppet: Puppet,
    finished: dict[str, NDArray[np.bool_]],
    wins: dict[str, NDArray[np.bool_]],
    seam_cover: dict[str, list[str]],
) -> dict[str, NDArray[np.bool_]]:
    """Visible mask per part: finished minus higher paint minus seam wins."""
    by_z: dict[int, list[str]] = {}
    for name, part in puppet.parts.items():
        by_z.setdefault(part.z, []).append(name)
    blank = np.zeros_like(next(iter(finished.values())), dtype=np.bool_)
    above: dict[str, NDArray[np.bool_]] = {}
    acc = blank
    for z in sorted(by_z, reverse=True):
        for name in by_z[z]:
            above[name] = acc
        level = blank
        for name in by_z[z]:
            level = np.asarray(level | finished[name], dtype=np.bool_)
        acc = np.asarray(acc | level, dtype=np.bool_)
    visible: dict[str, NDArray[np.bool_]] = {}
    for name in puppet.parts:
        seen = np.asarray(finished[name] & ~above[name], dtype=np.bool_)
        for plate in seam_cover.get(name, []):
            if plate in wins:
                seen = np.asarray(seen & ~wins[plate], dtype=np.bool_)
        visible[name] = seen
    return visible


def _repivot_config(puppet: Puppet) -> PuppetConfig:
    """The joint tree from a built puppet (names, parents, order, hiding)."""
    parts = [
        PartSpec(
            name=name,
            parent=part.parent,
            concepts=["repivot"] if not part.hidden else [],
            z=part.z,
            hidden=part.hidden,
        )
        for name, part in puppet.parts.items()
    ]
    return PuppetConfig(parts=parts)


def repivot_view(view_dir: Path) -> dict[str, dict[str, Any]]:
    """Recompute one built view's pivots in place; return per-joint changes."""
    puppet, report, images, finished = _load_view(view_dir)
    line_width = float(puppet.line_width_px)
    height, width = next(iter(finished.values())).shape
    wins = _load_wins(view_dir, report)
    order = sorted(images, key=lambda n: (puppet.parts[n].z, n))
    rest = recompose({n: images[n] for n in order}, order, wins or None)
    line = np.asarray(line_mask(np.asarray(rest[..., :3], dtype=np.uint8)), dtype=np.bool_)
    visible = _visible_masks(puppet, finished, wins, report.get("seam_cover", {}))
    cfg = _repivot_config(puppet)
    result = compute_pivots(visible, cfg, line, line_width)
    white_rest = np.asarray(rest[..., :3].min(axis=2) >= PAPER_MIN, dtype=np.bool_)
    pre_covered = np.zeros((height, width), dtype=np.bool_)
    for name in order:
        pre_covered = np.asarray(pre_covered | finished[name], dtype=np.bool_)
    joints: dict[str, Any] = report.get("joints", {})
    changes: dict[str, dict[str, Any]] = {}
    for name, part in puppet.parts.items():
        if part.parent is None or part.hidden:
            continue
        old = [float(part.pivot[0]), float(part.pivot[1])]
        entry: dict[str, Any] = joints.get(name, {})
        built_paper = dict(entry.get("paper_px", {}))
        if name in result.domes:
            new = [float(result.pivots[name][0]), float(result.pivots[name][1])]
            part.pivot = [new[0], new[1]]
            part.estimated = False
            entry.update(
                {
                    "parent": part.parent,
                    "pivot": [new[0], new[1]],
                    "radius": float(result.radii[name]),
                }
            )
            changes[name] = {"old": old, "new": new, "moved": True, "dome": result.domes[name]}
        else:
            new = old
            changes[name] = {"old": old, "new": new, "moved": False, "dome": None}
        pivot = (float(new[0]), float(new[1]))
        moving = build_subtree_names(cfg, name)
        sweep_r = sweep_radius(pivot, {n: finished[n] for n in moving}, line_width)
        static = np.zeros((height, width), dtype=np.bool_)
        for other in order:
            if other not in moving:
                static = np.asarray(static | finished[other], dtype=np.bool_)
        zone = rotation_zone((height, width), pivot, sweep_r / 2.0, static)
        paper: dict[str, int] = {}
        for angle in ROTATION_ANGLES:
            turned = {n: (rotate_rgba(images[n], pivot, angle) if n in moving else images[n]) for n in order}
            back = recompose(turned, order)
            white = np.asarray(back[..., :3].min(axis=2) >= PAPER_MIN, dtype=np.bool_)
            paper[f"{angle:+.0f}"] = int(np.logical_and(white, np.logical_and(zone, ~white_rest)).sum())
        entry["paper_px"] = paper
        joints[name] = entry
        changes[name]["paper_built"] = built_paper
        changes[name]["paper_new"] = dict(paper)
        changes[name]["own_px"] = entry.get("own_px")
    report["joints"] = joints
    (view_dir / "puppet.json").write_text(puppet.model_dump_json(indent=1) + "\n", encoding="utf-8")
    (view_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return changes


def repivot_set(set_dir: Path, out_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Copy a built set and refresh every view's pivots; return the changes."""
    src = Path(set_dir)
    dst = Path(out_dir)
    if not (src / "set.json").is_file():
        raise ValueError(f"no set.json at {src / 'set.json'}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    pset = PuppetSet.model_validate_json((dst / "set.json").read_text(encoding="utf-8"))
    changes: dict[str, dict[str, dict[str, Any]]] = {}
    for group in ("views", "head_views"):
        for name, entry in getattr(pset, group).items():
            changes[f"{group}/{name}"] = repivot_view(dst / entry.dir)
    return changes
