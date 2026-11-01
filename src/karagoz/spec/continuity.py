"""Per-scene continuity files merged into one view (WP-52).

Each scene's continuity task writes ``<episode>/continuity/scene<N>.yaml`` (same
``Continuity`` model, ``scene: N``); readers merge them in scene order. A lone
``<episode>/continuity.yaml`` without a ``continuity/`` directory is the legacy
single-scene layout and loads as is.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import karagoz.spec.io as _io
from karagoz.spec.findings import Finding, SpecError
from karagoz.spec.io import load
from karagoz.spec.models import Continuity

SCENE_FILE_RE = re.compile(r"^scene(\d+)\.yaml$")


class _SceneKinds(dict[str, str]):
    """File-kind table that also knows scene files.

    ``io.kind_of`` only recognises exact file names; scene files are
    ``scene<N>.yaml`` for any N, so this table answers the pattern. Installed at
    import, so ``load()`` and the studio checkers treat scene files as
    continuity files. (``io.py`` is the natural home; it was out of scope for
    WP-52, so the extension lives here next to the merge.)
    """

    def __contains__(self, name: object) -> bool:
        return super().__contains__(name) or (isinstance(name, str) and SCENE_FILE_RE.match(name) is not None)

    def __missing__(self, name: str) -> str:
        if SCENE_FILE_RE.match(name) is not None:
            return "continuity"
        raise KeyError(name)


_io._YAML_FILES = _SceneKinds(_io._YAML_FILES)


def continuity_paths(episode_dir: Path) -> list[Path]:
    """Continuity files in merge order: scene files, or the legacy single file, or nothing."""
    directory = episode_dir / "continuity"
    if directory.is_dir():
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return []
        numbered: list[tuple[int, Path]] = []
        for child in children:
            if not child.is_file():
                continue
            match = SCENE_FILE_RE.match(child.name)
            if match is not None:
                numbered.append((int(match.group(1)), child))
        numbered.sort(key=lambda item: item[0])
        return [path for _, path in numbered]
    legacy = episode_dir / "continuity.yaml"
    return [legacy] if legacy.is_file() else []


def _rel(path: Path, episode_dir: Path) -> str:
    try:
        rel = os.path.relpath(path, episode_dir)
    except ValueError:
        return path.as_posix()
    return Path(rel).as_posix()


def merge_continuities(parts: list[tuple[Path, Continuity]], episode_dir: Path) -> Continuity:
    """Merge scene files in order; a clashing prop_states list names both files."""
    states = [state for _, part in parts for state in part.states]
    props = [entry for _, part in parts for entry in (part.props or [])]
    merged_states: dict[str, list[str]] | None = None
    owners: dict[str, str] = {}
    for path, part in parts:
        if part.prop_states is None:
            continue
        if merged_states is None:
            merged_states = {}
        rel = _rel(path, episode_dir)
        for prop, states_list in part.prop_states.items():
            if prop in merged_states:
                if merged_states[prop] != states_list:
                    raise SpecError(
                        [
                            Finding(
                                "error",
                                "prop.conflict",
                                rel,
                                f"prop_states.{prop}",
                                f"Prop '{prop}' has conflicting state lists in '{owners[prop]}' and '{rel}'.",
                            )
                        ]
                    )
                continue
            merged_states[prop] = list(states_list)
            owners[prop] = rel
    return Continuity(
        version=1,
        scene=parts[0][1].scene,
        states=states,
        prop_states=merged_states,
        props=props if any(part.props is not None for _, part in parts) else None,
    )


def load_continuity(episode_dir: Path) -> Continuity | None:
    """The episode's continuity as one view: merged scene files, the legacy file, or None."""
    paths = continuity_paths(episode_dir)
    if not paths:
        return None
    parts: list[tuple[Path, Continuity]] = []
    for path in paths:
        obj = load(path)
        if not isinstance(obj, Continuity):
            raise SpecError([Finding("error", "file.unknown_kind", str(path), "", f"unknown file kind: {path.name}")])
        parts.append((path, obj))
    return merge_continuities(parts, episode_dir)
