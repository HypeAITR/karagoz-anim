"""Typed models for the layered puppet (rig v2, package K-0).

A puppet is built once per character from one reference drawing. Part names are
data from the parts config, never hard-coded here or in the other modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Puppet models reject unknown keys: a misspelled preset key must fail
    loudly, never drop silently (K-0d item 1)."""

    model_config = ConfigDict(extra="forbid")


class FeatureSpec(_Strict):
    """A named probe point in a view's reference drawing (preset data).

    The point is reference pixels, e.g. a one-sided chest detail or a
    front-surface midpoint used for the yaw sign and depth cue. It may sit
    on ink (a seam is a line); only the canvas bounds apply.
    """

    part: str
    point: list[float]
    front: bool = False

    @field_validator("point")
    @classmethod
    def _point_xy(cls, value: list[float]) -> list[float]:
        if len(value) != 2:
            raise ValueError("feature point must be [x, y]")
        return [float(value[0]), float(value[1])]


class PartSpec(_Strict):
    """One part from the parts config: how to find it and where it hangs."""

    name: str
    parent: str | None
    concepts: list[str] = Field(default_factory=list)
    seed: list[list[float]] | None = None
    z: int = 0
    # Optional joint rotation limits [lo, hi] in degrees, relative to rest.
    # Declared here so a preset can pin a torso-mounted cap; the fit search
    # clips every candidate into the range (K-1 REV1).
    limits: list[float] | None = None
    # Hidden in this view: no region is claimed for the part, it gets no file,
    # and downstream views (head variants) render in its place. The seed, if
    # any, is ignored. The root can never be hidden.
    hidden: bool = False
    # How many disconnected islands the part legitimately has in this view
    # (claw tips, eyes); the outline measure counts components after merging
    # 1-3 px crumbs and must equal this.
    islands: int = Field(default=1, ge=1)

    @field_validator("seed", mode="before")
    @classmethod
    def _seed_xy(cls, value: object) -> object:
        if value is None:
            return None
        pairs = value if isinstance(value, list) and value and isinstance(value[0], list) else [value]
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("seed must be [x, y] or a list of [x, y]")
        return pairs

    @field_validator("limits", mode="before")
    @classmethod
    def _limits_pair(cls, value: object) -> object:
        if value is None:
            return None
        pair = [float(v) for v in value] if isinstance(value, (list, tuple)) else None
        if pair is None or len(pair) != 2:
            raise ValueError("limits must be [lo, hi] degrees")
        if not pair[0] <= pair[1]:
            raise ValueError(f"limits need lo <= hi, got {pair}")
        return pair


class PuppetConfig(_Strict):
    """The parts config file: the tree plus how to find each part."""

    parts: list[PartSpec]
    line_max_luma: float = 25.0
    # Declared turn of this view toward the viewer's right, in degrees. Read
    # from the preset, reported in report.json; the set build compares the
    # measured yaw against the set.yaml value.
    yaw_deg: float | None = None
    # When true, every mirrored `_l`/`_r` pair must match in area and bounding
    # box size within 0.85-1.18; otherwise the build reports a symmetry failure.
    # Only declare this for a frontal character; a turned character (far side
    # narrower) legitimately fails it. Undeclared, the ratios are report only.
    symmetric: bool = False
    # Named probe points in this view's reference drawing (see FeatureSpec).
    features: dict[str, FeatureSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _tree_ok(self) -> PuppetConfig:
        names = [p.name for p in self.parts]
        if len(set(names)) != len(names):
            raise ValueError("part names must be unique")
        roots = [p for p in self.parts if p.parent is None]
        if len(roots) != 1:
            raise ValueError(f"config needs exactly one root, found {len(roots)}")
        known = set(names)
        for part in self.parts:
            if part.parent is not None and part.parent not in known:
                raise ValueError(f"part {part.name!r} has unknown parent {part.parent!r}")
        seen: set[str] = set()
        for part in self.parts:
            node: PartSpec | None = part
            chain: set[str] = set()
            while node is not None and node.parent is not None:
                if node.name in chain:
                    raise ValueError(f"cycle in parent tree at {node.name!r}")
                chain.add(node.name)
                node = next((p for p in self.parts if p.name == node.parent), None)
            seen.add(part.name)
        if self.root.hidden:
            raise ValueError(f"root part {self.root.name!r} cannot be hidden")
        void = [p.name for p in self.parts if not p.hidden and not p.concepts and p.seed is None]
        if void:
            raise ValueError(f"parts without concepts and without seed: {void}")
        for feature in self.features.values():
            if feature.part not in known:
                raise ValueError(f"feature on unknown part {feature.part!r}")
        return self

    @property
    def root(self) -> PartSpec:
        for part in self.parts:
            if part.parent is None:
                return part
        raise AssertionError("validated to have a root")

    @property
    def by_name(self) -> dict[str, PartSpec]:
        return {p.name: p for p in self.parts}


def load_config(path: Path) -> PuppetConfig:
    """Read and validate a parts config YAML file."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("parts"), list):
        raise ValueError(f"{path} must map to {'parts': [...]}")
    return PuppetConfig.model_validate(raw)


def load_set_config(path: Path) -> SetConfig:
    """Read a set.yaml build config; preset/reference resolve against its directory."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must map to a puppet set config")
    cfg = SetConfig.model_validate(raw)
    base = path.parent
    views: dict[str, ViewSpec] = {}
    for name, spec in cfg.views.items():
        preset = spec.preset if spec.preset.is_absolute() else base / spec.preset
        reference = spec.reference if spec.reference.is_absolute() else base / spec.reference
        views[name] = spec.model_copy(update={"preset": preset, "reference": reference})
    head_views: dict[str, ViewSpec] = {}
    for name, spec in cfg.head_views.items():
        preset = spec.preset if spec.preset.is_absolute() else base / spec.preset
        reference = spec.reference if spec.reference.is_absolute() else base / spec.reference
        head_views[name] = spec.model_copy(update={"preset": preset, "reference": reference})
    if cfg.default_view not in views:
        raise ValueError(f"default_view {cfg.default_view!r} is not one of the set views")
    return cfg.model_copy(update={"views": views, "head_views": head_views})


def _read_view_puppet(out_dir: Path, entry: ViewEntry) -> Puppet:
    puppet_path = out_dir / entry.dir / "puppet.json"
    if not puppet_path.is_file():
        raise ValueError(f"view {entry.dir!r} has no puppet.json at {puppet_path}")
    return Puppet.model_validate_json(puppet_path.read_text(encoding="utf-8"))


def load_set(out_dir: Path) -> PuppetSet:
    """Read set.json plus every view puppet.json and validate the set hangs together.

    Part names must be the same set in every body view (a part may be hidden,
    never missing); the same holds across head views. The default view must
    exist; mirror targets must exist. The head part must exist in every body
    view where it is not hidden. Hidden parts have no file; visible parts must.
    """
    set_path = out_dir / "set.json"
    if not set_path.is_file():
        raise ValueError(f"no set.json at {set_path}")
    pset = PuppetSet.model_validate_json(set_path.read_text(encoding="utf-8"))
    if pset.default_view not in pset.views:
        raise ValueError(f"default_view {pset.default_view!r} is not one of the set views")

    def check_group(group: dict[str, ViewEntry], kind: str, own_names: set[str]) -> None:
        reference: set[str] | None = None
        for name, entry in group.items():
            if entry.mirror_of is not None and entry.mirror_of not in own_names:
                raise ValueError(f"{kind} view {name!r} mirrors unknown view {entry.mirror_of!r}")
            puppet = _read_view_puppet(out_dir, entry)
            names = set(puppet.parts)
            if reference is None:
                reference = names
            if names != reference:
                missing = sorted(reference - names)
                extra = sorted(names - reference)
                raise ValueError(
                    f"{kind} view {name!r} has a different part set"
                    + (f"; missing part(s): {missing}" if missing else "")
                    + (f"; extra part(s): {extra}" if extra else "")
                )
            for pname, part in puppet.parts.items():
                location = out_dir / entry.dir / part.file if part.file else None
                if part.hidden:
                    if part.file:
                        raise ValueError(f"hidden part {pname!r} in {kind} view {name!r} has a file")
                elif not part.file or location is None or not location.is_file():
                    raise ValueError(f"visible part {pname!r} in {kind} view {name!r} has no file")

    check_group(pset.views, "body", set(pset.views) | set(pset.head_views))
    check_group(pset.head_views, "head", set(pset.views) | set(pset.head_views))
    for name, entry in pset.views.items():
        puppet = _read_view_puppet(out_dir, entry)
        if pset.head_part not in puppet.parts:
            raise ValueError(f"head part {pset.head_part!r} missing from body view {name!r}")
    return pset


class PuppetPart(_Strict):
    """One entry of ``puppet.json``: where the part hangs and how it was filled."""

    parent: str | None
    pivot: list[float]
    z: int
    file: str
    # Optional joint rotation limits [lo, hi] in degrees, relative to rest
    # (see PartSpec.limits); the fit search clips every candidate into range.
    limits: list[float] | None = None
    # Optional joint slack in reference pixels: how far the joint may slide
    # from its modelled pivot (the animator nudging a shoulder). The chain
    # fit searches an offset inside this radius; default 0 disables the
    # search exactly.
    slack_px: float = 0.0
    completed: list[str] = Field(default_factory=list)
    approved: bool = False
    # Hidden in this view: not drawn here (a head variant renders instead),
    # so the part has no file. Never missing: hidden parts are still listed.
    hidden: bool = False
    # True when the pivot was projected onto the parent boundary because a
    # third part hides the joint (no visible touch to measure).
    estimated: bool = False

    @field_validator("limits", mode="before")
    @classmethod
    def _limits_pair(cls, value: object) -> object:
        if value is None:
            return None
        pair = [float(v) for v in value] if isinstance(value, (list, tuple)) else None
        if pair is None or len(pair) != 2:
            raise ValueError("limits must be [lo, hi] degrees")
        if not pair[0] <= pair[1]:
            raise ValueError(f"limits need lo <= hi, got {pair}")
        return pair

    @field_validator("slack_px", mode="before")
    @classmethod
    def _slack_nonnegative(cls, value: object) -> object:
        if value is None:
            return 0.0
        if not isinstance(value, (int, float, str)):
            raise ValueError("slack_px must be a number")
        number = float(value)
        if number < 0.0:
            raise ValueError("slack_px must be non-negative")
        return number


class Puppet(_Strict):
    """The ``puppet.json`` document, exactly as in ``_wp/kukla-ortak/PUPPET-FORMAT.md``."""

    version: int = 1
    reference: str
    canvas: list[int]
    line_width_px: float
    root: str
    parts: dict[str, PuppetPart]


class ViewEntry(_Strict):
    """One view in ``set.json``: where the built puppet lives and how far it turns."""

    yaw_deg: float = 0.0
    dir: str
    mirror_of: str | None = None


class PuppetSet(_Strict):
    """The ``set.json`` document: one character, several view puppets."""

    version: int = 1
    character: str
    views: dict[str, ViewEntry]
    head_views: dict[str, ViewEntry] = Field(default_factory=dict)
    head_part: str = "head"
    default_view: str


class ViewSpec(_Strict):
    """How to build one view: preset plus reference drawing plus declared turn."""

    preset: Path
    reference: Path
    yaw_deg: float = 0.0
    mirror_of: str | None = None


class SetConfig(_Strict):
    """The ``set.yaml`` document: how to build every view of a character."""

    character: str
    default_view: str
    head_part: str = "head"
    views: dict[str, ViewSpec]
    head_views: dict[str, ViewSpec] = Field(default_factory=dict)
