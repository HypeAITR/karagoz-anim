"""The puppet package: one layered character from one reference drawing."""

from __future__ import annotations

from karagoz.engine.puppet.build import (
    PuppetFiles,
    PuppetSetFiles,
    ViewPuppet,
    build_puppet,
    build_puppet_set,
    head_for,
    neighbours,
    summarize_set,
    view_order,
)
from karagoz.engine.puppet.model import (
    FeatureSpec,
    Puppet,
    PuppetConfig,
    PuppetPart,
    PuppetSet,
    SetConfig,
    ViewEntry,
    ViewSpec,
    load_config,
    load_set,
    load_set_config,
)

__all__ = [
    "FeatureSpec",
    "Puppet",
    "PuppetConfig",
    "PuppetFiles",
    "PuppetPart",
    "PuppetSet",
    "PuppetSetFiles",
    "SetConfig",
    "ViewEntry",
    "ViewPuppet",
    "ViewSpec",
    "build_puppet",
    "build_puppet_set",
    "head_for",
    "load_config",
    "load_set",
    "load_set_config",
    "neighbours",
    "summarize_set",
    "view_order",
]
