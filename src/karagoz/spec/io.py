"""File-kind routing and loading with agent-readable findings."""

from __future__ import annotations

import json
import re
import types
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

from karagoz.spec.findings import Finding, SpecError
from karagoz.spec.models import Align, Continuity, Lines, Mouth, Poses, Shots, Style, Xsheet

KINDS: dict[str, type[BaseModel]] = {
    "lines": Lines,
    "shots": Shots,
    "continuity": Continuity,
    "poses": Poses,
    "xsheet": Xsheet,
    "style": Style,
    "align": Align,
    "mouth": Mouth,
}

_YAML_FILES = {
    "lines.yaml": "lines",
    "shots.yaml": "shots",
    "continuity.yaml": "continuity",
    "poses.yaml": "poses",
    "xsheet.yaml": "xsheet",
    "style.yaml": "style",
}

_ANCHOR_EXAMPLE_HINT = "example: L01.end + beat.react"

_BRANCH_NAMES = frozenset({"Hold", "Transition", "EyeEvent", "SequenceEvent", "LoopEvent", "CameraEvent", "FxEvent"})
_BRANCH_RE = re.compile(r"^function-(before|after|wrap|plain)\[.*\]$")
_ALIAS_FIX = {"in_": "in", "from_": "from"}


def kind_of(path: Path) -> str:
    """Return the spec kind for a file name, or raise ``SpecError``."""
    name = path.name
    if name in _YAML_FILES:
        return _YAML_FILES[name]
    if name.endswith(".align.json"):
        return "align"
    if name == "mouth.json" or (
        name.startswith("mouth.") and name.endswith(".json") and len(name) > len("mouth..json")
    ):
        return "mouth"
    raise SpecError([Finding("error", "file.unknown_kind", str(path), "", f"unknown file kind: {name}")])


def load(path: Path) -> BaseModel:
    """Read a spec file and validate it; raise ``SpecError`` on any problem."""
    model = KINDS[kind_of(path)]
    file = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError([Finding("error", "file.unreadable", file, "", f"cannot read file: {exc}")]) from exc
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            message = f"invalid JSON at line {exc.lineno}: {exc.msg}"
            raise SpecError([Finding("error", "json.syntax", file, "", message)]) from exc
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            detail = getattr(exc, "problem", None) or exc
            if mark is None:
                message = f"invalid YAML (line unknown): {detail}"
            else:
                message = f"invalid YAML at line {mark.line + 1}: {detail}"
            raise SpecError([Finding("error", "yaml.syntax", file, "", message)]) from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise SpecError(findings_from_validation(file, exc)) from exc


def _is_branch_name(part: int | str) -> bool:
    return isinstance(part, str) and (part in _BRANCH_NAMES or _BRANCH_RE.match(part) is not None)


def _format_path(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        elif _is_branch_name(part):
            continue
        else:
            part = _ALIAS_FIX.get(part, part)
            out += part if not out else f".{part}"
    return out


def _strip_annotated(tp: Any) -> Any:
    while get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    return tp


def _union_args(tp: Any) -> tuple[Any, ...] | None:
    if get_origin(tp) in (Union, types.UnionType):
        return tuple(arg for arg in get_args(tp) if arg is not type(None))
    return None


def _fields_of(tp: Any) -> set[str]:
    tp = _strip_annotated(tp)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return {(field.alias or name) for name, field in tp.model_fields.items()}
    args = _union_args(tp)
    if args is not None:
        found: set[str] = set()
        for arg in args:
            found |= _fields_of(arg)
        return found
    return set()


def _step(tp: Any, part: int | str) -> Any | None:
    args = _union_args(tp)
    if args is not None:
        for arg in args:
            hit = _step(_strip_annotated(arg), part)
            if hit is not None:
                return hit
        return None
    if isinstance(part, int):
        if get_origin(tp) in (list, tuple, set, frozenset):
            items = get_args(tp)
            return items[0] if items else None
        return None
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        for name, field in tp.model_fields.items():
            if part == name or (field.alias is not None and part == field.alias):
                return field.annotation
        return None
    if get_origin(tp) is dict:
        items = get_args(tp)
        return items[1] if len(items) == 2 else None
    return None


def _resolve(tp: Any, loc: tuple[int | str, ...]) -> Any | None:
    for part in loc:
        tp = _step(_strip_annotated(tp), part)
        if tp is None:
            return None
    return tp


def findings_from_validation(file: str, err: ValidationError) -> list[Finding]:
    """Turn every pydantic error into a stable, agent-readable finding."""
    try:
        root: type[BaseModel] | None = KINDS[kind_of(Path(file))]
    except SpecError:
        root = None
    findings: list[Finding] = []
    for error in err.errors():
        code = f"schema.{error['type']}"
        path = _format_path(error["loc"])
        hint: str | None = None
        if error["type"] == "extra_forbidden" and root is not None:
            user_loc = tuple(p for p in error["loc"] if not _is_branch_name(p))
            allowed = _fields_of(_resolve(root, user_loc[:-1]))
            if allowed:
                hint = f"allowed fields: {', '.join(sorted(allowed))}"
        elif error["type"] == "anchor_syntax":
            hint = _ANCHOR_EXAMPLE_HINT
        findings.append(Finding("error", code, file, path, error["msg"], hint))
    return findings
