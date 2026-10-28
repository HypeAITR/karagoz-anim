"""Per-task context compiler: cores, department guide, chosen sections, index, own notes.

Why each piece exists (one line each):
- Read-everything buries what matters, but nothing may be unreachable: the rest are listed with "read when ...".
- If code cannot classify a task, the agent gets all of its department's sections (more context, never less).
- Agents start without memory; what they learn on a production lives in notes/<department>/, which they write.
- What the agent opened itself is logged: a section that is always opened later should have been given up front.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from karagoz.spec import Continuity, Poses, Shots, SpecError, load
from karagoz.studio.briefs import _strip_front_matter

_LESSON_START_RE = re.compile(r"^(\d+)\. ")
_TAGS_LINE_RE = re.compile(r"^\s*tags\s*:\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    read_when: str
    departments: tuple[str, ...]
    always_for: tuple[str, ...]
    tags: tuple[str, ...]
    body: str  # text after the front matter, stripped


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def load_sections(guide_dir: Path) -> dict[str, Section]:
    """Read every ``sections/*.md`` (sorted); front matter gives id, title, read_when and the routing lists."""
    out: dict[str, Section] = {}
    for path in sorted((guide_dir / "sections").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError(f"{path.name}: missing front matter")
        rest = text[len("---") :]
        lines = rest.splitlines(keepends=True)
        end: int | None = None
        for i, line in enumerate(lines):
            if line.strip() == "---":
                end = i
                break
        if end is None:
            raise ValueError(f"{path.name}: missing front matter")
        try:
            data = yaml.safe_load("".join(lines[:end]))
        except yaml.YAMLError as exc:
            raise ValueError(f"{path.name}: invalid front matter: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: missing front matter")
        sid = str(data.get("id", ""))
        if sid != path.stem:
            raise ValueError(f"{path.name}: id {sid!r} differs from file stem {path.stem!r}")
        out[sid] = Section(
            id=sid,
            title=str(data.get("title", "")),
            read_when=str(data.get("read_when", "")),
            departments=_str_tuple(data.get("departments")),
            always_for=_str_tuple(data.get("always_for")),
            tags=_str_tuple(data.get("tags")),
            body="".join(lines[end + 1 :]).strip(),
        )
    return out


RULE_TAGS: dict[str, str] = {
    "handover": "a prop changes holder in this shot",
    "props": "a prop has an entry for this shot",
    "stand-up": "a pose goes from sit to stand or crouch",
    "sit-down": "a pose goes from stand or crouch to sit",
    "turn-exit": "a character turns or leaves the frame",
    "body-move": "layout writes the poses: all body moves",
}
"""Tags matchable from task data; all other tags come only via always_for (or the orchestrator)."""


def structure_problems(sections: dict[str, Section], departments: Iterable[str]) -> list[str]:
    """One string per guide structure problem; empty list when fine."""
    known = set(departments)
    rule_tags = set(RULE_TAGS)
    problems: list[str] = []
    for sid, sec in sections.items():
        if not sec.departments:
            problems.append(f"{sid}: no departments entry")
            continue
        if not sec.always_for and not (set(sec.tags) & rule_tags):
            problems.append(f"{sid}: neither always_for nor a rule tag")
        for dept in sec.always_for:
            if dept not in sec.departments:
                problems.append(f"{sid}: always_for {dept!r} not in departments")
        for dept in sec.departments:
            if dept not in known:
                problems.append(f"{sid}: unknown department {dept!r}")
    return problems


@dataclass
class Selection:
    given: list[str]  # section ids, in load order
    reasons: dict[str, str]  # id -> why it was given
    fallback: bool = False


def _load_or_none(path: Path) -> BaseModel | None:
    """spec.load; a file that is missing or fails to load counts as missing."""
    try:
        return load(path)
    except SpecError:
        return None


def select_sections(
    department: str, prod: Path, episode: str, shot: str | None, sections: dict[str, Section]
) -> Selection:
    """Choose the department's sections for the task: always_for, data rules, layout body moves, fallback."""
    dept_ids = [sid for sid, sec in sections.items() if department in sec.departments]
    poses: Poses | None = None
    if shot is not None:
        loaded = _load_or_none(prod / episode / "shots" / shot / "poses.yaml")
        poses = loaded if isinstance(loaded, Poses) else None
    if department in ("timer", "checker") and shot is not None and poses is None:
        return Selection(
            given=list(dept_ids),
            reasons={sid: "fallback: task could not be classified" for sid in dept_ids},
            fallback=True,
        )
    given: dict[str, str] = {}
    for sid in dept_ids:
        if department in sections[sid].always_for:
            given[sid] = f"always for {department}"
    matched: set[str] = set()
    if shot is not None:
        continuity = _load_or_none(prod / episode / "continuity.yaml")
        if isinstance(continuity, Continuity) and continuity.props:
            entries = [e for e in continuity.props if e.shot == shot]
            if entries:
                matched.add("props")
                for e in entries:
                    if e.in_.holder is not None and e.out.holder is not None and e.in_.holder != e.out.holder:
                        matched.add("handover")
                        break
    if poses is not None:
        for char in poses.characters.values():
            plist = char.poses
            for first, second in zip(plist, plist[1:], strict=False):
                if first.support == "sit" and second.support in ("stand", "crouch"):
                    matched.add("stand-up")
                if first.support in ("stand", "crouch") and second.support == "sit":
                    matched.add("sit-down")
                if first.view != second.view:
                    matched.add("turn-exit")
    if shot is not None and "turn-exit" not in matched:
        maybe_shots = _load_or_none(prod / episode / "shots.yaml")
        shot_entry = None
        if isinstance(maybe_shots, Shots):
            shot_entry = next((s for s in maybe_shots.shots if s.id == shot), None)
        if (
            shot_entry is not None
            and shot_entry.characters is not None
            and any(c.offscreen is True for c in shot_entry.characters)
        ):
            matched.add("turn-exit")
    for sid in dept_ids:
        if sid in given:
            continue
        for tag in sections[sid].tags:
            if tag in matched:
                given[sid] = f"rule {tag}: {RULE_TAGS[tag]}"
                break
    if department == "layout":
        for sid in dept_ids:
            if sid not in given and "body-move" in sections[sid].tags:
                given[sid] = "layout writes the poses: all body moves"
    ordered = [sid for sid in dept_ids if sid in given]
    return Selection(given=ordered, reasons={sid: given[sid] for sid in ordered})


def apply_override(selection: Selection, override: dict[str, Any] | None, sections: dict[str, Section]) -> Selection:
    """Apply the orchestrator's add/remove override; unknown ids are ignored; keeps load order."""
    if override is None:
        return Selection(given=list(selection.given), reasons=dict(selection.reasons), fallback=selection.fallback)
    raw_add = override.get("add", [])
    raw_remove = override.get("remove", [])
    add_list = [str(x) for x in raw_add] if isinstance(raw_add, list) else []
    remove_list = [str(x) for x in raw_remove] if isinstance(raw_remove, list) else []
    reason_text = str(override.get("reason", ""))
    add_set = {sid for sid in add_list if sid in sections}
    remove_set = {sid for sid in remove_list if sid in sections}
    old_set = set(selection.given)
    new_set = (old_set - remove_set) | add_set
    new_given = [sid for sid in sections if sid in new_set]
    for sid in selection.given:
        if sid not in sections and sid in new_set and sid not in new_given:
            new_given.append(sid)
    new_reasons: dict[str, str] = {}
    for sid in new_given:
        if sid in old_set and sid not in remove_set and sid in selection.reasons:
            new_reasons[sid] = selection.reasons[sid]
        else:
            new_reasons[sid] = f"added by orchestrator: {reason_text}"
    return Selection(given=new_given, reasons=new_reasons, fallback=selection.fallback)


@dataclass(frozen=True)
class Lesson:
    number: int
    text: str  # the whole block, as written
    tags: tuple[str, ...]


def read_lessons(path: Path) -> list[Lesson]:
    """Read numbered lesson blocks from LESSONS.md; missing file -> []."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = text.splitlines()
    starts: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = _LESSON_START_RE.match(line)
        if m:
            starts.append((i, int(m.group(1))))
    lessons: list[Lesson] = []
    for idx, (start, number) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        block = lines[start:end]
        tags: tuple[str, ...] = ()
        for line in block:
            tm = _TAGS_LINE_RE.match(line)
            if tm:
                tags = tuple(p.strip() for p in tm.group(1).split(",") if p.strip())
                break
        lessons.append(Lesson(number=number, text="\n".join(block).strip(), tags=tags))
    return lessons


def select_lessons(lessons: list[Lesson], wanted: set[str], limit: int = 5) -> list[Lesson]:
    """Untagged lessons plus tag matches, newest first up to limit; returned in ascending number order."""
    untagged = sorted((x for x in lessons if not x.tags), key=lambda x: x.number, reverse=True)
    matching = sorted((x for x in lessons if x.tags and set(x.tags) & wanted), key=lambda x: x.number, reverse=True)
    chosen = (untagged + matching)[:limit]
    return sorted(chosen, key=lambda x: x.number)


def shot_summary(prod: Path, episode: str, shot: str | None) -> str | None:
    """The shot's goal and its moves, from data; None when the shot is unknown."""
    if shot is None:
        return None
    shots = _load_or_none(prod / episode / "shots.yaml")
    if not isinstance(shots, Shots):
        return None
    entry = next((s for s in shots.shots if s.id == shot), None)
    if entry is None:
        return None
    lines = ["## This shot", "", f"{entry.id}, {entry.framing}, location {entry.location}."]
    if entry.note:
        lines.append(f"Goal: {entry.note}")
    continuity = _load_or_none(prod / episode / "continuity.yaml")
    if isinstance(continuity, Continuity):
        for st in continuity.states:
            if st.shot == shot:
                lines.append(f"- {st.character}: {st.in_} → {st.out}")
        if continuity.props:
            for pe in continuity.props:
                if pe.shot == shot:
                    in_where = pe.in_.holder or pe.in_.place or "nowhere"
                    out_where = pe.out.holder or pe.out.place or "nowhere"
                    lines.append(f"- prop {pe.prop}: {in_where} ({pe.in_.state}) → {out_where} ({pe.out.state})")
    return "\n".join(lines)


def compile_context(
    *,
    task_id: str,
    department: str,
    guide_dir: Path,
    skills_dir: Path,
    sections: dict[str, Section],
    selection: Selection,
    lessons: list[Lesson],
    summary: str | None,
    check_command: str | None,
) -> str:
    """Compile one task's context: critical things first, then guide, sections, index, notes, check."""
    lines = [f"# Task: {task_id}", "", f"Department: {department}", ""]
    if summary:
        lines += [summary.strip(), ""]
    for core in ("studio.md", "notes.md"):
        core_path = guide_dir / "core" / core
        if core_path.is_file():
            lines += [core_path.read_text(encoding="utf-8").strip(), ""]
    skill_path = skills_dir / department / "SKILL.md"
    if skill_path.is_file():
        lines += ["# Your department guide", "", _strip_front_matter(skill_path.read_text(encoding="utf-8")), ""]
    if department == "layout":
        draw_path = skills_dir / "draw-prompts" / "SKILL.md"
        if draw_path.is_file():
            lines += [
                "# Image-model behaviour (draw-prompts)",
                "",
                _strip_front_matter(draw_path.read_text(encoding="utf-8")),
                "",
            ]
    for sid in selection.given:
        sec = sections.get(sid)
        if sec is None:
            continue
        lines += [f"# Section `{sid}`", "", sec.body, ""]
    dept_ids = [sid for sid, sec in sections.items() if department in sec.departments]
    not_given = [sid for sid in dept_ids if sid not in set(selection.given)]
    if not_given:
        lines += [
            "# Index of sections you were not given",
            "",
            "Before you write, read this list. If your task contains one of these "
            "situations, open the file `sections/<id>.md` in your working folder "
            "and read it before you write your output.",
            "",
        ]
        for sid in not_given:
            lines.append(f"- `{sid}`: read when {sections[sid].read_when}")
        lines.append("")
    if lessons:
        lines += ["# Your notes (lessons chosen for this task)", ""]
        for lesson in lessons:
            lines += [lesson.text, ""]
        lines += [f"All your notes are in `notes/{department}/`.", ""]
    if check_command:
        lines += ["# Check your work", "", f"Run `{check_command}` and fix every error. Read the warnings.", ""]
    return "\n".join(lines)
