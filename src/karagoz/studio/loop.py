"""Orchestrator loop: one review per shot, decide keep / redraw / retake, at most three rounds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from karagoz.studio.orchestrator import Studio


class RoundLimit(RuntimeError):
    """Raised when a shot's review would go past MAX_ROUNDS."""


MAX_ROUNDS = 3

ROUND_LIMIT_NOTE = "round limit reached (3): a human decides"

INTERVIEW_PROMPT = (
    "Your task is finished; do not change any file. Answer briefly: which sentence in your brief "
    "or the retake note was unclear or contradicted another? What did you need that was missing?"
)


def _review_dir(prod: Path, episode: str, shot: str) -> Path:
    return prod / episode / "review" / shot


def current_round(prod: Path, episode: str, shot: str) -> int:
    """The shot's current round: rounds.json "round", 1 when missing."""
    try:
        data = json.loads((_review_dir(prod, episode, shot) / "rounds.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1
    if isinstance(data, dict):
        value = data.get("round")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
    return 1


def next_round(prod: Path, episode: str, shot: str) -> int:
    """Count one round; past MAX_ROUNDS raise RoundLimit and save nothing."""
    current = current_round(prod, episode, shot)
    new = current + 1
    if new > MAX_ROUNDS:
        raise RoundLimit(ROUND_LIMIT_NOTE)
    path = _review_dir(prod, episode, shot) / "rounds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"round": new}) + "\n", encoding="utf-8")
    return new


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _finding_id(part: str, copy: int, index: int) -> str:
    return f"finding:{part}#{copy}:{index}"


def _copy_key(entry: dict[str, Any]) -> int:
    value = entry.get("copy")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def write_review(prod: Path, episode: str, shot: str) -> Path:
    """Write review/<shot>/REVIEW.md for a reader who decides; return its path."""
    review_dir = _review_dir(prod, episode, shot)
    review_dir.mkdir(parents=True, exist_ok=True)
    round_n = current_round(prod, episode, shot)

    gate_data = _read_json_dict(review_dir / "gate.json")
    gate_missing = gate_data is None
    raw_flags: list[Any] = []
    if gate_data is not None:
        candidate = gate_data.get("flags")
        if isinstance(candidate, list):
            raw_flags = candidate
    flags: list[dict[str, Any]] = [f for f in raw_flags if isinstance(f, dict)]

    checks_data = _read_json_dict(review_dir / "checks.json")
    checks_missing = checks_data is None
    raw_parts: list[Any] = []
    agreement_data: dict[str, Any] = {}
    if checks_data is not None:
        candidate = checks_data.get("parts")
        if isinstance(candidate, list):
            raw_parts = candidate
        candidate_agreement = checks_data.get("agreement")
        if isinstance(candidate_agreement, dict):
            agreement_data = candidate_agreement
    parts: list[dict[str, Any]] = [p for p in raw_parts if isinstance(p, dict)]

    kept_data = _read_json_dict(review_dir / "kept.json") or {}
    ack_data = _read_json_dict(review_dir / "gate-ack.json") or {}

    total_findings = 0
    failed = 0
    for entry in parts:
        error = entry.get("error")
        if error is not None:
            failed += 1
        findings = entry.get("findings")
        if isinstance(findings, list):
            total_findings += sum(1 for f in findings if isinstance(f, dict))

    lines: list[str] = [f"# Review {shot} · round {round_n}"]
    lines.append(f"{len(flags)} shown gate flags, {total_findings} checker findings, {failed} failed checks")
    if gate_missing:
        lines.append("gate.json missing or unreadable: counted as empty.")
    if checks_missing:
        lines.append("checks.json missing or unreadable: counted as empty.")

    # --- where they meet ---
    lines.append("## Where they meet")
    findings_index: list[tuple[str, str | None]] = []  # (finding id, drawing)
    for entry in parts:
        part_id = entry.get("part")
        copy = entry.get("copy")
        if not isinstance(part_id, str) or not isinstance(copy, int) or isinstance(copy, bool):
            continue
        findings = entry.get("findings")
        if not isinstance(findings, list):
            continue
        for index, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                continue
            drawing = finding.get("drawing")
            drawing_s = drawing if isinstance(drawing, str) else None
            findings_index.append((_finding_id(part_id, copy, index), drawing_s))
    meet_lines: list[str] = []
    seen_meet: set[tuple[str, str]] = set()
    for flag in flags:
        flag_id = flag.get("id")
        if not isinstance(flag_id, str):
            continue
        character = flag.get("character")
        character_s = character if isinstance(character, str) else None
        drawings = flag.get("drawings")
        if not isinstance(drawings, list):
            continue
        tags: list[str] = []
        for drawing in drawings:
            if not isinstance(drawing, str):
                continue
            # most flags name bare drawings plus a character; prop_count names "<char>/<name>" with no character
            tags.append(drawing if character_s is None or "/" in drawing else f"{character_s}/{drawing}")
        for fid, drawing in findings_index:
            if drawing is None:
                continue
            finding_tag = drawing[:-4] if drawing.endswith(".png") else drawing
            for tag in tags:
                if tag == finding_tag and (flag_id, fid) not in seen_meet:
                    seen_meet.add((flag_id, fid))
                    meet_lines.append(f"- {flag_id} + {fid}: {tag}")
    if meet_lines:
        lines.extend(meet_lines)
    else:
        lines.append("none")

    # --- gate flags ---
    lines.append("## Gate flags")
    lines.append("The checkers did not see these.")
    for flag in flags:
        flag_id = flag.get("id", "?")
        message = flag.get("message", "")
        frames = flag.get("frames")
        text = f"- {flag_id}: {message}"
        if frames is not None:
            text += f" frames {frames}"
        if isinstance(flag_id, str):
            ack_entry = ack_data.get(flag_id)
            if isinstance(ack_entry, dict) and ack_entry.get("shas") == flag.get("shas"):
                note = ack_entry.get("note", "")
                text += f" (kept: {note})"
        lines.append(text)

    # --- checker findings ---
    lines.append("## Checker findings")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in parts:
        part_id = entry.get("part")
        if not isinstance(part_id, str):
            continue
        grouped.setdefault(part_id, []).append(entry)
    for part_id, entries in grouped.items():
        lines.append(f"### {part_id}")
        if len(entries) >= 2:
            both: list[str] = []
            one: list[str] = []
            stored = agreement_data.get(part_id)
            both_raw = stored.get("both") if isinstance(stored, dict) else None
            one_raw = stored.get("one") if isinstance(stored, dict) else None
            if isinstance(both_raw, list) and isinstance(one_raw, list):
                both = [d for d in both_raw if isinstance(d, str)]
                one = [d for d in one_raw if isinstance(d, str)]
            else:
                sets: list[set[str]] = []
                for entry in entries:
                    drawings_set: set[str] = set()
                    findings = entry.get("findings")
                    if isinstance(findings, list):
                        for item in findings:
                            if isinstance(item, dict):
                                drawing = item.get("drawing")
                                if isinstance(drawing, str) and drawing:
                                    drawings_set.add(drawing)
                    sets.append(drawings_set)
                both = sorted(set.intersection(*sets)) if sets else []
                union: set[str] = set()
                for s in sets:
                    union |= s
                one = sorted(union - set(both))
            lines.append(f"agreement: both: {', '.join(both)}; one: {', '.join(one)}")
        ordered = sorted(entries, key=_copy_key)
        for entry in ordered:
            copy = entry.get("copy")
            copy_s = copy if isinstance(copy, int) else "?"
            error = entry.get("error")
            if error is not None:
                lines.append(f"- {part_id}#{copy_s}: check failed: {error}")
                continue
            findings = entry.get("findings")
            if not isinstance(findings, list):
                continue
            copy_int = copy if isinstance(copy, int) and not isinstance(copy, bool) else 0
            for index, finding in enumerate(findings, start=1):
                if not isinstance(finding, dict):
                    continue
                fid = _finding_id(part_id, copy_int, index)
                severity = finding.get("severity", "")
                look = finding.get("look", "")
                drawing = finding.get("drawing")
                frames = finding.get("frames")
                if isinstance(drawing, str) and drawing:
                    target = drawing
                elif frames is not None:
                    target = f"frames {frames}"
                else:
                    target = ""
                problem = finding.get("problem", "")
                reason = finding.get("reason", "")
                text = f"- {fid} [{severity}] {look} {target}: {problem} — {reason}"
                kept_note = kept_data.get(fid)
                if isinstance(kept_note, str):
                    text += f" (kept: {kept_note})"
                lines.append(text)

    # --- commands ---
    lines.append("## Commands")
    lines.append(f"karagoz sheet {prod} {episode} {shot}")
    lines.append(f"karagoz redraw {prod} {episode} {shot} <char/name> [...] --note TEXT")
    lines.append(f"karagoz keep {prod} {episode} {shot} <item-id> --note TEXT")
    lines.append(f"karagoz retake {prod} {episode} <task> <note>")
    lines.append(f"karagoz context {prod} {episode} <task> [--add ID ...] [--remove ID ...] --reason TEXT")
    lines.append(f"karagoz approve {prod} {episode} {shot}/review")

    out = review_dir / "REVIEW.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _count_round_or_hold(studio: Studio, shot: str) -> int:
    """Return the round being decided; on RoundLimit hold the review and re-raise."""
    prod = studio.prod
    episode = studio.episode
    current = current_round(prod, episode, shot)
    try:
        next_round(prod, episode, shot)
    except RoundLimit:
        review_id = f"{shot}/review"
        if review_id in studio.board.tasks:
            studio.hold(review_id, ROUND_LIMIT_NOTE)
        raise
    return current


def redraw(studio: Studio, shot: str, drawings: list[str], note: str) -> list[str]:
    """Move drawings to history so the draw task draws them again; count one round."""
    prod = studio.prod
    episode = studio.episode
    assets = prod / episode / "assets" / shot
    missing = [d for d in drawings if "/" not in d or not (assets / f"{d}.png").is_file()]
    if missing:
        raise ValueError(f"unknown drawing(s): {', '.join(missing)}")
    current = _count_round_or_hold(studio, shot)
    for drawing in drawings:
        char, _, name = drawing.partition("/")
        char_dir = assets / char
        history = char_dir / "history"
        history.mkdir(parents=True, exist_ok=True)
        (char_dir / f"{name}.png").rename(history / f"{name}.{current}.png")
        landmarks = char_dir / f"{name}.landmarks.json"
        if landmarks.is_file():
            landmarks.rename(history / f"{name}.{current}.landmarks.json")
    review_dir = _review_dir(prod, episode, shot)
    review_dir.mkdir(parents=True, exist_ok=True)
    redraw_path = review_dir / "redraw.json"
    try:
        existing = json.loads(redraw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing.append({"round": current, "drawings": list(drawings), "note": note})
    redraw_path.write_text(json.dumps(existing, indent=1) + "\n", encoding="utf-8")
    studio.log.append(f"{shot}/review", "decision", action="redraw", drawings=list(drawings), note=note, round=current)
    return list(drawings)


def keep(studio: Studio, shot: str, item_id: str, note: str) -> None:
    """Keep one gate flag or checker finding with a note; no round is counted."""
    from karagoz.engine.gate import Flag, acknowledge

    prod = studio.prod
    episode = studio.episode
    review_dir = _review_dir(prod, episode, shot)
    if item_id.startswith("finding:"):
        checks = _read_json_dict(review_dir / "checks.json")
        known: set[str] = set()
        if checks is not None:
            raw_parts = checks.get("parts")
            if isinstance(raw_parts, list):
                for entry in raw_parts:
                    if not isinstance(entry, dict):
                        continue
                    part_id = entry.get("part")
                    copy = entry.get("copy")
                    if not isinstance(part_id, str) or not isinstance(copy, int) or isinstance(copy, bool):
                        continue
                    findings = entry.get("findings")
                    if not isinstance(findings, list):
                        continue
                    for index, finding in enumerate(findings, start=1):
                        if isinstance(finding, dict):
                            known.add(_finding_id(part_id, copy, index))
        if item_id not in known:
            raise ValueError(f"unknown finding: {item_id}")
        kept_path = review_dir / "kept.json"
        kept = _read_json_dict(kept_path) or {}
        kept[item_id] = note
        review_dir.mkdir(parents=True, exist_ok=True)
        kept_path.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    else:
        gate = _read_json_dict(review_dir / "gate.json")
        candidates: list[dict[str, Any]] = []
        if gate is not None:
            for key in ("flags", "silenced"):
                raw = gate.get(key)
                if isinstance(raw, list):
                    candidates.extend([f for f in raw if isinstance(f, dict)])
        match = next((f for f in candidates if f.get("id") == item_id), None)
        if match is None:
            raise ValueError(f"unknown gate flag: {item_id}")
        kind = match.get("kind", "")
        character = match.get("character")
        drawings = tuple(d for d in (match.get("drawings") or []) if isinstance(d, str))
        size = match.get("size", 0.0)
        try:
            size_f = float(size)
        except (TypeError, ValueError):
            size_f = 0.0
        message = match.get("message", "")
        shas = tuple(s for s in (match.get("shas") or []) if isinstance(s, str))
        frames_raw = match.get("frames")
        frames: tuple[int, int] | None = None
        if isinstance(frames_raw, (list, tuple)) and len(frames_raw) == 2:
            try:
                frames = (int(frames_raw[0]), int(frames_raw[1]))
            except (TypeError, ValueError):
                frames = None
        flag = Flag(
            kind=str(kind),
            character=character if isinstance(character, str) else None,
            drawings=drawings,
            size=size_f,
            message=str(message),
            shas=shas,
            frames=frames,
        )
        acknowledge(prod, episode, shot, flag, note)
    studio.log.append(f"{shot}/review", "decision", action="keep", item=item_id, note=note)


def retake(studio: Studio, task_id: str, note: str) -> None:
    """Retake a department; with a review task for the shot this counts one round."""
    shot = task_id.rpartition("/")[0]
    review_id = f"{shot}/review"
    if review_id not in studio.board.tasks:
        studio.retake(task_id, note)
        return
    current = _count_round_or_hold(studio, shot)
    studio.retake(task_id, note)
    studio.log.append(review_id, "decision", action="retake", target=task_id, note=note, round=current)


def set_context(prod: Path, episode: str, task_id: str, add: list[str], remove: list[str], reason: str) -> None:
    """Write one task's entry in <episode>/run/context.yaml, keeping the others."""
    path = prod / episode / "run" / "context.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        raw = None
    except yaml.YAMLError:
        raw = None
    data: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    data[task_id] = {"add": list(add), "remove": list(remove), "reason": reason}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
