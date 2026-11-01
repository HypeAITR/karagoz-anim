"""Cross-file checks for one episode (spec v1)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from karagoz.spec.anchor import parse_anchor
from karagoz.spec.findings import Finding, SpecError
from karagoz.spec.io import load
from karagoz.spec.models import (
    Continuity,
    Hand,
    Hold,
    Lines,
    LoopEvent,
    Name,
    Place,
    Poses,
    Regions,
    SequenceEvent,
    Shots,
    Style,
    Transition,
    Xsheet,
)

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")

MIN_SUPPORT_CHANGE_FRAMES = 12


def norm_word(w: str) -> str:
    """Turkish-aware lower case with surrounding punctuation stripped."""
    t = w.replace("I", "ı").replace("İ", "i").lower()
    i = 0
    j = len(t)
    while i < j and not t[i].isalnum():
        i += 1
    while j > i and not t[j - 1].isalnum():
        j -= 1
    return t[i:j]


def words(text: str) -> list[str]:
    """Normalised words of a line in order."""
    out: list[str] = []
    for m in _WORD_RE.finditer(text):
        nw = norm_word(m.group(0))
        if nw:
            out.append(nw)
    return out


def _holds_prop(holds: dict[Hand, Name | None] | None, prop: str) -> bool:
    """True when either hand holds ``prop``."""
    return holds is not None and prop in holds.values()


def _view_side(view: str | None) -> int | None:
    if view in ("three_quarter_right", "profile_right"):
        return 1
    if view in ("three_quarter_left", "profile_left"):
        return -1
    return None


def _side_word(side: int) -> str:
    return "right" if side == 1 else "left"


def _place_x(place: Place) -> int:
    if place.seat is not None:
        return place.seat[0]
    return place.floor[0]


def _rel(path: Path, episode_dir: Path) -> str:
    try:
        rel = os.path.relpath(path, episode_dir)
    except ValueError:
        return path.as_posix()
    return Path(rel).as_posix()


def _find_episode_dir(start: Path) -> Path | None:
    cur = start
    seen: set[Path] = set()
    while cur not in seen:
        seen.add(cur)
        try:
            if (cur / "shots.yaml").is_file():
                return cur
        except OSError:
            pass
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def _check_support_change(
    findings: list[Finding], poses: Poses | None, char: str, prev: str | None, item: Transition, xrel: str, base: str
) -> None:
    if prev is None or poses is None or char not in poses.characters:
        return
    cposes = poses.characters[char]
    by_id = {p.id: p for p in cposes.poses}
    from_pose = by_id.get(prev)
    to_pose = by_id.get(item.to)
    pose_trans = next((t for t in (cposes.transitions or []) if t.from_ == prev and t.to == item.to), None)
    if from_pose is None or to_pose is None or pose_trans is None:
        return
    if from_pose.support is None or to_pose.support is None or from_pose.support == to_pose.support:
        return
    n = len(pose_trans.inbetweens)
    cadence_word = "ones" if item.cadence == "ones" else "twos"
    frames = n * (1 if item.cadence == "ones" else 2)
    if frames >= MIN_SUPPORT_CHANGE_FRAMES:
        return
    findings.append(
        Finding(
            "warn",
            "transition.too_short",
            xrel,
            base,
            f"{char} {prev} -> {item.to} ({from_pose.support} -> {to_pose.support})"
            f" is on screen for {frames} frames ({n} inbetweens on {cadence_word});"
            f" a whole-body move needs at least {MIN_SUPPORT_CHANGE_FRAMES}",
            "give the transition 6 inbetweens on twos (Layout) instead of stretching the hold",
        )
    )


def _check_characters_file(episode_dir: Path, shots: Shots | None, findings: list[Finding]) -> None:
    """Validate the production-root characters.yaml, when it exists (agent workdirs have none)."""
    prod = episode_dir.parent
    path = prod / "characters.yaml"
    if not path.is_file():
        return
    rel = _rel(path, episode_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        findings.append(Finding("error", "production.characters_invalid", rel, "", "Cannot read 'characters.yaml'."))
        return
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        findings.append(Finding("error", "production.characters_invalid", rel, "", "Cannot parse 'characters.yaml'."))
        return
    if not isinstance(data, dict) or not isinstance(data.get("characters"), dict):
        findings.append(
            Finding(
                "error",
                "production.characters_invalid",
                rel,
                "characters",
                "'characters.yaml' must contain 'characters: {name: {...}}'.",
            )
        )
        return
    chars = data["characters"]
    for name in sorted(chars.keys(), key=str):
        entry = chars[name]
        base = f"characters.{name}"
        if not isinstance(entry, dict):
            findings.append(
                Finding("error", "production.characters_invalid", rel, base, f"Character '{name}' must be a mapping.")
            )
            continue
        ref = entry.get("ref")
        ref_ok = isinstance(ref, str) and ref and (prod / ref).is_file()
        if not ref_ok:
            message = (
                f"Character '{name}' has no ref."
                if not isinstance(ref, str) or not ref
                else f"Character '{name}' ref '{ref}' does not exist."
            )
            findings.append(Finding("error", "production.characters_invalid", rel, f"{base}.ref", message))
        if entry.get("natural_facing") not in ("left", "right"):
            findings.append(
                Finding(
                    "error",
                    "production.characters_invalid",
                    rel,
                    f"{base}.natural_facing",
                    f"Character '{name}' natural_facing must be 'left' or 'right'.",
                )
            )
        if "mirror_ok" in entry and not isinstance(entry["mirror_ok"], bool):
            findings.append(
                Finding(
                    "error",
                    "production.characters_invalid",
                    rel,
                    f"{base}.mirror_ok",
                    f"Character '{name}' mirror_ok must be a bool.",
                )
            )
        if "height_cm" in entry:
            height = entry["height_cm"]
            if isinstance(height, bool) or not isinstance(height, (int, float)) or height <= 0:
                findings.append(
                    Finding(
                        "error",
                        "production.characters_invalid",
                        rel,
                        f"{base}.height_cm",
                        f"Character '{name}' height_cm must be a positive number.",
                    )
                )
    if shots is not None:
        known = set(chars.keys())
        seen: set[str] = set()
        for shot in shots.shots:
            for char in shot.characters or []:
                if char.name not in known and char.name not in seen:
                    seen.add(char.name)
                    findings.append(
                        Finding(
                            "error",
                            "production.character_unknown",
                            rel,
                            f"characters.{char.name}",
                            f"Character '{char.name}' is in shots.yaml but not in 'characters.yaml'.",
                        )
                    )


def _check_locations_file(episode_dir: Path, shots: Shots | None, findings: list[Finding]) -> None:
    """Validate the production-root locations.yaml, when it exists."""
    prod = episode_dir.parent
    path = prod / "locations.yaml"
    if not path.is_file():
        return
    rel = _rel(path, episode_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        findings.append(Finding("error", "production.locations_invalid", rel, "", "Cannot read 'locations.yaml'."))
        return
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        findings.append(Finding("error", "production.locations_invalid", rel, "", "Cannot parse 'locations.yaml'."))
        return
    if not isinstance(data, dict) or not isinstance(data.get("locations"), dict):
        findings.append(
            Finding(
                "error",
                "production.locations_invalid",
                rel,
                "locations",
                "'locations.yaml' must contain 'locations: {name: {description: str}}'.",
            )
        )
        return
    locations = data["locations"]
    for name in sorted(locations.keys(), key=str):
        entry = locations[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("description"), str):
            findings.append(
                Finding(
                    "error",
                    "production.locations_invalid",
                    rel,
                    f"locations.{name}",
                    f"Location '{name}' must be a mapping with a 'description' string.",
                )
            )
    if shots is not None:
        known = set(locations.keys())
        seen: set[str] = set()
        for shot in shots.shots:
            if shot.location in known or shot.location in seen:
                continue
            if (prod / "locations" / shot.location / "background.png").is_file():
                continue
            seen.add(shot.location)
            findings.append(
                Finding(
                    "warn",
                    "production.location_unknown",
                    rel,
                    f"locations.{shot.location}",
                    f"Location '{shot.location}' is in shots.yaml but not in 'locations.yaml' "
                    "and has no background.png.",
                )
            )


def _check_props_file(episode_dir: Path, findings: list[Finding]) -> None:
    """Validate the production-root props.yaml, when it exists."""
    path = episode_dir.parent / "props.yaml"
    if not path.is_file():
        return
    rel = _rel(path, episode_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        findings.append(Finding("error", "production.props_invalid", rel, "", "Cannot read 'props.yaml'."))
        return
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        findings.append(Finding("error", "production.props_invalid", rel, "", "Cannot parse 'props.yaml'."))
        return
    if not isinstance(data, dict) or not isinstance(data.get("props"), dict):
        findings.append(
            Finding(
                "error",
                "production.props_invalid",
                rel,
                "props",
                "'props.yaml' must contain 'props: {name: {description: str}}'.",
            )
        )
        return
    props = data["props"]
    for name in sorted(props.keys(), key=str):
        entry = props[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("description"), str):
            findings.append(
                Finding(
                    "error",
                    "production.props_invalid",
                    rel,
                    f"props.{name}",
                    f"Prop '{name}' must be a mapping with a 'description' string.",
                )
            )


def check_episode(episode_dir: Path, style_path: Path | None = None) -> list[Finding]:
    """Check every file of one episode; never raises for content problems."""
    findings: list[Finding] = []
    if style_path is None:
        style_path = episode_dir.parent / "style.yaml"

    def _load(path: Path) -> BaseModel | None:
        if not path.is_file():
            return None
        try:
            return load(path)
        except SpecError as exc:
            rel = _rel(path, episode_dir)
            for f in exc.findings:
                findings.append(Finding(f.level, f.code, rel, f.path, f.message, f.hint))
            return None

    lines_path = episode_dir / "lines.yaml"
    shots_path = episode_dir / "shots.yaml"
    continuity_path = episode_dir / "continuity.yaml"
    lines_obj = _load(lines_path)
    lines = lines_obj if isinstance(lines_obj, Lines) else None
    shots_obj = _load(shots_path)
    shots = shots_obj if isinstance(shots_obj, Shots) else None
    continuity_obj = _load(continuity_path)
    continuity = continuity_obj if isinstance(continuity_obj, Continuity) else None
    style_obj = _load(style_path)
    style = style_obj if isinstance(style_obj, Style) else None

    regions_by_location: dict[str, Regions] = {}
    if shots is not None:
        for location in sorted({sh.location for sh in shots.shots}):
            regions_path = episode_dir.parent / "locations" / location / "regions.yaml"
            if regions_path.is_file():
                regions_obj = _load(regions_path)
                if isinstance(regions_obj, Regions):
                    regions_by_location[location] = regions_obj

    poses_loaded: list[tuple[str, str, Poses]] = []
    xsheet_loaded: list[tuple[str, str, Xsheet]] = []
    shots_dir = episode_dir / "shots"
    try:
        children = sorted(shots_dir.iterdir(), key=lambda p: p.name) if shots_dir.is_dir() else []
    except OSError:
        children = []
    for child in children:
        if not child.is_dir():
            continue
        folder = child.name
        poses_path = child / "poses.yaml"
        if poses_path.is_file():
            obj = _load(poses_path)
            if isinstance(obj, Poses):
                poses_loaded.append((folder, _rel(poses_path, episode_dir), obj))
        xsheet_path = child / "xsheet.yaml"
        if xsheet_path.is_file():
            obj = _load(xsheet_path)
            if isinstance(obj, Xsheet):
                xsheet_loaded.append((folder, _rel(xsheet_path, episode_dir), obj))

    audio_dir = episode_dir / "audio"
    try:
        align_paths = sorted(audio_dir.glob("*.align.json")) if audio_dir.is_dir() else []
    except OSError:
        align_paths = []
    for ap in align_paths:
        _load(ap)

    lines_rel = _rel(lines_path, episode_dir)
    shots_rel = _rel(shots_path, episode_dir)
    continuity_rel = _rel(continuity_path, episode_dir)

    lines_by_id: dict[str, tuple[int, str, str]] = {}
    sounds_by_id: set[str] = set()
    line_words: dict[str, list[str]] = {}
    if lines is not None:
        for idx, le in enumerate(lines.lines):
            lines_by_id[le.id] = (idx, le.speaker, le.text)
            line_words[le.id] = words(le.text)
        if lines.sounds is not None:
            for se in lines.sounds:
                sounds_by_id.add(se.id)

    shots_by_id: dict[str, tuple[int, set[str]]] = {}
    shot_order: dict[str, int] = {}
    if shots is not None:
        for idx, sh in enumerate(shots.shots):
            chars: set[str] = set()
            if sh.characters is not None:
                for c in sh.characters:
                    chars.add(c.name)
            shots_by_id[sh.id] = (idx, chars)
            shot_order[sh.id] = idx

    beats: set[str] = set()
    if style is not None:
        beats = set(style.live.beats.keys())
    beats_hint = ", ".join(sorted(beats)) if beats else "(no known beats)"

    poses_by_shot: dict[str, Poses] = {}
    for _folder, _relf, pm in poses_loaded:
        if pm.shot not in poses_by_shot:
            poses_by_shot[pm.shot] = pm

    # Anchors wherever they appear.
    anchors: list[tuple[str, str, str, bool]] = []
    if lines is not None:
        for idx, le in enumerate(lines.lines):
            anchors.append((lines_rel, f"lines[{idx}].at", le.at, True))
        if lines.sounds is not None:
            for idx, se in enumerate(lines.sounds):
                anchors.append((lines_rel, f"sounds[{idx}].at", se.at, True))
    if shots is not None:
        for idx, sh in enumerate(shots.shots):
            anchors.append((shots_rel, f"shots[{idx}].in", sh.in_, False))
            anchors.append((shots_rel, f"shots[{idx}].out", sh.out, False))
    for _folder, xrel, xs in xsheet_loaded:
        for track_name, items in xs.tracks.items():
            for j, item in enumerate(items):
                base = f"tracks.{track_name}[{j}]"
                anchors.append((xrel, f"{base}.at", item.at, False))
                if isinstance(item, LoopEvent):
                    anchors.append((xrel, f"{base}.until", item.until, False))
        if xs.camera is not None:
            for k, cam in enumerate(xs.camera):
                anchors.append((xrel, f"camera[{k}].at", cam.at, False))
                if cam.until is not None:
                    anchors.append((xrel, f"camera[{k}].until", cam.until, False))
        if xs.mouth is not None:
            for char, mt in xs.mouth.items():
                if mt.edits is not None:
                    for m_idx, ed in enumerate(mt.edits):
                        anchors.append((xrel, f"mouth.{char}.edits[{m_idx}].at", ed.at, False))
        if xs.fx is not None:
            for f_idx, fx in enumerate(xs.fx):
                anchors.append((xrel, f"fx[{f_idx}].at", fx.at, False))
                if fx.until is not None:
                    anchors.append((xrel, f"fx[{f_idx}].until", fx.until, False))

    for afile, apath, atext, in_lines in anchors:
        try:
            a = parse_anchor(atext)
        except ValueError:
            continue
        if a.point in ("line", "word") and a.ref is not None:
            if lines is not None and a.ref not in lines_by_id:
                findings.append(
                    Finding(
                        "error", "anchor.unknown_line", afile, apath, f"Unknown line '{a.ref}' in anchor '{atext}'."
                    )
                )
            elif a.point == "word" and lines is not None and a.ref in lines_by_id and a.word is not None:
                lw = line_words.get(a.ref, [])
                nw = norm_word(a.word)
                count = sum(1 for w in lw if w == nw)
                if count < a.occurrence:
                    if a.occurrence == 1:
                        msg = f"Word '{a.word}' is not in line '{a.ref}'."
                    else:
                        msg = f"Word '{a.word}' occurs fewer than {a.occurrence} times in line '{a.ref}'."
                    hint = ", ".join(lw) if lw else "(line has no words)"
                    findings.append(Finding("error", "anchor.unknown_word", afile, apath, msg, hint))
        if a.point == "sound" and a.ref is not None and lines is not None and a.ref not in sounds_by_id:
            findings.append(
                Finding("error", "anchor.unknown_sound", afile, apath, f"Unknown sound '{a.ref}' in anchor '{atext}'.")
            )
        if a.point == "shot" and a.ref is not None and shots is not None and a.ref not in shots_by_id:
            findings.append(
                Finding("error", "anchor.unknown_shot", afile, apath, f"Unknown shot '{a.ref}' in anchor '{atext}'.")
            )
        if style is not None:
            for off in a.offsets:
                if off.kind == "beat" and isinstance(off.value, str) and off.value not in beats:
                    findings.append(
                        Finding(
                            "error",
                            "anchor.unknown_beat",
                            afile,
                            apath,
                            f"Unknown beat '{off.value}' in anchor '{atext}'.",
                            beats_hint,
                        )
                    )
        if in_lines and a.point in ("shot.in", "shot.out"):
            findings.append(
                Finding("error", "anchor.out_of_scope", afile, apath, f"'{a.point}' is only meaningful inside a shot.")
            )
        if any(off.kind == "frames" for off in a.offsets):
            findings.append(
                Finding("info", "anchor.raw_frames", afile, apath, f"Anchor '{atext}' uses a raw frame offset.")
            )

    # Shots referencing lines.
    if shots is not None:
        first_seen: dict[str, int] = {}
        for idx, sh in enumerate(shots.shots):
            if sh.lines is None:
                continue
            for j, lid in enumerate(sh.lines):
                if lines is not None and lid not in lines_by_id:
                    findings.append(
                        Finding(
                            "error",
                            "shot.unknown_line",
                            shots_rel,
                            f"shots[{idx}].lines[{j}]",
                            f"Unknown line '{lid}' in shot '{sh.id}'.",
                        )
                    )
                if lid not in first_seen:
                    first_seen[lid] = idx
                elif first_seen[lid] != idx:
                    findings.append(
                        Finding(
                            "error",
                            "shot.line_twice",
                            shots_rel,
                            f"shots[{idx}].lines[{j}]",
                            f"Line '{lid}' is listed by two shots.",
                        )
                    )
        # Line speaker missing from the shot that lists it (reported in lines.yaml).
        if lines is not None:
            for idx, sh in enumerate(shots.shots):
                if sh.lines is None:
                    continue
                _sidx, chars = shots_by_id.get(sh.id, (idx, set()))
                for lid in sh.lines:
                    entry = lines_by_id.get(lid)
                    if entry is None:
                        continue
                    lidx, speaker, _text = entry
                    if speaker not in chars:
                        findings.append(
                            Finding(
                                "warn",
                                "line.speaker_missing",
                                lines_rel,
                                f"lines[{lidx}].speaker",
                                f"Speaker '{speaker}' of line '{lid}' is not in shot '{sh.id}'.",
                            )
                        )

    # Poses/xsheet shot folders and characters.
    for folder, prel, pm in poses_loaded:
        in_shots = True
        if shots is not None:
            in_shots = pm.shot in shots_by_id
        differs = pm.shot != folder
        if not in_shots or differs:
            if not in_shots and differs:
                msg = f"Shot '{pm.shot}' is not in shots.yaml and differs from folder '{folder}'."
            elif not in_shots:
                msg = f"Unknown shot '{pm.shot}' in poses.yaml."
            else:
                msg = f"Shot '{pm.shot}' differs from folder '{folder}'."
            findings.append(Finding("error", "poses.unknown_shot", prel, "shot", msg))
        if shots is not None and pm.shot in shots_by_id:
            _sidx, chars = shots_by_id[pm.shot]
            for char in pm.characters:
                if char not in chars:
                    findings.append(
                        Finding(
                            "error",
                            "poses.unknown_character",
                            prel,
                            f"characters.{char}",
                            f"Unknown character '{char}' in poses for shot '{pm.shot}'.",
                        )
                    )
    for folder, xrel, xs in xsheet_loaded:
        in_shots = True
        if shots is not None:
            in_shots = xs.shot in shots_by_id
        differs = xs.shot != folder
        if not in_shots or differs:
            if not in_shots and differs:
                msg = f"Shot '{xs.shot}' is not in shots.yaml and differs from folder '{folder}'."
            elif not in_shots:
                msg = f"Unknown shot '{xs.shot}' in xsheet.yaml."
            else:
                msg = f"Shot '{xs.shot}' differs from folder '{folder}'."
            findings.append(Finding("error", "xsheet.unknown_shot", xrel, "shot", msg))
        if shots is not None and xs.shot in shots_by_id:
            _sidx, chars = shots_by_id[xs.shot]
            for track_name in xs.tracks:
                char = track_name.rsplit(".", 1)[0]
                if char not in chars:
                    findings.append(
                        Finding(
                            "error",
                            "xsheet.unknown_character",
                            xrel,
                            f"tracks.{track_name}",
                            f"Unknown character '{char}' in track '{track_name}'.",
                        )
                    )
            if xs.mouth is not None:
                for char in xs.mouth:
                    if char not in chars:
                        findings.append(
                            Finding(
                                "error",
                                "xsheet.unknown_character",
                                xrel,
                                f"mouth.{char}",
                                f"Unknown character '{char}' in mouth track.",
                            )
                        )
            if xs.camera is not None:
                hint = ", ".join(sorted(chars))
                for k, cam in enumerate(xs.camera):
                    if cam.target is None:
                        continue
                    char = cam.target.split(".", 1)[0]
                    if char not in chars:
                        findings.append(
                            Finding(
                                "error",
                                "camera.unknown_target",
                                xrel,
                                f"camera[{k}].target",
                                f"Unknown character '{char}' in camera target '{cam.target}'.",
                                hint,
                            )
                        )

    # Poses.yaml internal references: duplicate ids, transitions and sequences naming undefined poses.
    for _pfolder, prel, pm in poses_loaded:
        for pchar, pcposes in pm.characters.items():
            seen_ids: set[str] = set()
            for k, pose in enumerate(pcposes.poses):
                if pose.id in seen_ids:
                    findings.append(
                        Finding(
                            "error",
                            "pose.duplicate",
                            prel,
                            f"characters.{pchar}.poses[{k}]",
                            f"Duplicate pose id '{pose.id}' for '{pchar}' in shot '{pm.shot}'.",
                        )
                    )
                else:
                    seen_ids.add(pose.id)
            for i, ptrans in enumerate(pcposes.transitions or []):
                for side, ref in (("from", ptrans.from_), ("to", ptrans.to)):
                    if ref not in seen_ids:
                        findings.append(
                            Finding(
                                "error",
                                "pose.undefined",
                                prel,
                                f"characters.{pchar}.transitions[{i}].{side}",
                                f"Transition '{ptrans.from_}' -> '{ptrans.to}' of '{pchar}': "
                                f"pose '{ref}' is not defined.",
                            )
                        )
            for i, seq in enumerate(pcposes.sequences or []):
                refs: list[tuple[str, str]] = [("from", seq.from_)]
                if seq.to is not None:
                    refs.append(("to", seq.to))
                for side, ref in refs:
                    if ref not in seen_ids:
                        findings.append(
                            Finding(
                                "error",
                                "pose.undefined",
                                prel,
                                f"characters.{pchar}.sequences[{i}].{side}",
                                f"Sequence '{seq.id}' of '{pchar}': pose '{ref}' is not defined.",
                            )
                        )

    # Pose references, first hold, transitions.
    for _folder, xrel, xs in xsheet_loaded:
        poses = poses_by_shot.get(xs.shot)
        for track_name, items in xs.tracks.items():
            char = track_name.rsplit(".", 1)[0]
            kind = track_name.rsplit(".", 1)[-1] if "." in track_name else ""
            if kind == "body":
                if not items:
                    findings.append(
                        Finding(
                            "error",
                            "track.first_not_hold",
                            xrel,
                            f"tracks.{track_name}",
                            f"Track '{track_name}' must start with a Hold at shot.in.",
                        )
                    )
                else:
                    first = items[0]
                    if not isinstance(first, Hold) or first.at != "shot.in":
                        findings.append(
                            Finding(
                                "error",
                                "track.first_not_hold",
                                xrel,
                                f"tracks.{track_name}[0].at",
                                f"Track '{track_name}' must start with a Hold at shot.in.",
                            )
                        )
            prev: str | None = None
            for j, item in enumerate(items):
                base = f"tracks.{track_name}[{j}]"
                if isinstance(item, Hold):
                    defined = False
                    if poses is not None and char in poses.characters:
                        defined = any(p.id == item.pose for p in poses.characters[char].poses)
                    if not defined:
                        findings.append(
                            Finding(
                                "error",
                                "pose.undefined",
                                xrel,
                                f"{base}.pose",
                                f"Pose '{item.pose}' is not defined for '{char}' in shot '{xs.shot}'.",
                            )
                        )
                    prev = item.pose
                elif isinstance(item, Transition):
                    defined = False
                    if poses is not None and char in poses.characters:
                        defined = any(p.id == item.to for p in poses.characters[char].poses)
                    if not defined:
                        findings.append(
                            Finding(
                                "error",
                                "pose.undefined",
                                xrel,
                                f"{base}.to",
                                f"Pose '{item.to}' is not defined for '{char}' in shot '{xs.shot}'.",
                            )
                        )
                    if item.via == "inbetween" and prev is not None and poses is not None and char in poses.characters:
                        cposes = poses.characters[char]
                        ids = {p.id for p in cposes.poses}
                        if prev in ids and item.to in ids:
                            trans = cposes.transitions or []
                            if not any(t.from_ == prev and t.to == item.to for t in trans):
                                findings.append(
                                    Finding(
                                        "warn",
                                        "pose.no_transition",
                                        xrel,
                                        f"{base}.to",
                                        f"No inbetween transition from '{prev}' to '{item.to}' for '{char}'.",
                                    )
                                )
                    if kind == "body" and item.via == "inbetween":
                        _check_support_change(findings, poses, char, prev, item, xrel, base)
                    prev = item.to
                elif isinstance(item, LoopEvent):
                    if item.step is not None and style is not None:
                        beat_name = item.step.split(".", 1)[1] if "." in item.step else item.step
                        if beat_name not in beats:
                            findings.append(
                                Finding(
                                    "error",
                                    "loop.unknown_beat",
                                    xrel,
                                    f"{base}.step",
                                    f"Unknown beat '{beat_name}' in loop step '{item.step}'.",
                                    beats_hint,
                                )
                            )
                    for k, loop_item in enumerate(item.loop):
                        if loop_item.startswith("p"):
                            defined = False
                            if poses is not None and char in poses.characters:
                                defined = any(p.id == loop_item for p in poses.characters[char].poses)
                            if not defined:
                                findings.append(
                                    Finding(
                                        "error",
                                        "pose.undefined",
                                        xrel,
                                        f"{base}.loop[{k}]",
                                        f"Pose '{loop_item}' is not defined for '{char}' in shot '{xs.shot}'.",
                                    )
                                )
                        else:
                            defined = False
                            if poses is not None and char in poses.characters:
                                seqs = poses.characters[char].sequences or []
                                defined = any(s.id == loop_item for s in seqs)
                            if not defined:
                                findings.append(
                                    Finding(
                                        "error",
                                        "pose.undefined",
                                        xrel,
                                        f"{base}.loop[{k}]",
                                        f"Sequence '{loop_item}' is not defined for '{char}' in shot '{xs.shot}'.",
                                    )
                                )
                elif isinstance(item, SequenceEvent):
                    defined = False
                    if poses is not None and char in poses.characters:
                        seqs = poses.characters[char].sequences or []
                        defined = any(s.id == item.sequence for s in seqs)
                    if not defined:
                        findings.append(
                            Finding(
                                "error",
                                "pose.undefined",
                                xrel,
                                f"{base}.sequence",
                                f"Sequence '{item.sequence}' is not defined for '{char}' in shot '{xs.shot}'.",
                            )
                        )

    # Continuity and props.
    if continuity is not None:
        if shots is not None:
            for i, st in enumerate(continuity.states):
                if st.shot not in shots_by_id:
                    findings.append(
                        Finding(
                            "error",
                            "continuity.unknown_shot",
                            continuity_rel,
                            f"states[{i}].shot",
                            f"Unknown shot '{st.shot}' in continuity.",
                        )
                    )
            if continuity.props is not None:
                for j, pr in enumerate(continuity.props):
                    if pr.shot not in shots_by_id:
                        findings.append(
                            Finding(
                                "error",
                                "continuity.unknown_shot",
                                continuity_rel,
                                f"props[{j}].shot",
                                f"Unknown shot '{pr.shot}' in continuity.",
                            )
                        )
            for i, st in enumerate(continuity.states):
                if st.shot in shots_by_id and st.character not in shots_by_id[st.shot][1]:
                    findings.append(
                        Finding(
                            "error",
                            "continuity.unknown_character",
                            continuity_rel,
                            f"states[{i}].character",
                            f"Unknown character '{st.character}' for shot '{st.shot}' in continuity.",
                        )
                    )
            if continuity.props is not None:
                for j, pr in enumerate(continuity.props):
                    if pr.shot not in shots_by_id:
                        continue
                    chars = shots_by_id[pr.shot][1]
                    if pr.in_.holder is not None and pr.in_.holder not in chars:
                        findings.append(
                            Finding(
                                "error",
                                "continuity.unknown_character",
                                continuity_rel,
                                f"props[{j}].in.holder",
                                f"Unknown holder '{pr.in_.holder}' for shot '{pr.shot}' in continuity.",
                            )
                        )
                    if pr.out.holder is not None and pr.out.holder not in chars:
                        findings.append(
                            Finding(
                                "error",
                                "continuity.unknown_character",
                                continuity_rel,
                                f"props[{j}].out.holder",
                                f"Unknown holder '{pr.out.holder}' for shot '{pr.shot}' in continuity.",
                            )
                        )
        prop_states = continuity.prop_states
        if continuity.props is not None:
            for j, pr in enumerate(continuity.props):
                if prop_states is None or pr.prop not in prop_states:
                    findings.append(
                        Finding(
                            "error",
                            "prop.unknown_kind",
                            continuity_rel,
                            f"props[{j}].prop",
                            f"Unknown prop '{pr.prop}'.",
                        )
                    )
            if prop_states is not None:
                for j, pr in enumerate(continuity.props):
                    if pr.prop not in prop_states:
                        continue
                    allowed = prop_states[pr.prop]
                    if pr.in_.state not in allowed:
                        findings.append(
                            Finding(
                                "error",
                                "prop.unknown_state",
                                continuity_rel,
                                f"props[{j}].in.state",
                                f"Unknown state '{pr.in_.state}' for prop '{pr.prop}'.",
                            )
                        )
                    if pr.out.state not in allowed:
                        findings.append(
                            Finding(
                                "error",
                                "prop.unknown_state",
                                continuity_rel,
                                f"props[{j}].out.state",
                                f"Unknown state '{pr.out.state}' for prop '{pr.prop}'.",
                            )
                        )
                for j, pr in enumerate(continuity.props):
                    if pr.prop not in prop_states:
                        continue
                    allowed = prop_states[pr.prop]
                    if (
                        pr.in_.state in allowed
                        and pr.out.state in allowed
                        and allowed.index(pr.out.state) < allowed.index(pr.in_.state)
                    ):
                        findings.append(
                            Finding(
                                "error",
                                "prop.backwards",
                                continuity_rel,
                                f"props[{j}].out.state",
                                f"Prop '{pr.prop}' moves backwards from '{pr.in_.state}' to '{pr.out.state}'.",
                            )
                        )
            if shots is not None and continuity.props is not None:
                groups: dict[str, list[tuple[int, int]]] = {}
                for j, pr in enumerate(continuity.props):
                    if pr.shot not in shot_order:
                        continue
                    groups.setdefault(pr.prop, []).append((shot_order[pr.shot], j))
                for prop, lst in groups.items():
                    lst.sort()
                    for k in range(1, len(lst)):
                        prev_j = lst[k - 1][1]
                        cur_j = lst[k][1]
                        prev_pr = continuity.props[prev_j]
                        cur_pr = continuity.props[cur_j]
                        if prev_pr.shot == cur_pr.shot:
                            continue
                        oh = prev_pr.out.holder
                        ih = cur_pr.in_.holder
                        os_ = prev_pr.out.state
                        is_ = cur_pr.in_.state
                        if oh != ih or os_ != is_:
                            oh_s = oh if oh is not None else "(none)"
                            ih_s = ih if ih is not None else "(none)"
                            findings.append(
                                Finding(
                                    "error",
                                    "prop.discontinuity",
                                    continuity_rel,
                                    f"props[{cur_j}].in",
                                    f"Prop '{prop}' jumps from holder '{oh_s}' state '{os_}' "
                                    f"to holder '{ih_s}' state '{is_}'.",
                                )
                            )
                        if prop_states is not None and prop in prop_states:
                            allowed = prop_states[prop]
                            if os_ in allowed and is_ in allowed and allowed.index(is_) < allowed.index(os_):
                                findings.append(
                                    Finding(
                                        "error",
                                        "prop.backwards",
                                        continuity_rel,
                                        f"props[{cur_j}].in.state",
                                        f"Prop '{prop}' moves backwards from '{os_}' to '{is_}'.",
                                    )
                                )

    if lines is not None and shots is not None:
        from karagoz.engine.xsheet import check_timing

        findings.extend(check_timing(episode_dir, style_path))

    # Spec v1.1: named places and objects from regions.yaml, pose support, held props.
    shot_locations: dict[str, str] = {}
    shot_places: dict[str, dict[str, str | None]] = {}
    if shots is not None:
        for sh in shots.shots:
            shot_locations[sh.id] = sh.location
            shot_places[sh.id] = {c.name: c.place for c in (sh.characters or [])}

    if shots is not None:
        for idx, sh in enumerate(shots.shots):
            regions = regions_by_location.get(sh.location)
            if regions is None or sh.characters is None:
                continue
            place_names = {p.name for p in regions.places}
            place_hint = ", ".join(sorted(place_names))
            for j, c in enumerate(sh.characters):
                if c.place is not None and c.place not in place_names:
                    findings.append(
                        Finding(
                            "error",
                            "place.unknown",
                            shots_rel,
                            f"shots[{idx}].characters[{j}].place",
                            f"Unknown place '{c.place}' for '{c.name}' in shot '{sh.id}'.",
                            place_hint,
                        )
                    )

    prop_hint: str | None = None
    if continuity is not None and continuity.prop_states is not None:
        prop_hint = ", ".join(sorted(continuity.prop_states))

    for _folder, prel, pm in poses_loaded:
        for char, cposes in pm.characters.items():
            for k, pose in enumerate(cposes.poses):
                base = f"characters.{char}.poses[{k}]"
                if pose.support is None:
                    findings.append(
                        Finding(
                            "warn",
                            "support.missing",
                            prel,
                            base,
                            f"Pose '{pose.id}' of '{char}' has no support.",
                        )
                    )
                if pose.holds is not None and continuity is not None and continuity.prop_states is not None:
                    for hand, held in pose.holds.items():
                        if held is not None and held not in continuity.prop_states:
                            findings.append(
                                Finding(
                                    "warn",
                                    "holds.unknown_prop",
                                    prel,
                                    f"{base}.holds.{hand}",
                                    f"Unknown prop '{held}' held by '{char}'.",
                                    prop_hint,
                                )
                            )
        for char, cposes in pm.characters.items():
            poses_by_id = {p.id: p for p in cposes.poses}
            for i, tr in enumerate(cposes.transitions or []):
                tbase = f"characters.{char}.transitions[{i}].holds"
                if tr.holds is not None and len(tr.holds) != len(tr.inbetweens):
                    findings.append(
                        Finding(
                            "error",
                            "holds.inbetween_count",
                            prel,
                            tbase,
                            f"Transition '{tr.from_}' -> '{tr.to}' of '{char}' has "
                            f"{len(tr.holds)} holds entries for {len(tr.inbetweens)} inbetweens.",
                        )
                    )
                from_pose = poses_by_id.get(tr.from_)
                to_pose = poses_by_id.get(tr.to)
                if (
                    from_pose is not None
                    and to_pose is not None
                    and from_pose.holds is not None
                    and to_pose.holds is not None
                    and from_pose.holds != to_pose.holds
                    and (tr.holds is None or any(e is None for e in tr.holds))
                ):
                    findings.append(
                        Finding(
                            "warn",
                            "holds.inbetween_missing",
                            prel,
                            tbase,
                            f"{char} {tr.from_} -> {tr.to}: the hands change; write each "
                            "inbetween's hands in the transition's holds "
                            "(otherwise its drawings get no HANDS line)",
                        )
                    )
                if tr.holds is not None and continuity is not None and continuity.prop_states is not None:
                    for k_entry, hold_entry in enumerate(tr.holds):
                        if hold_entry is None:
                            continue
                        for hand, held in hold_entry.items():
                            if held is not None and held not in continuity.prop_states:
                                findings.append(
                                    Finding(
                                        "warn",
                                        "holds.unknown_prop",
                                        prel,
                                        f"{tbase}[{k_entry}].{hand}",
                                        f"Unknown prop '{held}' held by '{char}'.",
                                        prop_hint,
                                    )
                                )
        shot_location = shot_locations.get(pm.shot)
        regions = regions_by_location.get(shot_location) if shot_location is not None else None
        if regions is None:
            continue
        place_names = {p.name for p in regions.places}
        place_kinds = {p.name: p.kind for p in regions.places}
        object_names = {o.name for o in regions.objects}
        place_hint = ", ".join(sorted(place_names))
        object_hint = ", ".join(sorted(object_names))
        char_places = shot_places.get(pm.shot, {})
        for char, cposes in pm.characters.items():
            for k, pose in enumerate(cposes.poses):
                base = f"characters.{char}.poses[{k}]"
                if pose.on is not None and pose.on not in place_names:
                    findings.append(
                        Finding(
                            "error",
                            "place.unknown",
                            prel,
                            f"{base}.on",
                            f"Unknown place '{pose.on}' for '{char}' pose '{pose.id}'.",
                            place_hint,
                        )
                    )
                if pose.rests_on is not None:
                    for m, rest in enumerate(pose.rests_on):
                        if rest not in object_names:
                            findings.append(
                                Finding(
                                    "error",
                                    "rests_on.unknown",
                                    prel,
                                    f"{base}.rests_on[{m}]",
                                    f"Unknown object '{rest}' for '{char}' pose '{pose.id}'.",
                                    object_hint,
                                )
                            )
                if pose.support == "sit":
                    at = pose.on if pose.on is not None else char_places.get(char)
                    if at is not None and place_kinds.get(at) == "floor":
                        findings.append(
                            Finding(
                                "error",
                                "support.not_seat",
                                prel,
                                f"{base}.support",
                                f"Pose '{pose.id}' of '{char}' sits on floor place '{at}'.",
                            )
                        )

    # Handover inside one shot: the giver lets go at the contact key, the receiver takes there (D34). Only
    # when continuity names a giver and a receiver: picking a prop up over several inbetweens is fine.
    if continuity is not None and continuity.props is not None:
        for _folder, prel, pm in poses_loaded:
            for pr in continuity.props:
                if pr.shot != pm.shot:
                    continue
                giver = pr.in_.holder
                receiver = pr.out.holder
                if giver is None or receiver is None or giver == receiver:
                    continue
                giver_poses = pm.characters.get(giver)
                if giver_poses is not None:
                    by_id = {p.id: p for p in giver_poses.poses}
                    for i, tr in enumerate(giver_poses.transitions or []):
                        fp = by_id.get(tr.from_)
                        tp = by_id.get(tr.to)
                        if fp is None or tp is None or tr.holds is None:
                            continue
                        if not _holds_prop(fp.holds, pr.prop) or _holds_prop(tp.holds, pr.prop):
                            continue
                        for k, entry_holds in enumerate(tr.holds):
                            if entry_holds is None or not _holds_prop(entry_holds, pr.prop):
                                continue
                            findings.append(
                                Finding(
                                    "warn",
                                    "handover.holds_after_release",
                                    prel,
                                    f"characters.{giver}.transitions[{i}].holds[{k}]",
                                    f"{giver} {tr.from_} -> {tr.to}: the giver lets go at {tr.from_}, but "
                                    f"inbetween {k + 1} still holds '{pr.prop}' (two props on screen)",
                                )
                            )
                receiver_poses = pm.characters.get(receiver)
                if receiver_poses is not None:
                    by_id = {p.id: p for p in receiver_poses.poses}
                    for i, tr in enumerate(receiver_poses.transitions or []):
                        fp = by_id.get(tr.from_)
                        tp = by_id.get(tr.to)
                        if fp is None or tp is None or tr.holds is None:
                            continue
                        if _holds_prop(fp.holds, pr.prop) or not _holds_prop(tp.holds, pr.prop):
                            continue
                        for k, entry_holds in enumerate(tr.holds):
                            if entry_holds is None or not _holds_prop(entry_holds, pr.prop):
                                continue
                            findings.append(
                                Finding(
                                    "warn",
                                    "handover.holds_before_take",
                                    prel,
                                    f"characters.{receiver}.transitions[{i}].holds[{k}]",
                                    f"{receiver} {tr.from_} -> {tr.to}: the receiver takes at {tr.to}, but "
                                    f"inbetween {k + 1} already holds '{pr.prop}' (two props on screen)",
                                )
                            )

    # WP-20: gaze and the 180-degree line, from names only.
    poses_rel_by_shot: dict[str, list[tuple[str, Poses]]] = {}
    for _pfolder, _prel, _pm in poses_loaded:
        poses_rel_by_shot.setdefault(_pm.shot, []).append((_prel, _pm))

    expected_by_shot: dict[str, dict[str, tuple[int | None, str | None]]] = {}
    if shots is not None:
        for s_idx, sh in enumerate(shots.shots):
            regions = regions_by_location.get(sh.location)
            place_map: dict[str, Place] = {}
            if regions is not None:
                place_map = {p.name: p for p in regions.places}
            object_map = {o.name: o for o in regions.objects} if regions is not None else {}
            g_chars = sh.characters or []
            char_by_name = {c.name: c for c in g_chars}
            char_x: dict[str, float | None] = {}
            for c in g_chars:
                cx: float | None = None
                if regions is not None and c.place is not None:
                    pl = place_map.get(c.place)
                    if pl is not None:
                        cx = float(_place_x(pl))
                char_x[c.name] = cx
            expected: dict[str, tuple[int | None, str | None]] = {}
            for c_idx, c in enumerate(g_chars):
                if c.offscreen is True:
                    expected[c.name] = (None, c.facing)
                    continue
                if c.facing is None:
                    expected[c.name] = (None, None)
                    continue
                target = c.facing
                is_character = target in char_by_name and target != c.name
                if not is_character:
                    if regions is None:
                        findings.append(
                            Finding(
                                "warn",
                                "gaze.unknown_target",
                                shots_rel,
                                f"shots[{s_idx}].characters[{c_idx}].facing",
                                f"{c.name} faces '{target}', which is not a character of {sh.id}",
                            )
                        )
                    elif target not in object_map and target not in place_map:
                        findings.append(
                            Finding(
                                "warn",
                                "gaze.unknown_target",
                                shots_rel,
                                f"shots[{s_idx}].characters[{c_idx}].facing",
                                f"{c.name} faces '{target}', which is not a character of {sh.id}"
                                f" nor an object/place of {sh.location}",
                            )
                        )
                target_x: float | None = None
                if target in char_by_name and target != c.name:
                    t_char = char_by_name[target]
                    if t_char.offscreen is not True:
                        target_x = char_x.get(t_char.name)
                elif regions is not None:
                    if target in object_map:
                        box = object_map[target].box
                        target_x = (box[0] + box[2]) / 2
                    elif target in place_map:
                        target_x = float(_place_x(place_map[target]))
                self_x = char_x.get(c.name)
                exp_side: int | None = None
                if self_x is not None and target_x is not None:
                    diff = target_x - self_x
                    if diff >= 1:
                        exp_side = 1
                    elif diff <= -1:
                        exp_side = -1
                expected[c.name] = (exp_side, target)
                if exp_side is not None:
                    view_side = _view_side(c.view)
                    if view_side is not None and view_side != exp_side:
                        findings.append(
                            Finding(
                                "warn",
                                "gaze.wrong_side",
                                shots_rel,
                                f"shots[{s_idx}].characters[{c_idx}].view",
                                f"{c.name} faces {target} on screen {_side_word(exp_side)}"
                                f" but view '{c.view}' looks {_side_word(view_side)}",
                            )
                        )
            expected_by_shot[sh.id] = expected

    if shots is not None:
        for sh in shots.shots:
            exp_map = expected_by_shot.get(sh.id)
            if not exp_map:
                continue
            for prel_file, pm in poses_rel_by_shot.get(sh.id, []):
                for char_name, (exp_side, p_target) in exp_map.items():
                    if exp_side is None or p_target is None:
                        continue
                    g_cposes = pm.characters.get(char_name)
                    if g_cposes is None:
                        continue
                    for k, pose in enumerate(g_cposes.poses):
                        pose_side = _view_side(pose.view)
                        if pose_side is not None and pose_side != exp_side:
                            findings.append(
                                Finding(
                                    "info",
                                    "gaze.pose_away",
                                    prel_file,
                                    f"characters.{char_name}.poses[{k}].view",
                                    f"{char_name} {pose.id} looks {_side_word(pose_side)}, away from {p_target}",
                                )
                            )

    if shots is not None:
        for ax_i in range(len(shots.shots) - 1):
            s1 = shots.shots[ax_i]
            s2 = shots.shots[ax_i + 1]
            if s1.scene != s2.scene:
                continue
            on1 = {c.name: c for c in (s1.characters or []) if c.offscreen is not True}
            on2 = {c.name: c for c in (s2.characters or []) if c.offscreen is not True}
            common = sorted(set(on1) & set(on2))
            if len(common) < 2:
                continue
            same_location = s1.location == s2.location
            reg1 = regions_by_location.get(s1.location)
            reg2 = regions_by_location.get(s2.location)
            pmap1: dict[str, Place] = {}
            if reg1 is not None:
                pmap1 = {p.name: p for p in reg1.places}
            pmap2: dict[str, Place] = {}
            if reg2 is not None:
                pmap2 = {p.name: p for p in reg2.places}
            ax1: dict[str, float | None] = {}
            for ax_name, ax_c in on1.items():
                ax_v: float | None = None
                if reg1 is not None and ax_c.place is not None:
                    ax_pl = pmap1.get(ax_c.place)
                    if ax_pl is not None:
                        ax_v = float(_place_x(ax_pl))
                ax1[ax_name] = ax_v
            ax2: dict[str, float | None] = {}
            for ax_name, ax_c in on2.items():
                ax_v2: float | None = None
                if reg2 is not None and ax_c.place is not None:
                    ax_pl2 = pmap2.get(ax_c.place)
                    if ax_pl2 is not None:
                        ax_v2 = float(_place_x(ax_pl2))
                ax2[ax_name] = ax_v2
            for ia in range(len(common)):
                for ib in range(ia + 1, len(common)):
                    pair_a = common[ia]
                    pair_b = common[ib]
                    if same_location and reg1 is not None:
                        xa1 = ax1.get(pair_a)
                        xb1 = ax1.get(pair_b)
                        xa2 = ax2.get(pair_a)
                        xb2 = ax2.get(pair_b)
                        if xa1 is not None and xb1 is not None and xa2 is not None and xb2 is not None:
                            d1 = xb1 - xa1
                            d2 = xb2 - xa2
                            sgn1 = 1 if d1 > 0 else (-1 if d1 < 0 else 0)
                            sgn2 = 1 if d2 > 0 else (-1 if d2 < 0 else 0)
                            if sgn1 != 0 and sgn2 != 0 and sgn1 != sgn2:
                                findings.append(
                                    Finding(
                                        "warn",
                                        "axis.crossed",
                                        shots_rel,
                                        f"shots[{ax_i + 1}].characters",
                                        f"{pair_a} and {pair_b} swap screen sides between {s1.id} and {s2.id}"
                                        " (crossing the 180° line)",
                                    )
                                )
                            continue
                    pa1 = on1[pair_a]
                    pb1 = on1[pair_b]
                    pa2 = on2[pair_a]
                    pb2 = on2[pair_b]
                    crossed_done = False
                    if pa1.facing == pair_b and pa2.facing == pair_b:
                        va1 = _view_side(pa1.view)
                        va2 = _view_side(pa2.view)
                        if va1 is not None and va2 is not None and va1 != va2:
                            findings.append(
                                Finding(
                                    "warn",
                                    "axis.crossed",
                                    shots_rel,
                                    f"shots[{ax_i + 1}].characters",
                                    f"{pair_a} looks {_side_word(va1)} at {pair_b} in {s1.id}"
                                    f" but {_side_word(va2)} in {s2.id} (crossing the 180° line)",
                                )
                            )
                            crossed_done = True
                    if not crossed_done and pb1.facing == pair_a and pb2.facing == pair_a:
                        vb1 = _view_side(pb1.view)
                        vb2 = _view_side(pb2.view)
                        if vb1 is not None and vb2 is not None and vb1 != vb2:
                            findings.append(
                                Finding(
                                    "warn",
                                    "axis.crossed",
                                    shots_rel,
                                    f"shots[{ax_i + 1}].characters",
                                    f"{pair_b} looks {_side_word(vb1)} at {pair_a} in {s1.id}"
                                    f" but {_side_word(vb2)} in {s2.id} (crossing the 180° line)",
                                )
                            )

    _check_characters_file(episode_dir, shots, findings)
    _check_locations_file(episode_dir, shots, findings)
    _check_props_file(episode_dir, findings)

    findings.sort(key=lambda f: (f.file, f.path, f.code))
    return findings


def check_paths(paths: list[Path]) -> list[Finding]:
    """Check episode dirs or single files."""
    out: list[Finding] = []
    for p in paths:
        if not p.exists():
            out.append(Finding("error", "file.missing", str(p), "", f"Path does not exist: '{p}'."))
            continue
        if p.is_dir():
            if (p / "shots.yaml").is_file():
                out.extend(check_episode(p))
            else:
                ep = _find_episode_dir(p)
                if ep is None:
                    out.extend(check_episode(p))
                else:
                    try:
                        rel_dir = Path(os.path.relpath(p, ep)).as_posix()
                    except ValueError:
                        continue
                    for f in check_episode(ep):
                        if f.file == rel_dir or f.file.startswith(rel_dir + "/"):
                            out.append(f)
            continue
        ep = _find_episode_dir(p.parent)
        if ep is None:
            try:
                load(p)
            except SpecError as exc:
                out.extend(exc.findings)
            continue
        rel_file = _rel(p, ep)
        for f in check_episode(ep):
            if f.file == rel_file:
                out.append(f)
    out.sort(key=lambda f: (f.file, f.path, f.code))
    return out
