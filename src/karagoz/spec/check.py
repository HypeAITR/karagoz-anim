"""Cross-file checks for one episode (spec v1)."""

from __future__ import annotations

import os
import re
from pathlib import Path

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
    Poses,
    Regions,
    SequenceEvent,
    Shots,
    Style,
    Transition,
    Xsheet,
)

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


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
                    prev = item.to
                elif isinstance(item, LoopEvent):
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
