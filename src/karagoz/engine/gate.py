"""Mechanical gate: flags sudden or impossible drawings, never rejects (WP-25)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import expand
from karagoz.spec import Align, Lines, Poses, Regions, Shots, SpecError, Style, Xsheet, load

KINDS = ("prop_count", "reverse", "jump", "scale_jump", "placement", "world")

# First guesses for typical object heights, to be calibrated on real locations (WP-38).
WORLD_SIZES: dict[str, tuple[int, int]] = {
    "door": (180, 230),
    "tv": (30, 120),
    "television": (30, 120),
    "coffee table": (30, 60),
    "table": (65, 85),
    "sofa": (70, 110),
    "couch": (70, 110),
    "armchair": (70, 110),
    "chair": (75, 110),
    "bed": (40, 120),
    "fridge": (140, 200),
    "refrigerator": (140, 200),
    "wardrobe": (170, 240),
}
WORLD_TOL = 1.5

JUMP_HEADS = 0.6
REVERSE_MIN_MOVE = 0.25
FALLBACK_TOL = 0.1
WRONG_WAY_TOL = 0.1
SCALE_JUMP_RATIO = 1.25
FACE_TO_HEAD = 0.82
CUP_CONFIDENCE = 0.5
PX_RE = re.compile(r"(\d+)\s*px")
SUPPORT_RANK = {"lie": 0, "sit": 1, "crouch": 2, "stand": 3}
_UNKNOWN_FRAME = 10**9


@dataclass(frozen=True)
class Flag:
    kind: str
    character: str | None
    drawings: tuple[str, ...]
    size: float
    message: str
    shas: tuple[str, ...]
    frames: tuple[int, int] | None = None
    silenced: bool = False

    @property
    def id(self) -> str:
        who = self.character if self.character is not None else "-"
        return f"{self.kind}:{who}:{'+'.join(self.drawings)}"


@dataclass(frozen=True)
class _Head:
    x: float
    y: float
    height: float


def sequence(cells: list[str | None]) -> list[tuple[str, int, int]]:
    """Runs of the same non-None cell as (drawing, first frame, end frame exclusive)."""
    runs: list[tuple[str, int, int]] = []
    current: str | None = None
    start = 0
    for frame, cell in enumerate(cells):
        if cell is None:
            if current is not None:
                runs.append((current, start, frame))
                current = None
            continue
        if current is None:
            current = cell
            start = frame
        elif cell != current:
            runs.append((current, start, frame))
            current = cell
            start = frame
    if current is not None:
        runs.append((current, start, len(cells)))
    return runs


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _box(entry: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(entry, dict):
        return None
    box = entry.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    coords: list[float] = []
    for value in box:
        parsed = _as_float(value)
        if parsed is None:
            return None
        coords.append(parsed)
    return (coords[0], coords[1], coords[2], coords[3])


def _head_of(landmarks: dict[str, Any], placement: Any) -> _Head | None:
    """Placed head point and on-screen head height, or None when not measurable."""
    if not isinstance(placement, dict):
        return None
    scale = _as_float(placement.get("scale"))
    x = _as_float(placement.get("x"))
    y = _as_float(placement.get("y"))
    if scale is None or x is None or y is None:
        return None
    box = _box(landmarks.get("head"))
    from_head = True
    if box is None:
        box = _box(landmarks.get("face"))
        from_head = False
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if y1 - y0 <= 0:
        return None
    height = scale * (y1 - y0) if from_head else scale * (y1 - y0) / FACE_TO_HEAD
    if height <= 0:
        return None
    px = (x0 + x1) / 2.0
    py = (y0 + y1) / 2.0
    if bool(placement.get("mirror", False)):
        width: float | None = None
        size = landmarks.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            width = _as_float(size[0])
        if width is None:
            return None
        px = width - px
    return _Head(x + scale * px, y + scale * py, height)


def _sha(entry: Any) -> str:
    if isinstance(entry, dict):
        sha = entry.get("sha256")
        if isinstance(sha, str):
            return sha
    return ""


def _cup_count(landmarks: dict[str, dict[str, dict[str, Any]]], character: str, drawing: str) -> int:
    char_landmarks = landmarks.get(character)
    if char_landmarks is None:
        return 0
    entry = char_landmarks.get(drawing)
    if entry is None:
        return 0
    cups = entry.get("cups")
    if not isinstance(cups, list):
        return 0
    count = 0
    for cup in cups:
        if not isinstance(cup, dict):
            continue
        confidence = cup.get("confidence")
        if isinstance(confidence, bool):
            continue
        if isinstance(confidence, (int, float)) and confidence >= CUP_CONFIDENCE:
            count += 1
    return count


def _occurrences(run_drawings: list[str], pattern: list[str]) -> list[int]:
    if not pattern or len(run_drawings) < len(pattern):
        return []
    width = len(pattern)
    return [i for i in range(len(run_drawings) - width + 1) if run_drawings[i : i + width] == pattern]


def gate_flags(
    cells: dict[str, list[str | None]],
    placement: dict[str, Any],
    landmarks: dict[str, dict[str, dict[str, Any]]],
    poses: Poses | None,
    expected_props: int,
) -> list[Flag]:
    """Flag sudden or impossible drawings. Never rejects; order is KINDS, character, appearance."""
    placement_chars: dict[str, Any] = {}
    raw_chars = placement.get("characters", {})
    if isinstance(raw_chars, dict):
        placement_chars = raw_chars
    chars = sorted(cells)

    def sha_of(character: str, drawing: str) -> str:
        char_landmarks = landmarks.get(character)
        if char_landmarks is None:
            return ""
        return _sha(char_landmarks.get(drawing))

    heads: dict[str, dict[str, _Head]] = {}
    ref: dict[str, float | None] = {}
    seqs: dict[str, list[tuple[str, int, int]]] = {}
    first_frame: dict[str, dict[str, int]] = {}
    for char in chars:
        runs = sequence(cells[char])
        seqs[char] = runs
        seen_at: dict[str, int] = {}
        for frame, drawing in enumerate(cells[char]):
            if drawing is not None and drawing not in seen_at:
                seen_at[drawing] = frame
        first_frame[char] = seen_at
        char_landmarks = landmarks.get(char, {})
        char_placements = placement_chars.get(char, {})
        if not isinstance(char_placements, dict):
            char_placements = {}
        char_heads: dict[str, _Head] = {}
        for drawing in set(char_landmarks) | set(char_placements):
            if not isinstance(drawing, str):
                continue
            entry = char_landmarks.get(drawing)
            if entry is None:
                continue
            head = _head_of(entry, char_placements.get(drawing))
            if head is not None:
                char_heads[drawing] = head
        heads[char] = char_heads
        ref[char] = median([head.height for head in char_heads.values()]) if char_heads else None

    flagged: list[tuple[Flag, int]] = []

    for char in chars:  # jump + scale_jump over consecutive drawings
        runs = seqs[char]
        char_ref = ref[char]
        for i in range(len(runs) - 1):
            prev = runs[i][0]
            nxt, next_first, _ = runs[i + 1]
            if prev == nxt:
                continue
            head_prev = heads[char].get(prev)
            head_next = heads[char].get(nxt)
            if head_prev is None or head_next is None:
                continue
            if char_ref is not None and char_ref > 0:
                ratio = math.dist((head_prev.x, head_prev.y), (head_next.x, head_next.y)) / char_ref
                if ratio > JUMP_HEADS:
                    flagged.append(
                        (
                            Flag(
                                "jump",
                                char,
                                (prev, nxt),
                                ratio,
                                f"{char} {prev} -> {nxt}: the head moves {ratio:.2f} head heights in one drawing",
                                (sha_of(char, prev), sha_of(char, nxt)),
                            ),
                            next_first,
                        )
                    )
            size_ratio = max(head_prev.height, head_next.height) / min(head_prev.height, head_next.height)
            if size_ratio > SCALE_JUMP_RATIO:
                pct = round((size_ratio - 1) * 100)
                flagged.append(
                    (
                        Flag(
                            "scale_jump",
                            char,
                            (prev, nxt),
                            size_ratio - 1,
                            f"{char} {prev} -> {nxt}: the head changes size by {pct}%",
                            (sha_of(char, prev), sha_of(char, nxt)),
                        ),
                        next_first,
                    )
                )

    if poses is not None:  # reverse: progress inside a rise + support direction
        seen_reverse: set[Flag] = set()
        for char in chars:
            char_poses = poses.characters.get(char)
            if char_poses is None:
                continue
            transitions = char_poses.transitions or []
            if not transitions:
                continue
            supports = {pose.id: pose.support for pose in char_poses.poses}
            runs = seqs[char]
            run_drawings = [drawing for drawing, _, _ in runs]
            char_ref = ref[char]
            for transition in transitions:
                a = transition.from_
                b = transition.to
                inbetweens = [f"{a}-{b}.{k}" for k in range(1, len(transition.inbetweens) + 1)]
                head_a = heads[char].get(a)
                head_b = heads[char].get(b)
                if char_ref is not None and char_ref > 0 and head_a is not None and head_b is not None:
                    ya, yb = head_a.y, head_b.y
                    pattern = [a, *inbetweens, b]
                    for start in _occurrences(run_drawings, pattern):
                        if abs(yb - ya) >= REVERSE_MIN_MOVE * char_ref:
                            progress = 0.0
                            y_prev = ya
                            for j, drawing in enumerate(inbetweens):
                                head_i = heads[char].get(drawing)
                                if head_i is None:
                                    continue
                                yi = head_i.y
                                p_i = (yi - ya) / (yb - ya)
                                flag: Flag | None = None
                                if p_i < progress - FALLBACK_TOL:
                                    flag = Flag(
                                        "reverse",
                                        char,
                                        (drawing,),
                                        abs(yi - y_prev) / char_ref,
                                        f"{char} {drawing}: goes back between {a} and {b}",
                                        (sha_of(char, drawing),),
                                    )
                                elif p_i < -WRONG_WAY_TOL:
                                    flag = Flag(
                                        "reverse",
                                        char,
                                        (drawing,),
                                        abs(yi - ya) / char_ref,
                                        f"{char} {drawing}: goes back between {a} and {b}",
                                        (sha_of(char, drawing),),
                                    )
                                if flag is not None and flag not in seen_reverse:
                                    seen_reverse.add(flag)
                                    flagged.append((flag, runs[start + 1 + j][1]))
                                progress = p_i
                                y_prev = yi
                    support_a = supports.get(a)
                    support_b = supports.get(b)
                    if (
                        support_a is not None
                        and support_b is not None
                        and support_a in SUPPORT_RANK
                        and support_b in SUPPORT_RANK
                    ):
                        b_frame: int | None = None
                        seen_a = False
                        for drawing, first, _ in runs:
                            if drawing == a:
                                seen_a = True
                            if drawing == b and seen_a:
                                b_frame = first
                                break
                        if b_frame is not None:
                            rank_a = SUPPORT_RANK[support_a]
                            rank_b = SUPPORT_RANK[support_b]
                            support_flag: Flag | None = None
                            if rank_a < rank_b and yb > ya + REVERSE_MIN_MOVE * char_ref:
                                support_flag = Flag(
                                    "reverse",
                                    char,
                                    (b,),
                                    abs(yb - ya) / char_ref,
                                    f"{char} {a} -> {b}: {support_a} to {support_b} but the head goes down",
                                    (sha_of(char, b),),
                                )
                            elif rank_a > rank_b and yb < ya - REVERSE_MIN_MOVE * char_ref:
                                support_flag = Flag(
                                    "reverse",
                                    char,
                                    (b,),
                                    abs(yb - ya) / char_ref,
                                    f"{char} {a} -> {b}: {support_a} to {support_b} but the head goes up",
                                    (sha_of(char, b),),
                                )
                            if support_flag is not None and support_flag not in seen_reverse:
                                seen_reverse.add(support_flag)
                                flagged.append((support_flag, b_frame))

    if expected_props > 0 and cells:  # prop_count over frames
        nframes = max(len(row) for row in cells.values())
        group_start = 0
        group_end = 0
        group_seen = 0
        group_key: tuple[str, ...] = ()
        group_drawings: tuple[str, ...] = ()

        def flush() -> None:
            if group_end > group_start:
                shas = tuple(sha_of(tag.split("/", 1)[0], tag.split("/", 1)[1]) for tag in group_drawings)
                flagged.append(
                    (
                        Flag(
                            "prop_count",
                            None,
                            group_drawings,
                            group_seen - expected_props,
                            f"frames {group_start}-{group_end - 1}: "
                            f"{group_seen} props seen, {expected_props} expected "
                            f"({', '.join(group_drawings)})",
                            shas,
                            (group_start, group_end),
                        ),
                        group_start,
                    )
                )

        for frame in range(nframes):
            seen = 0
            current: list[str] = []
            bearing: list[str] = []
            for char in chars:
                row = cells[char]
                drawing = row[frame] if frame < len(row) else None
                if drawing is None:
                    continue
                tag = f"{char}/{drawing}"
                current.append(tag)
                count = _cup_count(landmarks, char, drawing)
                seen += count
                if count >= 1:
                    bearing.append(tag)
            key = tuple(sorted(current))
            if seen > expected_props and frame == group_end and key == group_key and seen == group_seen:
                group_end = frame + 1
            elif seen > expected_props:
                flush()
                group_start = frame
                group_end = frame + 1
                group_seen = seen
                group_key = key
                group_drawings = tuple(sorted(bearing))
            else:
                flush()
                group_start = frame + 1
                group_end = frame + 1
        flush()

    raw_warnings = placement.get("warnings", [])  # placement warnings
    if isinstance(raw_warnings, list):
        for warning in raw_warnings:
            if not isinstance(warning, str):
                continue
            words = warning.split()
            if len(words) < 2 or not words[1].endswith(":"):
                continue
            char = words[0]
            drawing = words[1][:-1]
            if not char or not drawing:
                continue
            match = PX_RE.search(warning)
            char_ref = ref.get(char)
            if match is not None and char_ref is not None and char_ref > 0:
                size = int(match.group(1)) / char_ref
            else:
                size = 0.3
            flagged.append(
                (
                    Flag("placement", char, (drawing,), size, warning, (sha_of(char, drawing),)),
                    first_frame.get(char, {}).get(drawing, _UNKNOWN_FRAME),
                )
            )

    kind_order = {kind: index for index, kind in enumerate(KINDS)}
    flagged.sort(key=lambda item: (kind_order[item[0].kind], item[0].character or "", item[1]))
    return [flag for flag, _ in flagged]


def select_flags(flags: list[Flag], limit: int = 5) -> tuple[list[Flag], list[Flag], int]:
    """Split flags into (shown, silenced, dropped count). Prop flags are always shown."""
    silenced = [flag for flag in flags if flag.silenced]
    rest = [(index, flag) for index, flag in enumerate(flags) if not flag.silenced]
    prop_shown = {index for index, flag in rest if flag.kind == "prop_count"}
    others = [(index, flag) for index, flag in rest if flag.kind != "prop_count"]
    ranked = sorted(others, key=lambda item: item[1].size, reverse=True)
    taken = {index for index, _ in ranked[: max(limit, 0)]}
    shown_at = prop_shown | taken
    shown = [flag for index, flag in enumerate(flags) if index in shown_at]
    return (shown, silenced, len(others) - len(taken))


_WORD_RE = re.compile(r"[a-z0-9]+")


def _world_key(kind: str) -> str | None:
    """Longest WORLD_SIZES key matching the lowercased kind as whole word(s), or None."""
    words = _WORD_RE.findall(kind.lower())
    best: str | None = None
    for key in WORLD_SIZES:
        parts = key.split()
        if len(parts) > len(words):
            continue
        for start in range(len(words) - len(parts) + 1):
            if words[start : start + len(parts)] == parts:
                if best is None or len(key) > len(best):
                    best = key
                break
    return best


def world_flags(regions: Regions | None, heights: dict[str, float]) -> list[Flag]:
    """Flag objects and characters whose measured sizes do not fit the world. Never rejects."""
    if regions is None or regions.scale is None:
        return []
    px_per_cm = regions.scale.px_per_cm
    sized: list[tuple[str, str, float]] = []  # (name, kind key, height_cm) in region-map order
    for obj in regions.objects:
        key = _world_key(obj.kind)
        if key is None:
            continue
        sized.append((obj.name, key, (obj.box[3] - obj.box[1]) / px_per_cm))
    flags: list[Flag] = []
    for name, key, height_cm in sized:
        if name == regions.scale.source:
            continue
        lo, hi = WORLD_SIZES[key]
        if height_cm > hi * WORLD_TOL or height_cm < lo / WORLD_TOL:
            size = max(height_cm / hi, lo / height_cm)
            flags.append(
                Flag(
                    "world",
                    None,
                    (name,),
                    size,
                    f"{name} ({key}): {round(height_cm)} cm tall, expected {lo}–{hi} cm",
                    (),
                )
            )
    for name, key, door_cm in sized:
        if key != "door":
            continue
        for character in sorted(heights):
            height = heights[character]
            if height > door_cm:
                flags.append(
                    Flag(
                        "world",
                        character,
                        (name,),
                        height / door_cm,
                        f"{character} is {round(height)} cm tall, taller than door {name} ({round(door_cm)} cm)",
                        (),
                    )
                )
    return flags


def _shot_regions(prod: Path, episode: str, shot: str) -> Regions | None:
    """The shot location's region map, or None when there is none to measure against."""
    shots_path = prod / episode / "shots.yaml"
    if not shots_path.is_file():
        return None
    shots_obj = load(shots_path)
    if not isinstance(shots_obj, Shots):
        return None
    location: str | None = None
    for entry in shots_obj.shots:
        if entry.id == shot:
            location = entry.location
            break
    if location is None:
        return None
    regions_path = prod / "locations" / location / "regions.yaml"
    if not regions_path.is_file():
        return None
    try:
        regions_obj = load(regions_path)
    except SpecError:
        return None
    return regions_obj if isinstance(regions_obj, Regions) else None


def _shot_heights(prod: Path, cells: dict[str, list[str | None]]) -> dict[str, float]:
    """Known heights in cm for the characters that have body cells in the shot."""
    present = {char for char, row in cells.items() if any(cell is not None for cell in row)}
    if not present:
        return {}
    chars_path = prod / "characters.yaml"
    if not chars_path.is_file():
        return {}
    data = yaml.safe_load(chars_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    chars = data.get("characters")
    if not isinstance(chars, dict):
        return {}
    heights: dict[str, float] = {}
    for char in present:
        info = chars.get(char)
        if not isinstance(info, dict):
            continue
        value = info.get("height_cm")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        heights[char] = float(value)
    return heights


def exposure(prod: Path, episode: str, shot: str) -> dict[str, list[str | None]]:
    """Body cells per character per frame, built the way plan_draw builds them."""
    episode_dir = prod / episode
    shot_dir = episode_dir / "shots" / shot

    lines_obj = load(episode_dir / "lines.yaml")
    shots_obj = load(episode_dir / "shots.yaml")
    style_obj = load(prod / "style.yaml")
    if not isinstance(lines_obj, Lines):
        raise ValueError(f"{episode_dir / 'lines.yaml'}: expected a lines file")
    if not isinstance(shots_obj, Shots):
        raise ValueError(f"{episode_dir / 'shots.yaml'}: expected a shots file")
    if not isinstance(style_obj, Style):
        raise ValueError(f"{prod / 'style.yaml'}: expected a style file")
    lines: Lines = lines_obj
    shots: Shots = shots_obj
    style: Style = style_obj

    aligns: dict[str, Align] = {}
    audio_dir = episode_dir / "audio"
    if audio_dir.is_dir():
        for align_path in sorted(audio_dir.glob("*.align.json")):
            obj = load(align_path)
            if isinstance(obj, Align):
                aligns[obj.line] = obj

    timing = resolve_episode(lines, shots, aligns, style)

    xsheet_obj = load(shot_dir / "xsheet.yaml")
    if not isinstance(xsheet_obj, Xsheet):
        raise ValueError(f"{shot_dir / 'xsheet.yaml'}: expected an xsheet file")
    xs: Xsheet = xsheet_obj

    poses: Poses | None = None
    poses_path = shot_dir / "poses.yaml"
    if poses_path.is_file():
        poses_obj = load(poses_path)
        if not isinstance(poses_obj, Poses):
            raise ValueError(f"{poses_path}: expected a poses file")
        poses = poses_obj

    return expand(xs, poses, timing, style).body


def _expected_props(path: Path, shot: str) -> int:
    if not path.is_file():
        return 0
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return 0
    entries = data.get("props")
    if not isinstance(entries, list):
        return 0
    names: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("shot") == shot:
            prop = entry.get("prop")
            if isinstance(prop, str) and prop:
                names.add(prop)
    return len(names)


def _flag_json(flag: Flag) -> dict[str, Any]:
    return {
        "id": flag.id,
        "kind": flag.kind,
        "character": flag.character,
        "drawings": list(flag.drawings),
        "frames": list(flag.frames) if flag.frames is not None else None,
        "size": round(flag.size, 3),
        "message": flag.message,
        "shas": list(flag.shas),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_ack(path: Path) -> tuple[dict[str, Any], str | None]:
    """The acknowledged flags, or ({}, warning) when the ack file is missing or corrupt."""
    if not path.is_file():
        return {}, None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, f"gate-ack.json cannot be read ({exc}); treated as no acknowledgements"
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"gate-ack.json is corrupt (invalid JSON: {exc.msg}); treated as no acknowledgements"
    if not isinstance(loaded, dict):
        return {}, "gate-ack.json is corrupt (expected an object); treated as no acknowledgements"
    return loaded, None


def acknowledge(prod: Path, episode: str, shot: str, flag: Flag, note: str) -> None:
    """Record that the orchestrator looked at a flag and keeps the drawing."""
    path = prod / episode / "review" / shot / "gate-ack.json"
    ack, _ = _read_ack(path)
    ack[flag.id] = {"shas": list(flag.shas), "note": note}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ack, indent=1) + "\n", encoding="utf-8")


def gate_shot(prod: Path, episode: str, shot: str, *, limit: int = 5) -> dict[str, Any]:
    """Flag one shot, silence acknowledged drawings, write and return gate.json."""
    assets_dir = prod / episode / "assets" / shot
    placement_path = assets_dir / "placement.json"
    if not placement_path.is_file():
        raise ValueError(f"no placement.json for {shot}: render the shot with a region map first")
    placement_data = _read_json_object(placement_path)
    landmarks_data: dict[str, dict[str, dict[str, Any]]] = {}
    raw_chars = placement_data.get("characters", {})
    if isinstance(raw_chars, dict):
        for char, drawings in raw_chars.items():
            if not isinstance(char, str) or not isinstance(drawings, dict):
                continue
            char_landmarks: dict[str, dict[str, Any]] = {}
            for drawing in drawings:
                if not isinstance(drawing, str):
                    continue
                landmarks_path = assets_dir / char / f"{drawing}.landmarks.json"
                if not landmarks_path.is_file():
                    continue
                parsed = json.loads(landmarks_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    char_landmarks[drawing] = parsed
            landmarks_data[char] = char_landmarks

    poses: Poses | None = None
    poses_path = prod / episode / "shots" / shot / "poses.yaml"
    if poses_path.is_file():
        poses_obj = load(poses_path)
        if not isinstance(poses_obj, Poses):
            raise ValueError(f"{poses_path}: expected a poses file")
        poses = poses_obj

    expected = _expected_props(prod / episode / "continuity.yaml", shot)
    cells = exposure(prod, episode, shot)
    flags = gate_flags(cells, placement_data, landmarks_data, poses, expected)
    flags = [*flags, *world_flags(_shot_regions(prod, episode, shot), _shot_heights(prod, cells))]

    ack, ack_warning = _read_ack(prod / episode / "review" / shot / "gate-ack.json")
    marked: list[Flag] = []
    for flag in flags:
        entry = ack.get(flag.id)
        stored = entry.get("shas") if isinstance(entry, dict) else None
        if isinstance(stored, list) and stored == list(flag.shas):
            marked.append(replace(flag, silenced=True))
        else:
            marked.append(flag)

    shown, silenced, dropped = select_flags(marked, limit)
    out: dict[str, Any] = {
        "shot": shot,
        "flags": [_flag_json(flag) for flag in shown],
        "silenced": [_flag_json(flag) for flag in silenced],
        "dropped": dropped,
        "warnings": [ack_warning] if ack_warning is not None else [],
    }
    gate_path = prod / episode / "review" / shot / "gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out
