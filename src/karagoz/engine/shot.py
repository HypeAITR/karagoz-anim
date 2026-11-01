"""Render one shot or a whole episode from the episode's files to MP4."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from karagoz.engine.camera import BBox, camera_track
from karagoz.engine.compose import ShotAssets, character_bboxes, cutout, render_frames
from karagoz.engine.framing import frame_targets
from karagoz.engine.landmarks import body_drawings
from karagoz.engine.mouth import mouth_track
from karagoz.engine.placement import (
    CharacterInput,
    DrawingPlacement,
    PoseInfo,
    solve,
    target_head_px,
    to_json,
)
from karagoz.engine.register import refine
from karagoz.engine.render import (
    FPS,
    TARGET_LUFS,
    Clip,
    RenderError,
    add_into,
    duck,
    encode,
    fade_edges,
    loudness,
    mix,
    normalize,
    sound_file,
)
from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import Exposure, expand, lint, style_lint
from karagoz.spec import Align, Lines, Mouth, MouthCue, Poses, Regions, Shot, Shots, SpecError, Style, Xsheet, load

STANDING_HEIGHT = 0.80

#: Frames of black/fade per transition edge while no production sets the beat (half a second at 24 fps).
BLACK_HOLD_DEFAULT = 12
#: MP4 comment + report line when the episode ran with human review off.
REVIEW_OFF_COMMENT = "human review: off"
#: Orchestrator's event-log note for a review-off run (``Studio._auto_approve``).
REVIEW_OFF_NOTE = "auto-approved"


@dataclass
class ShotRender:
    shot: str
    mp4: Path
    frames: int
    seconds: float
    lufs: float | None
    warnings: list[str]


def character_scale(ref: Image.Image, canvas: tuple[int, int], frame_h: int = 1080) -> float:
    """Scale so the reference (standing, full body) covers STANDING_HEIGHT of the frame."""
    cut = cutout(ref)
    resized = cut if cut.size == canvas else cut.resize(canvas, Image.Resampling.LANCZOS)
    alpha = np.array(resized.convert("RGBA"))[:, :, 3]
    ys, _ = np.nonzero(alpha > 0)
    if ys.size == 0:
        return 1.0
    opaque_h = int(ys.max()) - int(ys.min()) + 1
    if opaque_h <= 0:
        return 1.0
    return STANDING_HEIGHT * frame_h / opaque_h


def _load_characters(prod: Path) -> dict[str, dict[str, Any]]:
    path = prod / "characters.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    chars = data.get("characters")
    if not isinstance(chars, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, entry in chars.items():
        if isinstance(name, str) and isinstance(entry, dict):
            out[name] = dict(entry)
    return out


def _rest_mouth(shot: str, character: str, length: int) -> Mouth:
    return Mouth(
        version=1,
        shot=shot,
        character=character,
        fps=24,
        cues=[MouthCue(start=0, end=length, shape="X", src="auto")],
    )


def _ffmpeg_bin() -> str:
    override = os.environ.get("KARAGOZ_FFMPEG")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found is None:
        raise RenderError("ffmpeg not found on PATH (override with KARAGOZ_FFMPEG)")
    return found


def _join_shot_mp4s(mp4s: list[Path], out_mp4: Path, comment: str | None = None) -> Path:
    if len(mp4s) == 1:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mp4s[0], out_mp4)
        return out_mp4
    if comment is None and review_off(out_mp4.parent.parent):
        # Episode assembly is <episode>/render/<episode>.mp4 and the CLI joins here directly,
        # so detect review-off from the episode's event log (no cli.py change needed).
        comment = REVIEW_OFF_COMMENT
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_mp4.parent / "concat.txt"
    lines: list[str] = []
    for mp4 in mp4s:
        posix = mp4.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{posix}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        cmd = [
            _ffmpeg_bin(),
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-filter:a",
            "apad",
            "-shortest",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
        ]
        if comment is not None:
            cmd += ["-metadata", f"comment={comment}"]
        cmd += [
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            str(out_mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            tail = proc.stderr[-400:] if len(proc.stderr) > 400 else proc.stderr
            raise RenderError(f"concat failed: {tail}")
    finally:
        with contextlib.suppress(OSError):
            list_path.unlink()
    return out_mp4


def _height_cm(info: dict[str, Any]) -> float | None:
    value = info.get("height_cm")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if result > 0 else None
    return None


def _ref_head_height(data: Any) -> float | None:
    if not isinstance(data, dict):
        return None
    head = data.get("head")
    if not isinstance(head, dict):
        return None
    box = head.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    y0_raw, y1_raw = box[1], box[3]
    if isinstance(y0_raw, bool) or isinstance(y1_raw, bool):
        return None
    if not isinstance(y0_raw, (int, float)) or not isinstance(y1_raw, (int, float)):
        return None
    height = float(y1_raw) - float(y0_raw)
    return height if height > 0 else None


def _ref_face_height(data: Any) -> float | None:
    if not isinstance(data, dict):
        return None
    face = data.get("face")
    if not isinstance(face, dict):
        return None
    box = face.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    y0_raw, y1_raw = box[1], box[3]
    if isinstance(y0_raw, bool) or isinstance(y1_raw, bool):
        return None
    if not isinstance(y0_raw, (int, float)) or not isinstance(y1_raw, (int, float)):
        return None
    height = float(y1_raw) - float(y0_raw)
    return height if height > 0 else None


def _opaque_height(img: Image.Image) -> int | None:
    cut = cutout(img)
    alpha = np.array(cut.convert("RGBA"))[:, :, 3]
    ys, _ = np.nonzero(alpha > 0)
    if ys.size == 0:
        return None
    height = int(ys.max()) - int(ys.min()) + 1
    return height if height > 0 else None


def _placement_mode(
    prod: Path,
    episode_dir: Path,
    shot: str,
    shot_entry: Shot,
    order: list[str],
    mirror: frozenset[str],
    characters: dict[str, dict[str, Any]],
    poses: Poses | None,
) -> tuple[dict[str, dict[str, DrawingPlacement]], dict[str, np.ndarray], list[str]] | None:
    """Build solver placements and occluder masks, or None when placement mode is off."""
    regions_path = prod / "locations" / shot_entry.location / "regions.yaml"
    if not regions_path.is_file():
        return None
    try:
        regions_obj = load(regions_path)
    except (SpecError, OSError):
        return None
    if not isinstance(regions_obj, Regions):
        return None
    regions: Regions = regions_obj
    if regions.scale is None:
        return None
    assets_dir = episode_dir / "assets" / shot
    chars = [c for c in order if body_drawings(assets_dir, c)]
    place_by_char: dict[str, str | None] = {}
    if shot_entry.characters:
        for entry in shot_entry.characters:
            place_by_char[entry.name] = entry.place
    poses_by_char: dict[str, dict[str, PoseInfo]] = {}
    seqs_by_char: dict[str, dict[str, tuple[str, str | None]]] = {}
    if poses is not None:
        for char_name, char_poses in poses.characters.items():
            mapping: dict[str, PoseInfo] = {}
            for pose in char_poses.poses:
                mapping[pose.id] = PoseInfo(support=pose.support, on=pose.on, rests_on=tuple(pose.rests_on or []))
            poses_by_char[char_name] = mapping
            seqs_by_char[char_name] = {seq.id: (seq.from_, seq.to) for seq in (char_poses.sequences or [])}
    inputs: list[CharacterInput] = []
    for character in chars:
        info = characters.get(character)
        if info is None:
            return None
        height = _height_cm(info)
        if height is None:
            return None
        ref_rel = info.get("ref")
        if not isinstance(ref_rel, str) or not ref_rel:
            return None
        ref_path = prod / ref_rel
        if not ref_path.is_file():
            return None
        ref_lm_path = assets_dir / character / "ref.landmarks.json"
        if not ref_lm_path.is_file():
            return None
        try:
            ref_lm_data: Any = json.loads(ref_lm_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        head_h = _ref_head_height(ref_lm_data)
        if head_h is None:
            return None
        face_h = _ref_face_height(ref_lm_data)
        head_per_face: float | None = head_h / face_h if face_h is not None and face_h > 0 else None
        try:
            ref_img = Image.open(ref_path).convert("RGB")
        except OSError:
            return None
        opaque_h = _opaque_height(ref_img)
        if opaque_h is None or opaque_h <= 0:
            return None
        head_ratio = head_h / float(opaque_h)
        if head_ratio <= 0:
            return None
        target = target_head_px(height, regions.scale.px_per_cm, head_ratio)
        char_dir = assets_dir / character
        try:
            lm_files = sorted(char_dir.glob("*.landmarks.json"), key=lambda p: p.name)
        except OSError:
            lm_files = []
        landmarks: dict[str, dict[str, Any]] = {}
        for lm_path in lm_files:
            if lm_path.name == "ref.landmarks.json":
                continue
            if not lm_path.name.endswith(".landmarks.json"):
                continue
            drawing_name = lm_path.name[: -len(".landmarks.json")]
            if not drawing_name:
                continue
            try:
                data: Any = json.loads(lm_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                landmarks[drawing_name] = data
        inputs.append(
            CharacterInput(
                name=character,
                target_head_px=target,
                mirror=(character in mirror),
                place=place_by_char.get(character),
                poses=poses_by_char.get(character, {}),
                landmarks=landmarks,
                head_per_face=head_per_face,
                sequences=seqs_by_char.get(character, {}),
            )
        )
    placements, solver_warnings = solve(inputs, regions)
    placements, reg_warnings = refine(inputs, placements, regions, ShotAssets(assets_dir).drawing)
    solver_warnings = solver_warnings + reg_warnings
    out_path = assets_dir / "placement.json"
    out_path.write_text(json.dumps(to_json(placements, solver_warnings), indent=1) + "\n", encoding="utf-8")
    occluders: dict[str, np.ndarray] = {}
    frame_w, frame_h = 1920, 1080
    for obj in regions.objects:
        if not obj.front:
            continue
        mask_path = regions_path.parent / obj.mask
        if not mask_path.is_file():
            continue
        try:
            mask_img = Image.open(mask_path).convert("L")
        except OSError:
            continue
        if mask_img.size != (frame_w, frame_h):
            mask_img = mask_img.resize((frame_w, frame_h), Image.Resampling.NEAREST)
        arr = np.array(mask_img) > 127
        occluders[obj.name] = arr if arr.dtype == bool else arr.astype(bool)
    return (placements, occluders, solver_warnings)


def transition_frames(style: Style) -> int:
    """Frames of black/fade per transition edge.

    No existing beat fits a fade (react/shot_lead/gesture_lead/blink_lead are dialogue-timing beats),
    so this reads a ``black_hold`` beat productions can tune; the default is half a second (12 frames).
    """
    beat = style.live.beats.get("black_hold")
    if beat is None:
        return BLACK_HOLD_DEFAULT
    return max(0, round(beat.value))


def _fade_frame(frames_dir: Path, index: int, factor: float) -> None:
    path = frames_dir / f"{index:05d}.png"
    img = Image.open(path).convert("RGB")
    if factor <= 0.0:
        Image.new("RGB", img.size, (0, 0, 0)).save(path, format="PNG")
        return
    if factor >= 1.0:
        return
    scaled = np.rint(np.array(img).astype(np.float64) * factor).astype(np.uint8)
    Image.fromarray(scaled).save(path, format="PNG")


def apply_frame_transitions(
    frames_dir: Path, length: int, transition_in: str | None, transition_out: str | None, fade: int
) -> None:
    """Bake the shot's edge transitions into its PNG frames, in place (the shot keeps its length).

    ``from_black``/``to_black`` paint hard black frames; ``fade_*`` scales pixels linearly so the outer
    frame is exactly black; ``cut``/None leaves frames untouched. Spans clamp to the shot; overlapping
    edges multiply.
    """
    n = min(max(fade, 0), max(length, 0))
    if n <= 0 or length <= 0:
        return
    if transition_in in ("from_black", "fade_from_black"):
        hard = transition_in == "from_black"
        for j in range(n):
            _fade_frame(frames_dir, j, 0.0 if hard else j / n)
    if transition_out in ("to_black", "fade_to_black"):
        hard = transition_out == "to_black"
        for j in range(n):
            _fade_frame(frames_dir, length - n + j, 0.0 if hard else (n - 1 - j) / n)


def _screen_side(view: str | None) -> int | None:
    if view in ("three_quarter_right", "profile_right"):
        return 1
    if view in ("three_quarter_left", "profile_left"):
        return -1
    return None


def _side_word(side: int) -> str:
    return "right" if side == 1 else "left"


def facing_mirror(
    base: frozenset[str],
    order: list[str],
    bboxes: dict[str, BBox],
    facing: dict[str, str | None],
    views: Mapping[str, str | None],
    characters: dict[str, dict[str, Any]],
) -> tuple[frozenset[str], list[str]]:
    """Adjust the view-based ``base`` mirror so ``facing`` characters look at their target on screen.

    The desired direction comes from the two screen positions (bbox centres); the drawn direction comes
    from ``natural_facing`` flipped by the base mirror (the view describes the target look, the drawing
    the start; the view stands in when the natural facing is unknown). ``mirror_ok: false`` keeps the
    base mirror and warns once with the reason — unless the view already claimed that refusal. Unknown,
    self, missing-position and same-x targets keep the base mirror silently (``check`` owns validation).
    """
    out = set(base)
    warnings: list[str] = []
    for character in order:
        target = facing.get(character)
        if not target or target == character:
            continue
        self_box = bboxes.get(character)
        target_box = bboxes.get(target)
        if self_box is None or target_box is None:
            continue
        diff = (target_box[0] + target_box[2]) - (self_box[0] + self_box[2])
        if diff == 0:
            continue
        desired = 1 if diff > 0 else -1
        info = characters.get(character, {})
        raw = info.get("natural_facing")
        natural = 1 if raw == "right" else (-1 if raw == "left" else None)
        view_side = _screen_side(views.get(character))
        drawn = natural if natural is not None else view_side
        if drawn is None:
            continue
        if character in out:
            drawn = -drawn
        if drawn == desired:
            continue
        if info.get("mirror_ok") is True:
            if character in out:
                out.discard(character)
            else:
                out.add(character)
            continue
        refused_for_view = view_side is not None and natural is not None and view_side != natural
        if refused_for_view and character not in base:
            continue  # the view-based warning already fired for this character
        known = _side_word(natural) if natural is not None else "unknown"
        warnings.append(
            f"{character} faces {target} on screen {_side_word(desired)} "
            f"but mirror_ok is false (natural {known}); keeping unmirrored"
        )
    return frozenset(out), warnings


def shot_lint_warnings(
    exp: Exposure, style: Style, speaking: dict[str, list[tuple[str, int, int]]], mouths: dict[str, Mouth]
) -> list[str]:
    """The same timing/style lint ``check`` runs (``xsheet.lint`` + ``xsheet.style_lint``), as warning lines."""
    cells: dict[str, list[str | None]] = {}
    for character, track in mouths.items():
        row: list[str | None] = [None] * exp.length
        for cue in track.cues:
            for frame in range(max(cue.start, 0), min(cue.end, exp.length)):
                row[frame] = cue.shape
        cells[character] = row
    findings = [*lint(exp, style), *style_lint(exp, style, speaking, cells)]
    return [f"lint: {finding}" for finding in findings]


def review_off(episode_dir: Path) -> bool:
    """True when the episode's event log carries the orchestrator's review-off note."""
    try:
        text = (episode_dir / "run" / "events.jsonl").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return REVIEW_OFF_NOTE in text


def render_shot(prod: Path, episode: str, shot: str) -> ShotRender:
    """Render one shot: frames, mix, MP4."""
    episode_dir = prod / episode
    lines_obj = load(episode_dir / "lines.yaml")
    shots_obj = load(episode_dir / "shots.yaml")
    style_obj = load(prod / "style.yaml")
    if not isinstance(lines_obj, Lines):
        raise ValueError(f"expected lines file: {episode_dir / 'lines.yaml'}")
    if not isinstance(shots_obj, Shots):
        raise ValueError(f"expected shots file: {episode_dir / 'shots.yaml'}")
    if not isinstance(style_obj, Style):
        raise ValueError(f"expected style file: {prod / 'style.yaml'}")
    lines: Lines = lines_obj
    shots: Shots = shots_obj
    style: Style = style_obj
    aligns: dict[str, Align] = {}
    audio_dir = episode_dir / "audio"
    if audio_dir.is_dir():
        for align_path in sorted(audio_dir.glob("*.align.json")):
            align_obj = load(align_path)
            if isinstance(align_obj, Align):
                aligns[align_obj.line] = align_obj
    xsheet_path = episode_dir / "shots" / shot / "xsheet.yaml"
    xsheet_obj = load(xsheet_path)
    if not isinstance(xsheet_obj, Xsheet):
        raise ValueError(f"expected xsheet file: {xsheet_path}")
    xs: Xsheet = xsheet_obj
    poses: Poses | None = None
    poses_path = episode_dir / "shots" / shot / "poses.yaml"
    if poses_path.is_file():
        poses_obj = load(poses_path)
        if isinstance(poses_obj, Poses):
            poses = poses_obj
    characters = _load_characters(prod)
    timing = resolve_episode(lines, shots, aligns, style)
    exp = expand(xs, poses, timing, style)
    shot_entry = next((s for s in shots.shots if s.id == shot), None)
    if shot_entry is None:
        raise ValueError(f"unknown shot '{shot}'")
    lines_by_id = {entry.id: entry for entry in lines.lines}
    shot_line_ids = list(shot_entry.lines or [])
    mouth_chars: set[str] = set()
    if xs.mouth is not None:
        mouth_chars.update(xs.mouth.keys())
    for lid in shot_line_ids:
        entry = lines_by_id.get(lid)
        if entry is not None:
            mouth_chars.add(entry.speaker)
    mouths: dict[str, Mouth] = {}
    for character in mouth_chars:
        try:
            mouths[character] = mouth_track(shot, character, lines, shots, xs, timing, style)
        except ValueError:
            mouths[character] = _rest_mouth(shot, character, exp.length)
    order = [c.name for c in (shot_entry.characters or []) if f"{c.name}.body" in xs.tracks and not c.offscreen]
    views = {c.name: c.view for c in (shot_entry.characters or [])}
    mirror_chars: list[str] = []
    warnings: list[str] = []
    for character in order:
        view = views.get(character)
        if view is None or view in ("front", "back"):
            continue
        info = characters.get(character)
        if info is None:
            continue
        natural = info.get("natural_facing")
        if not isinstance(natural, str):
            continue
        natural_norm = natural.lower()
        need = (natural_norm == "right" and view.endswith("_left")) or (
            natural_norm == "left" and view.endswith("_right")
        )
        if not need:
            continue
        if info.get("mirror_ok") is True:
            mirror_chars.append(character)
        else:
            warnings.append(f"{character} needs mirror for view {view} (natural {natural}) but mirror_ok is false")
    mirror = frozenset(mirror_chars)
    speaking: dict[str, list[tuple[str, int, int]]] = {}
    if shot_entry.lines:
        lint_shot_start = timing.shot(shot).start
        for line_id in shot_entry.lines:
            line_entry = lines_by_id.get(line_id)
            if line_entry is None or line_entry.speaker not in exp.body:
                continue
            span = timing.line(line_id)
            speaking.setdefault(line_entry.speaker, []).append(
                (line_id, span.start - lint_shot_start, span.end - lint_shot_start)
            )
    warnings.extend(shot_lint_warnings(exp, style, speaking, mouths))
    assets = ShotAssets(episode_dir / "assets" / shot)
    scales: dict[str, float] = {}
    for character in order:
        info = characters.get(character)
        if info is None:
            continue
        ref_rel = info.get("ref")
        if not isinstance(ref_rel, str) or not ref_rel:
            continue
        ref_path = prod / ref_rel
        if not ref_path.is_file():
            continue
        try:
            ref_img = Image.open(ref_path).convert("RGB")
        except OSError:
            continue
        canvas: tuple[int, int] | None = None
        for cell in exp.body.get(character, []):
            if cell is None:
                continue
            img = assets.drawing(character, cell)
            if img is not None:
                canvas = img.size
                break
            if "." in cell and "-" in cell:
                base, _k = cell.rsplit(".", 1)
                if "-" not in base:
                    continue
                a, b = base.split("-", 1)
                for cand in (a, b):
                    fallback_img = assets.drawing(character, cand)
                    if fallback_img is not None:
                        canvas = fallback_img.size
                        break
                if canvas is not None:
                    break
        if canvas is None:
            continue
        try:
            scales[character] = character_scale(ref_img, canvas)
        except (ValueError, ZeroDivisionError, OSError):
            continue
    facing_map: dict[str, str | None] = {c.name: c.facing for c in (shot_entry.characters or [])}
    placement_data = _placement_mode(prod, episode_dir, shot, shot_entry, order, mirror, characters, poses)
    placements_map: dict[str, dict[str, DrawingPlacement]] | None = None
    occluder_masks: dict[str, np.ndarray] | None = None
    if placement_data is not None:
        placements_map, occluder_masks, _ = placement_data
        # The solver boxes are the true screen positions: face from them, re-solving once on a flip.
        faced, facing_warnings = facing_mirror(
            mirror,
            order,
            character_bboxes(exp, assets, order, placements=placements_map),
            facing_map,
            views,
            characters,
        )
        if faced != mirror:
            mirror = faced
            placement_data = _placement_mode(prod, episode_dir, shot, shot_entry, order, mirror, characters, poses)
            if placement_data is not None:
                placements_map, occluder_masks, _ = placement_data
        warnings.extend(facing_warnings)
    else:
        probe = character_bboxes(exp, assets, order, mirror=mirror, scales=scales)
        mirror, facing_warnings = facing_mirror(mirror, order, probe, facing_map, views, characters)
        warnings.extend(facing_warnings)
    if placement_data is not None:
        warnings.extend(placement_data[2])
    frames_dir = episode_dir / "render" / shot / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    if placements_map is not None:
        bboxes = character_bboxes(exp, assets, order, placements=placements_map)
        targets = frame_targets(exp, assets, order, placements_map)
        camera_boxes, camera_warnings = camera_track(
            xs.camera, shot_entry.framing, shot, exp.length, timing, style, bboxes, frame_targets=targets
        )
    else:
        bboxes = character_bboxes(exp, assets, order, mirror=mirror, scales=scales)
        camera_boxes, camera_warnings = camera_track(
            xs.camera, shot_entry.framing, shot, exp.length, timing, style, bboxes
        )
    warnings.extend(camera_warnings)
    if placements_map is not None:
        render_warnings = render_frames(
            exp,
            mouths,
            assets,
            order,
            frames_dir,
            camera=camera_boxes,
            placements=placements_map,
            occluders=occluder_masks,
        )
    else:
        render_warnings = render_frames(
            exp, mouths, assets, order, frames_dir, mirror=mirror, scales=scales, camera=camera_boxes
        )
    warnings.extend(render_warnings)
    fade = transition_frames(style)
    apply_frame_transitions(frames_dir, exp.length, shot_entry.transition_in, shot_entry.transition_out, fade)
    shot_span = timing.shot(shot)
    clips: list[Clip] = []
    speech: list[tuple[int, int]] = []
    for lid in shot_line_ids:
        line_span = timing.line(lid)
        voice_path = sound_file(audio_dir, lid)
        if voice_path is None:
            warnings.append(f"line {lid}: no audio file in audio/")
            continue
        clips.append(Clip(voice_path, line_span.start - shot_span.start))
        speech.append((line_span.start - shot_span.start, line_span.end - shot_span.start))
    sounds_by_id = {sound.id: sound for sound in (lines.sounds or [])}
    for sid in shot_entry.sounds or []:
        sentry = sounds_by_id.get(sid)
        if sentry is None or sentry.kind != "sfx":
            continue
        sfx_path = sound_file(audio_dir, sid)
        if sfx_path is None:
            warnings.append(f"sound {sid}: no audio file in audio/")
            continue
        clips.append(Clip(sfx_path, timing.frame(sentry.at, shot) - shot_span.start, gain_db=sentry.gain_db))
    music_clips: list[Clip] = []
    for sound in lines.sounds or []:
        if sound.kind != "music":
            continue
        music_path = sound_file(audio_dir, sound.id)
        if music_path is None:
            warnings.append(f"sound {sound.id}: no audio file in audio/")
            continue
        music_clips.append(Clip(music_path, timing.frame(sound.at, shot) - shot_span.start, gain_db=sound.gain_db))
    mix_wav = episode_dir / "render" / shot / "mix.wav"
    mix(clips, length=exp.length, out_wav=mix_wav)
    if music_clips:
        music_wav = episode_dir / "render" / shot / "music.wav"
        mix(music_clips, length=exp.length, out_wav=music_wav)
        duck(music_wav, speech)
        add_into(mix_wav, music_wav)
    in_n = fade if shot_entry.transition_in in ("from_black", "fade_from_black") else 0
    out_n = fade if shot_entry.transition_out in ("to_black", "fade_to_black") else 0
    if in_n or out_n:
        # Fade before normalize so the loudness target still holds with faded edges.
        fade_edges(
            mix_wav,
            in_n,
            out_n,
            hard_in=shot_entry.transition_in == "from_black",
            hard_out=shot_entry.transition_out == "to_black",
        )
    if clips or music_clips:
        normalize(mix_wav)
    off = review_off(episode_dir)
    if off:
        warnings.append(f"{REVIEW_OFF_COMMENT} (episode auto-approved)")
    mp4_path = episode_dir / "render" / f"{shot}.mp4"
    encode(frames_dir, mix_wav, mp4_path, comment=REVIEW_OFF_COMMENT if off else None)
    lufs = loudness(mp4_path) if (clips or music_clips) else None
    if lufs is not None and abs(lufs - TARGET_LUFS) > 1.0:
        warnings.append(f"audio: {lufs:.1f} LUFS, target -14 (peak limit)")
    return ShotRender(
        shot=shot,
        mp4=mp4_path,
        frames=exp.length,
        seconds=exp.length / FPS,
        lufs=lufs,
        warnings=warnings,
    )


def render_episode(prod: Path, episode: str) -> Path:
    """Render every shot in order, then join them into one MP4."""
    episode_dir = prod / episode
    shots_obj = load(episode_dir / "shots.yaml")
    if not isinstance(shots_obj, Shots):
        raise ValueError(f"expected shots file: {episode_dir / 'shots.yaml'}")
    mp4s: list[Path] = []
    for shot_entry in shots_obj.shots:
        rendered = render_shot(prod, episode, shot_entry.id)
        mp4s.append(rendered.mp4)
    out_mp4 = episode_dir / "render" / f"{episode}.mp4"
    off = REVIEW_OFF_COMMENT if review_off(episode_dir) else None
    return _join_shot_mp4s(mp4s, out_mp4, comment=off)
