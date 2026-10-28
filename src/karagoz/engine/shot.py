"""Render one shot or a whole episode from the episode's files to MP4."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from karagoz.engine.camera import camera_track
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
from karagoz.engine.render import FPS, Clip, RenderError, encode, loudness, mix
from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import expand
from karagoz.spec import Align, Lines, Mouth, MouthCue, Poses, Regions, Shot, Shots, SpecError, Style, Xsheet, load

STANDING_HEIGHT = 0.80


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


def _join_shot_mp4s(mp4s: list[Path], out_mp4: Path) -> Path:
    if len(mp4s) == 1:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mp4s[0], out_mp4)
        return out_mp4
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
    if poses is not None:
        for char_name, char_poses in poses.characters.items():
            mapping: dict[str, PoseInfo] = {}
            for pose in char_poses.poses:
                mapping[pose.id] = PoseInfo(support=pose.support, on=pose.on, rests_on=tuple(pose.rests_on or []))
            poses_by_char[char_name] = mapping
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
            )
        )
    placements, solver_warnings = solve(inputs, regions)
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
    order = [c.name for c in (shot_entry.characters or []) if f"{c.name}.body" in xs.tracks]
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
    placement_data = _placement_mode(prod, episode_dir, shot, shot_entry, order, mirror, characters, poses)
    placements_map: dict[str, dict[str, DrawingPlacement]] | None = None
    occluder_masks: dict[str, np.ndarray] | None = None
    if placement_data is not None:
        placements_map, occluder_masks, solver_warnings = placement_data
        warnings.extend(solver_warnings)
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
    shot_span = timing.shot(shot)
    clips: list[Clip] = []
    for lid in shot_line_ids:
        line_span = timing.line(lid)
        clips.append(Clip(episode_dir / "audio" / f"{lid}.mp3", line_span.start - shot_span.start))
    mix_wav = episode_dir / "render" / shot / "mix.wav"
    mix(clips, length=exp.length, out_wav=mix_wav)
    mp4_path = episode_dir / "render" / f"{shot}.mp4"
    encode(frames_dir, mix_wav, mp4_path)
    lufs = loudness(mp4_path) if clips else None
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
    return _join_shot_mp4s(mp4s, out_mp4)
