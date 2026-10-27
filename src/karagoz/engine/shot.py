"""Render one shot or a whole episode from the episode's files to MP4."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from karagoz.engine.compose import ShotAssets, cutout, render_frames
from karagoz.engine.mouth import mouth_track
from karagoz.engine.render import FPS, Clip, RenderError, encode, loudness, mix
from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import expand
from karagoz.spec import Align, Lines, Mouth, MouthCue, Poses, Shots, Style, Xsheet, load

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
    frames_dir = episode_dir / "render" / shot / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    render_warnings = render_frames(exp, mouths, assets, order, frames_dir, mirror=mirror, scales=scales)
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
