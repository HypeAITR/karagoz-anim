"""Deterministic render: numbered PNG frames + voice lines -> MP4."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FPS: int = 24

_SAMPLE_RATE = 48000
_CHANNELS = 2
_SAMPLE_WIDTH = 2  # 16-bit


@dataclass(frozen=True)
class Clip:
    path: Path
    frame: int


class RenderError(RuntimeError):
    """Render failure; message includes the last 400 chars of ffmpeg's stderr."""


def _ffmpeg_bin() -> str:
    override = os.environ.get("KARAGOZ_FFMPEG")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found is None:
        raise RenderError("ffmpeg not found on PATH (override with KARAGOZ_FFMPEG)")
    return found


def _ffprobe_bin() -> str:
    override = os.environ.get("KARAGOZ_FFPROBE")
    if override:
        return override
    found = shutil.which("ffprobe")
    if found is None:
        raise RenderError("ffprobe not found on PATH (override with KARAGOZ_FFPROBE)")
    return found


def _tail(stderr: str) -> str:
    return stderr[-400:] if len(stderr) > 400 else stderr


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RenderError(f"{what} failed: {_tail(proc.stderr)}")
    return proc


def _write_silence(out_wav: Path, samples: int) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(_CHANNELS)
        w.setsampwidth(_SAMPLE_WIDTH)
        w.setframerate(_SAMPLE_RATE)
        if samples > 0:
            w.writeframes(b"\x00" * (samples * _CHANNELS * _SAMPLE_WIDTH))
    return out_wav


def mix(clips: list[Clip], length: int, out_wav: Path) -> Path:
    """Mix clips into one 48 kHz stereo 16-bit WAV exactly ``length`` frames long."""
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    total = length * _SAMPLE_RATE // FPS
    if length == 0 or not clips:
        return _write_silence(out_wav, total)
    ffmpeg = _ffmpeg_bin()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [ffmpeg, "-y", "-v", "error"]
    for clip in clips:
        cmd += ["-i", str(clip.path)]
    parts: list[str] = []
    for i, clip in enumerate(clips):
        if clip.frame >= 0:
            ms = clip.frame * 1000 / FPS
            parts.append(
                f"[{i}:a]aresample={_SAMPLE_RATE},"
                f"aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"adelay={ms:.15g}:all=1[a{i}]"
            )
        else:
            skip = -clip.frame * _SAMPLE_RATE // FPS
            parts.append(
                f"[{i}:a]aresample={_SAMPLE_RATE},"
                f"aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"atrim=start_sample={skip},asetpts=N-SR-TB[a{i}]"
            )
    ins = "".join(f"[a{i}]" for i in range(len(clips)))
    parts.append(f"{ins}amix=inputs={len(clips)}:normalize=0:duration=longest[mix0]")
    parts.append(
        f"[mix0]apad,atrim=end_sample={total},asetpts=N-SR-TB,"
        f"aformat=sample_fmts=s16:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo[mixed]"
    )
    cmd += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[mixed]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(_SAMPLE_RATE),
        "-ac",
        str(_CHANNELS),
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        str(out_wav),
    ]
    _run(cmd, "mix")
    return out_wav


def _pattern_to_glob(pattern: str) -> str:
    return re.sub(r"%0?\d*d", "*", pattern)


def encode(frames_dir: Path, audio: Path | None, out_mp4: Path, *, pattern: str = "%05d.png", crf: int = 18) -> Path:
    """Encode numbered PNG frames (+ optional audio) into a deterministic MP4."""
    glob = _pattern_to_glob(pattern)
    try:
        found = any(frames_dir.glob(glob))
    except OSError:
        found = False
    if not frames_dir.is_dir() or not found:
        # Let ffmpeg produce the authoritative error, but fail fast with a clear message.
        ffmpeg = _ffmpeg_bin()
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-framerate",
            str(FPS),
            "-start_number",
            "0",
            "-i",
            str(frames_dir / pattern),
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        try:
            _run(cmd, "encode")
        except RenderError as exc:
            raise RenderError(f"encode failed (no frames in {frames_dir} matching {pattern}): {exc}") from exc
        raise RenderError(f"encode failed: no frames in {frames_dir} matching {pattern}")
    ffmpeg = _ffmpeg_bin()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-framerate",
        str(FPS),
        "-start_number",
        "0",
        "-i",
        str(frames_dir / pattern),
    ]
    if audio is not None:
        cmd += ["-i", str(audio)]
    cmd += ["-map", "0:v:0"]
    if audio is not None:
        cmd += ["-map", "1:a:0"]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-r",
        str(FPS),
    ]
    if audio is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-filter:a", "apad", "-shortest"]
    cmd += [
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
    _run(cmd, "encode")
    return out_mp4


_LUFS_RE = re.compile(r"Integrated loudness:\s*\n?\s*I:\s*(-inf(?:inity)?|-?\d+(?:\.\d+)?)", re.IGNORECASE)
_LUFS_FALLBACK_RE = re.compile(r"\bI:\s*(-inf(?:inity)?|-?\d+(?:\.\d+)?)\s*LUFS", re.IGNORECASE)


def loudness(path: Path) -> float:
    """Measure integrated loudness (LUFS) of the file's audio with ebur128."""
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-filter:a",
        "ebur128",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RenderError(f"loudness failed: {_tail(proc.stderr)}")
    match = _LUFS_RE.search(proc.stderr) or _LUFS_FALLBACK_RE.search(proc.stderr)
    if match is None:
        # No parsable loudness: either no audio or an unexpected filter output.
        raise RenderError(f"loudness failed (no audio in {path}): {_tail(proc.stderr)}")
    raw = match.group(1).lower()
    if raw.startswith("-inf"):
        return float("-inf")
    return float(raw)


def _parse_rate(rate: str) -> float | None:
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(rate)
    except ValueError:
        return None


def probe(path: Path) -> dict[str, Any]:
    """Probe video geometry, frame rate, counted frames, duration and audio presence."""
    ffprobe = _ffprobe_bin()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate,duration,nb_read_frames,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RenderError(f"probe failed: {_tail(proc.stderr)}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RenderError(f"probe failed: invalid ffprobe output for {path}") from exc
    streams: list[dict[str, Any]] = data.get("streams", [])
    video: dict[str, Any] | None = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RenderError(f"probe failed: no video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    try:
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError(f"probe failed: missing geometry for {path}") from exc
    fps: float | None = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = video.get(key)
        if isinstance(raw, str):
            fps = _parse_rate(raw)
            if fps is not None and fps > 0:
                break
    frames: int | None = None
    for key in ("nb_read_frames", "nb_frames"):
        raw = video.get(key)
        if isinstance(raw, str) and raw.isdigit():
            frames = int(raw)
            break
        if isinstance(raw, int):
            frames = raw
            break
    duration: float | None = None
    fmt = data.get("format", {})
    if isinstance(fmt, dict):
        raw_dur = fmt.get("duration")
        if isinstance(raw_dur, str):
            try:
                duration = float(raw_dur)
            except ValueError:
                duration = None
    if duration is None:
        raw_dur = video.get("duration")
        if isinstance(raw_dur, str):
            try:
                duration = float(raw_dur)
            except ValueError:
                duration = None
    if fps is None:
        if frames is not None and duration:
            fps = frames / duration
        else:
            raise RenderError(f"probe failed: unknown fps for {path}")
    if frames is None:
        if duration is not None and fps > 0:
            frames = round(duration * fps)
        else:
            raise RenderError(f"probe failed: unknown frame count for {path}")
    if duration is None:
        duration = frames / fps if fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": float(fps),
        "frames": int(frames),
        "duration": float(duration),
        "has_audio": bool(has_audio),
    }
