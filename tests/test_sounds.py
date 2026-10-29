"""Acceptance tests for WP-31 (sound effects and music in the shot mix). Written by the maintainer; do not edit.

Needs ffmpeg. The render fixture is test_shot's mini production: sh01 = [0, 100), L01 (ayse) at 8, L02 (ali) at 60.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from test_shot import EP, prod  # noqa: F401  (fixture)

from karagoz.engine.render import (
    DUCK_RAMP_FRAMES,
    FPS,
    MUSIC_DUCK_DB,
    Clip,
    add_into,
    duck,
    mix,
    sound_file,
)
from karagoz.engine.shot import render_shot
from karagoz.spec import SoundEntry

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

RATE = 48000
SPF = RATE // FPS  # samples per frame


def tone(path: Path, seconds: float, amp: float = 0.3, freq: float = 440.0) -> Path:
    n = int(seconds * RATE)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(
            b"".join(struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / RATE))) for i in range(n))
        )
    return path


def samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())[:, 0]


def rms(data: np.ndarray, f0: int, f1: int) -> float:
    seg = data[f0 * SPF : f1 * SPF].astype(np.float64)
    return float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0


def steady(tmp_path: Path, name: str, frames: int = 120) -> Path:
    src = tone(tmp_path / f"{name}-src.wav", frames / FPS + 0.5)
    return mix([Clip(src, 0)], length=frames, out_wav=tmp_path / name)


def test_constants() -> None:
    assert MUSIC_DUCK_DB == -12.0 and DUCK_RAMP_FRAMES == 6


def test_sound_entry_defaults_and_kind() -> None:
    s = SoundEntry.model_validate({"id": "S01", "description": "cup on table", "at": "L01.end"})
    assert s.kind == "sfx" and s.gain_db == 0.0
    m = SoundEntry.model_validate({"id": "S02", "description": "bed", "at": "sh01.in", "kind": "music", "gain_db": -18})
    assert m.kind == "music" and m.gain_db == -18.0
    with pytest.raises(ValidationError):
        SoundEntry.model_validate({"id": "S03", "description": "x", "at": "sh01.in", "kind": "voice"})


def test_clip_gain(tmp_path: Path) -> None:
    src = tone(tmp_path / "a.wav", 1.0)
    full = samples(mix([Clip(src, 0)], length=24, out_wav=tmp_path / "full.wav"))
    half = samples(mix([Clip(src, 0, gain_db=-6.0)], length=24, out_wav=tmp_path / "half.wav"))
    assert rms(half, 2, 20) / rms(full, 2, 20) == pytest.approx(10 ** (-6 / 20), rel=0.03)


def test_duck_lowers_only_the_span_with_ramps(tmp_path: Path) -> None:
    wav = steady(tmp_path, "m.wav")
    before = samples(wav).copy()
    duck(wav, [(40, 70)])
    after = samples(wav)
    ratio_in = rms(after, 42, 68) / rms(before, 42, 68)
    assert ratio_in == pytest.approx(10 ** (MUSIC_DUCK_DB / 20), rel=0.05)
    assert np.array_equal(after[: 33 * SPF], before[: 33 * SPF])  # before the ramp (40 - 6 = 34)
    assert np.array_equal(after[77 * SPF :], before[77 * SPF :])  # after the ramp (70 + 6 = 76)
    r1 = rms(after, 34, 36) / rms(before, 34, 36)
    r2 = rms(after, 37, 39) / rms(before, 37, 39)
    assert 1.0 > r1 > r2 > ratio_in  # ramp goes down towards the span


def test_duck_without_spans_is_untouched_and_deterministic(tmp_path: Path) -> None:
    wav = steady(tmp_path, "m.wav")
    raw = wav.read_bytes()
    duck(wav, [])
    assert wav.read_bytes() == raw
    a, b = steady(tmp_path, "a.wav"), steady(tmp_path, "b.wav")
    duck(a, [(10, 30), (25, 50)])
    duck(b, [(10, 30), (25, 50)])
    assert a.read_bytes() == b.read_bytes()


def test_add_into(tmp_path: Path) -> None:
    dst = mix([], length=48, out_wav=tmp_path / "dst.wav")
    src = steady(tmp_path, "src.wav", frames=48)
    add_into(dst, src, gain_db=-6.0)
    assert rms(samples(dst), 2, 46) / rms(samples(src), 2, 46) == pytest.approx(10 ** (-6 / 20), rel=0.03)
    short = mix([], length=24, out_wav=tmp_path / "short.wav")
    with pytest.raises(ValueError):
        add_into(dst, short)


def test_sound_file(tmp_path: Path) -> None:
    assert sound_file(tmp_path, "S01") is None
    (tmp_path / "S01.wav").write_bytes(b"x")
    assert sound_file(tmp_path, "S01") == tmp_path / "S01.wav"
    (tmp_path / "S01.mp3").write_bytes(b"x")
    assert sound_file(tmp_path, "S01") == tmp_path / "S01.mp3"


def add_sounds(prod_dir: Path, with_sfx_file: bool = True) -> None:
    ep = prod_dir / EP
    lines = yaml.safe_load((ep / "lines.yaml").read_text(encoding="utf-8"))
    lines["sounds"] = [
        {"id": "S01", "description": "cup on table", "at": "L01.end + 4f"},
        {"id": "S02", "description": "music bed", "at": "sh01.in", "kind": "music", "gain_db": -6},
    ]
    (ep / "lines.yaml").write_text(yaml.safe_dump(lines, allow_unicode=True), encoding="utf-8")
    shots = yaml.safe_load((ep / "shots.yaml").read_text(encoding="utf-8"))
    shots["shots"][0]["sounds"] = ["S01"]
    (ep / "shots.yaml").write_text(yaml.safe_dump(shots, allow_unicode=True), encoding="utf-8")
    if with_sfx_file:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=1000:d=0.2", str(ep / "audio" / "S01.mp3")],
            check=True,
        )
    tone(ep / "audio" / "S02.wav", 6.0, amp=0.2, freq=220)


def test_render_shot_mixes_sfx_and_ducked_music(prod: Path) -> None:  # noqa: F811
    add_sounds(prod)
    r = render_shot(prod, EP, "sh01")
    assert not any(w.startswith("sound") for w in r.warnings)
    assert r.lufs is not None and r.lufs == pytest.approx(-14.0, abs=1.0)
    music = samples(prod / EP / "render" / "sh01" / "music.wav")
    # L01 is spoken from frame 8, L02 from frame 60: music is ducked there, full between the lines' ramps
    quiet = rms(music, 12, 30)
    loud = rms(music, 0, 2)
    assert quiet / loud == pytest.approx(10 ** (MUSIC_DUCK_DB / 20), rel=0.1)


def test_missing_sfx_file_is_a_warning(prod: Path) -> None:  # noqa: F811
    add_sounds(prod, with_sfx_file=False)
    r = render_shot(prod, EP, "sh01")
    assert any(w == "sound S01: no audio file in audio/" for w in r.warnings)
