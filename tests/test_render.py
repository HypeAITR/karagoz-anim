"""Acceptance tests for WP-08 (render + mix). Written by the maintainer; do not edit. Needs ffmpeg on PATH."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from karagoz.engine.render import FPS, Clip, RenderError, encode, loudness, mix, probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(path: Path, seconds: float, freq: float = 440.0, amp: float = 0.3) -> Path:
    rate = 48000
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))
        )
    return path


def read_wav(path: Path) -> tuple[int, int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(-1, ch)
    return rate, ch, data


def first_sound(data: np.ndarray, rate: int) -> float:
    idx = np.nonzero(np.abs(data[:, 0]) > 500)[0]
    return idx[0] / rate if len(idx) else -1.0


def frames(d: Path, n: int, size: tuple[int, int] = (1920, 1080)) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, (i * 10 % 256, 80, 160)).save(d / f"{i:05d}.png")
    return d


def test_mix_places_clips_inside_the_audio(tmp_path: Path) -> None:
    a = tone(tmp_path / "a.wav", 0.5)
    b = tone(tmp_path / "b.wav", 0.5, freq=660)
    out = mix([Clip(a, 12), Clip(b, 48)], length=72, out_wav=tmp_path / "mix.wav")
    rate, ch, data = read_wav(out)
    assert rate == 48000 and ch == 2
    assert len(data) == 72 * 48000 // FPS
    assert first_sound(data, rate) == pytest.approx(0.5, abs=0.01)  # frame 12 = 0.5 s
    assert np.abs(data[int(2.05 * rate) : int(2.45 * rate), 0]).max() > 500  # second clip: frame 48 = 2.0 s
    assert np.abs(data[int(1.1 * rate) : int(1.9 * rate), 0]).max() < 50  # silence between 1.0 s and 2.0 s


def test_mix_cuts_and_silence(tmp_path: Path) -> None:
    a = tone(tmp_path / "a.wav", 3.0)
    rate, _, data = read_wav(mix([Clip(a, 0)], length=24, out_wav=tmp_path / "m.wav"))
    assert len(data) == 48000
    _, _, silent = read_wav(mix([], length=48, out_wav=tmp_path / "s.wav"))
    assert len(silent) == 96000 and np.abs(silent).max() == 0


def test_encode_probe_and_duration(tmp_path: Path) -> None:
    fd = frames(tmp_path / "f", 30)
    wav = mix([Clip(tone(tmp_path / "a.wav", 2.0), 0)], length=30, out_wav=tmp_path / "m.wav")
    mp4 = encode(fd, wav, tmp_path / "out.mp4")
    info = probe(mp4)
    assert (info["width"], info["height"]) == (1920, 1080)
    assert info["fps"] == pytest.approx(24.0)
    assert info["frames"] == 30
    assert info["duration"] == pytest.approx(30 / 24, abs=0.05)
    assert info["has_audio"] is True


def test_encode_without_audio(tmp_path: Path) -> None:
    mp4 = encode(frames(tmp_path / "f", 12), None, tmp_path / "out.mp4")
    assert probe(mp4)["has_audio"] is False
    with pytest.raises(RenderError):
        loudness(mp4)


def test_encode_is_deterministic(tmp_path: Path) -> None:
    fd = frames(tmp_path / "f", 24)
    wav = mix([Clip(tone(tmp_path / "a.wav", 0.5), 6)], length=24, out_wav=tmp_path / "m.wav")
    a = encode(fd, wav, tmp_path / "a.mp4").read_bytes()
    b = encode(fd, wav, tmp_path / "b.mp4").read_bytes()
    assert a == b


def test_loudness(tmp_path: Path) -> None:
    wav = mix([Clip(tone(tmp_path / "a.wav", 3.0, amp=0.3), 0)], length=72, out_wav=tmp_path / "m.wav")
    lufs = loudness(wav)
    assert -20 < lufs < -5


def test_missing_frames_raise(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(RenderError):
        encode(tmp_path / "empty", None, tmp_path / "x.mp4")
