"""Acceptance tests for WP-28 (dialogue mix to -14 LUFS). Written by the maintainer; do not edit. Needs ffmpeg."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
from test_shot import EP, prod  # noqa: F401  (fixture: mini production, sh01 with two sine lines)

from karagoz.engine.render import PEAK_DBFS, TARGET_LUFS, Clip, loudness, mix, normalize
from karagoz.engine.shot import render_shot

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

PEAK_MAX = round(32767 * 10 ** (PEAK_DBFS / 20)) + 1


def tone(path: Path, seconds: float, amp: float, spike: float | None = None) -> Path:
    rate = 48000
    n = int(seconds * rate)
    vals = [int(amp * 32767 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
    if spike is not None:
        vals[n // 2] = int(spike * 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", v) for v in vals))
    return path


def read(path: Path) -> tuple[tuple[int, int, int, int], np.ndarray]:
    with wave.open(str(path), "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return params, data


def mixed(tmp_path: Path, name: str, amp: float, spike: float | None = None) -> Path:
    src = tone(tmp_path / f"{name}-src.wav", 3.0, amp, spike)
    return mix([Clip(src, 0)], length=72, out_wav=tmp_path / f"{name}.wav")


def test_constants() -> None:
    assert TARGET_LUFS == -14.0 and PEAK_DBFS == -1.0


def test_quiet_mix_is_raised_to_target(tmp_path: Path) -> None:
    wav = mixed(tmp_path, "quiet", 0.05)
    before = loudness(wav)
    gain = normalize(wav)
    assert gain == pytest.approx(TARGET_LUFS - before, abs=0.05)
    assert loudness(wav) == pytest.approx(TARGET_LUFS, abs=0.5)


def test_loud_mix_is_lowered_to_target(tmp_path: Path) -> None:
    wav = mixed(tmp_path, "loud", 0.9)
    gain = normalize(wav)
    assert gain < 0
    assert loudness(wav) == pytest.approx(TARGET_LUFS, abs=0.5)


def test_peak_limits_the_gain(tmp_path: Path) -> None:
    # quiet tone with one near-full-scale sample: loudness wants a big boost, the peak allows ~0.55 dB
    wav = mixed(tmp_path, "spike", 0.05, spike=0.95)
    _, before = read(wav)
    peak = int(np.abs(before.astype(np.int32)).max()) / 32767  # the mix's own peak (mono -> stereo lowers it)
    gain = normalize(wav)
    assert gain == pytest.approx(PEAK_DBFS - 20 * math.log10(peak), abs=0.05)
    _, data = read(wav)
    assert int(np.abs(data.astype(np.int32)).max()) <= PEAK_MAX
    assert loudness(wav) < TARGET_LUFS - 1.0


def test_format_and_length_unchanged(tmp_path: Path) -> None:
    wav = mixed(tmp_path, "fmt", 0.05)
    params_before, _ = read(wav)
    normalize(wav)
    params_after, data = read(wav)
    assert params_after == params_before
    assert len(data) == params_before[0] * params_before[3]


def test_silence_is_untouched(tmp_path: Path) -> None:
    wav = mix([], length=48, out_wav=tmp_path / "s.wav")
    before = wav.read_bytes()
    assert normalize(wav) == 0.0
    assert wav.read_bytes() == before


def test_deterministic(tmp_path: Path) -> None:
    a = mixed(tmp_path, "a", 0.05)
    b = mixed(tmp_path, "b", 0.05)
    normalize(a)
    normalize(b)
    assert a.read_bytes() == b.read_bytes()


def test_render_shot_is_at_target(prod: Path) -> None:  # noqa: F811
    r = render_shot(prod, EP, "sh01")
    assert r.lufs is not None and r.lufs == pytest.approx(TARGET_LUFS, abs=1.0)
    assert not any(w.startswith("audio:") for w in r.warnings)
