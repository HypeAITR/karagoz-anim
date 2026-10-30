"""Acceptance tests for WP-34 (alignment clamped to the audio duration). Written by the maintainer; do not edit."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from test_voice import EP, FakeTransport, client, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.spec import Align, load
from karagoz.voice.elevenlabs import audio_duration, clamp_align, synthesize_episode


def _align(n: int, step: float = 0.1) -> Align:
    return Align(
        version=1,
        line="L01",
        text="x" * n,
        characters=["x"] * n,
        starts=[round(i * step, 6) for i in range(n)],
        ends=[round((i + 1) * step, 6) for i in range(n)],
    )


def test_clamp_cuts_every_time_to_the_duration() -> None:
    a = clamp_align(_align(10), 0.75)
    assert max(a.ends) == pytest.approx(0.75) and max(a.starts) <= 0.75
    assert a.ends[:7] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert a.starts[:8] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert a.characters == ["x"] * 10 and a.text == "x" * 10 and a.line == "L01"
    Align.model_validate(a.model_dump())  # still a valid alignment


def test_clamp_keeps_an_alignment_that_fits() -> None:
    a = _align(5)
    assert clamp_align(a, 2.0) == a


def test_new_lines_are_written_clamped(prod: Path) -> None:  # noqa: F811
    durations = {"L01.mp3": 1.62, "L02.mp3": 1.04}
    synthesize_episode(prod, EP, client(FakeTransport()), duration=lambda p: durations[p.name])
    audio = prod / EP / "audio"
    a1, a2 = load(audio / "L01.align.json"), load(audio / "L02.align.json")
    assert isinstance(a1, Align) and isinstance(a2, Align)
    assert a1.ends[-1] == pytest.approx(1.62) and max(a1.starts + a1.ends) <= 1.62  # 17 letters, fake ends at 1.7
    assert a2.ends[-1] == pytest.approx(1.04) and max(a2.starts + a2.ends) <= 1.04  # 12 letters, fake ends at 1.2


def test_cached_alignment_is_clamped_without_a_new_call(prod: Path) -> None:  # noqa: F811
    t = FakeTransport()
    synthesize_episode(prod, EP, client(t), duration=lambda p: None)  # as today: nothing clamped
    audio = prod / EP / "audio"
    before = load(audio / "L01.align.json")
    assert isinstance(before, Align) and before.ends[-1] == pytest.approx(1.7)
    again = synthesize_episode(prod, EP, client(t), duration=lambda p: 1.5)
    assert all(x.cached for x in again) and len(t.calls) == 2
    after = load(audio / "L01.align.json")
    assert isinstance(after, Align) and after.ends[-1] == pytest.approx(1.5) and after.characters == before.characters


def test_unknown_duration_changes_nothing(prod: Path) -> None:  # noqa: F811
    synthesize_episode(prod, EP, client(FakeTransport()), duration=lambda p: None)
    a = load(prod / EP / "audio" / "L02.align.json")
    assert isinstance(a, Align) and a.ends[-1] == pytest.approx(1.2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")
def test_audio_duration_of_a_real_mp3(tmp_path: Path) -> None:
    mp3 = tmp_path / "s.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5", str(mp3)], check=True
    )
    d = audio_duration(mp3)
    assert d is not None and 0.45 <= d <= 0.6


def test_audio_duration_of_a_file_that_is_not_audio_is_none(tmp_path: Path) -> None:
    fake = tmp_path / "x.mp3"
    fake.write_bytes(b"MP3:not audio")
    assert audio_duration(fake) is None
    assert audio_duration(tmp_path / "missing.mp3") is None
