"""Tests for WP-43 (Meta MMS-TTS voice adapter). Written with the package.

A fake VITS backend (fixed frames per token, sine waveform) drives every test, so the default suite needs
neither torch nor a model download. Only ``test_live_mms_synthesis`` loads the real model (``@pytest.mark.live``).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import karagoz.voice.mms as mms_module
from karagoz.voice.elevenlabs import VoiceConfig, audio_duration
from karagoz.voice.mms import MmsTts, Synthesis, TokenSpan

NEEDS_FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
NEEDS_FFPROBE = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")


@dataclass
class FakeBackend:
    """Deterministic stand-in for the VITS forward pass: fixed frames per token, sine waveform."""

    sample_rate: int = 16000
    hop_length: int = 256
    char_frames: int = 20
    gap_frames: int = 6  # spaces are voiced gaps, like the real tokenizer's space token
    blank_frames: int = 4
    drop: str = ",."  # punctuation the fake tokenizer drops
    calls: list[dict[str, Any]] = field(default_factory=list)

    def synthesize(self, text: str, *, model: str, speed: float, seed: int) -> Synthesis:
        self.calls.append({"text": text, "model": model, "speed": speed, "seed": seed})
        kept = [i for i, ch in enumerate(text) if ch not in self.drop]
        spans = [TokenSpan(None, self.blank_frames)]
        for i in kept:
            frames = self.gap_frames if text[i] == " " else self.char_frames
            spans.append(TokenSpan(i, frames))
            spans.append(TokenSpan(None, self.blank_frames))
        total = sum(s.frames for s in spans)
        n = total * self.hop_length
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return Synthesis(samples=samples, sample_rate=self.sample_rate, hop_length=self.hop_length, spans=tuple(spans))


def mms_voice(**overrides: Any) -> VoiceConfig:
    base: dict[str, Any] = {"voice_id": "", "provider": "mms", "model": "facebook/mms-tts-eng"}
    base.update(overrides)
    return VoiceConfig(**base)


@NEEDS_FFMPEG
@NEEDS_FFPROBE
def test_speak_alignment_matches_text_and_audio(tmp_path: Path) -> None:
    mp3, align = MmsTts(FakeBackend()).speak("Su ver.", mms_voice())
    assert align.characters == list("Su ver.")
    assert all(a <= b for a, b in zip(align.starts, align.starts[1:], strict=False))
    assert all(s <= e for s, e in zip(align.starts, align.ends, strict=True))
    assert len(mp3) > 1000
    path = tmp_path / "L01.mp3"
    path.write_bytes(mp3)
    duration = audio_duration(path)
    assert duration is not None
    # 6 kept letters (5 x 20 + 1 gap x 6 frames) plus 7 blanks x 4 frames, all at 256/16000 s per frame.
    assert align.ends[-1] == pytest.approx((5 * 20 + 6 + 7 * 4) * 256 / 16000)
    assert align.ends[-1] <= duration


@NEEDS_FFMPEG
def test_speak_dropped_punctuation_gets_zero_length_timings_in_order() -> None:
    _, align = MmsTts(FakeBackend()).speak("Hi, Jo.", mms_voice())
    assert align.characters == list("Hi, Jo.")
    assert align.starts[0] == 0.0
    comma, space, stop = 2, 3, 6
    assert align.starts[comma] == align.ends[comma] == align.ends[comma - 1]  # ',' sits between 'i' and ' '
    assert align.ends[space] > align.starts[space] == align.ends[comma]  # the space is a voiced gap
    assert align.starts[stop] == align.ends[stop] == align.ends[stop - 1]  # trailing '.' at the very end


@NEEDS_FFMPEG
def test_speak_splits_blank_frames_between_neighbouring_letters() -> None:
    _, align = MmsTts(FakeBackend()).speak("ab", mms_voice())
    # Leading/trailing blanks go wholly to the only neighbour; inner blanks split half/half:
    # 'a' gets 4 + 20 + 2 frames, 'b' gets 2 + 20 + 4 frames.
    assert align.starts == pytest.approx([0.0, 26 * 256 / 16000])
    assert align.ends == pytest.approx([26 * 256 / 16000, 52 * 256 / 16000])


@NEEDS_FFMPEG
def test_speak_same_input_gives_same_bytes() -> None:
    tts = MmsTts(FakeBackend())
    voice = mms_voice()
    mp3_a, align_a = tts.speak("Su ver.", voice)
    mp3_b, align_b = tts.speak("Su ver.", voice)
    assert mp3_a == mp3_b
    assert align_a == align_b


@NEEDS_FFMPEG
def test_speak_passes_model_speed_and_seed_to_the_backend() -> None:
    backend = FakeBackend()
    MmsTts(backend).speak("hi", mms_voice())
    assert backend.calls == [{"text": "hi", "model": "facebook/mms-tts-eng", "speed": 1.0, "seed": 0}]
    MmsTts(backend).speak("hi", mms_voice(model="facebook/mms-tts-tur", speed=1.2, seed=7))
    assert backend.calls[1] == {"text": "hi", "model": "facebook/mms-tts-tur", "speed": 1.2, "seed": 7}


def test_speak_rejects_a_nonpositive_speed() -> None:
    tts = MmsTts(FakeBackend())
    with pytest.raises(ValueError, match="speed"):
        tts.speak("hi", mms_voice(speed=0.0))
    with pytest.raises(ValueError, match="speed"):
        tts.speak("hi", mms_voice(speed=-1.0))


@NEEDS_FFMPEG
@NEEDS_FFPROBE
def test_speak_pitch_changes_the_audio_but_keeps_duration_and_timings(tmp_path: Path) -> None:
    tts = MmsTts(FakeBackend())
    mp3_plain, align_plain = tts.speak("Su ver.", mms_voice())
    mp3_pitched, align_pitched = tts.speak("Su ver.", mms_voice(pitch=2.0))
    assert mp3_pitched != mp3_plain
    assert align_pitched.starts == align_plain.starts and align_pitched.ends == align_plain.ends
    plain, pitched = tmp_path / "plain.mp3", tmp_path / "pitched.mp3"
    plain.write_bytes(mp3_plain)
    pitched.write_bytes(mp3_pitched)
    # atempo keeps the duration; mp3 header durations still quantize to whole frames (~26 ms).
    assert audio_duration(plain) == pytest.approx(audio_duration(pitched), abs=0.06)


def test_mms_imports_torch_and_transformers_only_inside_functions() -> None:
    src = Path(mms_module.__file__ or "").read_text(encoding="utf-8")
    heavy = {"torch", "transformers"}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {"a.name.split('.')[0]" for a in node.names}
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
            assert not (names & heavy), "torch/transformers must be imported inside functions only"
    assert "import torch" in src and "transformers" in src  # the real backend does use them


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="needs an env without torch")
def test_speak_without_the_mms_extra_names_the_extra() -> None:
    with pytest.raises(RuntimeError, match="extra mms"):
        MmsTts().speak("hi", mms_voice())


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("KARAGOZ_LIVE") != "1", reason="loads the MMS model; set KARAGOZ_LIVE=1")
@NEEDS_FFMPEG
@NEEDS_FFPROBE
def test_live_mms_synthesis(tmp_path: Path) -> None:
    mp3, align = MmsTts().speak("Pass me the tea.", mms_voice())
    assert len(mp3) > 1000
    assert align.characters == list("Pass me the tea.")
    path = tmp_path / "live.mp3"
    path.write_bytes(mp3)
    duration = audio_duration(path)
    assert duration is not None and align.ends[-1] <= duration
    again_mp3, again_align = MmsTts().speak("Pass me the tea.", mms_voice())
    assert again_mp3 == mp3 and again_align == align
