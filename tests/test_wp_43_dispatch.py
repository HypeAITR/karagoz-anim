"""Tests for WP-43 (mms voices in config, per-provider dispatch, mms sound effects)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
import yaml
from test_sfx import SOUNDS, FakeBytes
from test_voice import EP, FIX, FakeTransport
from test_wp_43_mms import FakeBackend

from karagoz.voice.elevenlabs import (
    ElevenLabs,
    VoiceConfig,
    cache_key,
    load_voices,
    synthesize_episode,
    synthesize_sounds,
)
from karagoz.voice.mms import MmsTts

NEEDS_FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
FAKE_KEY = "fake-key-for-tests"


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / EP).mkdir(parents=True)
    shutil.copy(FIX / "lines.yaml", p / EP / "lines.yaml")
    (p / "voices.yaml").write_text(
        'version: 1\nvoices:\n  ali: { provider: mms }\n  ayse: { voice_id: "v-ayse" }\n', encoding="utf-8"
    )
    return p


@pytest.fixture
def mms_prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / EP).mkdir(parents=True)
    shutil.copy(FIX / "lines.yaml", p / EP / "lines.yaml")
    (p / "voices.yaml").write_text(
        "version: 1\nvoices:\n  ali: { provider: mms }\n  ayse: { provider: mms, pitch: -2.0 }\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sounds_prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / EP).mkdir(parents=True)
    lines = yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8"))
    lines["sounds"] = SOUNDS
    (p / EP / "lines.yaml").write_text(yaml.safe_dump(lines, allow_unicode=True), encoding="utf-8")
    (p / "voices.yaml").write_text(
        "version: 1\nvoices:\n  ali: { provider: mms }\n  ayse: { provider: mms }\n", encoding="utf-8"
    )
    return p


def el_client(t: FakeTransport) -> ElevenLabs:
    return ElevenLabs(transport=t, api_key=lambda: FAKE_KEY)


def test_provider_defaults_to_elevenlabs_and_mms_has_its_own_model_default(prod: Path) -> None:
    voices = load_voices(prod / "voices.yaml")
    assert voices["ayse"] == VoiceConfig(voice_id="v-ayse")
    assert voices["ayse"].provider == "elevenlabs" and voices["ayse"].model == "eleven_v4"
    assert voices["ayse"].pitch is None
    assert voices["ali"].provider == "mms" and voices["ali"].model == "facebook/mms-tts-eng"
    assert voices["ali"].voice_id == ""  # mms needs no voice id


def test_mms_voice_options_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "voices.yaml"
    path.write_text(
        "version: 1\nvoices:\n  ali: { provider: mms, model: facebook/mms-tts-tur, speed: 1.1, pitch: 2.0, seed: 3 }\n",
        encoding="utf-8",
    )
    assert load_voices(path)["ali"] == VoiceConfig(
        voice_id="", provider="mms", model="facebook/mms-tts-tur", speed=1.1, pitch=2.0, seed=3
    )


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "voices.yaml"
    path.write_text('version: 1\nvoices:\n  ali: { provider: azure, voice_id: "v" }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="provider"):
        load_voices(path)


def test_cache_key_covers_provider_and_pitch() -> None:
    v = VoiceConfig(voice_id="v1")
    assert cache_key("a", v) == cache_key("a", VoiceConfig(voice_id="v1"))
    assert cache_key("a", v) != cache_key("a", VoiceConfig(voice_id="v1", provider="mms"))
    assert cache_key("a", v) != cache_key("a", VoiceConfig(voice_id="v1", pitch=2.0))


@NEEDS_FFMPEG
def test_synthesize_episode_picks_the_client_per_provider(prod: Path) -> None:
    t = FakeTransport()
    backend = FakeBackend()
    out = synthesize_episode(prod, EP, {"elevenlabs": el_client(t), "mms": MmsTts(backend)})
    assert [a.line for a in out] == ["L01", "L02"] and not any(a.cached for a in out)
    assert [call[1]["text"] for call in t.calls] == ["Yarın pazar, Ali."]  # L01 speaker ayse
    assert [call["text"] for call in backend.calls] == ["Su ver bana."]  # L02 speaker ali
    assert (prod / EP / "audio" / "L02.mp3").stat().st_size > 1000
    again = synthesize_episode(prod, EP, {"elevenlabs": el_client(t), "mms": MmsTts(backend)})
    assert all(a.cached for a in again) and len(t.calls) == 1 and len(backend.calls) == 1


@NEEDS_FFMPEG
def test_a_single_client_serves_only_its_own_provider(mms_prod: Path) -> None:
    backend = FakeBackend()
    out = synthesize_episode(mms_prod, EP, MmsTts(backend))
    assert [a.line for a in out] == ["L01", "L02"]
    assert [call["text"] for call in backend.calls] == ["Yarın pazar, Ali.", "Su ver bana."]


def test_an_unknown_provider_string_fails_dispatch(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import karagoz.voice.elevenlabs as el

    def bad_voices(_path: Path) -> dict[str, VoiceConfig]:
        return {"ali": VoiceConfig(voice_id="v"), "ayse": VoiceConfig(voice_id="v", provider="azure")}  # type: ignore[typeddict-item]

    monkeypatch.setattr(el, "load_voices", bad_voices)
    with pytest.raises(ValueError, match="provider"):
        synthesize_episode(prod, EP, {"elevenlabs": el_client(FakeTransport())})


def test_mms_production_sounds_use_only_provided_files(sounds_prod: Path, caplog: pytest.LogCaptureFixture) -> None:
    audio = sounds_prod / EP / "audio"
    audio.mkdir(parents=True)
    (audio / "S03.wav").write_bytes(b"MY OWN DOOR")
    b = FakeBytes()
    with caplog.at_level(logging.WARNING, logger="karagoz.voice.elevenlabs"):
        out = synthesize_sounds(sounds_prod, EP, ElevenLabs(bytes_transport=b, api_key=lambda: FAKE_KEY))
    assert [(s.sound, s.status) for s in out] == [("S03", "provided")]
    assert (audio / "S03.wav").read_bytes() == b"MY OWN DOOR" and not (audio / "S03.mp3").exists()
    assert b.calls == []  # ElevenLabs is never called for sounds in an mms production
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1 and "S01" in warnings[0]  # one warning per missing effect; music stays silent


def test_mixed_production_sounds_still_generate(sounds_prod: Path) -> None:
    (sounds_prod / "voices.yaml").write_text(
        'version: 1\nvoices:\n  ali: { provider: mms }\n  ayse: { voice_id: "v-ayse" }\n', encoding="utf-8"
    )
    b = FakeBytes()
    out = synthesize_sounds(sounds_prod, EP, ElevenLabs(bytes_transport=b, api_key=lambda: FAKE_KEY))
    assert [(s.sound, s.status) for s in out] == [("S01", "new"), ("S03", "new")]
    assert len(b.calls) == 2


def test_sounds_without_a_voices_file_still_generate(tmp_path: Path) -> None:
    p = tmp_path / "prod"
    (p / EP).mkdir(parents=True)
    lines = yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8"))
    lines["sounds"] = SOUNDS
    (p / EP / "lines.yaml").write_text(yaml.safe_dump(lines, allow_unicode=True), encoding="utf-8")
    b = FakeBytes()
    out = synthesize_sounds(p, EP, ElevenLabs(bytes_transport=b, api_key=lambda: FAKE_KEY))
    assert [(s.sound, s.status) for s in out] == [("S01", "new"), ("S03", "new")]


def test_provider_names_supported_by_dispatch() -> None:
    assert ElevenLabs.provider == "elevenlabs"
    assert MmsTts.provider == "mms"
    assert set(VoiceConfig.__dataclass_fields__) >= {"provider", "pitch", "voice_id", "model", "speed"}
