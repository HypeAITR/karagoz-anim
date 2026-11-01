"""Extra tests for WP-47 (empty TTS audio, voices.yaml provider knobs)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from karagoz.voice.elevenlabs import ElevenLabs, VoiceConfig, load_voices, synthesize_episode

FAKE_KEY = "fake-key-for-tests"


class _EmptyAudioTransport:
    """ElevenLabs transport with a valid alignment but zero audio bytes."""

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        text = body["text"]
        return 200, {
            "audio_base64": base64.b64encode(b"").decode(),
            "alignment": {
                "characters": list(text),
                "character_start_times_seconds": [i * 0.1 for i in range(len(text))],
                "character_end_times_seconds": [(i + 1) * 0.1 for i in range(len(text))],
            },
        }


def test_empty_tts_audio_writes_no_cache_files(tmp_path: Path) -> None:
    prod = tmp_path / "prod"
    ep = "ep"
    (prod / ep).mkdir(parents=True)
    (prod / "voices.yaml").write_text('version: 1\nvoices:\n  ali: { voice_id: "v-ali" }\n', encoding="utf-8")
    (prod / ep / "lines.yaml").write_text(
        'version: 1\nepisode: ep\nlines:\n  - { id: L01, speaker: ali, text: "Su ver.", at: start }\n',
        encoding="utf-8",
    )
    client = ElevenLabs(transport=_EmptyAudioTransport(), api_key=lambda: FAKE_KEY)
    with pytest.raises(RuntimeError, match="empty audio"):
        synthesize_episode(prod, ep, client, duration=lambda path: None)
    assert list((prod / ep / "audio").iterdir()) == []


def test_huge_finite_pitch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "voices.yaml"
    path.write_text("version: 1\nvoices:\n  ali: { provider: mms, pitch: 20000 }\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pitch"):
        load_voices(path)


def test_valid_provider_knobs_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "voices.yaml"
    path.write_text(
        "version: 1\nvoices:\n"
        '  ali: { voice_id: "v-ali", stability: 0.4, similarity: 0.7, speed: 1.1, seed: 3 }\n'
        "  ayse: { provider: mms, pitch: 2.0, speed: 0.9, seed: 5 }\n",
        encoding="utf-8",
    )
    voices = load_voices(path)
    assert voices["ali"] == VoiceConfig(voice_id="v-ali", stability=0.4, similarity=0.7, speed=1.1, seed=3)
    assert voices["ayse"].provider == "mms" and voices["ayse"].pitch == 2.0
