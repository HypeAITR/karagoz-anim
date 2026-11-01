"""BB-1 bug hunt: voice adapter bugs (ElevenLabs + voices.yaml).

Each test fails on today's code and proves one user-facing bug. Do not fix
the source to make them pass here; they are the bug report in executable form.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from karagoz.voice.elevenlabs import ElevenLabs, VoiceConfig, load_voices

FAKE_KEY = "fake-key-for-tests"


class _EmptyAudioTransport:
    """ElevenLabs transport that returns a valid alignment with zero audio bytes."""

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


def test_bb1_empty_tts_audio_is_rejected() -> None:
    """speak() must reject an empty audio payload instead of caching a 0-byte mp3.

    sound() already raises "empty audio response"; speak() writes the empty bytes
    to <line>.mp3 plus a key file, so the corrupt line stays cached and the render
    dies later with an ffmpeg tail ("Failed to find two consecutive MPEG frames").
    """
    client = ElevenLabs(transport=_EmptyAudioTransport(), api_key=lambda: FAKE_KEY)
    with pytest.raises(RuntimeError, match="empty audio"):
        client.speak("Su ver.", VoiceConfig(voice_id="v1"))


def test_bb1_elevenlabs_voice_rejects_pitch(tmp_path: Path) -> None:
    """pitch is an MMS-only knob; an ElevenLabs voice must not accept it silently.

    Today the pitch changes the cache key (so every tweak pays for a re-synthesis)
    while the request body never contains it, i.e. the paid call returns identical
    bytes. _parse_voice already validates provider-specific fields (voice_id), so
    fail fast there instead of ignoring the field.
    """
    path = tmp_path / "voices.yaml"
    path.write_text('version: 1\nvoices:\n  ali: { voice_id: "v-ali", pitch: 2.0 }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="pitch"):
        load_voices(path)


def test_bb1_nonfinite_pitch_is_rejected(tmp_path: Path) -> None:
    """Non-finite pitch must be rejected: inf hangs the render, nan breaks ffmpeg.

    pitch inf -> 2**(inf/12) == inf -> _atempo_chain(0.0) loops forever
    (0.0 / 0.5 stays 0.0); pitch nan -> filter "atempo=nan" -> cryptic ffmpeg
    failure. A finite but huge pitch (20000) overflows to inf the same way.
    """
    path = tmp_path / "voices.yaml"
    for raw in (".inf", "-.inf", ".nan"):
        path.write_text(f"version: 1\nvoices:\n  ali: {{ provider: mms, pitch: {raw} }}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pitch"):
            load_voices(path)


def test_bb1_mms_voice_rejects_elevenlabs_only_knobs(tmp_path: Path) -> None:
    """stability/similarity are ElevenLabs-only; an MMS voice must not take them.

    Today they are parsed, change the cache key (so every tweak re-synthesizes),
    and then are silently dropped by MmsTts.speak, which reads only speed/seed/pitch.
    """
    path = tmp_path / "voices.yaml"
    for field in ("stability", "similarity"):
        path.write_text(f"version: 1\nvoices:\n  ali: {{ provider: mms, {field}: 0.5 }}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            load_voices(path)
