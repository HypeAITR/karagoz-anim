"""Acceptance tests for WP-05 (ElevenLabs voice adapter). Written by the maintainer; do not edit."""

from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Any

import pytest

from karagoz import cli
from karagoz.spec import Align, load
from karagoz.voice.elevenlabs import ElevenLabs, VoiceConfig, cache_key, load_voices, synthesize_episode

FIX = Path(__file__).parent / "fixtures" / "spec"
FAKE_KEY = "fake-key-for-tests"
EP = "ep-test"


class FakeTransport:
    def __init__(self, status: int = 200) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.status = status

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, body, headers))
        if self.status != 200:
            return self.status, {"detail": {"message": "quota exceeded"}}
        text = body["text"]
        return 200, {
            "audio_base64": base64.b64encode(b"MP3:" + text.encode()).decode(),
            "alignment": {
                "characters": list(text),
                "character_start_times_seconds": [i * 0.1 for i in range(len(text))],
                "character_end_times_seconds": [(i + 1) * 0.1 for i in range(len(text))],
            },
        }


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / EP).mkdir(parents=True)
    shutil.copy(FIX / "lines.yaml", p / EP / "lines.yaml")
    (p / "voices.yaml").write_text(
        "version: 1\nvoices:\n"
        '  ali: { voice_id: "v-ali", stability: 0.4, speed: 0.9, seed: 7 }\n'
        '  ayse: { voice_id: "v-ayse" }\n',
        encoding="utf-8",
    )
    return p


def client(t: FakeTransport) -> ElevenLabs:
    return ElevenLabs(transport=t, api_key=lambda: FAKE_KEY)


def test_load_voices(prod: Path) -> None:
    v = load_voices(prod / "voices.yaml")
    assert v["ali"] == VoiceConfig(voice_id="v-ali", stability=0.4, speed=0.9, seed=7)
    assert v["ayse"].model == "eleven_v4" and v["ayse"].language == "tr"


def test_speak_request_and_align() -> None:
    t = FakeTransport()
    audio, align = client(t).speak("Su ver.", VoiceConfig(voice_id="v1", stability=0.4, seed=3))
    url, body, headers = t.calls[0]
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/v1/with-timestamps?output_format=mp3_44100_128"
    assert body == {
        "text": "Su ver.",
        "model_id": "eleven_v4",
        "language_code": "tr",
        "voice_settings": {"stability": 0.4},
        "seed": 3,
    }
    assert headers["xi-api-key"] == FAKE_KEY
    assert audio == b"MP3:Su ver."
    assert isinstance(align, Align) and align.characters == list("Su ver.")
    assert align.ends[-1] == pytest.approx(0.7)


def test_http_error_hides_key() -> None:
    with pytest.raises(RuntimeError) as e:
        client(FakeTransport(status=401)).speak("x", VoiceConfig(voice_id="v1"))
    assert "401" in str(e.value) and FAKE_KEY not in str(e.value)


def test_synthesize_writes_files_and_caches(prod: Path) -> None:
    t = FakeTransport()
    first = synthesize_episode(prod, EP, client(t))
    assert [(a.line, a.cached) for a in first] == [("L01", False), ("L02", False)]
    assert len(t.calls) == 2
    assert t.calls[0][0].endswith("/v-ayse/with-timestamps?output_format=mp3_44100_128")  # L01 speaker ayse
    assert "delivery" not in t.calls[0][1] and "neşeli" not in str(t.calls[0][1])
    audio_dir = prod / EP / "audio"
    a = load(audio_dir / "L01.align.json")
    assert isinstance(a, Align) and a.line == "L01" and a.text == "Yarın pazar, Ali."
    assert (audio_dir / "L02.mp3").read_bytes() == b"MP3:Su ver bana."
    second = synthesize_episode(prod, EP, client(t))
    assert all(x.cached for x in second) and len(t.calls) == 2
    for p in audio_dir.iterdir():
        assert FAKE_KEY not in p.read_text(encoding="utf-8", errors="ignore")


def test_changed_voice_or_text_recalls_only_that_line(prod: Path) -> None:
    t = FakeTransport()
    synthesize_episode(prod, EP, client(t))
    lines = prod / EP / "lines.yaml"
    lines.write_text(lines.read_text(encoding="utf-8").replace("Su ver bana.", "Su ver."), encoding="utf-8")
    again = synthesize_episode(prod, EP, client(t))
    assert [(a.line, a.cached) for a in again] == [("L01", True), ("L02", False)]
    assert len(t.calls) == 3


def test_cache_key_depends_on_everything() -> None:
    v = VoiceConfig(voice_id="v1")
    assert cache_key("a", v) == cache_key("a", VoiceConfig(voice_id="v1"))
    assert cache_key("a", v) != cache_key("b", v)
    assert cache_key("a", v) != cache_key("a", VoiceConfig(voice_id="v1", speed=1.1))
    assert len(cache_key("a", v)) == 64


def test_missing_voice_fails_before_any_call(prod: Path) -> None:
    (prod / "voices.yaml").write_text('version: 1\nvoices:\n  ali: { voice_id: "v-ali" }\n', encoding="utf-8")
    t = FakeTransport()
    with pytest.raises(ValueError, match="ayse"):
        synthesize_episode(prod, EP, client(t))
    assert t.calls == []


def test_cli(prod: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    t = FakeTransport()
    import karagoz.voice.elevenlabs as el

    monkeypatch.setattr(el, "ElevenLabs", lambda: client(t))
    assert cli.main(["voice", str(prod), EP]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["L01 new 1.70 s", "L02 new 1.20 s"]
    assert cli.main(["voice", str(prod), EP]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "L01 cached 1.70 s"
