"""ElevenLabs TTS adapter: MP3 audio plus per-letter timings for every voice line."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from karagoz import config
from karagoz.engine.render import _ffprobe_bin
from karagoz.spec import Align, Lines, load

BASE_URL = "https://api.elevenlabs.io"
TIMEOUT_SECONDS = 300.0
OUTPUT_FORMAT = "mp3_44100_128"
SFX_MODEL = "eleven_text_to_sound_v2"

Transport = Callable[[str, dict[str, Any], dict[str, str]], tuple[int, dict[str, Any]]]
BytesTransport = Callable[[str, dict[str, Any], dict[str, str]], tuple[int, bytes]]


@dataclass(frozen=True)
class VoiceConfig:
    voice_id: str
    model: str = "eleven_v4"
    language: str = "tr"
    stability: float | None = None
    similarity: float | None = None
    speed: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class LineAudio:
    line: str
    mp3: Path
    align: Path
    cached: bool


@dataclass(frozen=True)
class SoundAudio:
    sound: str
    path: Path
    status: Literal["new", "cached", "provided"]


def urllib_transport(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """POST JSON with urllib; returns (status, parsed JSON body)."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload: dict[str, Any] = json.load(response)
            return response.status, payload
    except urllib.error.HTTPError as err:
        text = err.read().decode("utf-8", errors="replace")
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            return err.code, {"error": {"message": text}}
        if isinstance(data, dict):
            return err.code, data
        return err.code, {"error": {"message": json.dumps(data)[:300]}}


def urllib_bytes_transport(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, bytes]:
    """POST JSON with urllib; returns (status, raw response bytes)."""
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


_VOICE_FIELDS = ("voice_id", "model", "language", "stability", "similarity", "speed", "seed")


def load_voices(path: Path) -> dict[str, VoiceConfig]:
    """Read voices.yaml into {speaker: VoiceConfig}; ValueError with a clear message if malformed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read voices file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with 'version' and 'voices'")
    if data.get("version") != 1:
        raise ValueError(f"{path}: expected 'version: 1'")
    voices = data.get("voices")
    if not isinstance(voices, dict):
        raise ValueError(f"{path}: 'voices' must be a mapping of speaker name to voice settings")
    out: dict[str, VoiceConfig] = {}
    for speaker, entry in voices.items():
        if not isinstance(speaker, str):
            raise ValueError(f"{path}: speaker name must be a string, got {speaker!r}")
        out[speaker] = _parse_voice(path, speaker, entry)
    return out


def _parse_voice(path: Path, speaker: str, entry: Any) -> VoiceConfig:
    where = f"{path}: voices.{speaker}"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected a mapping, got {entry!r}")
    unknown = sorted(set(entry) - set(_VOICE_FIELDS))
    if unknown:
        raise ValueError(f"{where}: unknown field(s): {', '.join(unknown)}")
    voice_id = entry.get("voice_id")
    if not isinstance(voice_id, str) or not voice_id:
        raise ValueError(f"{where}: 'voice_id' must be a non-empty string")
    model = entry.get("model", "eleven_v4")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{where}: 'model' must be a non-empty string")
    language = entry.get("language", "tr")
    if not isinstance(language, str) or not language:
        raise ValueError(f"{where}: 'language' must be a non-empty string")
    return VoiceConfig(
        voice_id=voice_id,
        model=model,
        language=language,
        stability=_optional_float(entry, where, "stability"),
        similarity=_optional_float(entry, where, "similarity"),
        speed=_optional_float(entry, where, "speed"),
        seed=_optional_int(entry, where, "seed"),
    )


def _optional_float(entry: dict[Any, Any], where: str, field: str) -> float | None:
    value = entry.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: '{field}' must be a number, got {value!r}")
    return float(value)


def _optional_int(entry: dict[Any, Any], where: str, field: str) -> int | None:
    value = entry.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: '{field}' must be an integer, got {value!r}")
    return value


def _error_message(result: dict[str, Any]) -> str:
    """Best-effort human message from an ElevenLabs error payload."""
    for field in ("detail", "error", "message"):
        detail = result.get(field)
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, dict):
            message = detail.get("message")
            if isinstance(message, str) and message:
                return message
            return json.dumps(detail, sort_keys=True)
    return json.dumps(result, sort_keys=True)


def _without_key(message: str, key: str) -> str:
    return message.replace(key, "[redacted]") if key else message


@dataclass
class ElevenLabs:
    transport: Transport = urllib_transport
    bytes_transport: BytesTransport = urllib_bytes_transport
    api_key: Callable[[], str] = lambda: config.secret("ELEVENLABS_API_KEY")
    base_url: str = BASE_URL

    def speak(self, text: str, voice: VoiceConfig) -> tuple[bytes, Align]:
        """Synthesize one line; returns (mp3 bytes, per-letter alignment with line unset)."""
        key = self.api_key()
        url = f"{self.base_url}/v1/text-to-speech/{voice.voice_id}/with-timestamps?output_format={OUTPUT_FORMAT}"
        body: dict[str, Any] = {"text": text, "model_id": voice.model, "language_code": voice.language}
        settings: dict[str, float] = {}
        if voice.stability is not None:
            settings["stability"] = voice.stability
        if voice.similarity is not None:
            settings["similarity_boost"] = voice.similarity
        if voice.speed is not None:
            settings["speed"] = voice.speed
        if settings:
            body["voice_settings"] = settings
        if voice.seed is not None:
            body["seed"] = voice.seed
        headers = {"xi-api-key": key, "Content-Type": "application/json"}
        status, result = self.transport(url, body, headers)
        if status != 200:
            raise RuntimeError(f"ElevenLabs HTTP {status}: {_without_key(_error_message(result), key)[:300]}")
        try:
            audio = base64.b64decode(result["audio_base64"])
            alignment = result["alignment"]
            align = Align(
                version=1,
                line="L00",
                text=text,
                characters=alignment["characters"],
                starts=alignment["character_start_times_seconds"],
                ends=alignment["character_end_times_seconds"],
            )
        except ValidationError as exc:
            raise RuntimeError(f"ElevenLabs: invalid alignment in response ({exc.error_count()} error(s))") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"ElevenLabs: malformed response: {exc}") from exc
        return audio, align

    def sound(self, text: str) -> bytes:
        """Generate one sound effect; returns the raw MP3 bytes."""
        key = self.api_key()
        url = f"{self.base_url}/v1/sound-generation?output_format={OUTPUT_FORMAT}"
        body: dict[str, Any] = {"text": text, "model_id": SFX_MODEL}
        headers = {"xi-api-key": key, "Content-Type": "application/json"}
        status, data = self.bytes_transport(url, body, headers)
        if status != 200:
            text_body = data.decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(text_body)
            except json.JSONDecodeError:
                message = text_body
            else:
                message = _error_message(parsed) if isinstance(parsed, dict) else text_body
            raise RuntimeError(f"ElevenLabs HTTP {status}: {_without_key(message, key)[:300]}")
        if not data:
            raise RuntimeError("ElevenLabs: empty audio response")
        return data


def cache_key(text: str, voice: VoiceConfig) -> str:
    """sha256 hex of a canonical JSON of the text plus all voice fields."""
    canonical = json.dumps({"text": text, **asdict(voice)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sound_cache_key(description: str) -> str:
    """sha256 hex of a canonical JSON of the SFX model plus the description."""
    canonical = json.dumps({"model": SFX_MODEL, "text": description}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audio_duration(path: Path) -> float | None:
    """Audio duration in seconds via ffprobe; None when unknown (no ffprobe, failure, not audio)."""
    try:
        ffprobe = _ffprobe_bin()
    except RuntimeError:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    try:
        value = float(proc.stdout.strip())
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def clamp_align(align: Align, duration_s: float) -> Align:
    """Copy of align with every start and end capped at duration_s; nothing else changes."""
    return align.model_copy(
        update={
            "starts": [min(s, duration_s) for s in align.starts],
            "ends": [min(e, duration_s) for e in align.ends],
        }
    )


def synthesize_episode(
    prod: Path,
    episode: str,
    client: ElevenLabs | None = None,
    duration: Callable[[Path], float | None] = audio_duration,
) -> list[LineAudio]:
    """Synthesize every line of the episode; cached lines (mp3 + align + matching key) are skipped.

    Alignments are clamped to the audio's own duration, both for new lines and for
    cached lines whose align file on disk still runs past the audio.
    """
    voices = load_voices(prod / "voices.yaml")
    lines = load(prod / episode / "lines.yaml")
    if not isinstance(lines, Lines):
        raise ValueError(f"{prod / episode / 'lines.yaml'}: expected a lines file")
    for entry in lines.lines:
        if entry.speaker not in voices:
            raise ValueError(f"{episode}: no voice configured for speaker {entry.speaker!r} (line {entry.id})")
    active = ElevenLabs() if client is None else client
    audio_dir = prod / episode / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out: list[LineAudio] = []
    for entry in lines.lines:
        voice = voices[entry.speaker]
        key = cache_key(entry.text, voice)
        mp3_path = audio_dir / f"{entry.id}.mp3"
        align_path = audio_dir / f"{entry.id}.align.json"
        key_path = audio_dir / f"{entry.id}.key"
        if (
            mp3_path.is_file()
            and align_path.is_file()
            and key_path.is_file()
            and key_path.read_text(encoding="utf-8").strip() == key
        ):
            d = duration(mp3_path)
            if d is not None:
                existing = load(align_path)
                if isinstance(existing, Align):
                    clamped = clamp_align(existing, d)
                    if clamped != existing:
                        align_path.write_text(clamped.model_dump_json(), encoding="utf-8")
            out.append(LineAudio(line=entry.id, mp3=mp3_path, align=align_path, cached=True))
            continue
        audio, align = active.speak(entry.text, voice)
        mp3_path.write_bytes(audio)
        d = duration(mp3_path)
        if d is not None:
            align = clamp_align(align, d)
        align_path.write_text(align.model_copy(update={"line": entry.id}).model_dump_json(), encoding="utf-8")
        key_path.write_text(key, encoding="utf-8")
        out.append(LineAudio(line=entry.id, mp3=mp3_path, align=align_path, cached=False))
    return out


def synthesize_sounds(prod: Path, episode: str, client: ElevenLabs | None = None) -> list[SoundAudio]:
    """Generate every effect (kind sfx) of the episode; cached and provided files are skipped.

    Music is never generated. A file the production put there itself (mp3/wav without a key
    file) is reported as provided and never touched.
    """
    lines = load(prod / episode / "lines.yaml")
    if not isinstance(lines, Lines):
        raise ValueError(f"{prod / episode / 'lines.yaml'}: expected a lines file")
    effects = [entry for entry in (lines.sounds or []) if entry.kind == "sfx"]
    if not effects:
        return []
    audio_dir = prod / episode / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    active = client
    out: list[SoundAudio] = []
    for entry in effects:
        key = sound_cache_key(entry.description)
        mp3_path = audio_dir / f"{entry.id}.mp3"
        wav_path = audio_dir / f"{entry.id}.wav"
        key_path = audio_dir / f"{entry.id}.key"
        if key_path.is_file() and key_path.read_text(encoding="utf-8").strip() == key and mp3_path.is_file():
            out.append(SoundAudio(sound=entry.id, path=mp3_path, status="cached"))
            continue
        if not key_path.is_file() and (mp3_path.is_file() or wav_path.is_file()):
            provided = mp3_path if mp3_path.is_file() else wav_path
            out.append(SoundAudio(sound=entry.id, path=provided, status="provided"))
            continue
        if active is None:
            active = ElevenLabs()
        audio = active.sound(entry.description)
        mp3_path.write_bytes(audio)
        key_path.write_text(key, encoding="utf-8")
        out.append(SoundAudio(sound=entry.id, path=mp3_path, status="new"))
    return out
