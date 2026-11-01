"""Meta MMS-TTS voice adapter: MP3 audio plus per-letter timings from a local VITS model.

Runtime (hackathon) TTS: ``facebook/mms-tts-eng`` (VITS, CPU, 16 kHz mono, CC-BY-NC 4.0) through Hugging Face
``transformers`` (``VitsModel``). Same contract as ``ElevenLabs``: ``speak`` returns MP3 bytes plus an ``Align``
with one timing per character of the original text.

Per-letter timings come from the VITS duration predictor of the same forward pass that makes the audio.
``transformers`` does not return the predicted durations, so they are captured with a forward hook on
``model.duration_predictor`` (log-durations per input token) and converted with the exact formula the forward
pass uses (``ceil(exp(log_d) * mask / speaking_rate)``). Token positions map back to the original characters
through the tokenizer's own vocabulary: dropped punctuation gets zero-length timings, spaces keep their voiced
gap. Interleaved blank positions (id 0, when ``add_blank`` is on) carry transition frames and are split between
the neighbouring letters. The capture is validated against the waveform (frames times hop length must equal the
sample count within one frame); anything unexpected falls back to uniform per-character timings with a logged
warning, never silence and never a crash.

Determinism: a fixed seed (``voice.seed`` or 0) is set before every forward pass, the model stays in eval mode,
and ``noise_scale`` / ``noise_scale_duration`` are pinned to 0.667 / 0.8 (the checkpoint's own values, the VITS
reference defaults), so the same input gives the same bytes. The model loads lazily and is kept once per process
per model name.

``torch`` and ``transformers`` are an optional extra (``uv sync --extra mms``) and are imported only inside the
functions that need them, so the base install never requires them.
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from karagoz.engine.render import _ffmpeg_bin
from karagoz.spec import Align

if TYPE_CHECKING:
    from karagoz.voice.elevenlabs import VoiceConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/mms-tts-eng"
FIXED_SEED = 0
NOISE_SCALE = 0.667
NOISE_SCALE_DURATION = 0.8
MP3_BITRATE = "128k"
MP3_RATE = 44100  # match ElevenLabs' mp3_44100_128 so the rest of the pipeline is unchanged


@dataclass(frozen=True)
class TokenSpan:
    """One input token's duration: ``char`` indexes the original text, ``None`` is an interleaved blank."""

    char: int | None
    frames: int


@dataclass(frozen=True)
class Synthesis:
    """One backend forward pass: mono waveform plus per-token durations in spectrogram frames."""

    samples: np.ndarray
    sample_rate: int
    hop_length: int
    spans: tuple[TokenSpan, ...]


class Backend(Protocol):
    """Something that turns text into a waveform with per-token durations (torch VITS, or a test fake)."""

    def synthesize(self, text: str, *, model: str, speed: float, seed: int) -> Synthesis: ...


def align_from_spans(
    text: str,
    spans: tuple[TokenSpan, ...] | list[TokenSpan],
    *,
    hop_length: int,
    sample_rate: int,
    audio_seconds: float,
    line: str = "L00",
) -> Align:
    """Per-character timings for ``text`` from token spans; blanks split between neighbouring letters.

    Characters with no span (punctuation the tokenizer dropped) get zero-length timings at the cursor, in
    order; every start and end is capped at ``audio_seconds``.
    """
    if hop_length <= 0 or sample_rate <= 0:
        raise ValueError(f"hop_length and sample_rate must be positive, got {hop_length}/{sample_rate}")
    if audio_seconds < 0:
        raise ValueError(f"audio_seconds must be >= 0, got {audio_seconds}")
    frames = [0] * len(text)
    for pos, span in enumerate(spans):
        if span.frames < 0:
            raise ValueError(f"token span {pos} has negative frames: {span.frames}")
        if span.char is None:
            continue
        if not 0 <= span.char < len(text):
            raise ValueError(f"token span {pos} points at character {span.char} of a {len(text)}-char text")
        frames[span.char] += span.frames
    prev_of: dict[int, int | None] = {}
    last: int | None = None
    for pos, span in enumerate(spans):
        prev_of[pos] = last
        if span.char is not None:
            last = pos
    next_of: dict[int, int | None] = {}
    nxt: int | None = None
    for pos in range(len(spans) - 1, -1, -1):
        next_of[pos] = nxt
        if spans[pos].char is not None:
            nxt = pos
    for pos, span in enumerate(spans):
        if span.char is not None or span.frames == 0:
            continue
        prev_pos = prev_of[pos]
        next_pos = next_of[pos]
        before = spans[prev_pos].char if prev_pos is not None else None
        after = spans[next_pos].char if next_pos is not None else None
        if before is None:
            if after is None:
                continue  # no letters at all; the audio runs but nothing is shown
            frames[after] += span.frames
        elif after is None:
            frames[before] += span.frames
        else:
            half = span.frames // 2
            frames[before] += half
            frames[after] += span.frames - half
    sec_per_frame = hop_length / sample_rate
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0
    for count in frames:
        starts.append(min(cursor, audio_seconds))
        cursor += count * sec_per_frame
        ends.append(min(cursor, audio_seconds))
    return Align(version=1, line=line, text=text, characters=list(text), starts=starts, ends=ends)


def uniform_spans(text: str, total_frames: int) -> tuple[TokenSpan, ...]:
    """Fallback spans: ``total_frames`` spread evenly over every character (spaces included)."""
    if total_frames < 0:
        raise ValueError(f"total_frames must be >= 0, got {total_frames}")
    if not text or total_frames == 0:
        return ()
    base, extra = divmod(total_frames, len(text))
    return tuple(TokenSpan(i, base + (1 if i < extra else 0)) for i in range(len(text)))


def _atempo_chain(value: float) -> list[str]:
    """``atempo`` only takes 0.5..2.0, so chain filters for larger tempo changes."""
    chain: list[str] = []
    while value > 2.0:
        chain.append("atempo=2.0")
        value /= 2.0
    while value < 0.5:
        chain.append("atempo=0.5")
        value /= 0.5
    chain.append(f"atempo={value!r}")
    return chain


def _pitch_filters(sample_rate: int, pitch: float) -> list[str]:
    """Pitch shift by ``pitch`` semitones with the duration kept: faster playback, resample, stretch back."""
    if pitch == 0.0:
        return []
    ratio = 2.0 ** (pitch / 12.0)
    return [f"asetrate={sample_rate}*{ratio!r}", f"aresample={sample_rate}", *_atempo_chain(1.0 / ratio)]


def _encode_mp3(samples: np.ndarray, sample_rate: int, pitch: float) -> bytes:
    """Encode mono float samples as MP3 bytes (44100 Hz, 128k) through ffmpeg, like the ElevenLabs files."""
    mono = np.ascontiguousarray(samples, dtype="<f4")
    if mono.ndim != 1 or mono.size == 0:
        raise ValueError(f"cannot encode a waveform with shape {mono.shape}")
    cmd = [_ffmpeg_bin(), "-v", "error", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0"]
    filters = _pitch_filters(sample_rate, pitch)
    if filters:
        cmd += ["-filter:a", ",".join(filters)]
    cmd += ["-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, "-ar", str(MP3_RATE), "-f", "mp3", "pipe:1"]
    proc = subprocess.run(cmd, input=mono.tobytes(), capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"MMS MP3 encode failed: {detail}")
    if not proc.stdout:
        raise RuntimeError("MMS MP3 encode produced no audio")
    return proc.stdout


_LOADED: dict[str, tuple[Any, Any]] = {}
_LOAD_LOCK = threading.Lock()
_SYNTH_LOCK = threading.Lock()


def _load_model(name: str) -> tuple[Any, Any]:
    """The (model, tokenizer) pair for ``name``, loaded once per process; needs the ``mms`` extra."""
    with _LOAD_LOCK:
        if name not in _LOADED:
            try:
                from transformers import VitsModel, VitsTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    f"MMS voice {name!r} needs the 'mms' extra (torch + transformers): uv sync --extra mms"
                ) from exc
            tokenizer: Any = VitsTokenizer.from_pretrained(name)
            model: Any = VitsModel.from_pretrained(name)
            model.eval()
            model.noise_scale = NOISE_SCALE
            model.noise_scale_duration = NOISE_SCALE_DURATION
            _LOADED[name] = (model, tokenizer)
        return _LOADED[name]


def _kept_characters(text: str, tokenizer: Any, filtered: str) -> list[int] | None:
    """Original indices that survive the tokenizer's normalization, or None on any surprise."""
    try:
        encoder = tokenizer.encoder
        kept_chars = [tokenizer.normalize_text(ch) for ch in text]
    except (AttributeError, TypeError):
        return None
    kept = [i for i, ch in enumerate(kept_chars) if ch in encoder]
    while kept and kept_chars[kept[0]].strip() == "":
        kept.pop(0)
    while kept and kept_chars[kept[-1]].strip() == "":
        kept.pop()
    if "".join(kept_chars[i] for i in kept) != filtered:
        return None
    return kept


class _TorchBackend:
    """The real backend: one VITS forward pass per line, durations from the same pass via a hook."""

    def synthesize(self, text: str, *, model: str, speed: float, seed: int) -> Synthesis:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"MMS voice {model!r} needs the 'mms' extra (torch + transformers): uv sync --extra mms"
            ) from exc
        vits, tokenizer = _load_model(model)
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        captured: dict[str, Any] = {}

        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured.setdefault("log_duration", output.detach().cpu())

        with _SYNTH_LOCK:
            torch.manual_seed(seed)
            handle = vits.duration_predictor.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    out = vits(input_ids=input_ids, attention_mask=attention_mask, speaking_rate=speed)
            finally:
                handle.remove()
        samples = np.asarray(out.waveform[0].cpu().numpy(), dtype=np.float32).reshape(-1)
        sample_rate = int(vits.config.sampling_rate)
        hop_length = int(math.prod(int(r) for r in vits.config.upsample_rates))
        spans = self._spans_from_hook(text, tokenizer, input_ids, attention_mask, captured.get("log_duration"), speed)
        if spans is None:
            logger.warning("mms: duration capture failed for %r; using uniform timings", text[:40])
            spans = uniform_spans(text, int(round(samples.shape[0] / hop_length)))
        elif abs(sum(s.frames for s in spans) * hop_length - samples.shape[0]) > hop_length:
            logger.warning("mms: durations disagree with the audio for %r; using uniform timings", text[:40])
            spans = uniform_spans(text, int(round(samples.shape[0] / hop_length)))
        return Synthesis(samples=samples, sample_rate=sample_rate, hop_length=hop_length, spans=spans)

    @staticmethod
    def _spans_from_hook(
        text: str,
        tokenizer: Any,
        input_ids: Any,
        attention_mask: Any,
        log_duration: Any,
        speed: float,
    ) -> tuple[TokenSpan, ...] | None:
        """Token spans from the predictor's log-durations, or None when anything looks unexpected."""
        if log_duration is None or bool(tokenizer.phonemize):
            return None
        try:
            token_count = int(input_ids.shape[1])
            logs: list[float] = [float(v) for v in log_duration[0, 0].tolist()]
            mask: list[float] = (
                [float(v) for v in attention_mask[0].tolist()] if attention_mask is not None else [1.0] * token_count
            )
            filtered, _ = tokenizer.prepare_for_tokenization(text)
        except (ValueError, TypeError, IndexError, AttributeError):
            return None
        if len(logs) != token_count or len(mask) != token_count:
            return None
        voiced = list(range(1, token_count, 2)) if bool(tokenizer.add_blank) else list(range(token_count))
        if len(voiced) != len(filtered):
            return None
        kept = _kept_characters(text, tokenizer, filtered)
        if kept is None:
            return None
        scale = 1.0 / speed
        durations = [int(math.ceil(math.exp(lg) * m * scale)) for lg, m in zip(logs, mask, strict=True)]
        char_of = dict(zip(voiced, kept, strict=True))
        return tuple(TokenSpan(char_of.get(i), durations[i]) for i in range(token_count))


class MmsTts:
    """MMS text-to-speech: ``speak`` returns (MP3 bytes, per-letter alignment), like ``ElevenLabs``."""

    provider = "mms"

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend if backend is not None else _TorchBackend()

    def speak(self, text: str, voice: VoiceConfig) -> tuple[bytes, Align]:
        """Synthesize one line; the alignment has one timing per character of ``text`` (line unset)."""
        speed = voice.speed if voice.speed is not None else 1.0
        if not speed > 0.0:
            raise ValueError(f"mms speed (speaking rate) must be > 0, got {voice.speed}")
        seed = voice.seed if voice.seed is not None else FIXED_SEED
        synth = self._backend.synthesize(text, model=voice.model, speed=speed, seed=seed)
        samples = np.asarray(synth.samples)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError(f"backend returned a waveform with shape {samples.shape}")
        if synth.sample_rate <= 0 or synth.hop_length <= 0:
            raise ValueError(f"backend returned invalid rate {synth.sample_rate} / hop {synth.hop_length}")
        align = align_from_spans(
            text,
            synth.spans,
            hop_length=synth.hop_length,
            sample_rate=synth.sample_rate,
            audio_seconds=samples.shape[0] / synth.sample_rate,
        )
        pitch = voice.pitch if voice.pitch is not None else 0.0
        return _encode_mp3(samples, synth.sample_rate, pitch), align
