"""WP-45 (#18 transitions, #11 review mark): shot edges fade to/from black, audio follows, review-off is marked."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from karagoz.engine.render import encode, fade_edges
from karagoz.engine.shot import (
    REVIEW_OFF_COMMENT,
    _join_shot_mp4s,
    apply_frame_transitions,
    review_off,
    transition_frames,
)
from karagoz.spec import Style

FIX = Path(__file__).parent / "fixtures" / "spec"
NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)
COLOUR = (200, 100, 50)


def solid_frames(d: Path, n: int, size: tuple[int, int] = (32, 24)) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, COLOUR).save(d / f"{i:05d}.png")
    return d


def means(d: Path, n: int) -> list[float]:
    return [float(np.array(Image.open(d / f"{i:05d}.png").convert("RGB")).mean()) for i in range(n)]


def style_with_beats(extra_beats: dict[str, float]) -> Style:
    data = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    for name, value in extra_beats.items():
        data["live"]["beats"][name] = {"value": value, "source": "test"}
    return Style.model_validate(data)


# --- #18 picture transitions ------------------------------------------------------------------------------------------


def test_fade_to_black_last_frames_monotonic_and_last_black(tmp_path: Path) -> None:
    d = solid_frames(tmp_path / "f", 8)
    apply_frame_transitions(d, 8, "cut", "fade_to_black", 3)
    got = means(d, 8)
    base = sum(COLOUR) / 3
    assert got[:5] == pytest.approx([base] * 5)  # untouched head
    assert got[5] > got[6] > got[7] == 0.0  # monotonic to black
    assert np.array(Image.open(d / "00007.png")).max() == 0  # last frame exactly black


def test_from_black_first_frames_black(tmp_path: Path) -> None:
    d = solid_frames(tmp_path / "f", 8)
    apply_frame_transitions(d, 8, "from_black", "cut", 3)
    for i in range(3):
        assert np.array(Image.open(d / f"{i:05d}.png")).max() == 0
    assert means(d, 8)[3:] == pytest.approx([sum(COLOUR) / 3] * 5)


def test_to_black_last_frames_black(tmp_path: Path) -> None:
    d = solid_frames(tmp_path / "f", 8)
    apply_frame_transitions(d, 8, None, "to_black", 3)
    for i in (5, 6, 7):
        assert np.array(Image.open(d / f"{i:05d}.png")).max() == 0
    assert means(d, 8)[:5] == pytest.approx([sum(COLOUR) / 3] * 5)


def test_fade_from_black_ramps_up_from_black(tmp_path: Path) -> None:
    d = solid_frames(tmp_path / "f", 8)
    apply_frame_transitions(d, 8, "fade_from_black", None, 4)
    got = means(d, 8)
    assert np.array(Image.open(d / "00000.png")).max() == 0  # first frame exactly black
    assert got[0] < got[1] < got[2] < got[3]
    assert got[4:] == pytest.approx([sum(COLOUR) / 3] * 4)


def test_cut_leaves_frames_unchanged(tmp_path: Path) -> None:
    d = solid_frames(tmp_path / "f", 6)
    before = [np.array(Image.open(d / f"{i:05d}.png")).tobytes() for i in range(6)]
    apply_frame_transitions(d, 6, "cut", None, 3)
    after = [np.array(Image.open(d / f"{i:05d}.png")).tobytes() for i in range(6)]
    assert before == after


def test_two_shot_boundary_fades_to_and_from_black(tmp_path: Path) -> None:
    """A synthetic 2-shot episode boundary: A fades out, B fades in; middles stay intact."""
    a = solid_frames(tmp_path / "a", 6)
    b = solid_frames(tmp_path / "b", 6)
    apply_frame_transitions(a, 6, "cut", "fade_to_black", 3)
    apply_frame_transitions(b, 6, "fade_from_black", "cut", 3)
    ma, mb = means(a, 6), means(b, 6)
    assert ma[3] > ma[4] > ma[5] == 0.0
    assert 0.0 == mb[0] < mb[1] < mb[2]
    assert ma[:3] == pytest.approx(mb[3:])


def test_black_hold_beat_sets_fade_length() -> None:
    assert transition_frames(style_with_beats({"black_hold": 7})) == 7
    assert transition_frames(style_with_beats({})) == 12  # default while productions add the beat


# --- #18 audio follows the picture ------------------------------------------------------------------------------------


def const_wav(path: Path, frames: int, value: int = 10000) -> Path:
    n = frames * 48000 // 24
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.full((n, 2), value, dtype="<i2").tobytes())
    return path


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)


def test_audio_fades_with_the_picture(tmp_path: Path) -> None:
    wav = const_wav(tmp_path / "m.wav", 6)
    fade_edges(wav, 2, 2, hard_out=True)  # 2-frame linear fade in, 2-frame hard silence out
    data = read_wav(wav)
    spf = 48000 // 24
    assert data[0].tolist() == [0, 0]
    assert data[spf].tolist() == [5000, 5000]  # half-way through the linear ramp
    ramp = data[: 2 * spf, 0].astype(int)
    assert bool(np.all(ramp[1:] >= ramp[:-1]))  # monotonic up
    assert data[2 * spf : 4 * spf].min() == 10000 and data[2 * spf : 4 * spf].max() == 10000  # middle untouched
    assert data[4 * spf :].max() == 0  # hard black edge is silent


def test_audio_fade_out_touches_silence(tmp_path: Path) -> None:
    wav = const_wav(tmp_path / "m.wav", 4)
    fade_edges(wav, 0, 2)
    data = read_wav(wav)
    assert data[-1].tolist() == [0, 0]
    tail = data[2 * (48000 // 24) :, 0].astype(int)
    assert bool(np.all(tail[1:] <= tail[:-1]))  # monotonic down


# --- #11 review off ---------------------------------------------------------------------------------------------------


def test_review_off_detected_from_events(tmp_path: Path) -> None:
    run = tmp_path / "ep" / "run"
    run.mkdir(parents=True)
    assert review_off(tmp_path / "ep") is False  # no event log: unknown means off is not claimed
    (run / "events.jsonl").write_text('{"kind": "state", "text": "done"}\n', encoding="utf-8")
    assert review_off(tmp_path / "ep") is False
    with (run / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"task": "ep/render", "kind": "note", "text": "human review off: auto-approved"}\n')
    assert review_off(tmp_path / "ep") is True


@NEEDS_FFMPEG
def test_encode_writes_comment_metadata(tmp_path: Path) -> None:
    mp4 = encode(solid_frames(tmp_path / "f", 6, (64, 36)), None, tmp_path / "out.mp4", comment=REVIEW_OFF_COMMENT)
    assert ffprobe_comment(mp4) == "human review: off"


def ffprobe_comment(mp4: Path) -> str | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", str(mp4)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)["format"]["tags"].get("comment")


def two_shots(ep: Path) -> list[Path]:
    out = []
    for shot in ("sh01", "sh02"):
        mp4 = encode(solid_frames(ep / "frames" / shot, 4, (64, 36)), None, ep / "render" / f"{shot}.mp4")
        out.append(mp4)
    return out


@NEEDS_FFMPEG
def test_join_marks_review_off_from_the_episode_log(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    (ep / "run").mkdir(parents=True)
    (ep / "run" / "events.jsonl").write_text("human review off: auto-approved\n", encoding="utf-8")
    joined = _join_shot_mp4s(two_shots(ep), ep / "render" / "ep.mp4")
    assert ffprobe_comment(joined) == "human review: off"


@NEEDS_FFMPEG
def test_join_leaves_no_comment_when_reviewed(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    (ep / "run").mkdir(parents=True)
    (ep / "run" / "events.jsonl").write_text('{"kind": "state"}\n', encoding="utf-8")
    joined = _join_shot_mp4s(two_shots(ep), ep / "render" / "ep.mp4")
    assert ffprobe_comment(joined) is None
