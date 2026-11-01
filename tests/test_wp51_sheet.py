"""WP-51 extra tests: sheet movement_parts and frame_strip error paths."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image
from test_draw import EP, prod  # noqa: F401 (prod is a pytest fixture)

from karagoz.engine.render import _ffmpeg_bin
from karagoz.engine.sheet import frame_strip, movement_parts


def test_movement_parts_unreadable_poses_raises_value_error(prod: Path) -> None:  # noqa: F811
    (prod / EP / "shots" / "sh01" / "poses.yaml").write_text(
        "version: 1\nshot: sh01\ncharacters: {{{", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        movement_parts(prod, EP, "sh01")


def test_movement_parts_missing_poses_returns_empty(prod: Path) -> None:  # noqa: F811
    (prod / EP / "shots" / "sh01" / "poses.yaml").unlink()
    assert movement_parts(prod, EP, "sh01") == []


def _video(tmp_path: Path, n: int = 4) -> Path:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(n):
        Image.new("RGB", (320, 180), (30 * i, 0, 0)).save(frames / f"{i:03d}.png")
    out = tmp_path / "v.mp4"
    subprocess.run(
        [
            _ffmpeg_bin(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "24",
            "-i",
            str(frames / "%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "0",
            str(out),
        ],
        check=True,
    )
    return out


def test_frame_strip_inverted_crop_raises_value_error(tmp_path: Path) -> None:
    v = _video(tmp_path)
    with pytest.raises(ValueError):
        frame_strip(v, 0, 2, tmp_path / "strip.png", cell_height=90, crop=(200, 10, 50, 100))


def test_frame_strip_valid_crop_still_works(tmp_path: Path) -> None:
    v = _video(tmp_path)
    out = frame_strip(v, 0, 2, tmp_path / "strip.png", cell_height=90, crop=(10, 10, 100, 100))
    assert out.is_file()
