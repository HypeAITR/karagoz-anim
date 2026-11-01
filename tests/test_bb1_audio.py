"""BB-1 bug hunt: episode audio (mix assembly in render_shot).

Each test fails on today's code and proves one user-facing bug. Needs ffmpeg on
PATH (same mark as the other render tests).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from karagoz.engine.shot import render_shot

FIX = Path(__file__).parent / "fixtures" / "spec"
NEEDS_FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def figure(colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 250, 550), fill=colour, outline=(0, 0, 0), width=6)
    return img


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    """Mini production that renders (same shape as the WP-45 fixture)."""
    p = tmp_path / "prod"
    ep = p / "ep"
    a = ep / "assets" / "sh01"
    for d in (ep / "shots" / "sh01", ep / "audio", a / "ali", a / "ayse", p / "referans"):
        d.mkdir(parents=True)
    style = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    style["live"].update(
        {
            "transition": {"inbetweens": {"value": 4, "source": "test"}, "anticipate": {"value": 4, "source": "test"}},
            "blink": {"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}},
        }
    )
    (p / "style.yaml").write_text(yaml.safe_dump(style, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    shutil.copy(FIX / "poses.yaml", ep / "shots" / "sh01" / "poses.yaml")
    shutil.copy(FIX / "xsheet.yaml", ep / "shots" / "sh01" / "xsheet.yaml")
    for lid, text in (("L01", "Yarın pazar, Ali."), ("L02", "Su ver bana.")):
        n = len(text)
        align = {
            "version": 1, "line": lid, "text": text, "characters": list(text),
            "starts": [round(i * 0.1, 6) for i in range(n)], "ends": [round((i + 1) * 0.1, 6) for i in range(n)],
        }  # fmt: skip
        (ep / "audio" / f"{lid}.align.json").write_text(json.dumps(align), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d={n * 0.1}", "-q:a", "4",
             str(ep / "audio" / f"{lid}.mp3")],
            check=True,
        )  # fmt: skip
    Image.new("RGB", (1920, 1080), (60, 60, 80)).save(a / "background.png")
    figure((200, 120, 80)).save(a / "ali" / "p1.png")
    figure((200, 120, 80)).save(a / "ali" / "p2.png")
    figure((90, 160, 90)).save(a / "ayse" / "p1.png")
    figure((200, 120, 80)).save(p / "referans" / "ali.png")
    figure((90, 160, 90)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: left, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }\n",
        encoding="utf-8",
    )
    return p


@NEEDS_FFMPEG
def test_bb1_missing_voice_mp3_warns_like_missing_sfx(prod: Path) -> None:
    """A missing voice mp3 must warn and render, like a missing SFX file does.

    render_shot checks SFX/music files and warns ("sound S01: no audio file in
    audio/") but hands voice clips to ffmpeg unchecked, so one deleted L02.mp3
    (align still there) crashes the whole shot with an ffmpeg tail including a
    pointer address ("[in#1 @ 0000...] Error opening input ... No such file").
    """
    (prod / "ep" / "audio" / "L02.mp3").unlink()
    rendered = render_shot(prod, "ep", "sh01")
    assert rendered.mp4.is_file()
    assert any("L02" in w for w in rendered.warnings), f"no L02 warning in {rendered.warnings}"
