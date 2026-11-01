"""WP-50 extra tests: render_shot offscreen + missing voice (item 5)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from karagoz.engine.shot import render_shot

FIX = Path(__file__).parent / "fixtures" / "spec"
NEEDS_FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _figure(colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((75, 30, 125, 180), fill=colour, outline=(0, 0, 0), width=4)
    return img


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    ep = p / "ep"
    a = ep / "assets" / "sh01"
    for d in (ep / "shots" / "sh01", ep / "audio", a / "ali", a / "ayse", p / "referans"):
        d.mkdir(parents=True)
    style = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    style["live"].update(
        {
            "transition": {
                "inbetweens": {"value": 4, "source": "test"},
                "anticipate": {"value": 4, "source": "test"},
                "settle": {"value": 6, "source": "test"},
            },
            "blink": {"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}},
            "hold": {"min": {"value": 6, "source": "test"}},
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
    _figure((200, 120, 80)).save(a / "ali" / "p1.png")
    _figure((200, 120, 80)).save(a / "ali" / "p2.png")
    _figure((90, 160, 90)).save(a / "ayse" / "p1.png")
    _figure((200, 120, 80)).save(p / "referans" / "ali.png")
    _figure((90, 160, 90)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }\n",
        encoding="utf-8",
    )
    return p


@NEEDS_FFMPEG
def test_offscreen_character_is_not_drawn(prod: Path) -> None:
    shots = yaml.safe_load((prod / "ep" / "shots.yaml").read_text(encoding="utf-8"))
    shots["shots"][0]["characters"][1]["offscreen"] = True
    (prod / "ep" / "shots.yaml").write_text(yaml.safe_dump(shots, allow_unicode=True), encoding="utf-8")
    xs = yaml.safe_load((prod / "ep" / "shots" / "sh01" / "xsheet.yaml").read_text(encoding="utf-8"))
    xs["tracks"]["ayse.body"] = [{"at": "shot.in", "pose": "p1"}]
    (prod / "ep" / "shots" / "sh01" / "xsheet.yaml").write_text(
        yaml.safe_dump(xs, allow_unicode=True), encoding="utf-8"
    )
    render_shot(prod, "ep", "sh01")
    px = np.array(Image.open(prod / "ep" / "render" / "sh01" / "frames" / "00050.png").convert("RGB")).astype(int)
    ayse = (np.abs(px - np.array([90, 160, 90])).sum(axis=2) < 30).any()
    assert not ayse


@NEEDS_FFMPEG
def test_missing_voice_mp3_warns_naming_the_line_and_still_renders(prod: Path) -> None:
    (prod / "ep" / "audio" / "L01.mp3").unlink()
    rendered = render_shot(prod, "ep", "sh01")
    assert rendered.mp4.is_file()
    assert any("L01" in w for w in rendered.warnings), rendered.warnings
