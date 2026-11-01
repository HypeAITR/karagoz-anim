"""Acceptance tests for WP-09b (render a shot / episode). Written by the maintainer; do not edit. Needs ffmpeg.

Mini production from the spec fixtures: shot sh01 = [0, 100) frames; L01 (ayse) at 8, L02 (ali) at 60.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from karagoz import cli
from karagoz.engine.compose import ShotAssets, render_frames
from karagoz.engine.shot import STANDING_HEIGHT, character_scale, render_episode, render_shot
from karagoz.engine.xsheet import Exposure

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

FIX = Path(__file__).parent / "fixtures" / "spec"
EP = "ep"
EXTRA = {
    "transition": {"inbetweens": {"value": 4, "source": "test"}, "anticipate": {"value": 4, "source": "test"}},
    "blink": {"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}},
}


def figure(colour: tuple[int, int, int], marker: bool = False) -> Image.Image:
    """400x600 white canvas, figure 150..250 x 100..550; optional red marker on its right side (x 260..300)."""
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 250, 550), fill=colour, outline=(0, 0, 0), width=6)
    if marker:
        d.rectangle((250, 200, 300, 260), fill=(255, 0, 0), outline=(0, 0, 0), width=3)
    return img


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    ep = p / EP
    a = ep / "assets" / "sh01"
    for d in (ep / "shots" / "sh01", ep / "audio", a / "ali", a / "ayse", p / "referans"):
        d.mkdir(parents=True)
    style = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    style["live"].update(EXTRA)
    (p / "style.yaml").write_text(yaml.safe_dump(style, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    shutil.copy(FIX / "poses.yaml", ep / "shots" / "sh01" / "poses.yaml")
    xs = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    xs["tracks"]["ayse.body"] = [{"at": "shot.in", "pose": "p1"}]
    (ep / "shots" / "sh01" / "xsheet.yaml").write_text(yaml.safe_dump(xs, allow_unicode=True), encoding="utf-8")
    for lid, text in (("L01", "Yarın pazar, Ali."), ("L02", "Su ver bana.")):
        n = len(text)
        starts = [round(i * 0.1, 6) for i in range(n)]
        ends = [round((i + 1) * 0.1, 6) for i in range(n)]
        align = {"version": 1, "line": lid, "text": text, "characters": list(text), "starts": starts, "ends": ends}
        (ep / "audio" / f"{lid}.align.json").write_text(json.dumps(align), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d={n * 0.1}", "-q:a", "4",
             str(ep / "audio" / f"{lid}.mp3")],
            check=True,
        )  # fmt: skip
    Image.new("RGB", (1920, 1080), (60, 60, 80)).save(a / "background.png")
    figure((200, 120, 80)).save(a / "ali" / "p1.png")
    figure((200, 120, 80)).save(a / "ali" / "p2.png")
    figure((90, 160, 90), marker=True).save(a / "ayse" / "p1.png")
    figure((200, 120, 80)).save(p / "referans" / "ali.png")
    figure((90, 160, 90)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }\n",
        encoding="utf-8",
    )
    return p


def test_character_scale() -> None:
    ref = figure((1, 2, 3))  # opaque 100..550 -> 451 px tall on a 400x600 canvas
    assert character_scale(ref, (400, 600)) == pytest.approx(STANDING_HEIGHT * 1080 / 451, rel=0.01)
    assert character_scale(ref.resize((800, 1200)), (400, 600)) == pytest.approx(STANDING_HEIGHT * 1080 / 451, rel=0.02)


def test_render_shot(prod: Path) -> None:
    r = render_shot(prod, EP, "sh01")
    assert r.mp4 == prod / EP / "render" / "sh01.mp4" and r.mp4.exists()
    assert r.frames == 100 and r.seconds == pytest.approx(100 / 24, abs=0.05)
    assert r.lufs is not None and -40 < r.lufs < 0
    frames = sorted((prod / EP / "render" / "sh01" / "frames").iterdir())
    assert len(frames) == 100
    assert any("p1-p2" in w for w in r.warnings)  # inbetweens were not drawn: fallback warnings


def test_mirrored_character(prod: Path) -> None:
    render_shot(prod, EP, "sh01")
    # Frame 50: past the 12-frame from_black fade-in (WP-45); ayse still holds p1.
    px = np.array(Image.open(prod / EP / "render" / "sh01" / "frames" / "00050.png").convert("RGB")).astype(int)
    red = (px[:, :, 0] > 200) & (px[:, :, 1] < 60) & (px[:, :, 2] < 60)
    green = (px[:, :, 0] < 120) & (px[:, :, 1] > 140) & (px[:, :, 2] < 120)
    assert red.any() and green.any()
    # ayse faces left in the shot (three_quarter_left) but is drawn facing right -> flipped: marker on her left
    assert np.nonzero(red)[1].mean() < np.nonzero(green)[1].mean()


def test_same_scale_for_all_poses(prod: Path) -> None:
    """ali's poses share one scale from his reference: p1 and p2 cover the same height on screen."""
    render_shot(prod, EP, "sh01")
    d = prod / EP / "render" / "sh01" / "frames"

    def height(name: str) -> int:
        px = np.array(Image.open(d / name).convert("RGB")).astype(int)
        mask = np.abs(px - np.array([200, 120, 80])).sum(axis=2) < 30
        ys = np.nonzero(mask)[0]
        return int(ys.max() - ys.min())

    # 00012 is the first full frame past the 12-frame from_black fade-in (WP-45); ali still holds p1.
    assert abs(height("00012.png") - height("00099.png")) <= 2
    assert height("00012.png") == pytest.approx(STANDING_HEIGHT * 1080 * 439 / 451, abs=12)  # fill inside outline


def test_mirror_not_allowed_warns(prod: Path) -> None:
    c = prod / "characters.yaml"
    allowed = "ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }"
    c.write_text(c.read_text(encoding="utf-8").replace(allowed, allowed.replace("true", "false")), "utf-8")
    r = render_shot(prod, EP, "sh01")
    assert any("ayse" in w and "mirror" in w for w in r.warnings)


def test_render_frames_mirror_and_scale(tmp_path: Path) -> None:
    root = tmp_path / "a"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (0, 0, 0)).save(root / "background.png")
    figure((200, 120, 80), marker=True).save(root / "ali" / "p1.png")
    exp = Exposure(shot="sh01", length=1, body={"ali": ["p1"]}, eyes={"ali": ["open"]}, windows=[])
    render_frames(exp, {}, ShotAssets(root), ["ali"], tmp_path / "f", mirror=frozenset({"ali"}), scales={"ali": 1.0})
    px = np.array(Image.open(tmp_path / "f" / "00000.png").convert("RGB")).astype(int)
    red = (px[:, :, 0] > 200) & (px[:, :, 1] < 60)
    body = np.abs(px - np.array([200, 120, 80])).sum(axis=2) < 30
    assert np.nonzero(red)[1].mean() < np.nonzero(body)[1].mean()
    assert np.nonzero(body)[0].max() - np.nonzero(body)[0].min() == pytest.approx(439, abs=3)  # scale 1.0


def test_render_episode_and_cli(prod: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = render_episode(prod, EP)
    assert out == prod / EP / "render" / f"{EP}.mp4" and out.exists()
    assert cli.main(["render", str(prod), EP]) == 0
    text = capsys.readouterr().out
    assert "sh01 100 frames" in text and "LUFS" in text and f"{EP}.mp4" in text
