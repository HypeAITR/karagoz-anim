"""WP-45 (#21 facing mirror, #30 lint at render, end-to-end wiring through render_shot)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from karagoz.engine.shot import facing_mirror, render_shot, shot_lint_warnings
from karagoz.engine.xsheet import Exposure, Window
from karagoz.spec import Mouth, MouthCue, Style

FIX = Path(__file__).parent / "fixtures" / "spec"
NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)


def chars(natural_a: str = "left", mirror_ok: bool | None = True) -> dict[str, dict[str, Any]]:
    entry: dict[str, Any] = {"natural_facing": natural_a}
    if mirror_ok is not None:
        entry["mirror_ok"] = mirror_ok
    return {"a": entry, "b": {"natural_facing": "right", "mirror_ok": True}}


# --- #21 facing -------------------------------------------------------------------------------------------------------


def test_facing_mirrors_toward_target() -> None:
    """A left of B, A naturally facing left, facing B: A must look right, so mirror."""
    out, warnings = facing_mirror(
        frozenset(),
        ["a", "b"],
        {"a": (0, 0, 100, 100), "b": (200, 0, 300, 100)},
        {"a": "b"},
        {"a": None},
        chars(),
    )
    assert out == frozenset({"a"})
    assert warnings == []


def test_facing_keeps_a_consistent_view_mirror() -> None:
    """View-based mirror already looks at the target: facing changes nothing."""
    out, warnings = facing_mirror(
        frozenset({"ayse"}),
        ["ali", "ayse"],
        {"ali": (400, 0, 600, 200), "ayse": (1200, 0, 1400, 200)},
        {"ali": "ayse", "ayse": "ali"},
        {"ali": "three_quarter_right", "ayse": "three_quarter_left"},
        {
            "ali": {"natural_facing": "right", "mirror_ok": True},
            "ayse": {"natural_facing": "right", "mirror_ok": True},
        },
    )
    assert out == frozenset({"ayse"})
    assert warnings == []


def test_facing_wins_over_a_conflicting_view() -> None:
    """Staging (who stands where, who faces whom) is the authority; the view follows it."""
    out, warnings = facing_mirror(
        frozenset(),
        ["ali", "ayse"],
        {"ali": (1200, 0, 1400, 200), "ayse": (100, 0, 300, 200)},  # target on ali's left...
        {"ali": "ayse"},
        {"ali": "three_quarter_right"},  # ...but the view looks right: flip to face her
        {"ali": {"natural_facing": "right", "mirror_ok": True}},
    )
    assert out == frozenset({"ali"})
    assert warnings == []


def test_facing_mirror_ok_false_warns_once() -> None:
    out, warnings = facing_mirror(
        frozenset(),
        ["a", "b"],
        {"a": (0, 0, 100, 100), "b": (200, 0, 300, 100)},
        {"a": "b"},
        {"a": None},
        chars(mirror_ok=False),
    )
    assert out == frozenset()
    assert len(warnings) == 1
    assert "a" in warnings[0] and "b" in warnings[0] and "mirror_ok" in warnings[0]


def test_facing_ignores_unknown_self_and_tied_targets() -> None:
    boxes = {"a": (0, 0, 100, 100), "b": (200, 0, 300, 100)}
    for facing in ({"a": "nobody"}, {"a": "a"}, {"a": None}, {}):
        out, warnings = facing_mirror(frozenset(), ["a", "b"], boxes, facing, {"a": None}, chars())
        assert (out, warnings) == (frozenset(), [])
    tied = {"a": (0, 0, 100, 100), "b": (0, 0, 100, 100)}
    out, warnings = facing_mirror(frozenset(), ["a", "b"], tied, {"a": "b"}, {"a": None}, chars())
    assert (out, warnings) == (frozenset(), [])


# --- #30 lint at render -----------------------------------------------------------------------------------------------


def fixture_style() -> Style:
    return Style.model_validate(yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8")))


def test_shot_lint_reports_a_timing_issue() -> None:
    exp = Exposure(
        shot="sh01",
        length=10,
        body={"ali": ["p1"] * 10},
        eyes={"ali": ["open"] * 10},
        windows=[],
        unsupported=[("ali.body", 0)],
    )
    warnings = shot_lint_warnings(exp, fixture_style(), {}, {})
    assert any("timing.unsupported" in w for w in warnings)


def test_shot_style_lint_reports_too_many_moves() -> None:
    data = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    data["lint"] = {"pose_per_line_max": {"value": 1, "source": "test"}}
    style = Style.model_validate(data)
    exp = Exposure(
        shot="sh01",
        length=10,
        body={"ali": ["p1"] * 10},
        eyes={"ali": ["open"] * 10},
        windows=[
            Window("ali.body", 0, "transition", 0, 2, 3),
            Window("ali.body", 1, "transition", 4, 5, 6),
        ],
    )
    mouths = {
        "ali": Mouth(
            version=1, shot="sh01", character="ali", fps=24, cues=[MouthCue(start=0, end=10, shape="X", src="auto")]
        )
    }
    warnings = shot_lint_warnings(exp, style, {"ali": [("L01", 0, 10)]}, mouths)
    assert any("pose.too_many" in w for w in warnings)


# --- end-to-end wiring ------------------------------------------------------------------------------------------------


def figure(colour: tuple[int, int, int], marker: bool = False) -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150, 100, 250, 550), fill=colour, outline=(0, 0, 0), width=6)
    if marker:
        d.rectangle((250, 200, 300, 260), fill=(255, 0, 0), outline=(0, 0, 0), width=3)
    return img


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    """Mini production like test_shot's, but: ali has no view and faces left by nature (facing must mirror him),
    ayse's eyes carry an unsupported sequence event (lint must warn), review ran with human review off."""
    p = tmp_path / "prod"
    ep = p / "ep"
    a = ep / "assets" / "sh01"
    for d in (ep / "shots" / "sh01", ep / "audio", a / "ali", a / "ayse", p / "referans", ep / "run"):
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
    shots_text = (ep / "shots.yaml").read_text(encoding="utf-8")
    old = "{ name: ali, place: koltuk, facing: ayse, view: three_quarter_right }"
    assert old in shots_text
    new = shots_text.replace(old, "{ name: ali, place: koltuk, facing: ayse }")
    (ep / "shots.yaml").write_text(new, encoding="utf-8")
    shutil.copy(FIX / "poses.yaml", ep / "shots" / "sh01" / "poses.yaml")
    xs = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    xs["tracks"]["ayse.body"] = [{"at": "shot.in", "pose": "p1"}]
    xs["tracks"]["ayse.eyes"] = [{"at": "shot.in", "sequence": "s1"}]
    (ep / "shots" / "sh01" / "xsheet.yaml").write_text(yaml.safe_dump(xs, allow_unicode=True), encoding="utf-8")
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
    figure((200, 120, 80), marker=True).save(a / "ali" / "p1.png")
    figure((200, 120, 80), marker=True).save(a / "ali" / "p2.png")
    figure((90, 160, 90), marker=True).save(a / "ayse" / "p1.png")
    figure((200, 120, 80)).save(p / "referans" / "ali.png")
    figure((90, 160, 90)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: left, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }\n",
        encoding="utf-8",
    )
    (ep / "run" / "events.jsonl").write_text(
        '{"task": "ep/render", "kind": "note", "text": "human review off: auto-approved"}\n', encoding="utf-8"
    )
    return p


@NEEDS_FFMPEG
def test_render_shot_wires_transitions_facing_lint_and_review_off(prod: Path) -> None:
    r = render_shot(prod, "ep", "sh01")
    assert any("human review: off" in w for w in r.warnings)  # #11 report line
    assert any("timing.unsupported" in w for w in r.warnings)  # #30 lint warning
    frames = prod / "ep" / "render" / "sh01" / "frames"
    assert all(float(np.array(Image.open(frames / f"{i:05d}.png")).mean()) == 0.0 for i in range(12))  # #18 from_black
    assert float(np.array(Image.open(frames / "00012.png")).mean()) > 0.0
    px = np.array(Image.open(frames / "00020.png").convert("RGB")).astype(int)  # #21 ali mirrored toward ayse
    red = (px[:, :, 0] > 200) & (px[:, :, 1] < 60) & (px[:, :, 2] < 60)
    body = np.abs(px - np.array([200, 120, 80])).sum(axis=2) < 30
    left = np.nonzero(red[:, :960])
    assert left[0].size > 0 and left[1].mean() < np.nonzero(body)[1].mean()  # marker on his left: mirrored
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", str(r.mp4)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc.stdout)["format"]["tags"]["comment"] == "human review: off"  # #11 metadata
