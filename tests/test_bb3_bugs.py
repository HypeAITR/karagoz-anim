"""BB-3 bug hunt: engine picture path. Each test fails on today's code.

Bug list (see _wp/RAPOR-BB-3.md):
 1. register.refine never aligns sequence drawings (sN.k).
 2. compose._borrowed_patch never lends key patches to sequence drawings.
 3. xsheet loop `step: beat.<unknown>` silently falls back to 8 frames.
 4. xsheet drops wrong-track items silently (blink on body, hold on eyes).
 5. xsheet loop snaps when the inbetweens exactly fill the step (ib*c == step).
 6. camera.target_bbox treats any target suffix as `.face`.
 7. shot.render_shot ignores `offscreen: true` and renders the character.
"""

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

from karagoz.engine.camera import target_bbox
from karagoz.engine.compose import Placement, ShotAssets, frame_layers
from karagoz.engine.patch import Patch, save_patch
from karagoz.engine.placement import CharacterInput, DrawingPlacement, PoseInfo
from karagoz.engine.register import refine
from karagoz.engine.timing import EpisodeTiming, TimingError, resolve_episode
from karagoz.engine.xsheet import Exposure, expand, lint
from karagoz.spec import Align, Lines, Mouth, Poses, Regions, Shots, Style, Xsheet

FIX = Path(__file__).parent / "fixtures" / "spec"

STYLE_EXTRA = {
    "transition": {
        "inbetweens": {"value": 4, "source": "test"},
        "anticipate": {"value": 4, "source": "test"},
        "settle": {"value": 6, "source": "test"},
    },
    "blink": {"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}},
    "hold": {"min": {"value": 6, "source": "test"}},
}

ALIGNS = {"L01": "Yarın pazar, Ali.", "L02": "Su ver bana."}


def _style() -> Style:
    d = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    d["live"].update(STYLE_EXTRA)
    return Style.model_validate(d)


def _align(line_id: str, text: str) -> Align:
    n = len(text)
    return Align.model_validate(
        {
            "version": 1,
            "line": line_id,
            "text": text,
            "characters": list(text),
            "starts": [round(i * 0.1, 6) for i in range(n)],
            "ends": [round((i + 1) * 0.1, 6) for i in range(n)],
        }
    )


def _timing(style: Style) -> EpisodeTiming:
    lines = Lines.model_validate(yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8")))
    shots = Shots.model_validate(yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8")))
    return resolve_episode(lines, shots, {k: _align(k, v) for k, v in ALIGNS.items()}, style)


def _poses() -> Poses:
    return Poses.model_validate(yaml.safe_load((FIX / "poses.yaml").read_text(encoding="utf-8")))


def _xsheet(body: list[dict[str, Any]], eyes: list[dict[str, Any]] | None = None) -> Xsheet:
    d = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    d["tracks"]["ali.body"] = body
    if eyes is not None:
        d["tracks"]["ali.eyes"] = eyes
    return Xsheet.model_validate(d)


HOLD_P1 = {"at": "shot.in", "pose": "p1"}


# --- bug 1: register.refine ignores sequence drawings ----------------------------------------------


def _r_fig(lower: tuple[int, int] = (0, 0)) -> Image.Image:
    a = np.zeros((300, 200, 4), np.uint8)
    a[20:180, 60:140] = (200, 120, 80, 255)
    dx, dy = lower
    a[180 + dy : 290 + dy, 50 + dx : 150 + dx] = (40, 40, 120, 255)
    return Image.fromarray(a)


def _r_lm() -> dict[str, Any]:
    return {"size": [200, 300], "hip": [100, 180], "foot_low": [100, 289], "feet": [], "hands": []}


def _r_regions() -> Regions:
    return Regions.model_validate(
        {
            "version": 1,
            "location": "salon",
            "image": "background.png",
            "size": [1920, 1080],
            "scale": None,
            "objects": [
                {
                    "name": "sofa",
                    "kind": "sofa",
                    "box": [1, 1, 10, 10],
                    "floor_y": 10,
                    "confidence": 1.0,
                    "mask": "m.png",
                    "front": False,
                }
            ],
            "places": [{"name": "sofa_left", "object": "sofa", "kind": "floor", "floor": [100, 900]}],
        }
    )


def _r_refine(names: list[str], images: dict[str, Image.Image]) -> dict[str, Any]:
    poses = {"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit")}
    char = CharacterInput("ali", 120.0, False, "sofa_left", poses, {n: _r_lm() for n in names}, None)
    if "s1.1" in names:
        char = CharacterInput(
            "ali", 120.0, False, "sofa_left", poses, {n: _r_lm() for n in names}, None, {"s1": ("p1", "p2")}
        )
    placed = {
        "ali": {
            n: DrawingPlacement(scale=1.0, x=100, y=100, mirror=False, anchor="hip", occluders=(), depth=848)
            for n in names
        }
    }
    out, _ = refine([char], placed, _r_regions(), lambda c, d: images.get(d))
    return out["ali"]


def test_bb3_register_aligns_sequence_drawings() -> None:
    control = _r_refine(["p1", "p1-p2.1", "p2"], {"p1": _r_fig(), "p1-p2.1": _r_fig((15, 0)), "p2": _r_fig()})
    assert control["p1-p2.1"].x == 85  # control: inbetweens are pulled back (passes today)
    out = _r_refine(["p1", "s1.1", "p2"], {"p1": _r_fig(), "s1.1": _r_fig((15, 0)), "p2": _r_fig()})
    assert out["s1.1"].x == 85  # BUG 1: sequence drawing is left at x=100, still part misaligned


# --- bug 2: no borrowed patches on sequence drawings -----------------------------------------------

SKIN = (200, 120, 80)
RED = (120, 20, 20)


def _b_drawing(dx: int = 0) -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((150 + dx, 100, 250 + dx, 550), fill=SKIN, outline=(0, 0, 0), width=6)
    d.rectangle((185 + dx, 158, 215 + dx, 162), fill=(0, 0, 0))
    d.rectangle((175 + dx, 123, 225 + dx, 126), fill=(0, 0, 0))
    return img


def _b_landmarks(face: list[int]) -> dict[str, object]:
    return {
        "version": 1, "image": "x.png", "size": [400, 600], "facing": "right", "head": None,
        "face": {"box": face, "confidence": 0.9}, "feet": [], "feet_source": None, "foot_low": None,
        "pants": None, "hip": None, "hands": [], "cups": [],
    }  # fmt: skip


def _b_assets(tmp_path: Path) -> Path:
    root = tmp_path / "assets" / "sh01"
    (root / "ali").mkdir(parents=True)
    Image.new("RGB", (1920, 1080), (50, 50, 50)).save(root / "background.png")
    _b_drawing().save(root / "ali" / "p1.png")
    _b_drawing(20).save(root / "ali" / "s1.1.png")
    _b_drawing(20).save(root / "ali" / "p1-p2.1.png")
    save_patch(Patch(Image.new("RGBA", (40, 20), (*RED, 255)), 180, 150), root / "ali" / "p1.mouth-D")
    faces = {"p1": [150, 100, 250, 200], "s1.1": [170, 100, 270, 200], "p1-p2.1": [170, 100, 270, 200]}
    for name, face in faces.items():
        (root / "ali" / f"{name}.landmarks.json").write_text(json.dumps(_b_landmarks(face)), encoding="utf-8")
    return root


def _b_exposure(cell: str) -> Exposure:
    return Exposure(shot="sh01", length=4, body={"ali": [cell] * 4}, eyes={"ali": ["open"] * 4}, windows=[])


def _b_mouths() -> dict[str, Mouth]:
    cues = [{"start": 0, "end": 4, "shape": "D", "src": "auto"}]
    return {"ali": Mouth.model_validate({"version": 1, "shot": "sh01", "character": "ali", "fps": 24, "cues": cues})}


def _b_rgb(img: Image.Image, x: int, y: int) -> tuple[int, ...]:
    return tuple(int(v) for v in np.array(img.convert("RGBA"))[y, x, :3])


def test_bb3_sequence_frame_borrows_the_key_mouth(tmp_path: Path) -> None:
    root = _b_assets(tmp_path)
    layers, warn = frame_layers(_b_exposure("p1-p2.1"), _b_mouths(), ShotAssets(root), ["ali"],
                                {"ali": Placement(1.0, 0, 0)}, 0)  # fmt: skip
    assert _b_rgb(layers[0][0], 210, 160) == RED  # control: inbetween borrows (passes today)
    layers, warn = frame_layers(_b_exposure("s1.1"), _b_mouths(), ShotAssets(root), ["ali"],
                                {"ali": Placement(1.0, 0, 0)}, 0)  # fmt: skip
    assert "mouth-D" not in " ".join(warn), warn  # BUG 2a: "f=0 ali s1.1 mouth-D patch missing"
    assert _b_rgb(layers[0][0], 210, 160) == RED  # BUG 2b: mouth stays shut on sequence frames


# --- bug 3: loop step with an unknown beat ---------------------------------------------------------


def test_bb3_loop_step_with_unknown_beat_is_rejected() -> None:
    style = _style()
    xs = _xsheet([HOLD_P1, {"at": "shot.in + 10f", "loop": ["p1", "p2"], "step": "beat.nope",
                             "cadence": "twos", "until": "shot.in + 40f"}])  # fmt: skip
    with pytest.raises(TimingError) as exc:
        expand(xs, _poses(), _timing(style), style)
    assert exc.value.code == "anchor.unknown_beat"  # BUG 3: silently uses 8 today (no raise)


# --- bug 4: wrong-track items are silently dropped ------------------------------------------------


def test_bb3_wrong_track_items_are_reported() -> None:
    style = _style()
    xs = _xsheet(
        [HOLD_P1, {"at": "L02.start", "event": "blink"}],
        eyes=[{"at": "shot.in", "pose": "p1"}],
    )
    exp = expand(xs, _poses(), _timing(style), style)
    blink_seen = any(w.kind == "blink" for w in exp.windows) or ("ali.body", 1) in exp.unsupported
    assert blink_seen  # BUG 4a: blink on the body track vanishes (no window, no report)
    assert ("ali.eyes", 0) in exp.unsupported  # BUG 4b: hold on the eyes track vanishes silently
    codes = {f.code for f in lint(exp, style)}
    assert "timing.unsupported" in codes


# --- bug 5: loop inbetweens that exactly fill the step --------------------------------------------


def test_bb3_loop_plays_inbetweens_that_exactly_fill_the_step() -> None:
    style = _style()
    d = yaml.safe_load((FIX / "poses.yaml").read_text(encoding="utf-8"))
    d["characters"]["ali"]["transitions"][0]["inbetweens"] = [
        "Kol göğüs hizasında, bardak dik duruyor.",
        "Kol yarıya kadar uzanmış, bardak dik duruyor.",
        "Kol üç çeyrek uzanmış, bardak dik duruyor.",
        "Kol neredeyse tam uzanmış, bardak dik duruyor.",
    ]
    poses4 = Poses.model_validate(d)
    xs = _xsheet([HOLD_P1, {"at": "shot.in + 10f", "loop": ["p1", "p2"], "cadence": "twos",
                             "until": "shot.in + 42f"}])  # fmt: skip
    exp = expand(xs, poses4, _timing(style), style)
    assert exp.body["ali"][10:18] == ["p1-p2.1"] * 2 + ["p1-p2.2"] * 2 + ["p1-p2.3"] * 2 + ["p1-p2.4"] * 2


# --- bug 6: camera target with an unknown part ----------------------------------------------------


def test_bb3_camera_target_with_unknown_part_is_rejected() -> None:
    boxes = {"ali": (800, 200, 1000, 1000), "ayse": (1300, 300, 1500, 1000)}
    with pytest.raises(KeyError):
        target_bbox("ali.hand", boxes)  # BUG 6: silently returns the face slice today


# --- bug 7: offscreen characters are rendered -----------------------------------------------------

pytestmark_bb7 = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _bb7_figure(colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((150, 100, 250, 550), fill=colour, outline=(0, 0, 0), width=6)
    return img


@pytestmark_bb7
def test_bb3_offscreen_character_is_not_rendered(tmp_path: Path) -> None:
    from karagoz.engine.shot import render_shot

    p = tmp_path / "prod"
    ep = p / "ep"
    a = ep / "assets" / "sh01"
    for d in (ep / "shots" / "sh01", ep / "audio", a / "ali", a / "ayse", p / "referans"):
        d.mkdir(parents=True)
    style = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    style["live"].update(STYLE_EXTRA)
    (p / "style.yaml").write_text(yaml.safe_dump(style, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    shots = yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8"))
    shots["shots"][0]["characters"][1]["offscreen"] = True  # ayse is heard, not seen
    (ep / "shots.yaml").write_text(yaml.safe_dump(shots, allow_unicode=True), encoding="utf-8")
    shutil.copy(FIX / "poses.yaml", ep / "shots" / "sh01" / "poses.yaml")
    xs = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    xs["tracks"]["ayse.body"] = [{"at": "shot.in", "pose": "p1"}]
    (ep / "shots" / "sh01" / "xsheet.yaml").write_text(yaml.safe_dump(xs, allow_unicode=True), encoding="utf-8")
    for lid, text in ALIGNS.items():
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
    _bb7_figure((200, 120, 80)).save(a / "ali" / "p1.png")
    _bb7_figure((200, 120, 80)).save(a / "ali" / "p2.png")
    _bb7_figure((90, 160, 90)).save(a / "ayse" / "p1.png")
    _bb7_figure((200, 120, 80)).save(p / "referans" / "ali.png")
    _bb7_figure((90, 160, 90)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: true }\n",
        encoding="utf-8",
    )
    render_shot(p, "ep", "sh01")
    px = np.array(Image.open(ep / "render" / "sh01" / "frames" / "00050.png").convert("RGB")).astype(int)
    ayse = (np.abs(px - np.array([90, 160, 90])).sum(axis=2) < 30).any()
    assert not ayse  # BUG 7: offscreen ayse is drawn on screen today
