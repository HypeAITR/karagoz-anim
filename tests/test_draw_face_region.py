"""Acceptance tests for WP-13b (patches for inbetween frames, cut inside the face). Maintainer's; do not edit.

Mini production of the WP-09a tests (`test_draw.py`). The fake mouth edit paints the mouth (470..530 x 440..480);
here it also moves a "glass rim" below it (560..620 x 540..600), 60 px away: today both end up in one patch.
The face box is (400, 300, 600, 500) (eyes and mouth); with the 10 % pad the region is (380, 280, 620, 520).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from test_draw import EP, FakeImage, assets, by_kind, png, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz import cli
from karagoz.engine.mouth import mouth_track
from karagoz.engine.patch import load_patch
from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import expand
from karagoz.meta.sam import Segment
from karagoz.spec import Align, Lines, Poses, Shots, Style, Xsheet, load
from karagoz.studio.draw import plan_draw, recut_patches, run_draw, sam_face_of

FACE = (400.0, 300.0, 600.0, 500.0)
REGION = (380, 280, 620, 520)


class GlassImage(FakeImage):
    def edit(self, images: list[Any], prompt: str, **kw: Any) -> list[bytes]:
        out = super().edit(images, prompt, **kw)
        if "Change ONLY the mouth" not in prompt:
            return out
        img = Image.open(io.BytesIO(out[0])).convert("RGB")
        ImageDraw.Draw(img).rectangle((560, 540, 620, 600), fill=(255, 255, 255), outline=(20, 20, 20), width=4)
        return [png(img)]


def mouth_patches(prod: Path) -> list[Path]:  # noqa: F811
    return sorted(p for p in (assets(prod) / "ali").glob("*.mouth-*.json"))


def patch_box(json_path: Path) -> tuple[int, int, int, int]:
    p = load_patch(Path(str(json_path)[: -len(".json")]))
    a = np.array(p.image)[:, :, 3]
    ys, xs = np.nonzero(a)
    return (p.x + int(xs.min()), p.y + int(ys.min()), p.x + int(xs.max()) + 1, p.y + int(ys.max()) + 1)


# --- plan ----------------------------------------------------------------------------------------------------------


def test_inbetween_frames_get_patches_on_the_nearer_key(prod: Path) -> None:  # noqa: F811
    ep = prod / EP
    lines, shots, style = load(ep / "lines.yaml"), load(ep / "shots.yaml"), load(prod / "style.yaml")
    xs, poses = load(ep / "shots/sh01/xsheet.yaml"), load(ep / "shots/sh01/poses.yaml")
    assert isinstance(lines, Lines) and isinstance(shots, Shots) and isinstance(style, Style)
    assert isinstance(xs, Xsheet) and isinstance(poses, Poses)
    aligns = {}
    for lid in ("L01", "L02"):
        a = load(ep / "audio" / f"{lid}.align.json")
        assert isinstance(a, Align)
        aligns[lid] = a
    timing = resolve_episode(lines, shots, aligns, style)
    exp = expand(xs, poses, timing, style)
    mouth = mouth_track("sh01", "ali", lines, shots, xs, timing, style)
    shapes = ["X"] * exp.length
    for cue in mouth.cues:
        for f in range(max(0, cue.start), min(exp.length, cue.end)):
            shapes[f] = cue.shape
    want = set()
    for f, cell in enumerate(exp.body["ali"]):
        if cell is not None and "-" in cell and shapes[f] != "X":
            k = int(cell.rsplit(".", 1)[1])
            want.add(("p1" if k <= 1 else "p2", shapes[f]))  # 2 inbetweens: .1 -> p1, .2 -> p2
    assert want  # ali speaks during his move
    jobs = by_kind(plan_draw(prod, EP, "sh01"))["mouth"]
    assert want <= {(j.drawing, j.variant) for j in jobs}
    assert {j.drawing for j in jobs} <= {"p1", "p2"}  # never a patch job on an inbetween drawing


# --- run -----------------------------------------------------------------------------------------------------------


def test_without_a_face_the_rim_is_in_the_patch(prod: Path) -> None:  # noqa: F811
    run_draw(plan_draw(prod, EP, "sh01"), GlassImage())
    boxes = [patch_box(p) for p in mouth_patches(prod)]
    assert boxes and all(b[3] > REGION[3] for b in boxes)  # today's behaviour, kept when no face box is known


def test_face_region_keeps_the_rim_out(prod: Path) -> None:  # noqa: F811
    seen: list[Path] = []

    def face_of(path: Path) -> tuple[float, float, float, float] | None:
        seen.append(path)
        return FACE

    rep = run_draw(plan_draw(prod, EP, "sh01"), GlassImage(), face_of=face_of)
    assert rep.failed == []
    boxes = [patch_box(p) for p in mouth_patches(prod)]
    assert boxes
    for x0, y0, x1, y1 in boxes:
        assert REGION[0] <= x0 and REGION[1] <= y0 and x1 <= REGION[2] and y1 <= REGION[3]
        assert x1 > 500 and y1 > 470  # the mouth itself is still there
    assert all(p.parent.name == "ali" and "." not in p.stem for p in seen)  # asked about pose drawings


def test_landmarks_file_gives_the_face(prod: Path) -> None:  # noqa: F811
    jobs = plan_draw(prod, EP, "sh01")
    run_draw([j for j in jobs if j.kind not in ("mouth", "eyes")], GlassImage())
    for pose in ("p1", "p2"):
        lm = {"face": {"box": list(FACE), "confidence": 0.9}, "size": [1024, 1536]}
        (assets(prod) / "ali" / f"{pose}.landmarks.json").write_text(json.dumps(lm), encoding="utf-8")
    run_draw(jobs, GlassImage())
    assert all(patch_box(p)[3] <= REGION[3] for p in mouth_patches(prod))


# --- recut ---------------------------------------------------------------------------------------------------------


def test_recut_old_patches_without_image_calls(prod: Path, capsys: Any) -> None:  # noqa: F811
    run_draw(plan_draw(prod, EP, "sh01"), GlassImage())
    assert all(patch_box(p)[3] > REGION[3] for p in mouth_patches(prod))
    lm = {"face": {"box": list(FACE), "confidence": 0.9}, "size": [1024, 1536]}
    (assets(prod) / "ali" / "p1.landmarks.json").write_text(json.dumps(lm), encoding="utf-8")
    done, skipped = recut_patches(prod, EP, "sh01")
    assert done and all(p.name.startswith("p1.") for p in done) and done == sorted(done)
    assert all(patch_box(Path(str(p) + ".json"))[3] <= REGION[3] for p in done)
    assert any(s.startswith("ali/p2.") for s in skipped)  # p2 has no landmarks
    assert cli.main(["recut", str(prod), EP, "sh01"]) == 0
    assert f"recut {len(done)}, skipped {len(skipped)}" in capsys.readouterr().out


# --- SAM -----------------------------------------------------------------------------------------------------------


class FakeSam:
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls += 1
        if concept != "face":
            return []
        return [Segment((400, 300, 600, 500), (1024, 1536), 0.9, np.full((200, 200), 200, np.uint8))]


def test_sam_face_of_measures_once_and_caches(prod: Path) -> None:  # noqa: F811
    jobs = plan_draw(prod, EP, "sh01")
    run_draw([j for j in jobs if j.kind == "pose"], FakeImage())
    sam = FakeSam()
    face_of = sam_face_of(sam, prod)
    pose = assets(prod) / "ali" / "p1.png"
    assert face_of(pose) == FACE
    calls = sam.calls
    assert face_of(pose) == FACE and sam.calls == calls  # cached in prod/.cache/landmarks
    assert list((prod / ".cache" / "landmarks").glob("*-right.json"))  # ali's natural facing
