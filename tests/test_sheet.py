"""Acceptance tests for WP-24 (contact sheets, movement parts, close looks, frame strips). Written by the maintainer.

Do not edit. Uses the mini production of the WP-09a tests (`test_draw.py`): ali p1 -> p2 with two inbetweens,
ayse one pose.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz import cli
from karagoz.engine.render import _ffmpeg_bin
from karagoz.engine.sheet import (
    REGIONS,
    Part,
    contact_sheet,
    frame_strip,
    look,
    movement_parts,
    part_images,
    part_sheet,
)

ASSETS = f"{EP}/assets/sh01"


def solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def dark_pixels(img: Image.Image, box: tuple[int, int, int, int]) -> int:
    a = np.asarray(img.convert("L").crop(box))
    return int((a < 80).sum())


def drawings(prod: Path, names: dict[str, list[str]]) -> None:  # noqa: F811
    for char, items in names.items():
        for i, n in enumerate(items):
            solid(prod / ASSETS / char / f"{n}.png", (400, 600), (60 + 40 * i, 120, 200))


# --- contact sheet -------------------------------------------------------------------------------------------------


def test_contact_sheet_layout_and_labels(tmp_path: Path) -> None:
    imgs = [solid(tmp_path / f"d{i}.png", (400, 600), (0, 0, 255)) for i in range(3)]
    out = contact_sheet(imgs, tmp_path / "s" / "sheet.png", cell_height=300)
    assert out == tmp_path / "s" / "sheet.png" and out.is_file()
    sheet = Image.open(out)
    cell_w = 300 * 2 // 3
    assert sheet.size == (8 + 3 * (cell_w + 8), 8 + 300 + 8)
    assert sheet.convert("RGB").getpixel((2, 2)) == (200, 200, 200)  # gap colour
    assert dark_pixels(sheet, (8, 8, 8 + cell_w, 8 + 30)) > 20  # a label in the top-left corner of the first cell
    assert sheet.convert("RGB").getpixel((8 + cell_w // 2, 8 + 200)) == (0, 0, 255)  # the image itself


def test_contact_sheet_columns_rows_and_landscape(tmp_path: Path) -> None:
    imgs = [solid(tmp_path / f"p{i}.png", (400, 600), (0, 255, 0)) for i in range(7)]
    imgs.append(solid(tmp_path / "wide.png", (1920, 1080), (255, 0, 0)))
    sheet = Image.open(contact_sheet(imgs, tmp_path / "s.png", cell_height=180, columns=4))
    cell_w = 180 * 16 // 9  # one cell size for the whole sheet: the widest needed
    assert sheet.size == (8 + 4 * (cell_w + 8), 8 + 2 * (180 + 8))


def test_contact_sheet_errors_and_custom_labels(tmp_path: Path) -> None:
    img = solid(tmp_path / "a.png", (400, 600), (255, 255, 255))
    with pytest.raises(ValueError):
        contact_sheet([], tmp_path / "x.png")
    with pytest.raises(ValueError):
        contact_sheet([img], tmp_path / "x.png", labels=["a", "b"])
    sheet = Image.open(contact_sheet([img], tmp_path / "x.png", labels=["naime/p1.png"], cell_height=300))
    assert dark_pixels(sheet, (8, 8, 208, 38)) > 20


def test_transparent_drawing_on_white(tmp_path: Path) -> None:
    p = tmp_path / "t.png"
    Image.new("RGBA", (400, 600), (0, 0, 0, 0)).save(p)
    sheet = Image.open(contact_sheet([p], tmp_path / "s.png", cell_height=300)).convert("RGB")
    assert sheet.getpixel((8 + 100, 8 + 250)) == (255, 255, 255)


# --- movement parts ------------------------------------------------------------------------------------------------


def test_movement_parts(prod: Path) -> None:  # noqa: F811
    drawings(prod, {"ali": ["p1", "p1-p2.2", "p1-p2.1", "p2", "p1-p2.10"], "ayse": ["p1"]})
    parts = movement_parts(prod, EP, "sh01")
    assert parts == [
        Part("ali", "p1", "p2", ("p1", "p1-p2.1", "p1-p2.2", "p1-p2.10", "p2")),
        Part("ayse", "p1", "p1", ("p1",)),
    ]
    assert [p.id for p in parts] == ["ali:p1-p2", "ayse:p1"]
    imgs = part_images(prod, EP, "sh01", parts[0])
    assert [i.name for i in imgs] == ["p1.png", "p1-p2.1.png", "p1-p2.2.png", "p1-p2.10.png", "p2.png"]


def test_part_with_missing_files(prod: Path) -> None:  # noqa: F811
    drawings(prod, {"ali": ["p1", "p2"]})
    parts = movement_parts(prod, EP, "sh01")
    assert parts[0].drawings == ("p1", "p2")  # no inbetween files yet
    assert part_images(prod, EP, "sh01", parts[1]) == []  # ayse not drawn
    with pytest.raises(ValueError, match="ayse:p1"):
        part_sheet(prod, EP, "sh01", parts[1], prod / "x.png")
    out = part_sheet(prod, EP, "sh01", parts[0], prod / "s.png", cell_height=240)
    assert Image.open(out).size == (8 + 2 * (160 + 8), 8 + 240 + 8)


def test_no_poses_no_parts(prod: Path) -> None:  # noqa: F811
    (prod / EP / "shots" / "sh01" / "poses.yaml").unlink()
    assert movement_parts(prod, EP, "sh01") == []


# --- look ----------------------------------------------------------------------------------------------------------


def landmarked(tmp_path: Path) -> Path:
    img = Image.new("RGB", (1000, 1500), (255, 255, 255))
    img.paste((255, 0, 0), (400, 100, 600, 300))  # the "face"
    img.paste((0, 0, 255), (100, 800, 200, 900))  # two "hands"
    img.paste((0, 0, 255), (800, 800, 900, 900))
    p = tmp_path / "p1.png"
    img.save(p)
    lm = {"version": 1, "image": "p1.png", "size": [1000, 1500], "face": {"box": [400, 100, 600, 300]},
          "head": {"box": [350, 50, 650, 350]}, "hands": [{"box": [100, 800, 200, 900]}, {"box": [800, 800, 900, 900]}],
          "feet": [], "cups": []}  # fmt: skip
    (tmp_path / "p1.landmarks.json").write_text(json.dumps(lm), encoding="utf-8")
    return p


def test_look_face_is_padded_zoomed_and_labelled(tmp_path: Path) -> None:
    p = landmarked(tmp_path)
    out = look(p, tmp_path / "face.png", region="face")
    img = Image.open(out).convert("RGB")
    w, h = img.size
    assert max(w, h) >= 768
    assert abs(w / h - 1.0) < 0.02  # 200x200 box padded by 15 % each side: still square
    assert img.getpixel((w // 2, h // 2)) == (255, 0, 0)
    assert img.getpixel((w - 3, h // 2)) == (255, 255, 255)  # padding around the face
    assert dark_pixels(img, (0, 0, w // 2, 40)) > 20


def test_look_hands_union_and_halves(tmp_path: Path) -> None:
    p = landmarked(tmp_path)
    img = Image.open(look(p, tmp_path / "h.png", region="hands", zoom=1)).convert("RGB")
    w, h = img.size
    assert w / h > 5  # union of both hands: wide and short
    top = Image.open(look(p, tmp_path / "t.png", region="top", zoom=1))
    assert abs(top.size[0] / top.size[1] - 1000 / 750) < 0.02


def test_look_errors(tmp_path: Path) -> None:
    p = landmarked(tmp_path)
    assert set(REGIONS) == {"full", "face", "head", "hands", "feet", "props", "top", "bottom", "left", "right"}
    with pytest.raises(ValueError, match="feet not found"):
        look(p, tmp_path / "f.png", region="feet")
    with pytest.raises(ValueError):
        look(p, tmp_path / "f.png", region="elbow")
    bare = solid(tmp_path / "bare.png", (100, 100), (1, 2, 3))
    with pytest.raises(ValueError, match="face not found"):
        look(bare, tmp_path / "f.png", region="face")
    assert look(bare, tmp_path / "full.png").is_file()


# --- frame strip ---------------------------------------------------------------------------------------------------


def video(tmp_path: Path, n: int = 8) -> Path:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(n):
        Image.new("RGB", (320, 180), (30 * i, 0, 0)).save(frames / f"{i:03d}.png")
    out = tmp_path / "v.mp4"
    subprocess.run([_ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", "24", "-i", str(frames / "%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0", str(out)], check=True)  # fmt: skip
    return out


def test_frame_strip(tmp_path: Path) -> None:
    v = video(tmp_path)
    out = frame_strip(v, 2, 7, tmp_path / "strip.png", step=2, cell_height=180)
    sheet = Image.open(out).convert("RGB")
    cell_w = 180 * 16 // 9
    assert sheet.size == (8 + 3 * (cell_w + 8), 8 + 180 + 8)  # frames 2, 4, 6
    for j, frame in enumerate((2, 4, 6)):
        r, _, _ = sheet.getpixel((8 + j * (cell_w + 8) + cell_w // 2, 8 + 150))
        assert abs(r - 30 * frame) <= 12, (frame, r)
    with pytest.raises(ValueError):
        frame_strip(v, 5, 5, tmp_path / "x.png")


# --- cli -----------------------------------------------------------------------------------------------------------


def test_cli(prod: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    drawings(prod, {"ali": ["p1", "p1-p2.1", "p1-p2.2", "p2"], "ayse": ["p1"]})
    assert cli.main(["parts", str(prod), EP, "sh01"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[0] == "ali:p1-p2" and lines[0].split()[1:] == ["p1", "p1-p2.1", "p1-p2.2", "p2"]
    assert cli.main(["sheet", str(prod), EP, "sh01"]) == 0
    sheets = prod / EP / "review" / "sh01" / "sheets"
    assert sorted(x.name for x in sheets.iterdir()) == ["ali__p1-p2.png", "ayse__p1.png"]
    capsys.readouterr()
    assert cli.main(["sheet", str(prod), EP, "sh01", "--part", "ayse:p1", "--out", str(tmp_path / "o")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "o" / "ayse__p1.png")
    monkeypatch.chdir(tmp_path)
    one = prod / ASSETS / "ali" / "p1.png"
    assert cli.main(["look", str(one)]) == 0
    assert (tmp_path / "look-p1-full.png").is_file()
    assert cli.main(["look", str(one), "--region", "face"]) == 1  # no landmarks: an error, not a crash
    assert cli.main(["sheet-files", str(tmp_path / "f.png"), str(one), str(one)]) == 0
    assert (tmp_path / "f.png").is_file()
    v = video(tmp_path)
    assert cli.main(["strip", str(v), "0", "4"]) == 0
    assert (tmp_path / "strip-0-4.png").is_file()
