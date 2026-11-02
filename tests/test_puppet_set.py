"""K-0c: puppet sets — one character, several view puppets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import (
    ViewPuppet,
    build_puppet_set,
    head_for,
    neighbours,
    summarize_set,
    view_order,
)
from karagoz.engine.puppet.model import Puppet, PuppetSet, load_set, load_set_config
from karagoz.engine.puppet.view_geometry import (
    YawEstimate,
    estimate_yaw,
    long_axis_len,
    palette_distance,
    yaw_flag,
)

GREY = (70, 70, 70)
DARK = (30, 30, 30)
MAROON = (120, 40, 40)
BLUE = (40, 80, 200)


def make_front_parts() -> dict:
    return {
        "parts": [
            {"name": "torso", "parent": None, "concepts": [], "seed": [80, 70], "z": 0},
            {"name": "arm_l", "parent": "torso", "concepts": [], "seed": [120, 50], "z": 1},
            {"name": "arm_r", "parent": "torso", "concepts": [], "seed": [40, 50], "z": 1},
        ]
    }


def make_front_ref() -> NDArray[np.uint8]:
    img = Image.new("RGB", (160, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 30, 100, 110], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([100, 43, 140, 57], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([20, 43, 60, 57], fill=GREY, outline=(0, 0, 0), width=2)
    return np.asarray(img, dtype=np.uint8)


def _write(path: Path, rgb: NDArray[np.uint8]) -> Path:
    Image.fromarray(rgb).save(path)
    return path


def _foreshorten(
    masks: dict[str, NDArray[np.bool_]],
    pivots: dict[str, tuple[float, float]],
    yaw_deg: float,
    shrink_far: float = 1.0,
    far_side: str = "l",
    height: int = 120,
) -> tuple[dict, dict]:
    """Collapse lateral distances by cos(yaw) about the silhouette centre."""
    import cv2

    all_on = np.logical_or.reduce(list(masks.values()))
    ys, xs = np.nonzero(all_on)
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    factor = float(np.cos(np.radians(yaw_deg)))
    out_masks = {}
    for name, m in masks.items():
        side = name[-2:] if name.endswith(("_l", "_r")) else ""
        extra = shrink_far if side == f"_{far_side}" else 1.0
        h, w = m.shape
        mat = np.array([[factor * extra, 0.0, cx * (1.0 - factor * extra)], [0.0, 1.0, 0.0]])
        warped = cv2.warpAffine(m.astype(np.uint8), mat, (w, h), flags=cv2.INTER_NEAREST)
        out_masks[name] = np.asarray(warped > 0, dtype=np.bool_)
    out_pivots = {}
    for name, (px, py) in pivots.items():
        side = name[-2:] if name.endswith(("_l", "_r")) else ""
        extra = shrink_far if side == f"_{far_side}" else 1.0
        out_pivots[name] = (cx + (px - cx) * factor * extra, py)
    return out_masks, out_pivots


def _front_view() -> tuple[Puppet, dict]:
    tor = np.zeros((120, 160), np.bool_)
    tor[30:110, 60:100] = True
    al = np.zeros((120, 160), np.bool_)
    al[43:57, 100:140] = True
    ar = np.zeros((120, 160), np.bool_)
    ar[43:57, 20:60] = True
    masks = {"torso": tor, "arm_l": al, "arm_r": ar}
    puppet = Puppet.model_validate(
        {
            "version": 1,
            "reference": "front.png",
            "canvas": [160, 120],
            "line_width_px": 2.0,
            "root": "torso",
            "parts": {
                "torso": {
                    "parent": None,
                    "pivot": [80.0, 70.0],
                    "z": 0,
                    "file": "parts/torso.png",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                },
                "arm_l": {
                    "parent": "torso",
                    "pivot": [100.0, 50.0],
                    "z": 1,
                    "file": "parts/arm_l.png",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                },
                "arm_r": {
                    "parent": "torso",
                    "pivot": [60.0, 50.0],
                    "z": 1,
                    "file": "parts/arm_r.png",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                },
            },
        }
    )
    return puppet, masks


def test_yaw_of_foreshortened_set_within_3deg() -> None:
    front, front_masks = _front_view()
    for yaw in (20.0, 35.0, 60.0):
        masks, pivots = _foreshorten(
            front_masks, {"torso": (80.0, 70.0), "arm_l": (100.0, 50.0), "arm_r": (60.0, 50.0)}, yaw
        )
        view = front.model_copy(
            update={
                "parts": {
                    n: p.model_copy(update={"pivot": [pivots[n][0], pivots[n][1]]}) for n, p in front.parts.items()
                }
            }
        )
        est = estimate_yaw(view, masks, front, front_masks)
        assert isinstance(est, YawEstimate)
        assert est.method == "pairs"
        assert est.yaw_deg == pytest.approx(yaw, abs=3.0), (yaw, est)
        assert est.spread_deg >= 0.0


def test_yaw_sign_follows_far_side() -> None:
    front, front_masks = _front_view()
    pivots = {"torso": (80.0, 70.0), "arm_l": (100.0, 50.0), "arm_r": (60.0, 50.0)}
    masks, pv = _foreshorten(front_masks, pivots, 25.0, shrink_far=0.9, far_side="l")
    view = front.model_copy(
        update={"parts": {n: p.model_copy(update={"pivot": [pv[n][0], pv[n][1]]}) for n, p in front.parts.items()}}
    )
    est = estimate_yaw(view, masks, front, front_masks)
    assert est.yaw_deg > 0
    assert est.near_side == "r"


def test_yaw_profile_uses_shoulder_offset() -> None:
    front, front_masks = _front_view()
    pivots = {"torso": (80.0, 70.0), "arm_l": (100.0, 50.0), "arm_r": (60.0, 50.0)}
    masks, pv = _foreshorten(front_masks, pivots, 85.0)
    masks["arm_r"] = np.zeros_like(masks["arm_r"])
    view = front.model_copy(
        update={"parts": {n: p.model_copy(update={"pivot": [pv[n][0], pv[n][1]]}) for n, p in front.parts.items()}}
    )
    est = estimate_yaw(view, masks, front, front_masks)
    assert est.method == "profile"
    assert est.yaw_deg == pytest.approx(90.0, abs=8.0)


def test_yaw_flag_beyond_8deg() -> None:
    assert yaw_flag(measured=5.0, declared=25.0) is True
    assert yaw_flag(measured=23.0, declared=25.0) is False
    assert yaw_flag(measured=None, declared=25.0) is False


def test_palette_distance_zero_same_and_large_recoloured() -> None:
    a = np.full((20, 20, 3), GREY, np.uint8)
    b = np.full((20, 20, 3), GREY, np.uint8)
    c = np.full((20, 20, 3), BLUE, np.uint8)
    m = np.ones((20, 20), np.bool_)
    no_line = np.zeros((20, 20), np.bool_)
    assert palette_distance(a, m, no_line, b, m, no_line) == pytest.approx(0.0)
    assert palette_distance(a, m, no_line, c, m, no_line) > 40.0


def test_long_axis_len_measures_major_extent() -> None:
    m = np.zeros((40, 120), np.bool_)
    m[14:26, 10:110] = True
    assert long_axis_len(m) == pytest.approx(100.0, abs=2.0)


def _hand_set_dir(tmp_path: Path, missing: bool = False, hidden: bool = False) -> Path:
    out = tmp_path / "set"
    for view, wanted in (("a", {"x", "y"}), ("b", {"x"} if missing else {"x", "y"})):
        vdir = out / "views" / view
        (vdir / "parts").mkdir(parents=True, exist_ok=True)
        part_entries = {}
        for name in sorted(wanted):
            is_hidden = hidden and view == "b" and name == "y"
            if not is_hidden:
                Image.new("RGB", (30, 30), (255, 0, 0)).save(vdir / "parts" / f"{name}.png")
            part_entries[name] = {
                "parent": None if name == "x" else "x",
                "pivot": [15.0, 15.0],
                "z": 0,
                "file": "" if is_hidden else f"parts/{name}.png",
                "completed": [],
                "approved": False,
                "hidden": is_hidden,
            }
        (vdir / "puppet.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "reference": f"{view}.png",
                    "canvas": [30, 30],
                    "line_width_px": 2.0,
                    "root": "x",
                    "parts": part_entries,
                }
            ),
            encoding="utf-8",
        )
    set_doc = {
        "version": 1,
        "character": "t",
        "default_view": "a",
        "head_part": "x",
        "views": {"a": {"yaw_deg": 0.0, "dir": "views/a"}, "b": {"yaw_deg": 25.0, "dir": "views/b"}},
        "head_views": {},
    }
    (out / "set.json").write_text(json.dumps(set_doc), encoding="utf-8")
    return out


def test_set_missing_part_name_fails_clear(tmp_path: Path) -> None:
    out = _hand_set_dir(tmp_path, missing=True)
    with pytest.raises(ValueError, match="missing"):
        load_set(out)


def test_set_hidden_part_passes_without_file(tmp_path: Path) -> None:
    out = _hand_set_dir(tmp_path, hidden=True)
    got = load_set(out)
    assert isinstance(got, PuppetSet)
    assert got.views["b"].dir == "views/b"
    assert (out / "views" / "b" / "parts" / "y.png").exists() is False


def test_summarize_set_from_disk(tmp_path: Path) -> None:
    preset = tmp_path / "p.yaml"
    preset.write_text("parts:\n- {name: x, parent: null, concepts: [], seed: [15, 15], z: 0}\n", encoding="utf-8")
    ref = tmp_path / "r.png"
    red = np.full((30, 30, 3), 255, np.uint8)
    red[5:25, 5:25] = (200, 30, 30)
    _write(ref, red)
    out = tmp_path / "out"
    vdir = out / "views" / "v"
    (vdir / "parts").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 30), (200, 30, 30)).save(vdir / "parts" / "x.png")
    (vdir / "puppet.json").write_text(
        json.dumps(
            {
                "version": 1,
                "reference": "r.png",
                "canvas": [30, 30],
                "line_width_px": 2.0,
                "root": "x",
                "parts": {
                    "x": {
                        "parent": None,
                        "pivot": [15.0, 15.0],
                        "z": 0,
                        "file": "parts/x.png",
                        "completed": [],
                        "approved": False,
                        "hidden": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (out / "set.json").write_text(
        json.dumps(
            {
                "version": 1,
                "character": "t",
                "default_view": "v",
                "head_part": "x",
                "views": {"v": {"yaw_deg": 0.0, "dir": "views/v"}},
                "head_views": {},
            }
        ),
        encoding="utf-8",
    )
    set_yaml = tmp_path / "set.yaml"
    set_yaml.write_text(
        "character: t\ndefault_view: v\nhead_part: x\n"
        f"views:\n  v: {{preset: {preset.as_posix()}, reference: {ref.as_posix()}, yaw_deg: 0}}\n",
        encoding="utf-8",
    )
    report = summarize_set(out, load_set_config(set_yaml))
    assert report["views"]["v"]["yaw_measured"] == pytest.approx(0.0)
    assert report["views"]["v"]["yaw_flag"] is False
    assert report["parts"]["x"]["area_ratio"]["v"] == pytest.approx(1.0)
    assert report["parts"]["x"]["palette_flag"]["v"] is False
    assert (out / "set_report.json").is_file()
    assert (out / "set_sheet.png").is_file()


def test_view_order_and_neighbours() -> None:
    s = PuppetSet.model_validate(
        {
            "version": 1,
            "character": "t",
            "default_view": "b",
            "head_part": "x",
            "views": {
                "a": {"yaw_deg": -20.0, "dir": "views/a"},
                "b": {"yaw_deg": 0.0, "dir": "views/b"},
                "c": {"yaw_deg": 25.0, "dir": "views/c"},
            },
            "head_views": {},
        }
    )
    assert view_order(s) == ["a", "b", "c"]
    assert neighbours(s, "a") == ["b"]
    assert neighbours(s, "b") == ["a", "c"]
    assert neighbours(s, "c") == ["b"]


def _view_puppet(head_xy: tuple[float, float], neck_xy: tuple[float, float]) -> ViewPuppet:
    body_images: dict[str, NDArray[np.uint8]] = {}
    torso = np.zeros((60, 60, 4), np.uint8)
    torso[30:, 10:50, :3] = GREY
    torso[30:, 10:50, 3] = 255
    neck = np.zeros((60, 60, 4), np.uint8)
    neck[24:32, 26:34, :3] = DARK
    neck[24:32, 26:34, 3] = 255
    head = np.zeros((60, 60, 4), np.uint8)
    hx, hy = int(head_xy[0]), int(head_xy[1])
    head[hy - 6 : hy + 6, hx - 6 : hx + 6, :3] = MAROON
    head[hy - 6 : hy + 6, hx - 6 : hx + 6, 3] = 255
    body_images = {"torso": torso, "neck": neck, "head": head}
    body = Puppet.model_validate(
        {
            "version": 1,
            "reference": "b.png",
            "canvas": [60, 60],
            "line_width_px": 1.0,
            "root": "torso",
            "parts": {
                "torso": {
                    "parent": None,
                    "pivot": [30.0, 50.0],
                    "z": 0,
                    "file": "parts/torso.png",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                },
                "neck": {
                    "parent": "torso",
                    "pivot": [float(neck_xy[0]), float(neck_xy[1])],
                    "z": 1,
                    "file": "parts/neck.png",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                },
                "head": {
                    "parent": "neck",
                    "pivot": [float(head_xy[0]), float(head_xy[1])],
                    "z": 2,
                    "file": "parts/head.png",
                    "completed": [],
                    "approved": False,
                    "hidden": False,
                },
            },
        }
    )
    return ViewPuppet(name="body", puppet=body, images=body_images)


def test_head_for_puts_head_pivot_on_neck_pivot() -> None:
    body = _view_puppet(head_xy=(30.0, 10.0), neck_xy=(30.0, 28.0))
    head_view = _view_puppet(head_xy=(45.0, 12.0), neck_xy=(45.0, 40.0))
    head_view = ViewPuppet(name="headv", puppet=head_view.puppet, images={"head": head_view.images["head"]})
    head_view.puppet.parts["head"].parent = None
    got = head_for(body, head_view, head_part="head")
    neck = got.puppet.parts["neck"].pivot
    head_pivot = got.puppet.parts["head"].pivot
    assert head_pivot[0] == pytest.approx(neck[0], abs=0.5)
    assert head_pivot[1] == pytest.approx(neck[1], abs=0.5)
    assert set(got.images) == {"torso", "neck", "head"}
    assert got.puppet.parts["head"].parent == "neck"


def test_parallel_build_set_validates_and_reports(tmp_path: Path) -> None:
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    ref_a = _write(tmp_path / "a.png", make_front_ref())
    rgb_b = make_front_ref().copy()
    rgb_b[45:55, 102:138] = BLUE  # interior only: the black outline stays closed
    ref_b = _write(tmp_path / "b.png", rgb_b)
    (preset_dir / "a.yaml").write_text(
        "parts:\n"
        "- {name: torso, parent: null, concepts: [], seed: [80, 70], z: 0}\n"
        "- {name: arm_l, parent: torso, concepts: [], seed: [120, 50], z: 1}\n"
        "- {name: arm_r, parent: torso, concepts: [], seed: [40, 50], z: 1}\n",
        encoding="utf-8",
    )
    (preset_dir / "b.yaml").write_text(
        "parts:\n"
        "- {name: torso, parent: null, concepts: [], seed: [80, 70], z: 0}\n"
        "- {name: arm_l, parent: torso, concepts: [], seed: [120, 50], z: 1}\n"
        "- {name: arm_r, parent: torso, concepts: [], seed: [40, 50], z: 1}\n",
        encoding="utf-8",
    )
    set_yaml = tmp_path / "set.yaml"
    set_yaml.write_text(
        "character: t\ndefault_view: a\nhead_part: torso\n"
        "views:\n"
        f"  a: {{preset: {(preset_dir / 'a.yaml').as_posix()}, reference: {ref_a.as_posix()}, yaw_deg: 0}}\n"
        f"  b: {{preset: {(preset_dir / 'b.yaml').as_posix()}, reference: {ref_b.as_posix()}, yaw_deg: 30}}\n",
        encoding="utf-8",
    )
    cfg = load_set_config(set_yaml)
    assert set(cfg.views) == {"a", "b"}
    files = build_puppet_set(cfg, tmp_path / "out", workers=2, use_sam=False, use_image=False)
    assert (tmp_path / "out" / "set.json").is_file()
    assert (tmp_path / "out" / "views" / "a" / "puppet.json").is_file()
    assert (tmp_path / "out" / "views" / "b" / "puppet.json").is_file()
    assert (tmp_path / "out" / "set_sheet.png").is_file()
    assert set(files.set.views) == {"a", "b"}
    report = files.report
    assert report["views"]["b"]["yaw_measured"] == pytest.approx(0.0, abs=3.0)
    assert report["views"]["b"]["yaw_flag"] is True
    assert report["parts"]["arm_l"]["palette_flag"]["b"] is True
    assert report["parts"]["torso"]["palette_flag"]["b"] is False
    assert files.report["totals"]["cost_usd"] == 0.0
