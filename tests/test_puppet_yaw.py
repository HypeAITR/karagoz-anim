"""K-0d yaw: centreline estimator, back rule, hidden skip, shadow load, set wire-up."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw
from test_puppet_k0d import _cfg, _one_part, _write

from karagoz.engine.puppet.build import _load_view, build_puppet, build_puppet_set
from karagoz.engine.puppet.model import ViewEntry
from karagoz.engine.puppet.view_geometry import centerline_yaw, looks_like_back

GREY = (70, 70, 70)


def test_centerline_yaw_zero_and_sign() -> None:
    assert centerline_yaw(0.0, 100.0) == pytest.approx(0.0)
    assert centerline_yaw(50.0, 100.0) == pytest.approx(30.0)
    assert centerline_yaw(-50.0, 100.0) == pytest.approx(-30.0)


def test_centerline_yaw_clamps_edge_on() -> None:
    assert centerline_yaw(120.0, 100.0) == pytest.approx(90.0)
    assert centerline_yaw(-130.0, 100.0) == pytest.approx(-90.0)
    with pytest.raises(ValueError, match="ruler must be positive"):
        centerline_yaw(10.0, 0.0)


def test_looks_like_back_needs_all_front_features_absent() -> None:
    front = {"eye": SimpleNamespace(front=True), "seam": SimpleNamespace(front=True)}
    assert looks_like_back(front, {}) is True
    assert looks_like_back(front, {"seam": SimpleNamespace(front=True)}) is False
    assert looks_like_back({}, {}) is False
    assert looks_like_back({"deco": SimpleNamespace(front=False)}, {}) is False


def _torso_view(path: Path, box: tuple[int, int, int, int], seam_x: int | None) -> Path:
    img = Image.new("RGB", (600, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, fill=GREY, outline=(0, 0, 0), width=3)
    if seam_x is not None:
        draw.line([(seam_x, 210), (seam_x, 250)], fill=(0, 0, 0), width=3)
    return _write(path, np.asarray(img, dtype=np.uint8))


def _preset_yaml(path: Path, seed: list[int], features: dict[str, tuple[str, list[int], bool]]) -> Path:
    lines = [
        "parts:",
        "  - name: torso",
        "    parent: null",
        "    concepts: []",
        f"    seed: [{seed[0]}, {seed[1]}]",
        "    z: 0",
    ]
    if features:
        lines.append("features:")
        for name, (part, point, front) in features.items():
            flag = "true" if front else "false"
            lines.append(
                "  "
                + name
                + ": {part: "
                + part
                + ", point: ["
                + str(point[0])
                + ", "
                + str(point[1])
                + "], front: "
                + flag
                + "}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_hidden_part_skipped_in_outlines(tmp_path: Path) -> None:
    ref = _torso_view(tmp_path / "r.png", (200, 150, 400, 300), None)
    cfg = _cfg(
        [_one_part("body", seed=[300, 280]), {**_one_part("cape", seed=None), "parent": "body", "hidden": True, "z": 1}]
    )
    files = build_puppet(ref, cfg, tmp_path / "out")
    assert set(files.report["outlines"]) == {"body"}
    assert files.puppet.parts["cape"].hidden is True


def test_load_view_reads_shadow_layer(tmp_path: Path) -> None:
    ref = _torso_view(tmp_path / "r.png", (200, 150, 400, 300), None)
    cfg = _cfg([_one_part("body", seed=[300, 280])])
    preset = tmp_path / "p.yaml"
    preset.write_text(
        "parts:\n  - name: body\n    parent: null\n    concepts: []\n    seed: [300, 280]\n    z: 0\n", encoding="utf-8"
    )
    out = tmp_path / "views" / "v"
    build_puppet(ref, cfg, out)
    shadow = np.zeros((400, 600), np.uint8)
    shadow[330:360, 220:380] = 200
    Image.fromarray(np.dstack([np.full((400, 600, 3), 128, np.uint8), shadow])).save(out / "parts" / "_shadow.png")
    lv = _load_view(
        tmp_path, "views", "v", ViewEntry(yaw_deg=0.0, dir="views/v"), SimpleNamespace(preset=preset, reference=ref)
    )
    assert lv.shadow is not None
    assert int(np.asarray(lv.shadow, dtype=np.bool_).sum()) == 30 * 160


def test_set_yaw_end_to_end(tmp_path: Path) -> None:
    views = {
        "front": ((100, 150, 500, 300), 280, 0.0, {"seam": ("torso", [300, 230], True)}),
        "turned": ((100, 150, 500, 300), 280, 25.0, {"seam": ("torso", [386, 230], True)}),
        "profile": ((100, 150, 500, 300), 280, 90.0, {"seam": ("torso", [501, 230], True)}),
        "back": ((100, 150, 500, 300), 280, 180.0, {}),
    }
    set_lines = ["character: t", "default_view: front", "head_part: torso", "views:"]
    for name, (box, seed_y, yaw, feats) in views.items():
        seam_spec = feats.get("seam")
        seam_x: int | None = seam_spec[1][0] if seam_spec is not None else None
        ref = _torso_view(tmp_path / f"{name}.png", box, seam_x)
        preset = _preset_yaml(tmp_path / f"{name}.yaml", [300, seed_y], feats)
        set_lines += [
            f"  {name}:",
            f"    preset: {preset.as_posix()}",
            f"    reference: {ref.as_posix()}",
            f"    yaw_deg: {yaw}",
        ]
    set_yaml = tmp_path / "set.yaml"
    set_yaml.write_text("\n".join(set_lines) + "\n", encoding="utf-8")
    files = build_puppet_set(set_yaml, tmp_path / "out", workers=1, use_sam=False, use_image=False, max_image_calls=0)
    got = files.report["views"]
    assert got["front"]["yaw_measured"] == pytest.approx(0.0)
    assert got["turned"]["yaw_measured"] == pytest.approx(25.0, abs=2.0)
    assert got["turned"]["yaw_method"] == "seam"
    assert got["profile"]["yaw_measured"] == pytest.approx(90.0, abs=8.0)
    assert got["back"]["yaw_measured"] == pytest.approx(180.0)
    assert got["back"]["yaw_method"] == "back-absence"
    assert all(got[v]["yaw_flag"] is False for v in views)
