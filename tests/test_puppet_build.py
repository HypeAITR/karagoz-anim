"""K-0: the build writes the format files and names come only from the config."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.model import Puppet, PuppetConfig, load_config


def make_robot() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 40, 120, 140], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([30, 63, 122, 77], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.ellipse([120 - 16, 70 - 16, 120 + 16, 70 + 16], fill=(120, 40, 40), outline=(0, 0, 0), width=2)
    draw.rectangle([14, 58, 34, 82], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "torso", "parent": None, "concepts": [], "seed": [85, 110], "z": 0},
                {"name": "cap", "parent": "torso", "concepts": [], "seed": [130, 70], "z": 2},
                {"name": "arm", "parent": "cap", "concepts": [], "seed": [60, 70], "z": 1},
                {"name": "hand", "parent": "arm", "concepts": [], "seed": [24, 70], "z": 3},
            ]
        }
    )
    return rgb, cfg


def _write_ref(rgb: NDArray[np.uint8], tmp_path: Path) -> Path:
    ref = tmp_path / "ref.png"
    Image.fromarray(rgb).save(ref)
    return ref


def test_puppet_json_round_trips_through_model(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    raw = json.loads((tmp_path / "out" / "puppet.json").read_text(encoding="utf-8"))
    again = Puppet.model_validate(raw)
    assert again == files.puppet
    assert again.model_dump() == Puppet.model_validate(again.model_dump()).model_dump()
    assert again.root == "torso"
    assert set(again.parts) == {"torso", "cap", "arm", "hand"}
    for name, part in again.parts.items():
        assert part.file == f"parts/{name}.png"
        assert part.approved is False


def test_renamed_config_gives_renamed_files(tmp_path: Path) -> None:
    rgb, _cfg = make_robot()
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "body", "parent": None, "concepts": [], "seed": [85, 110], "z": 0},
                {"name": "dome", "parent": "body", "concepts": [], "seed": [130, 70], "z": 2},
                {"name": "tube", "parent": "dome", "concepts": [], "seed": [60, 70], "z": 1},
                {"name": "claw", "parent": "tube", "concepts": [], "seed": [24, 70], "z": 3},
            ]
        }
    )
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    assert {p.stem for p in (tmp_path / "out" / "parts").glob("*.png")} == {"body", "dome", "tube", "claw"}
    assert set(files.puppet.parts) == {"body", "dome", "tube", "claw"}
    assert set(files.report["joints"]) == {"dome", "tube", "claw"}


def test_report_has_required_keys_and_sheet_exists(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    report = files.report
    assert report["recomposition"]["outside_band_gt40"] == 0
    for name, info in report["outlines"].items():
        assert info["components"] == 1.0, name
        assert info["border_line_coverage"] >= 0.95, name
    for name, joint in report["joints"].items():
        assert set(joint["paper_px"]) == {"-45", "-20", "+20", "+45"}, name
        assert len(joint["pivot"]) == 2
    sheet = tmp_path / "out" / "sheet.png"
    assert sheet.is_file() and sheet.stat().st_size > 0


def test_robot_preset_loads_with_single_root() -> None:
    cfg = load_config(Path("src/karagoz/engine/puppet/presets/robot/3q_right.yaml"))
    assert cfg.root.name == "torso"
    assert len(cfg.parts) == 19
