"""K-0: the reference split gives one closed component per part (synthetic robot)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import split_reference

CAP_CENTRE = (120, 70)


def make_robot() -> tuple[NDArray[np.uint8], PuppetConfig]:
    """Tiny layered robot: torso rect, round cap, tube arm under the cap, hand."""
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


def test_split_gives_one_component_per_part() -> None:
    rgb, cfg = make_robot()
    split = split_reference(rgb, cfg, sam=None)
    assert split.sam_calls == 0
    assert set(split.part_masks) == {"torso", "cap", "arm", "hand"}
    for name, count in split.region_counts.items():
        assert count == 1, f"{name} has {count} components"


def test_recomposition_diff_zero_outside_line_band(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    assert files.report["recomposition"]["outside_band_gt40"] == 0


def test_seed_alone_decides_without_sam() -> None:
    rgb, cfg = make_robot()
    with_concepts = PuppetConfig.model_validate(
        {
            "parts": [
                {**p, "concepts": ["robot torso"] if p["name"] == "torso" else ["robot arm"]}
                for p in [part.model_dump() for part in cfg.parts]
            ]
        }
    )
    plain = split_reference(rgb, cfg, sam=None)
    named = split_reference(rgb, with_concepts, sam=None)
    assert named.sam_calls == 0
    for name in plain.part_masks:
        assert np.array_equal(named.part_masks[name], plain.part_masks[name])


def test_seed_list_claims_both_regions(tmp_path: Path) -> None:
    """One part with two seeds claims both of its regions (still one component)."""
    img = Image.new("RGB", (120, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 20, 50, 40], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([50, 20, 90, 40], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "tube", "parent": None, "concepts": [], "seed": [[30, 30], [70, 30]], "z": 0},
            ]
        }
    )
    split = split_reference(rgb, cfg, sam=None)
    assert cfg.by_name["tube"].seed == [[30.0, 30.0], [70.0, 30.0]]
    assert bool(split.part_masks["tube"][30, 30]) and bool(split.part_masks["tube"][30, 70])
    assert split.region_counts == {"tube": 1}


def test_shared_sam_mask_splits_twins_by_seed(tmp_path: Path) -> None:
    """Twins sharing one SAM mask (equal overlap) are split by their seeds."""
    from karagoz.meta.sam import Segment

    img = Image.new("RGB", (120, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 20, 50, 40], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([70, 20, 110, 40], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "arm_r", "parent": None, "concepts": ["arm"], "seed": [30, 30], "z": 0},
                {"name": "arm_l", "parent": "arm_r", "concepts": ["arm"], "seed": [90, 30], "z": 0},
            ]
        }
    )
    union = np.zeros((60, 120), np.uint8)
    union[20:41, 10:51] = 255
    union[20:41, 70:111] = 255

    class StubSam:
        def segment(self, image: object, concept: str, tag: str = "") -> list[Segment]:
            assert concept == "arm"
            return [Segment((0, 0, 120, 60), (120, 60), None, union)]

    split = split_reference(
        rgb,
        cfg,
        image_bytes=b"stub-image",
        sam=StubSam(),
        cache_dir=tmp_path / "cache",  # type: ignore[arg-type]
    )
    assert split.sam_calls == 1
    assert bool(split.part_masks["arm_r"][30, 30]) and not bool(split.part_masks["arm_r"][30, 90])
    assert bool(split.part_masks["arm_l"][30, 90]) and not bool(split.part_masks["arm_l"][30, 30])
    assert split.region_counts == {"arm_r": 1, "arm_l": 1}


def _write_ref(rgb: NDArray[np.uint8], tmp_path: Path) -> Path:
    ref = tmp_path / "ref.png"
    Image.fromarray(rgb).save(ref)
    return ref
