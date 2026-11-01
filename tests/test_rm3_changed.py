"""RM-3: changed region matches degisen on a torso crop, plus synthetic checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import changed, interval, masks, measure

CROP = (slice(350, 750), slice(250, 650))  # shoulder, arm and torso of the keys


def _experiment():
    root = rig_golden.fixture()
    os.environ["RIG_ARALIK"] = "xy"
    os.environ["RIG_CIZGI"] = "2.8"
    for p in ("deneme/r2/plan", "deneme/r2", "deneme/r2/d", ""):
        sys.path.insert(0, str(root / p) if p else str(root))
    import degisen

    return degisen


@pytest.fixture(scope="module")
def exp():
    return _experiment()


@pytest.fixture(scope="module")
def ctx() -> interval.IntervalContext:
    root = rig_golden.fixture()
    return interval.IntervalContext(
        key_a="poz_X",
        key_b="poz_Y",
        image_a=root / "deneme" / "r2" / "d" / "girdi" / "poz_X.png",
        image_b=root / "deneme" / "r2" / "d" / "girdi" / "poz_Y.png",
        skeleton_path=root / "deneme" / "r2" / "d" / "iskelet_xy.json",
        motion_path=root / "deneme" / "r2" / "d" / "hareket_xy.json",
        output_dir=root / "deneme" / "r2" / "plan" / "out" / "xy",
        line_width=2.8,
    )


@pytest.fixture(scope="module")
def full_inputs(ctx):
    gray_a = np.array(Image.open(ctx.image_a).convert("L"))
    gray_b = np.array(Image.open(ctx.image_b).convert("L"))
    cache = rig_golden.r2() / "d" / "cache" / "sam"
    loaded = {
        name: masks.load_masks(masks.sam_cache_key(path), gray_a.shape, cache)
        for name, path in (("poz_X", ctx.image_a), ("poz_Y", ctx.image_b))
    }
    arm = np.zeros(gray_a.shape, bool)
    head = np.zeros(gray_a.shape, bool)
    for name in loaded:
        arm |= masks.union(loaded[name], ["arm", "hand", "cup", "tea glass", "glass"], gray_a.shape)
        head |= masks.union(loaded[name], ["head", "face", "hair", "mustache"], gray_a.shape)
    hidden = masks.dilate(arm, ctx.px(changed.HIDDEN_DILATE)) | head
    foreground = measure.foreground(np.array(Image.open(ctx.image_a).convert("RGB"))) | measure.foreground(
        np.array(Image.open(ctx.image_b).convert("RGB"))
    )
    return {"gray_a": gray_a, "gray_b": gray_b, "hidden": hidden, "arm": arm, "head": head, "foreground": foreground}


@pytest.fixture(scope="module")
def crop_inputs(full_inputs):
    return {k: v[CROP] for k, v in full_inputs.items()}


@pytest.fixture(scope="module")
def exp_field(exp, crop_inputs):
    return exp.alan_olc(crop_inputs["gray_a"], crop_inputs["gray_b"], crop_inputs["hidden"])


@pytest.fixture(scope="module")
def field(ctx, crop_inputs):
    return changed.measure_field(ctx, crop_inputs["gray_a"], crop_inputs["gray_b"], crop_inputs["hidden"])


def _assert_fields_equal(got, want) -> None:
    assert np.array_equal(got["xs"], want["xs"])
    assert np.array_equal(got["ys"], want["ys"])
    for key in ("dx", "dy", "skor"):  # unmeasured blocks are NaN in both
        assert got[key].dtype == want[key].dtype
        assert np.array_equal(got[key], want[key], equal_nan=True)


@pytest.mark.rig_golden
def test_measure_field_matches_experiment_on_torso_crop(field, exp_field) -> None:
    _assert_fields_equal(field, exp_field)
    assert 100 < (~np.isnan(exp_field["dx"])).sum() < exp_field["dx"].size  # measured and unmeasured blocks both exist


@pytest.mark.rig_golden
def test_build_region_matches_experiment_on_torso_crop(ctx, exp, field, exp_field, crop_inputs) -> None:
    got = changed.build_region(
        ctx, field, crop_inputs["gray_a"].shape, crop_inputs["arm"], crop_inputs["head"], crop_inputs["foreground"]
    )
    want = exp.bolge_kur(
        exp_field, crop_inputs["gray_a"].shape, crop_inputs["arm"], crop_inputs["head"], crop_inputs["foreground"]
    )
    assert np.array_equal(got["bolge"], want["bolge"])
    assert np.array_equal(got["izgara_degisen"], want["izgara_degisen"])
    assert np.array_equal(got["izgara_olcu"], want["izgara_olcu"])
    assert got["adalar"] == want["adalar"]
    assert got["bolge"].any() and (~got["bolge"]).any()


@pytest.mark.rig_golden
def test_area_points_matches_experiment_on_torso_crop(ctx, exp, field, exp_field, crop_inputs) -> None:
    region = changed.build_region(
        ctx, field, crop_inputs["gray_a"].shape, crop_inputs["arm"], crop_inputs["head"], crop_inputs["foreground"]
    )["bolge"]
    existing = [[300.0, 450.0], [500.0, 600.0]]
    got = changed.area_points(
        ctx, field, crop_inputs["gray_a"], region, crop_inputs["hidden"], crop_inputs["head"], existing
    )
    want = exp.alan_noktalari(
        exp_field, crop_inputs["gray_a"], region, crop_inputs["hidden"], crop_inputs["head"], existing
    )
    assert got == want


@pytest.mark.rig_golden
def test_verify_rules_matches_experiment_on_crop_points(ctx, exp, crop_inputs) -> None:
    points = {
        "deltoid": {"a": [90.0, 160.0], "b": [63.0, 155.0], "tur": "hareketli"},
        "sabit_nokta": {"a": [200.0, 300.0], "b": [201.0, 301.0], "tur": "sabit"},
        "hareketli_nokta": {"a": [150.0, 200.0], "b": [140.0, 195.0], "tur": "hareketli"},
        "izlenemeyen": {"a": [250.0, 250.0], "b": [240.0, 245.0], "tur": "sabit", "izlenemedi": True},
    }
    got = changed.verify_rules(ctx, crop_inputs["gray_a"], crop_inputs["gray_b"], points, crop_inputs["hidden"])
    want = exp.kural_dogrula(crop_inputs["gray_a"], crop_inputs["gray_b"], points, crop_inputs["hidden"])
    assert got == want
    assert "deltoid" not in got and "sabit_nokta" not in got


@pytest.mark.rig_golden
def test_still_body_matches_experiment_on_torso_crop(ctx, exp, field, crop_inputs) -> None:
    region = changed.build_region(
        ctx, field, crop_inputs["gray_a"].shape, crop_inputs["arm"], crop_inputs["head"], crop_inputs["foreground"]
    )["bolge"]
    got = changed.still_body(ctx, region, crop_inputs["head"], crop_inputs["foreground"])
    want = exp.sabit_govde(region, crop_inputs["head"], crop_inputs["foreground"])
    assert np.array_equal(got, want)


@pytest.mark.rig_golden
def test_band_summary_matches_experiment(exp, field, exp_field) -> None:
    bands = {"ust": (0, 200), "alt": (200, 400)}
    assert changed.band_summary(field, bands) == exp.bant_ozet(exp_field, bands)


@pytest.mark.rig_golden
def test_draw_region_matches_experiment(ctx, exp, field, crop_inputs, tmp_path: Path) -> None:
    region = changed.build_region(
        ctx, field, crop_inputs["gray_a"].shape, crop_inputs["arm"], crop_inputs["head"], crop_inputs["foreground"]
    )["bolge"]
    rgb_a = np.stack([crop_inputs["gray_a"]] * 3, axis=-1)
    rgb_b = np.stack([crop_inputs["gray_b"]] * 3, axis=-1)
    got_path, want_path = tmp_path / "got.png", tmp_path / "want.png"
    changed.draw_region(ctx, rgb_a, rgb_b, region, got_path)
    exp.gorsel(rgb_a, rgb_b, region, want_path)
    assert got_path.read_bytes() == want_path.read_bytes()


def test_measure_field_flat_images_have_no_measured_block(tmp_path: Path) -> None:
    ctx = interval.IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
        line_width=2.8,
    )
    a = np.full((80, 80), 200, np.uint8)
    field = changed.measure_field(ctx, a, a.copy(), np.zeros((80, 80), bool))
    assert np.isnan(field["dx"]).all()
    assert field["dx"].shape == (len(field["ys"]), len(field["xs"]))


def test_still_body_full_region_is_empty(tmp_path: Path) -> None:
    ctx = interval.IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
        line_width=2.8,
    )
    out = changed.still_body(ctx, np.ones((80, 80), bool), np.zeros((80, 80), bool), np.ones((80, 80), bool))
    assert not out.any()


def test_band_summary_empty_bands_report_none() -> None:
    field = {
        "xs": np.arange(3),
        "ys": np.array([10, 20]),
        "dx": np.full((2, 3), np.nan, np.float32),
        "dy": np.full((2, 3), np.nan, np.float32),
        "skor": np.full((2, 3), np.nan, np.float32),
    }
    assert changed.band_summary(field, {"hepsi": (0, 30)}) == {"hepsi": {"blok": 0, "ortanca": None, "p90": None}}
