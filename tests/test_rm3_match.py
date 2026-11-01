"""RM-3: matched points equal the golden eslesme.json, plus per-function and synthetic checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import interval, masks, match


def _experiment():
    root = rig_golden.fixture()
    os.environ["RIG_ARALIK"] = "xy"
    os.environ["RIG_CIZGI"] = "2.8"
    for p in ("deneme/r2/plan", "deneme/r2", "deneme/r2/d", ""):
        sys.path.insert(0, str(root / p) if p else str(root))
    import eslesme

    return eslesme


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
def key_data(ctx):
    gray_a = np.array(Image.open(ctx.image_a).convert("L"))
    gray_b = np.array(Image.open(ctx.image_b).convert("L"))
    cache = rig_golden.r2() / "d" / "cache" / "sam"
    loaded = {}
    for name, path in (("poz_X", ctx.image_a), ("poz_Y", ctx.image_b)):
        raw = masks.load_masks(masks.sam_cache_key(path), gray_a.shape, cache)
        loaded[name] = {k.replace(" ", "_"): v for k, v in raw.items()}
    pivots = json.loads((rig_golden.fixture() / "_altin_r13" / "xy" / "deltoid.json").read_text(encoding="utf-8"))
    return {
        "gray_a": gray_a,
        "gray_b": gray_b,
        "masks_a": loaded["poz_X"],
        "masks_b": loaded["poz_Y"],
        "pivot_a": np.array(pivots["poz_X"]["pivot"], float),
        "pivot_b": np.array(pivots["poz_Y"]["pivot"], float),
        "dome_r": float(pivots["kubbe_yaricap"]),
    }


def _assert_side_points_equal(got, want) -> None:
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name]["p"], want[name]["p"]), name
        assert got[name]["kesik"] == want[name]["kesik"], name


@pytest.mark.rig_golden
def test_build_matches_golden_eslesme(ctx, key_data) -> None:
    got = match.build(
        ctx,
        key_data["gray_a"],
        key_data["gray_b"],
        key_data["masks_a"],
        key_data["masks_b"],
        key_data["pivot_a"],
        key_data["pivot_b"],
        key_data["dome_r"],
    )
    want = json.loads((rig_golden.fixture() / "_altin_r13" / "xy" / "r7" / "eslesme.json").read_text(encoding="utf-8"))
    assert got == want


@pytest.mark.rig_golden
def test_search_patch_matches_experiment_on_golden_points(exp, key_data) -> None:
    arm = max(
        key_data["masks_a"]["arm"],
        key=lambda m: int(m[int(key_data["pivot_a"][1]), int(key_data["pivot_a"][0])]) * int(m.sum()),
    )
    pa = match.side_points(key_data["masks_a"], key_data["pivot_a"], key_data["dome_r"])
    for name in ("aski_dis", "gobek", "koltuk_alti", "yaka_1", "yakin_kenar_1"):
        got_q, got_s = match.search_patch(key_data["gray_a"], key_data["gray_b"], pa[name]["p"], arm)
        want_q, want_s = exp.yama_ara(key_data["gray_a"], key_data["gray_b"], pa[name]["p"], arm)
        assert np.array_equal(got_q, want_q), name
        assert got_s == want_s, name


@pytest.mark.rig_golden
def test_find_armpit_matches_experiment_both_keys(exp, key_data) -> None:
    for key, pivot, near in (
        ("masks_a", key_data["pivot_a"], None),
        ("masks_b", key_data["pivot_b"], key_data["pivot_a"]),
    ):
        garment = max(key_data[key]["tank_top"], key=lambda m: int(m.sum()))
        arm = max(key_data[key]["arm"], key=lambda m: int(m[int(pivot[1]), int(pivot[0])]) * int(m.sum()))
        got = match.find_armpit(garment, arm, pivot, key_data["dome_r"], near=near)
        want = exp.kose_ara(garment, arm, pivot, key_data["dome_r"], yakin=near)
        assert np.array_equal(got[0], want[0]), key
        assert got[1] == want[1], key
        assert np.array_equal(got[2], want[2]), key


@pytest.mark.rig_golden
def test_arc_points_matches_experiment(exp, key_data) -> None:
    pa = match.side_points(key_data["masks_a"], key_data["pivot_a"], key_data["dome_r"])
    c = exp._kontur(max(key_data["masks_a"]["tank_top"], key=lambda m: int(m.sum())))
    cut = match.near_arm(max(key_data["masks_a"]["arm"], key=lambda m: int(m.sum())))
    got = match.arc_points(c, 10, 500, 1, cut, {"kol_agzi_1": 0.25, "kol_agzi_2": 0.5})
    want = exp.yay_noktalari(c, 10, 500, 1, cut, {"kol_agzi_1": 0.25, "kol_agzi_2": 0.5})
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name]["p"], want[name]["p"]), name
        assert got[name]["kesik"] == want[name]["kesik"], name
    assert pa["kol_agzi_1"]["kesik"] in (True, False)


@pytest.mark.rig_golden
def test_near_arm_matches_experiment(exp, key_data) -> None:
    arm = max(key_data["masks_a"]["arm"], key=lambda m: int(m.sum()))
    assert np.array_equal(match.near_arm(arm), exp.kol_yakini(arm))


@pytest.mark.rig_golden
def test_side_points_match_experiment_both_keys(exp, key_data) -> None:
    got_a = match.side_points(key_data["masks_a"], key_data["pivot_a"], key_data["dome_r"])
    want_a = exp.yon_noktalari(key_data["masks_a"], key_data["pivot_a"], key_data["dome_r"])
    _assert_side_points_equal(got_a, want_a)
    got_b = match.side_points(
        key_data["masks_b"], key_data["pivot_b"], key_data["dome_r"], armpit=got_a["koltuk_alti"]["p"]
    )
    want_b = exp.yon_noktalari(
        key_data["masks_b"], key_data["pivot_b"], key_data["dome_r"], koltuk=want_a["koltuk_alti"]["p"]
    )
    _assert_side_points_equal(got_b, want_b)


@pytest.mark.rig_golden
def test_chained_ceiling_uses_median_and_marks_untracked(ctx, key_data) -> None:
    got = match.build(
        ctx,
        key_data["gray_a"],
        key_data["gray_b"],
        key_data["masks_a"],
        key_data["masks_b"],
        key_data["pivot_a"],
        key_data["pivot_b"],
        key_data["dome_r"],
        chained=True,
    )
    want = json.loads((rig_golden.fixture() / "_altin_r13" / "xy" / "r7" / "eslesme.json").read_text(encoding="utf-8"))
    assert got["kayma_tavani"] == 33.0  # 2 * median(28.1, 16.0, 17.0, 10.0)
    assert got["izlenemeyen"] == want["izlenemeyen"] == ["kol_agzi_3", "yaka_3", "yakin_kenar_3"]
    for name in got["izlenemeyen"]:
        assert got["noktalar"][name]["izlenemedi"] is True
    stripped = {k: dict(v) for k, v in got["noktalar"].items()}
    for v in stripped.values():
        v.pop("izlenemedi", None)
    assert stripped == want["noktalar"]
    assert got["dusuk_skor"] == want["dusuk_skor"]


@pytest.mark.rig_golden
def test_load_or_build_returns_stored_when_fresh(ctx, tmp_path: Path) -> None:
    golden = rig_golden.fixture() / "_altin_r13" / "xy" / "r7" / "eslesme.json"
    copy = tmp_path / "eslesme.json"
    copy.write_bytes(golden.read_bytes())

    def _builder():
        raise AssertionError("must not rebuild a fresh match")

    assert match.load_or_build(copy, ctx, _builder) == json.loads(golden.read_text(encoding="utf-8"))


@pytest.mark.rig_golden
def test_load_or_build_rebuilds_when_stale(ctx, tmp_path: Path) -> None:
    stale = tmp_path / "eslesme.json"
    stale.write_text(json.dumps({"kaynak_ozetleri": {"poz_X": "dead", "poz_Y": "beef"}}), encoding="utf-8")
    assert match.load_or_build(stale, ctx, lambda: {"fresh": True}) == {"fresh": True}
    assert match.load_or_build(tmp_path / "missing.json", ctx, lambda: {"fresh": True}) == {"fresh": True}


@pytest.mark.rig_golden
def test_build_writes_panel_only_when_asked(ctx, key_data, tmp_path: Path) -> None:
    args = (
        ctx,
        key_data["gray_a"],
        key_data["gray_b"],
        key_data["masks_a"],
        key_data["masks_b"],
        key_data["pivot_a"],
        key_data["pivot_b"],
        key_data["dome_r"],
    )
    rgb_a = np.array(Image.open(ctx.image_a).convert("RGB"))
    rgb_b = np.array(Image.open(ctx.image_b).convert("RGB"))
    panel = tmp_path / "panel.png"
    match.build(*args, panel_path=panel, panel_rgb=(rgb_a, rgb_b))
    with Image.open(panel) as im:
        assert im.size == (1500, 1100)
        assert im.mode == "RGB"
    with pytest.raises(ValueError, match="panel_rgb"):
        match.build(*args, panel_path=tmp_path / "no.png")


def test_search_patch_finds_itself_on_identical_images() -> None:
    a = np.full((80, 80), 200, np.uint8)
    a[30:50, 30:50] = 40
    q, score = match.search_patch(a, a.copy(), np.array([40.0, 40.0]), np.zeros((80, 80), bool))
    assert np.array_equal(q, [40.0, 40.0])
    assert score > 0.99


def test_search_patch_covered_window_scores_zero() -> None:
    a = np.full((80, 80), 200, np.uint8)
    a[30:50, 30:50] = 40
    q, score = match.search_patch(a, a.copy(), np.array([40.0, 40.0]), np.ones((80, 80), bool))
    assert np.array_equal(q, [40.0, 40.0])
    assert score == 0.0
