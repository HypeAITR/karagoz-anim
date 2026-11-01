"""RM-1: rig part masks and reconciliation match the experiment, plus synthetic checks."""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import masks


def _experiment():
    root = rig_golden.fixture()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "deneme" / "r2"))
    import uzlastir_iskelet

    return uzlastir_iskelet


def _keys() -> tuple[str, str]:
    motion = json.loads((rig_golden.r2() / "d" / "hareket_xy.json").read_text(encoding="utf-8"))
    return motion["poz_a"], motion["poz_b"]


def _image(name: str):
    return rig_golden.r2() / "d" / "girdi" / f"{name}.png"


def _cache_dir():
    return rig_golden.r2() / "d" / "cache" / "sam"


def _shape(name: str) -> tuple[int, int]:
    with Image.open(_image(name)) as img:
        w, h = img.size
    return (h, w)


@pytest.fixture(scope="module")
def exp():
    return _experiment()


@pytest.fixture(scope="module")
def key_names() -> tuple[str, str]:
    return _keys()


@pytest.fixture(scope="module")
def port_loaded(key_names):
    return {name: masks.load_masks(masks.sam_cache_key(_image(name)), _shape(name), _cache_dir()) for name in key_names}


@pytest.fixture(scope="module")
def exp_results(exp, key_names):
    return {
        name: exp.uzlastir_ve_bol(name, girdi_path=_image(name), cache_dir=_cache_dir(), kaydet=False)
        for name in key_names
    }


@pytest.fixture(scope="module")
def port_results(key_names):
    return {name: masks.reconcile_and_split(name, _image(name), _cache_dir()) for name in key_names}


def _joint_point(joints: dict, name: str) -> np.ndarray:
    return np.array([joints[name]["x"], joints[name]["y"]], float)


@pytest.mark.rig_golden
def test_sam_cache_key_matches_experiment_both_keys(exp, key_names) -> None:
    for name in key_names:
        assert masks.sam_cache_key(_image(name)) == exp.sam_anahtari(_image(name))


@pytest.mark.rig_golden
def test_load_masks_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        shape = _shape(name)
        expected = exp.maskeleri_yukle(exp.sam_anahtari(_image(name)), shape, cache_dir=_cache_dir())
        got = port_loaded[name]
        assert set(got) == set(expected)
        for concept in expected:
            assert len(got[concept]) == len(expected[concept])
            for a, b in zip(got[concept], expected[concept], strict=True):
                assert np.array_equal(a, b)


@pytest.mark.rig_golden
def test_union_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    concepts = ["arm", "upper arm", "hand"]
    for name in key_names:
        shape = _shape(name)
        assert np.array_equal(
            masks.union(port_loaded[name], concepts, shape),
            exp.birlestir(port_loaded[name], concepts, shape),
        )


@pytest.mark.rig_golden
def test_components_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        blob = masks.union(port_loaded[name], ["arm", "hand"], _shape(name))
        for min_area in (64, 500):
            expected = exp.bilesenler(blob, min_alan=min_area)
            got = masks.components(blob, min_area=min_area)
            assert len(got) == len(expected)
            for a, b in zip(got, expected, strict=True):
                assert np.array_equal(a, b)


@pytest.mark.rig_golden
def test_dilate_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        blob = masks.union(port_loaded[name], ["hand"], _shape(name))
        assert np.array_equal(masks.dilate(blob, 3), exp.genislet(blob, 3))


@pytest.mark.rig_golden
def test_erode_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        blob = masks.union(port_loaded[name], ["hand"], _shape(name))
        assert np.array_equal(masks.erode(blob, 3), exp.daralt(blob, 3))


@pytest.mark.rig_golden
def test_opening_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        blob = masks.union(port_loaded[name], ["hand"], _shape(name))
        assert np.array_equal(masks.opening(blob, 2), exp.acma(blob, 2))


@pytest.mark.rig_golden
def test_fill_holes_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        blob = masks.union(port_loaded[name], ["torso", "tank top"], _shape(name))
        assert np.array_equal(masks.fill_holes(blob), exp.delik_doldur(blob))


@pytest.mark.rig_golden
def test_centroid_matches_experiment_both_keys(exp, key_names, port_loaded) -> None:
    for name in key_names:
        blob = masks.union(port_loaded[name], ["head", "face"], _shape(name))
        np.testing.assert_allclose(masks.centroid(blob), exp.merkez(blob), rtol=0, atol=1e-9)


@pytest.mark.rig_golden
def test_unit_matches_experiment_both_keys(exp, exp_results) -> None:
    for res in exp_results.values():
        joints = res["eklemler"]
        v = _joint_point(joints, "elbow_L") - _joint_point(joints, "shoulder_L")
        np.testing.assert_allclose(masks.unit(v), exp.birim(v), rtol=0, atol=1e-9)


@pytest.mark.rig_golden
def test_distance_matches_experiment_both_keys(exp, exp_results) -> None:
    for res in exp_results.values():
        joints = res["eklemler"]
        a = _joint_point(joints, "shoulder_L")
        b = _joint_point(joints, "wrist_L")
        assert math.isclose(masks.distance(a, b), exp.uzaklik(a, b), rel_tol=0, abs_tol=1e-9)


@pytest.mark.rig_golden
def test_bone_radius_matches_experiment_both_keys(exp, exp_results) -> None:
    for res in exp_results.values():
        joints = res["eklemler"]
        arm = res["anatomik_ek"]["arm_L_anatomik"]
        wrist = _joint_point(joints, "wrist_L")
        elbow = _joint_point(joints, "elbow_L")
        assert math.isclose(
            masks.bone_radius(arm, wrist, elbow), exp.kemik_yaricapi(arm, wrist, elbow), rel_tol=0, abs_tol=1e-9
        )


@pytest.mark.rig_golden
def test_elbow_knee_point_matches_experiment_both_keys(exp, exp_results) -> None:
    for res in exp_results.values():
        joints = res["eklemler"]
        arm = res["anatomik_ek"]["arm_L_anatomik"]
        shoulder = _joint_point(joints, "shoulder_L")
        wrist = _joint_point(joints, "wrist_L")
        np.testing.assert_allclose(
            masks.elbow_knee_point(arm, shoulder, wrist),
            exp.dirsek_diz_noktasi(arm, shoulder, wrist),
            rtol=0,
            atol=1e-9,
        )


@pytest.mark.rig_golden
def test_reconcile_and_split_matches_experiment_both_keys(exp_results, port_results) -> None:
    assert set(port_results) == set(exp_results)
    for name in exp_results:
        expected, got = exp_results[name], port_results[name]
        assert np.array_equal(got["rgb"], expected["rgb"])
        assert np.array_equal(got["fg"], expected["fg"])
        assert np.array_equal(got["fg_raw"], expected["fg_raw"])
        assert set(got["katmanlar"]) == set(expected["katmanlar"])
        for layer in expected["katmanlar"]:
            assert np.array_equal(got["katmanlar"][layer], expected["katmanlar"][layer]), layer
        assert got["eklemler"] == expected["eklemler"]
        assert got["kapsama"] == expected["kapsama"]
        assert set(got["anatomik_ek"]) == set(expected["anatomik_ek"])
        for key, value in expected["anatomik_ek"].items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(got["anatomik_ek"][key], value), key
            else:
                assert got["anatomik_ek"][key] == value, key


def test_components_finds_drawn_rectangles_and_drops_specks() -> None:
    m = np.zeros((200, 200), bool)
    m[10:40, 10:50] = True  # 1200 px
    m[100:110, 100:110] = True  # 100 px
    m[150, 150] = True  # 1 px speck
    parts = masks.components(m)
    assert [int(p.sum()) for p in parts] == [1200, 100]
    assert [int(p.sum()) for p in masks.components(m, min_area=1000)] == [1200]


def test_fill_holes_fills_a_drawn_ring() -> None:
    m = np.zeros((20, 20), bool)
    m[4:16, 4:16] = True
    m[6:14, 6:14] = False
    filled = masks.fill_holes(m)
    assert filled[4:16, 4:16].all()
    assert filled.sum() == 144
    empty = np.zeros((20, 20), bool)
    assert np.array_equal(masks.fill_holes(empty), empty)


def test_dilate_grows_a_pixel_to_a_plus() -> None:
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    expected = np.zeros((7, 7), bool)
    expected[3, 3] = expected[2, 3] = expected[4, 3] = expected[3, 2] = expected[3, 4] = True
    assert np.array_equal(masks.dilate(m, 1), expected)


def test_erode_shrinks_a_block_to_its_core() -> None:
    m = np.zeros((9, 9), bool)
    m[2:7, 2:7] = True
    expected = np.zeros((9, 9), bool)
    expected[3:6, 3:6] = True
    assert np.array_equal(masks.erode(m, 1), expected)


def test_opening_snaps_an_isolated_speck() -> None:
    m = np.zeros((9, 12), bool)
    m[2:7, 2:7] = True
    m[7, 9] = True  # isolated 1 px speck
    expected = np.zeros((9, 12), bool)
    expected[2:7, 2:7] = True
    expected[2, 2] = expected[2, 6] = expected[6, 2] = expected[6, 6] = False  # ellipse rounds the corners
    assert np.array_equal(masks.opening(m, 1), expected)


def test_union_merges_concepts_and_ignores_missing() -> None:
    a = np.zeros((10, 10), bool)
    a[1:3, 1:3] = True
    b = np.zeros((10, 10), bool)
    b[5:7, 5:7] = True
    out = masks.union({"a": [a], "b": [b]}, ["a", "missing"], (10, 10))
    assert np.array_equal(out, a)


def test_centroid_of_rectangle_is_its_middle() -> None:
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    np.testing.assert_allclose(masks.centroid(m), (9.5, 9.5), rtol=0, atol=1e-9)
    np.testing.assert_allclose(masks.centroid(np.zeros((20, 20), bool)), (0.0, 0.0), rtol=0, atol=0)


def test_unit_and_distance_on_small_vectors() -> None:
    np.testing.assert_allclose(masks.unit(np.array([3.0, 4.0])), (0.6, 0.8), rtol=0, atol=1e-12)
    np.testing.assert_allclose(masks.unit(np.array([0.0, 0.0])), (1.0, 0.0), rtol=0, atol=0)
    assert masks.distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_bone_radius_of_a_straight_bar_is_half_its_width() -> None:
    bar = np.zeros((100, 120), bool)
    bar[40:60, 10:110] = True  # 20 px wide, 100 px long
    r = masks.bone_radius(bar, np.array([15.0, 50.0]), np.array([105.0, 50.0]))
    assert 9.0 <= r <= 10.0


def test_elbow_knee_point_of_a_straight_bar_is_its_middle() -> None:
    bar = np.zeros((100, 120), bool)
    bar[40:60, 10:110] = True
    pt = masks.elbow_knee_point(bar, np.array([10.0, 50.0]), np.array([110.0, 50.0]))
    assert abs(pt[0] - 60.0) < 3.0
    assert abs(pt[1] - 50.0) < 2.0


def test_sam_cache_key_is_stable_and_content_addressed(tmp_path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    c = tmp_path / "c.png"
    c.write_bytes(b"other")
    assert masks.sam_cache_key(a) == masks.sam_cache_key(b)
    assert masks.sam_cache_key(a) != masks.sam_cache_key(c)
    assert masks.sam_cache_key(a).startswith("sha256_")
    assert len(masks.sam_cache_key(a)) == len("sha256_") + 16


def test_load_masks_missing_cache_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="SAM cache entry"):
        masks.load_masks("sha256_deadbeefdeadbeef", (20, 20), tmp_path / "cache")


def test_load_masks_skips_wrong_shape_empty_and_missing(tmp_path) -> None:
    key_dir = tmp_path / "cache" / "sha256_test"
    key_dir.mkdir(parents=True)
    (key_dir / "hand.json").write_text(json.dumps({"concept": "hand", "n": 2}), encoding="utf-8")
    Image.fromarray((np.zeros((20, 20), bool) | (np.arange(400).reshape(20, 20) < 50)).astype(np.uint8) * 255).save(
        key_dir / "hand_0.png"
    )
    # hand_1.png is missing and must be skipped.
    (key_dir / "arm.json").write_text(json.dumps({"concept": "arm", "n": 1}), encoding="utf-8")
    Image.fromarray(np.full((10, 10), 255, np.uint8)).save(key_dir / "arm_0.png")  # wrong shape
    (key_dir / "foot.json").write_text(json.dumps({"concept": "foot", "n": 1}), encoding="utf-8")
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(key_dir / "foot_0.png")  # empty
    out = masks.load_masks("sha256_test", (20, 20), tmp_path / "cache")
    assert set(out) == {"hand", "arm", "foot"}
    assert len(out["hand"]) == 1 and int(out["hand"][0].sum()) == 50
    assert out["arm"] == [] and out["foot"] == []
