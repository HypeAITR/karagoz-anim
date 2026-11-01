"""RM-3: body bending matches bukulme on the golden X->Y inputs, plus synthetic checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import bend, deform, interval


def _experiment():
    root = rig_golden.fixture()
    os.environ["RIG_ARALIK"] = "xy"
    os.environ["RIG_CIZGI"] = "2.8"
    for p in ("deneme/r2/plan", "deneme/r2", "deneme/r2/d", ""):
        sys.path.insert(0, str(root / p) if p else str(root))
    import ag_deformer
    import bukulme

    return bukulme, ag_deformer


@pytest.fixture(scope="module")
def exp_modules():
    return _experiment()


@pytest.fixture(scope="module")
def exp(exp_modules):
    return exp_modules[0]


@pytest.fixture(scope="module")
def exp_deform(exp_modules):
    return exp_modules[1]


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


def _rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _mask(path: Path) -> np.ndarray:
    return _gray(path) > 127


def _shadow(path: Path) -> np.ndarray:
    g = np.array(Image.open(path).convert("RGBA")).astype(np.float32)
    return deform.premultiplied(g[..., :3], g[..., 3] / 255.0)


@pytest.fixture(scope="module")
def golden(exp, ctx):
    alt = rig_golden.fixture() / "_altin_r13" / "xy"
    kan = alt / "kanonik"
    plan = json.loads((alt / "kare_plani_AB.json").read_text(encoding="utf-8"))
    return {
        "plan": plan,
        "z_a": _rgb(kan / "govde_kolsuz_A.png"),
        "giysi_a": _mask(kan / "giysi_A.png"),
        "disk": _mask(kan / "omuz_disk.png"),
        "region": _mask(kan / "degisen.png"),
        "still": _mask(kan / "degisen_sabit.png"),
        "head": _mask(kan / "poz_X_kafa_alfa.png") | _mask(kan / "poz_Y_kafa_alfa.png"),
        "patch_rgb": _rgb(kan / "poz_Y_yama_rgb.png"),
        "patch_alpha": cv2.GaussianBlur(_mask(kan / "poz_Y_yama.png").astype(np.float32), (0, 0), exp.YAMA_YUMUSAK),
        "patch_garment": _mask(kan / "poz_Y_yama_giysi.png"),
        "seam": json.loads((kan / "poz_Y_yama_dikis.json").read_text(encoding="utf-8")),
        "shadow_a": _shadow(kan / "poz_X_golge.png"),
        "shadow_b": _shadow(kan / "poz_Y_golge.png"),
        "far_shadow_rgba": np.array(Image.open(kan / "poz_X_golge_uzak.png").convert("RGBA")),
        "arm_casing_gray": _gray(kan / "poz_Y_kol_zarf.png"),
        "gap_gray": _gray(kan / "bosluk_A.png"),
        "opening_gray": _gray(kan / "poz_X_acilan.png"),
        "matched": json.loads((alt / "r7" / "eslesme.json").read_text(encoding="utf-8"))["noktalar"],
        "neck": json.loads((alt / "r9" / "kafa.json").read_text(encoding="utf-8")),
        "blocks": json.loads((alt / "r9" / "alan_noktalari.json").read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def exp_bend(exp, golden):
    return exp.Bukulme(
        golden["plan"],
        golden["z_a"],
        golden["giysi_a"],
        golden["disk"],
        golden["patch_rgb"],
        golden["patch_alpha"],
        golden["patch_garment"],
        golden["seam"],
        golge_a=golden["shadow_a"],
        golge_b=golden["shadow_b"],
        ara=None,
    )


@pytest.fixture(scope="module")
def port_bend(ctx, golden):
    return bend.Bend(
        ctx,
        golden["plan"],
        golden["z_a"],
        golden["giysi_a"],
        golden["disk"],
        golden["region"],
        patch_rgb=golden["patch_rgb"],
        patch_alpha=golden["patch_alpha"],
        patch_garment=golden["patch_garment"],
        seam=golden["seam"],
        shadow_a=golden["shadow_a"],
        shadow_b=golden["shadow_b"],
        still=golden["still"],
        head=golden["head"],
        far_shadow_rgba=golden["far_shadow_rgba"],
        arm_casing_gray=golden["arm_casing_gray"],
        gap_gray=golden["gap_gray"],
        opening_gray=golden["opening_gray"],
        matched_points=golden["matched"],
        neck=golden["neck"],
        blocks=golden["blocks"],
    )


def _assert_control_points_equal(got, want) -> None:
    assert set(got) == set(want)
    for key in ("a", "b", "pivot_a", "pivot_b"):
        assert np.array_equal(got[key], want[key]), key
    assert np.array_equal(got["hareketli"], want["hareketli"])
    if "k" in want:
        assert np.array_equal(got["k"], want["k"])
        assert got["t_k"] == want["t_k"]


@pytest.mark.rig_golden
def test_read_midpoints_none_for_two_sources(golden, tmp_path: Path) -> None:
    assert bend.read_midpoints(golden["plan"], tmp_path / "missing.json") is None


@pytest.mark.rig_golden
def test_read_midpoints_returns_mid_for_three_sources(tmp_path: Path) -> None:
    mid = {"t": 0.5, "noktalar": {"aski_dis": [330.0, 417.0]}}
    path = tmp_path / "ara.json"
    path.write_text(json.dumps(mid), encoding="utf-8")
    plan = {"kaynaklar": {"a": {"kare": 0}, "k": {"kare": 4}, "b": {"kare": 7}}}
    assert bend.read_midpoints(plan, path) == mid


@pytest.mark.rig_golden
def test_collect_points_matches_experiment(exp, golden) -> None:
    got = bend.collect_points(golden["matched"], golden["neck"], golden["blocks"])
    assert got == exp.nokta_sozlugu()
    assert got["boyun"]["a"] == golden["neck"]["boyun_a"]
    assert "blok_1" in got and "deltoid" in got


@pytest.mark.rig_golden
def test_control_points_match_experiment_plain_and_mid(exp, golden) -> None:
    collected = bend.collect_points(golden["matched"], golden["neck"], golden["blocks"])
    _assert_control_points_equal(bend.control_points(collected), exp.noktalar())
    n = bend.control_points(collected)
    assert len(n["a"]) == sum(1 for d in collected.values() if d["tur"] not in ("gorunmuyor", "denetimsiz")) - 1
    mid: dict = {"t": 0.5, "noktalar": {"aski_dis": [330.0, 417.0]}}
    _assert_control_points_equal(bend.control_points(collected, mid), exp.noktalar(mid))


@pytest.mark.rig_golden
def test_target_points_match_experiment_plain_and_mid(exp, golden) -> None:
    collected = bend.collect_points(golden["matched"], golden["neck"], golden["blocks"])
    n = bend.control_points(collected)
    for t in (0.0, 0.35, 1.0):
        assert np.array_equal(bend.target_points(n, t), exp.hedef_noktalar(exp.noktalar(), t))
    mid: dict = {"t": 0.5, "noktalar": {}}
    nm = bend.control_points(collected, mid)
    for t in (0.2, 0.7):
        assert np.array_equal(bend.target_points(nm, t), exp.hedef_noktalar(exp.noktalar(mid), t))


@pytest.mark.rig_golden
def test_head_mask_matches_experiment(exp, golden) -> None:
    shape = golden["z_a"].shape[:2]
    kan = rig_golden.fixture() / "_altin_r13" / "xy" / "kanonik"
    got = bend.head_mask(shape, [_gray(kan / "poz_X_kafa_alfa.png"), _gray(kan / "poz_Y_kafa_alfa.png")])
    assert np.array_equal(got, exp.kafa_maskesi(shape))
    assert got.any()


@pytest.mark.rig_golden
def test_support_map_matches_experiment_without_and_with_still(ctx, exp, golden) -> None:
    assert np.array_equal(bend.support_map(ctx, golden["region"]), exp.destek_haritasi(golden["region"]))
    got = bend.support_map(ctx, golden["region"], golden["still"], golden["head"])
    want = exp.destek_haritasi(golden["region"], golden["still"])
    assert np.array_equal(got, want)
    assert ((got == 0) | ((got > 0) & (got < 1)) | (got == 1)).all()
    assert (got[golden["still"]] == 0).all()


@pytest.mark.rig_golden
def test_inner_map_matches_experiment(ctx, exp, golden) -> None:
    assert np.array_equal(bend.inner_map(ctx, golden["patch_alpha"]), exp.ic_haritasi(golden["patch_alpha"]))


@pytest.mark.rig_golden
def test_shadow_strength_and_arrival_match_experiment(exp) -> None:
    for t, ts, tn in [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.5, 0.0, 1.0), (0.9, 1.0, 0.0)]:
        assert bend.shadow_strength(t, ts, tn) == exp.golge_gucu(t, ts, tn)
    for t in (0.0, 0.15, 0.5, 0.85, 1.0):
        assert bend.shadow_arrival(t) == exp.golge_gelis(t)


@pytest.mark.rig_golden
def test_support_at_matches_experiment_and_clips(ctx, exp, golden) -> None:
    weights = bend.support_map(ctx, golden["region"], golden["still"], golden["head"])
    vertices = np.array([[0.0, 0.0], [480.0, 720.0], [-50.0, 5000.0], [300.0, 600.0]])
    assert np.array_equal(bend.support_at(vertices, weights), exp.destek(vertices, weights))


@pytest.mark.rig_golden
def test_seam_weights_match_experiment(ctx, exp, golden) -> None:
    rng = np.random.default_rng(11)
    vertices = rng.uniform(200, 700, (200, 2))
    pivot = np.array([340.7, 512.3])
    axis = np.array([1.0, 0.0])
    got = bend.seam_weights(ctx, vertices, golden["giysi_a"], pivot, axis, 200.0)
    want = exp.dikis_agirligi(vertices, golden["giysi_a"], pivot, axis, 200.0)
    assert np.array_equal(got, want)
    assert got.min() >= 0.0 and got.max() <= 1.0


@pytest.mark.rig_golden
def test_blended_source_matches_experiment(exp, exp_deform, golden) -> None:
    small = golden["giysi_a"][400:700, 250:550]
    port_mesh = deform.build_mesh(small)
    exp_mesh = exp_deform.ag_kur(small)
    weight = np.linspace(0, 1, len(port_mesh.rest))

    def rigid(frame: int, v: np.ndarray) -> np.ndarray:
        return v + np.array([2.0, 0.0])

    def body(frame: int, v: np.ndarray) -> np.ndarray:
        return v + np.array([2.0, 6.0])

    visible = np.ones(len(port_mesh.tri), bool)
    got_source, got_strength = bend.blended_source(rigid, body, weight, visible, port_mesh)
    want_source, want_strength = exp.yapisik_kaynak(rigid, body, weight, visible, exp_mesh)
    assert np.array_equal(got_source(0, port_mesh.rest), want_source(0, exp_mesh.kose))
    assert got_strength == want_strength


@pytest.mark.rig_golden
def test_mls_ground_source_matches_experiment(ctx, exp, golden) -> None:
    collected = bend.collect_points(golden["matched"], golden["neck"], golden["blocks"])
    n = bend.control_points(collected)
    weights = bend.support_map(ctx, golden["region"], golden["still"], golden["head"])
    progress = bend.progress(golden["plan"])
    got = bend.mls_ground_source(n, progress, weights)
    want = exp.mls_a_kaynak(exp.noktalar(), progress, weights)
    vertices = np.array([[100.0, 100.0], [480.0, 720.0], [340.7, 512.3]])
    assert np.array_equal(got(3, vertices), want(3, vertices))
    assert np.array_equal(got(0, vertices), vertices)  # frame 0 has zero displacement


@pytest.mark.rig_golden
def test_mls_patch_source_matches_experiment_with_and_without_inner(ctx, exp, golden) -> None:
    collected = bend.collect_points(golden["matched"], golden["neck"], golden["blocks"])
    n = bend.control_points(collected)
    progress = bend.progress(golden["plan"])
    inner = bend.inner_map(ctx, golden["patch_alpha"])
    vertices = np.array([[100.0, 100.0], [480.0, 720.0], [340.7, 512.3]])
    got = bend.mls_patch_source(n, progress, golden["seam"], inner, 1.0, "b")
    want = exp.mls_b_kaynak(exp.noktalar(), progress, golden["seam"], inner, 1.0, "b")
    assert np.array_equal(got(3, vertices), want(3, vertices))
    got_plain = bend.mls_patch_source(n, progress)
    want_plain = exp.mls_b_kaynak(exp.noktalar(), progress)
    assert np.array_equal(got_plain(3, vertices), want_plain(3, vertices))


@pytest.mark.rig_golden
def test_progress_matches_experiment_angle_and_shared(exp, golden) -> None:
    assert bend.progress(golden["plan"]) == exp.ilerleme(golden["plan"])
    shared = {"t_kare": [0.0, 0.2, 0.5, 1.0], "kareler": []}
    assert bend.progress(shared) == exp.ilerleme(shared) == [0.0, 0.2, 0.5, 1.0]


@pytest.mark.rig_golden
def test_patch_mask_matches_experiment(ctx, exp, golden) -> None:
    hidden = golden["head"]
    assert np.array_equal(bend.patch_mask(ctx, golden["region"], hidden), exp.yama_maskesi(golden["region"], hidden))


@pytest.mark.rig_golden
def test_seam_constraints_match_experiment(ctx, exp, golden) -> None:
    patch = bend.patch_mask(ctx, golden["region"], golden["head"])
    b_lines = _gray(rig_golden.fixture() / "_altin_r13" / "xy" / "kanonik" / "poz_Y_yama_rgb.png") < 48
    ground_lines = _gray(rig_golden.fixture() / "deneme" / "r2" / "d" / "girdi" / "poz_X.png") < 48
    got = bend.seam_constraints(ctx, patch, golden["head"], b_lines, ground_lines)
    want = exp.dikis_kisitlari(patch, golden["head"], b_lines, ground_lines)
    assert got == want


@pytest.mark.rig_golden
def test_bend_frame_matches_experiment_unpatched_mid(exp_bend, port_bend) -> None:
    got = port_bend.frame(3, False)
    want = exp_bend.kare(3, False)
    assert np.array_equal(got["zemin"], want["zemin"])
    assert np.array_equal(got["giysi"], want["giysi"])
    assert np.array_equal(got["disk"], want["disk"])
    assert np.array_equal(got["pivot"], want["pivot"])
    assert np.array_equal(got["oran"]["a"], want["oran"]["a"])
    assert "b" not in got["oran"] and "b" not in want["oran"]


@pytest.mark.rig_golden
def test_bend_frame_matches_experiment_patched_mid(exp_bend, port_bend) -> None:
    got = port_bend.frame(3, True)
    want = exp_bend.kare(3, True)
    assert np.array_equal(got["zemin"], want["zemin"])
    assert np.array_equal(got["giysi"], want["giysi"])
    assert np.array_equal(got["disk"], want["disk"])
    assert np.array_equal(got["pivot"], want["pivot"])
    assert np.array_equal(got["oran"]["a"], want["oran"]["a"])
    assert np.array_equal(got["oran"]["b"], want["oran"]["b"])


@pytest.mark.rig_golden
def test_bend_accessors_return_ground_and_patch(port_bend) -> None:
    assert port_bend.a_name == "poz_X" and port_bend.b_name == "poz_Y"
    assert port_bend.source_of("poz_X") is port_bend.ground_source
    assert port_bend.source_of("poz_Y") is port_bend.patches["poz_Y"]["kaynak"]
    assert port_bend.mesh_of("poz_X") is port_bend.ground_mesh
    assert port_bend.mesh_of("poz_Y") is port_bend.patches["poz_Y"]["ag"]
    assert port_bend.garment_of("poz_X") is port_bend.garment_raw_a


def _synthetic_ctx(tmp_path: Path) -> interval.IntervalContext:
    return interval.IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
        line_width=2.8,
    )


def test_target_points_still_point_stays_at_rest() -> None:
    n = {
        "a": np.array([[10.0, 20.0], [30.0, 40.0]]),
        "b": np.array([[50.0, 60.0], [70.0, 80.0]]),
        "hareketli": np.array([True, False]),
    }
    np.testing.assert_allclose(bend.target_points(n, 0.5), [[30.0, 40.0], [30.0, 40.0]], rtol=0, atol=0)


def test_shadow_strength_is_one_at_source_and_zero_past_fade() -> None:
    assert bend.shadow_strength(0.0, 0.0, 1.0) == 1.0
    assert bend.shadow_strength(0.2, 0.0, 1.0) == 0.0
    assert bend.shadow_arrival(0.0) == 0.0
    assert bend.shadow_arrival(1.0) == 1.0


def test_patch_mask_drops_small_islands(tmp_path: Path) -> None:
    ctx = _synthetic_ctx(tmp_path)
    region = np.zeros((200, 200), bool)
    region[20:120, 20:120] = True  # 10000 px, kept
    region[150:155, 150:155] = True  # 25 px, dropped
    out = bend.patch_mask(ctx, region, np.zeros((200, 200), bool))
    assert out[20:120, 20:120].all()
    assert not out[150:155, 150:155].any()


def test_progress_passes_shared_t_kare_through() -> None:
    assert bend.progress({"t_kare": [0.0, 1.0], "kareler": []}) == [0.0, 1.0]


def test_read_midpoints_none_without_mid_key() -> None:
    assert bend.read_midpoints({"kaynaklar": {"a": {}, "b": {}}}, Path("nowhere.json")) is None
