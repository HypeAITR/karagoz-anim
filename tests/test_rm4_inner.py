"""RM-4: ported inner functions match the experiment's functions on the same inputs.

The experiment modules are imported read-only from the fixture (``RIG_ARALIK=xy``, ``RIG_CIZGI=2.8``,
the golden run's settings); nothing is written back.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rig_golden

from karagoz.engine.rig import canonical, fill, frames, head, layers
from karagoz.engine.rig.interval import IntervalContext

XY_LINE_WIDTH = 2.8


@pytest.fixture(scope="module")
def exp_modules():
    root = rig_golden.fixture()
    os.environ["RIG_ARALIK"] = "xy"
    os.environ["RIG_CIZGI"] = "2.8"
    for p in ("deneme/r2/plan", "deneme/r2", "deneme/r2/d", ""):
        sys.path.insert(0, str(root / p) if p else str(root))
    import cok_kaynak
    import kafa
    import kanonik
    import rig_kare

    return kanonik, cok_kaynak, kafa, rig_kare


def _ctx(tmp_path: Path) -> IntervalContext:
    return IntervalContext(
        key_a="poz_X",
        key_b="poz_Y",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
        line_width=XY_LINE_WIDTH,
    )


@pytest.mark.rig_golden
def test_fill_cache_name_matches_golden_filename() -> None:
    golden = sorted((rig_golden.fixture() / "_altin_r13" / "xy" / "kanonik").glob("muse_kolsuz_*.png"))
    d = rig_golden.r2() / "d"
    assert fill.cache_path("x", "A", d / "girdi" / "poz_X.png", canonical.PROMPT_ARMLESS_A).name == golden[0].name
    assert fill.cache_path("x", "B", d / "girdi" / "poz_Y.png", canonical.PROMPT_ARMLESS_B).name == golden[1].name


@pytest.mark.rig_golden
def test_background_matches_experiment(exp_modules) -> None:
    kanonik, _ck, _kafa, _rk = exp_modules
    rgb = np.full((60, 60, 3), 255, np.uint8)
    rgb[20:40, 20:40] = (240, 240, 240)  # light pocket, enclosed
    rgb[19:41, 19:20] = (10, 10, 10)
    rgb[19:41, 40:41] = (10, 10, 10)
    rgb[19:20, 19:41] = (10, 10, 10)
    rgb[40:41, 19:41] = (10, 10, 10)
    for limit in (236, 251):
        assert np.array_equal(canonical.background(rgb, limit), kanonik.arka_plan(rgb, limit))


@pytest.mark.rig_golden
def test_expand_contour_matches_experiment(exp_modules, tmp_path: Path) -> None:
    kanonik, _ck, _kafa, _rk = exp_modules
    ctx = _ctx(tmp_path)
    rgb = np.full((80, 80, 3), 255, np.uint8)
    rgb[30:50, 10:70] = (200, 120, 110)  # skin band
    rgb[28:30, 10:70] = (15, 15, 18)  # line above
    rgb[50:52, 10:70] = (15, 15, 18)  # line below
    mask = np.zeros((80, 80), bool)
    mask[32:48, 15:65] = True
    assert np.array_equal(canonical.expand_contour(ctx, mask, rgb), kanonik.kontur_genislet(mask, rgb))


@pytest.mark.rig_golden
def test_pure_cup_matches_experiment(exp_modules, tmp_path: Path) -> None:
    kanonik, _ck, _kafa, _rk = exp_modules
    ctx = _ctx(tmp_path)
    rgb = np.full((70, 70, 3), 255, np.uint8)
    rgb[20:50, 25:45] = (150, 200, 210)  # cup glass
    rgb[20:50, 25:27] = (15, 15, 18)
    rgb[20:50, 43:45] = (15, 15, 18)
    cup = np.zeros((70, 70), bool)
    cup[18:52, 20:50] = True  # SAM overshoot past the rim
    assert np.array_equal(canonical.pure_cup(ctx, cup, rgb), kanonik.bardak_saf(cup, rgb))


@pytest.mark.rig_golden
def test_head_matrices_match_experiment(exp_modules) -> None:
    _kanonik, _ck, kafa, _rk = exp_modules
    data = {"boyun_a": [460.2, 417.1], "oteleme": [-21.0, -5.0], "derece": -6.25}
    neck = np.array(data["boyun_a"], float)
    shift = np.array(data["oteleme"], float)
    np.testing.assert_array_equal(head.matrix(neck, shift, -6.25), kafa.matris(neck, shift, -6.25))
    for t, from_b in ((0.0, False), (0.35, False), (1.0, False), (0.35, True)):
        np.testing.assert_array_equal(head.frame_matrix(data, t, from_b), kafa.kare_matrisi(data, t, from_b))
    np.testing.assert_array_equal(head.mid_matrix(data, 0.4, 0.7), kafa.ara_matrisi(data, 0.4, 0.7))


@pytest.mark.rig_golden
def test_affine_matrix_matches_experiment(exp_modules) -> None:
    _kanonik, _ck, _kafa, rk = exp_modules
    cases = [
        ((100.0, 200.0), (110.0, 205.0), 12.5, 1.0, (1.0, 0.0)),
        ((340.7, 512.3), (313.0, 507.4), -25.9, 0.9476, (0.83, 0.55)),
        ((0.0, 0.0), (0.0, 0.0), 0.0, 1.0, (1.0, 0.0)),
    ]
    for p0, pk, phi, scale, axis in cases:
        np.testing.assert_array_equal(
            layers.affine_matrix(p0, pk, phi, scale, axis), rk.afin_matris(p0, pk, phi, scale, axis)
        )
    assert layers.is_identity_tf(cases[2]) == rk.kimlik_tf_mi(cases[2])
    assert layers.is_identity_tf(cases[1]) == rk.kimlik_tf_mi(cases[1])


@pytest.mark.rig_golden
def test_time_curve_matches_experiment(exp_modules) -> None:
    _kanonik, ck, _kafa, _rk = exp_modules
    plan = {"kareler": [{"kare": i} for i in range(8)]}
    got, exp = frames.time_curve(plan), ck.zaman_egrisi(plan)
    assert [got(i) for i in range(8)] == [exp(i) for i in range(8)]
    chained = {**plan, "t_kare": [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]}
    got, exp = frames.time_curve(chained), ck.zaman_egrisi(chained)
    assert [got(i) for i in range(8)] == [exp(i) for i in range(8)]


@pytest.mark.rig_golden
def test_shape_jump_matches_experiment(exp_modules, tmp_path: Path) -> None:
    _kanonik, ck, _kafa, _rk = exp_modules
    ctx = _ctx(tmp_path)
    a = np.full((120, 120, 3), 255, np.uint8)
    a[40:80, 30:90] = (20, 20, 22)
    b = a.copy()
    b[45:85, 35:95] = (20, 20, 22)
    assert frames.shape_jump(ctx, a, b, (60.0, 60.0)) == ck.bicim_sicramasi(a, b, (60.0, 60.0))
    assert frames.shape_jump(ctx, a, b, (60.0, 60.0), ctx.px(90)) == ck.bicim_sicramasi(
        a, b, (60.0, 60.0), ck.zincir.px(90)
    )
    assert frames.shape_jump(ctx, a, a, (60.0, 60.0)) == 0.0
    white = np.full((120, 120, 3), 255, np.uint8)
    assert math.isinf(frames.shape_jump(ctx, a, white, (60.0, 60.0)))


def _toy_plan() -> dict:
    joints_a = {
        "shoulder_L": [100.0, 100.0],
        "elbow_L": [150.0, 140.0],
        "wrist_L": [190.0, 190.0],
        "hand_center_L": [205.0, 210.0],
    }
    joints_b = {
        "shoulder_L": [102.0, 99.0],
        "elbow_L": [170.0, 130.0],
        "wrist_L": [230.0, 150.0],
        "hand_center_L": [248.0, 160.0],
    }

    def bones(e):
        out = {}
        for b, (j0, j1) in (
            ("upper_L", ("shoulder_L", "elbow_L")),
            ("fore_L", ("elbow_L", "wrist_L")),
            ("hand_L", ("wrist_L", "hand_center_L")),
        ):
            out[b] = round(float(np.hypot(e[j1][0] - e[j0][0], e[j1][1] - e[j0][1])), 1)
        return out

    frames_list = []
    for i in range(4):
        t = i / 3
        e = {
            j: [round(a * (1 - t) + c * t, 1) for a, c in zip(joints_a[j], joints_b[j], strict=True)] for j in joints_a
        }
        frames_list.append(
            {
                "kare": i,
                "kaynak": "A" if i < 2 else "B",
                "hedef_eklemler": e,
                "vekil_boy": {b: round(v * (1 + 0.05 * t), 2) for b, v in bones(joints_a).items()},
            }
        )
    return {
        "olcek_siniri": 0.15,
        "kaynaklar": {
            "A": {"kare": 0, "eklemler": joints_a, "boy": bones(joints_a)},
            "B": {"kare": 3, "eklemler": joints_b, "boy": bones(joints_b)},
        },
        "kareler": frames_list,
    }


@pytest.mark.rig_golden
def test_rekey_apply_cut_and_strain_match_experiment(exp_modules) -> None:
    _kanonik, ck, _kafa, _rk = exp_modules
    plan = _toy_plan()
    for k in plan["kareler"]:
        for name in ("A", "B"):
            assert frames.rekey(plan, k, name) == ck.kaynakla(plan, k, name)
    assert frames.apply_cut(plan, [2]) == ck.kesme_uygula(plan, [2])
    assert frames.joint_strain(plan, plan["kareler"], 0, 3, 2, "A", "B") == ck.eklem_sapmasi(
        plan, plan["kareler"], 0, 3, 2, "A", "B"
    )


@pytest.mark.rig_golden
def test_load_source_matches_experiment(exp_modules, tmp_path: Path) -> None:
    _kanonik, ck, _kafa, _rk = exp_modules
    ctx = _ctx(tmp_path)
    golden = rig_golden.fixture() / "_altin_r13" / "xy"
    plan = json.loads((golden / "kare_plani_AB.json").read_text(encoding="utf-8"))
    deltoid = json.loads((golden / "deltoid.json").read_text(encoding="utf-8"))
    for name in ("poz_X", "poz_Y"):
        # Both sides read the experiment's own canonical folder so the inputs are identical.
        got = canonical.load_source(
            ctx, name, plan["kaynaklar"][name]["eklemler"], golden / "kanonik", float(deltoid["kubbe_yaricap"])
        )
        exp = ck.kaynak_yukle(name, plan["kaynaklar"][name]["eklemler"])
        assert set(got["parcalar"]) == set(exp["parcalar"])
        for b, piece in got["parcalar"].items():
            assert np.array_equal(piece["mask"], exp["parcalar"][b]["mask"]), f"{name}.{b}.mask"
            assert np.array_equal(piece["rgb"], exp["parcalar"][b]["rgb"]), f"{name}.{b}.rgb"
            assert np.array_equal(piece["alfa"], exp["parcalar"][b]["alfa"]), f"{name}.{b}.alfa"
            assert np.array_equal(piece["kesik_bant"], exp["parcalar"][b]["kesik_bant"]), f"{name}.{b}.kesik_bant"
        assert got["kapaklar"] == exp["kapaklar"]
        np.testing.assert_array_equal(got["kubbe_olcu"], exp["kubbe_olcu"])
        assert np.array_equal(got["tam"], exp["tam"])
        assert np.array_equal(got["kivrim"], exp["kivrim"])
        assert np.array_equal(got["izinli"], exp["izinli"])


@pytest.mark.rig_golden
def test_frame_transforms_match_experiment(exp_modules, tmp_path: Path) -> None:
    _kanonik, ck, _kafa, _rk = exp_modules
    ctx = _ctx(tmp_path)
    golden = rig_golden.fixture() / "_altin_r13" / "xy"
    plan = json.loads((golden / "kare_plani_AB.json").read_text(encoding="utf-8"))
    # Transforms never touch the bend object or head layer; None keeps this test fast.
    setup = frames.FrameSetup(
        ctx=ctx, plan=plan, canonical_dir=golden / "kanonik", dome_radius=87.7, bend_obj=None, head_layer=None
    )
    for k in plan["kareler"]:
        got = frames.frame_transforms(setup, k)
        exp = ck.kare_donusumleri(plan, k)
        assert set(got) == set(exp)
        for b in got:
            for g, e in zip(got[b], exp[b], strict=True):
                assert np.array_equal(np.asarray(g, float), np.asarray(e, float)), f"k{k['kare']}.{b}"
