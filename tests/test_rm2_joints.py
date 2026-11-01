"""RM-2: pose joints match the golden skeleton, the experiment, and synthetic checks."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import joints
from karagoz.engine.rig.interval import IntervalContext

XY_LINE_WIDTH = 2.8  # panel line width p50 the experiment measured (plan/aralik_kur.py CIZGI)


def _xy_context(out_dir: Path) -> IntervalContext:
    d = rig_golden.r2() / "d"
    motion = json.loads((d / "hareket_xy.json").read_text(encoding="utf-8"))
    return IntervalContext(
        key_a=motion["poz_a"],
        key_b=motion["poz_b"],
        image_a=d / "girdi" / f"{motion['poz_a']}.png",
        image_b=d / "girdi" / f"{motion['poz_b']}.png",
        skeleton_path=out_dir / "iskelet_xy.json",
        motion_path=d / "hareket_xy.json",
        output_dir=out_dir,
        line_width=XY_LINE_WIDTH,
    )


def _dummy_ctx(tmp_path: Path) -> IntervalContext:
    return IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
    )


def _assert_json_close(got, exp, path: str = "root") -> None:
    if isinstance(exp, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(exp), f"{path}: key sets differ by {set(got) ^ set(exp)}"
        for k in exp:
            _assert_json_close(got[k], exp[k], f"{path}.{k}")
    elif isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), f"{path}: list lengths differ"
        for i, (g, e) in enumerate(zip(got, exp, strict=True)):
            _assert_json_close(g, e, f"{path}[{i}]")
    elif isinstance(exp, bool) or isinstance(got, bool):
        assert got is exp, f"{path}: {got!r} vs {exp!r}"
    elif isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9), f"{path}: {got} vs {exp}"
    else:
        assert type(got) is type(exp) and got == exp, f"{path}: {got!r} vs {exp!r}"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("rm2joints")
    ctx = _xy_context(out)
    d = rig_golden.r2() / "d"
    motion = json.loads((d / "hareket_xy.json").read_text(encoding="utf-8"))
    library = json.loads((rig_golden.r2() / "out" / motion["karakter"] / "iskelet.json").read_text(encoding="utf-8"))
    skeleton = joints.build_skeleton(ctx, d / "cache" / "sam", library, motion["karakter"], out / "bardak_katman.png")
    return ctx, skeleton, out


@pytest.fixture(scope="module")
def exp_eklem():
    root = rig_golden.fixture()
    before_path = list(sys.path)
    cwd = os.getcwd()
    try:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "deneme" / "r2"))
        sys.path.insert(0, str(root / "deneme" / "r2" / "d"))
        import eklem_olc

        return eklem_olc
    finally:
        os.chdir(cwd)  # the experiment import chdirs into the fixture
        sys.path[:] = before_path


@pytest.fixture(scope="module")
def exp_measured(exp_eklem, tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("rm2joints_exp")
    d = rig_golden.r2() / "d"
    motion = json.loads((d / "hareket_xy.json").read_text(encoding="utf-8"))
    measured = {
        name: exp_eklem.poz_eklem_olc(name, d / "girdi" / f"{name}.png", d / "cache" / "sam", {})
        for name in (motion["poz_a"], motion["poz_b"])
    }
    # The experiment relativises the layer path to its own tree; point that cosmetic constant at the sandbox.
    old_r2 = exp_eklem.R2_DIR
    exp_eklem.R2_DIR = out
    try:
        info = exp_eklem.bardak_katmani_cikar(measured[motion["poz_a"]], measured[motion["poz_b"]], out)
    finally:
        exp_eklem.R2_DIR = old_r2
    return measured, info, out


def _golden_skeleton() -> dict:
    return json.loads((rig_golden.fixture() / "_altin_r13" / "iskelet_xy.json").read_text(encoding="utf-8"))


@pytest.mark.rig_golden
def test_skeleton_json_matches_golden_structure_and_numbers(built) -> None:
    ctx, skeleton, _out = built
    on_disk = json.loads(ctx.skeleton_path.read_text(encoding="utf-8"))
    assert on_disk == skeleton  # the file carries the returned dict
    golden = _golden_skeleton()
    assert set(on_disk) == set(golden)
    for key in golden:
        if key == "bardak_katman":
            continue
        _assert_json_close(on_disk[key], golden[key], key)
    for key in golden["bardak_katman"]:
        if key == "dosya":  # golden path is experiment-relative; ours is the explicit output path
            continue
        _assert_json_close(on_disk["bardak_katman"][key], golden["bardak_katman"][key], f"bardak_katman.{key}")
    assert Path(on_disk["bardak_katman"]["dosya"]).name == "bardak_katman.png"
    assert Path(on_disk["bardak_katman"]["dosya"]).is_file()


@pytest.mark.rig_golden
def test_pose_joints_match_experiment_both_keys(built, exp_measured) -> None:
    _ctx, skeleton, _out = built
    measured, _info, _exp_out = exp_measured
    for name, exp in measured.items():
        got = skeleton["pozlar"][name]
        for key, value in exp.items():
            if key.startswith("_"):
                continue
            _assert_json_close(got[key], value, f"{name}.{key}")


@pytest.mark.rig_golden
def test_joint_masks_match_experiment_both_keys(built, exp_eklem, exp_measured) -> None:
    ctx, _skeleton, _out = built
    measured, _info, _exp_out = exp_measured
    d = rig_golden.r2() / "d"
    for name in (ctx.key_a, ctx.key_b):
        got = joints.measure_pose_joints(ctx, name, d / "girdi" / f"{name}.png", d / "cache" / "sam")
        for key in ("_bardak_mask", "_hand_mask", "_rgb"):
            assert np.array_equal(got[key], measured[name][key]), f"{name}.{key}"


@pytest.mark.rig_golden
def test_cup_layer_matches_experiment_file_and_pixels(built, exp_measured) -> None:
    _ctx, skeleton, out = built
    _measured, exp_info, exp_out = exp_measured
    got_info = skeleton["bardak_katman"]
    assert set(got_info) == set(exp_info)
    for key in exp_info:
        if key == "dosya":
            continue
        _assert_json_close(got_info[key], exp_info[key], f"bardak_katman.{key}")
    got_rgba = np.array(Image.open(out / "bardak_katman.png").convert("RGBA"))
    exp_rgba = np.array(Image.open(exp_out / "bardak_katman.png").convert("RGBA"))
    assert np.array_equal(got_rgba, exp_rgba)


@pytest.mark.rig_golden
def test_contact_center_matches_experiment(exp_eklem, tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    a = np.zeros((60, 60), bool)
    a[10:30, 10:30] = True
    b = np.zeros((60, 60), bool)
    b[25:45, 15:35] = True
    got_pt, got_gap = joints.contact_center(ctx, a, b)
    exp_pt, exp_gap = exp_eklem.temas_merkezi(a, b)
    np.testing.assert_allclose(got_pt, exp_pt, rtol=0, atol=1e-9)
    assert math.isclose(got_gap, exp_gap, rel_tol=0, abs_tol=1e-9)


@pytest.mark.rig_golden
def test_dome_center_matches_experiment(exp_eklem, tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    yy, xx = np.ogrid[:140, :140]
    upper = (xx - 70) ** 2 + (yy - 70) ** 2 <= 45**2
    silhouette = upper.copy()
    got = joints.dome_center(ctx, upper, silhouette)
    exp = exp_eklem.kubbe_merkezi(upper, silhouette)
    assert (got is None) == (exp is None)
    assert got is not None and exp is not None
    np.testing.assert_allclose(got[0], exp[0], rtol=0, atol=1e-9)
    assert math.isclose(got[1], exp[1], rel_tol=0, abs_tol=1e-9)


@pytest.mark.rig_golden
def test_forearm_near_end_matches_experiment(exp_eklem, tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    bar = np.zeros((120, 200), bool)
    bar[50:70, 20:180] = True
    wrist = np.array([25.0, 60.0])
    got = joints.forearm_near_end(ctx, bar, wrist)
    exp = exp_eklem.on_kol_yakin_ucu(bar, wrist)
    assert got is not None and exp is not None
    np.testing.assert_allclose(got, exp, rtol=0, atol=1e-9)


def test_angle_diff_wraps_to_half_circle(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    assert joints.angle_diff_deg(ctx, 170.0, -170.0) == pytest.approx(20.0)
    assert joints.angle_diff_deg(ctx, -170.0, 170.0) == pytest.approx(-20.0)
    assert joints.angle_diff_deg(ctx, 10.0, 30.0) == pytest.approx(20.0)


def test_degrees_from_down_and_std_on_cardinal_points(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    assert joints.degrees_from_down(ctx, (0.0, 0.0), (0.0, 10.0)) == pytest.approx(0.0)
    assert joints.degrees_from_down(ctx, (0.0, 0.0), (10.0, 0.0)) == pytest.approx(90.0)
    assert joints.degrees_std(ctx, (0.0, 0.0), (10.0, 0.0)) == pytest.approx(0.0)
    assert joints.degrees_std(ctx, (0.0, 0.0), (0.0, 10.0)) == pytest.approx(90.0)


def test_largest_component_picks_biggest_and_drops_specks(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    m = np.zeros((100, 100), bool)
    m[5:15, 5:15] = True  # 100 px
    m[50:70, 50:70] = True  # 400 px
    m[90, 90] = True  # 1 px speck
    assert int(joints.largest_component(ctx, m).sum()) == 400
    assert int(joints.largest_component(ctx, m, min_area=1000).sum()) == 0
    assert not joints.largest_component(ctx, np.zeros((10, 10), bool)).any()


def test_merge_masks_unions_and_empty_list_stays_empty(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    a = np.zeros((10, 10), bool)
    a[1:3, 1:3] = True
    b = np.zeros((10, 10), bool)
    b[5:7, 5:7] = True
    assert np.array_equal(joints.merge_masks(ctx, [a, b], (10, 10)), a | b)
    assert not joints.merge_masks(ctx, [], (10, 10)).any()


def test_contact_center_without_overlap_reports_no_contact(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    pt, gap = joints.contact_center(ctx, np.zeros((10, 10), bool), np.zeros((10, 10), bool))
    assert gap == 1e9
    np.testing.assert_allclose(pt, (0.0, 0.0), rtol=0, atol=0)


def test_dome_and_forearm_end_give_up_on_tiny_masks(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    tiny = np.zeros((30, 30), bool)
    tiny[5:8, 5:8] = True
    assert joints.dome_center(ctx, tiny, tiny) is None
    assert joints.forearm_near_end(ctx, tiny, np.array([5.0, 5.0])) is None


def test_build_skeleton_without_cache_raises(tmp_path: Path) -> None:
    for name in ("a.png", "b.png"):
        Image.fromarray(np.full((40, 40, 3), 255, np.uint8)).save(tmp_path / name)
    ctx = IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="SAM cache entry"):
        joints.build_skeleton(ctx, tmp_path / "cache", {}, "recep", tmp_path / "cup.png")
