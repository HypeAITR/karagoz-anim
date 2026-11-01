"""RM-2: frame plan matches the goldens, the experiment, and synthetic checks."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rig_golden

from karagoz.engine.rig import joints, pivot, plan
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
def chained(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("rm2plan")
    ctx = _xy_context(out)
    d = rig_golden.r2() / "d"
    motion = json.loads((d / "hareket_xy.json").read_text(encoding="utf-8"))
    library = json.loads((rig_golden.r2() / "out" / motion["karakter"] / "iskelet.json").read_text(encoding="utf-8"))
    skeleton = joints.build_skeleton(ctx, d / "cache" / "sam", library, motion["karakter"], out / "bardak_katman.png")
    deltoid = pivot.measure_pivots(ctx, skeleton, d / "cache" / "sam")
    planned, frame_plan = plan.build_plan(ctx, skeleton, deltoid, int(motion["kare_sayisi"]))
    return ctx, skeleton, deltoid, planned, frame_plan, out


@pytest.fixture(scope="module")
def exp_planla():
    rig_golden.fixture()
    before_path = list(sys.path)
    before_env = dict(os.environ)
    try:
        os.environ["RIG_ARALIK"] = "xy"  # zincir reads the motion file at import
        os.environ["RIG_CIZGI"] = str(XY_LINE_WIDTH)
        sys.path.insert(0, str(rig_golden.fixture()))
        sys.path.insert(0, str(rig_golden.r2()))
        sys.path.insert(0, str(rig_golden.r2() / "plan"))
        import planla

        return planla
    finally:
        os.environ.clear()
        os.environ.update(before_env)
        sys.path[:] = before_path


def _golden(name: str) -> dict:
    return json.loads((rig_golden.fixture() / "_altin_r13" / name).read_text(encoding="utf-8"))


@pytest.mark.rig_golden
def test_plan_matches_golden_structure_and_numbers(chained) -> None:
    ctx, _skeleton, _deltoid, planned, _frame_plan, _out = chained
    on_disk = json.loads((ctx.output_dir / "plan.json").read_text(encoding="utf-8"))
    assert on_disk == planned  # the file carries the returned dict
    _assert_json_close(on_disk, _golden("xy/plan.json"))


@pytest.mark.rig_golden
def test_frame_plan_standalone_equals_embedded_table(chained) -> None:
    ctx, _skeleton, _deltoid, planned, frame_plan, _out = chained
    on_disk = json.loads((ctx.output_dir / "kare_plani_AB.json").read_text(encoding="utf-8"))
    assert on_disk == frame_plan  # the file carries the returned dict
    assert frame_plan == planned["kare_plani"]  # one run writes both from the same table


@pytest.mark.rig_golden
def test_frame_plan_matches_golden_embedded_table_exactly(chained) -> None:
    # The plan step's contract is the pre-cut table (switch [4]): the golden
    # plan.json's embedded 'kare_plani', compared here with no exempt leaves.
    # The standalone golden kare_plani_AB.json is the post-cut plan (switch
    # [6]): the frame step (cok_kaynak.py main: kesme_sec, then kesme_uygula)
    # overwrites it after planla.py, so RM-4 must reproduce that file instead.
    _ctx, _skeleton, _deltoid, _planned, frame_plan, _out = chained
    golden = _golden("xy/plan.json")["kare_plani"]
    assert frame_plan["degisim_kareleri"] == [4]
    _assert_json_close(frame_plan, golden)


@pytest.mark.rig_golden
def test_trusted_joints_match_experiment(exp_planla, tmp_path: Path) -> None:
    d = rig_golden.r2() / "d"
    motion = json.loads((d / "hareket_xy.json").read_text(encoding="utf-8"))
    ctx = IntervalContext(
        key_a=motion["poz_a"],
        key_b=motion["poz_b"],
        image_a=d / "girdi" / f"{motion['poz_a']}.png",
        image_b=d / "girdi" / f"{motion['poz_b']}.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=d / "hareket_xy.json",
        output_dir=tmp_path,
        line_width=XY_LINE_WIDTH,
    )
    skeleton = _golden("iskelet_xy.json")
    got_a, got_b, got_notes = plan.trusted_joints(ctx, skeleton)
    exp_a, exp_b, exp_notes = exp_planla.guvenli_eklemler(skeleton)
    _assert_json_close(got_a, exp_a, "A")
    _assert_json_close(got_b, exp_b, "B")
    assert got_notes == exp_notes


@pytest.mark.rig_golden
def test_slerp_direction_inside_ik_match_experiment(exp_planla, tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(plan.slerp(ctx, a, b, 0.25), exp_planla.slerp(a, b, 0.25), rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.direction3(ctx, 90.0, 30.0, 1), exp_planla.yon3(90.0, 30.0, 1), rtol=0, atol=1e-12)
    square = [np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([10.0, 10.0]), np.array([0.0, 10.0])]
    assert plan.inside(ctx, np.array([5.0, 5.0]), square) is exp_planla.icinde(np.array([5.0, 5.0]), square)
    shoulder, elbow, wrist = np.array([0.0, 0.0]), np.array([50.0, 0.0]), np.array([100.0, 0.0])
    target = np.array([60.0, 40.0])
    got_el, got_wr = plan.two_bone_ik(ctx, shoulder, elbow, wrist, target)
    exp_el, exp_wr = exp_planla.iki_kemik_ik(shoulder, elbow, wrist, target)
    np.testing.assert_allclose(got_el, exp_el, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_wr, exp_wr, rtol=0, atol=1e-12)


def _joint(x: float, y: float, r: float = 50.0) -> dict:
    return {"x": x, "y": y, "r": r}


def _synthetic_skeleton(key_a: str, key_b: str) -> dict:
    lib = {
        k: _joint(0.0, 0.0, 100.0)
        for k in (
            "neck",
            "shoulder_L",
            "shoulder_R",
            "elbow_L",
            "elbow_R",
            "wrist_L",
            "wrist_R",
            "hip_L",
            "hip_R",
            "knee_L",
            "knee_R",
            "ankle_L",
            "ankle_R",
        )
    }

    def pose(dx: float) -> dict:
        return {
            "neck": _joint(450.0, 430.0),
            "shoulder_L": _joint(340.0, 510.0),
            "shoulder_R": _joint(550.0, 477.0),
            "elbow_L": _joint(280.0 + dx, 635.0),
            "elbow_R": _joint(749.0, 561.0),
            "wrist_L": _joint(410.0 + dx, 725.0),
            "wrist_R": _joint(859.0, 639.0),
            "hand_center_L": {"x": 470.0 + dx, "y": 748.0},
            "hip_L": _joint(367.0, 915.0),
            "hip_R": _joint(586.0, 913.0),
            "knee_L": _joint(267.0, 1079.0),
            "knee_R": _joint(675.0, 1018.0),
            "ankle_L": _joint(185.0, 1228.0),
            "ankle_R": _joint(677.0, 1192.0),
        }

    return {"kutuphane_iskelet": lib, "pozlar": {key_a: {"eklemler": pose(0.0)}, key_b: {"eklemler": pose(60.0)}}}


def test_inside_reports_square_membership(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    square = [np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([10.0, 10.0]), np.array([0.0, 10.0])]
    assert plan.inside(ctx, np.array([5.0, 5.0]), square) is True
    assert plan.inside(ctx, np.array([15.0, 5.0]), square) is False


def test_direction3_points_down_up_and_sideways(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    np.testing.assert_allclose(plan.direction3(ctx, 0.0, 0.0, 1), (0.0, 1.0, 0.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.direction3(ctx, 90.0, 0.0, 1), (1.0, 0.0, 0.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.direction3(ctx, 0.0, 90.0, 1), (0.0, 0.0, 1.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.direction3(ctx, 0.0, 90.0, -1), (0.0, 0.0, -1.0), rtol=0, atol=1e-12)


def test_slerp_hits_both_ends_and_the_middle(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(plan.slerp(ctx, a, b, 0.0), a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.slerp(ctx, a, b, 1.0), b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.slerp(ctx, a, b, 0.5), (math.sqrt(0.5), math.sqrt(0.5), 0.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.slerp(ctx, a, a, 0.3), a, rtol=0, atol=0)


def test_two_bone_ik_keeps_bone_lengths_and_reaches(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    elbow, wrist = plan.two_bone_ik(
        ctx, np.array([0.0, 0.0]), np.array([50.0, 0.0]), np.array([100.0, 0.0]), np.array([60.0, 0.0])
    )
    np.testing.assert_allclose(wrist, (60.0, 0.0), rtol=0, atol=1e-9)
    assert math.hypot(*(elbow - (0.0, 0.0))) == pytest.approx(50.0)
    assert math.hypot(*(wrist - elbow)) == pytest.approx(50.0)


def test_trusted_joints_inherit_a_covered_joint(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    skeleton = _synthetic_skeleton("a", "b")
    skeleton["pozlar"]["a"]["eklemler"]["elbow_L"]["r"] = 10.0  # below 40% of the library's 100
    a, _b, notes = plan.trusted_joints(ctx, skeleton)
    assert a["elbow_L"] == skeleton["pozlar"]["b"]["eklemler"]["elbow_L"]
    assert len(notes) == 1 and "A.elbow_L" in notes[0]


def test_build_plan_runs_on_a_synthetic_interval(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    ctx = IntervalContext(
        key_a="a",
        key_b="b",
        image_a=a,
        image_b=b,
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path / "out",
    )
    skeleton = _synthetic_skeleton("a", "b")
    skeleton["kaynak_ozetleri"] = ctx.source_digests()
    planned, frame_plan = plan.build_plan(ctx, skeleton, None, 3)
    assert planned["kare_sayisi"] == 3
    assert set(planned["kipler"]) == {"M0", "M1", "M2"}
    assert frame_plan["anahtarlar"] == [0, 2]
    assert [row["kare"] for row in frame_plan["kareler"]] == [0, 1, 2]
    assert (tmp_path / "out" / "plan.json").is_file()
    assert (tmp_path / "out" / "kare_plani_AB.json").is_file()
    timed, timed_frame = plan.build_plan(ctx, skeleton, None, 3, times=[0.0, 0.5, 1.0])
    assert timed_frame["t_kare"] == [0.0, 0.5, 1.0]
    assert "t_kare" not in frame_plan
    assert timed["kare_plani"]["t_kare"] == [0.0, 0.5, 1.0]
