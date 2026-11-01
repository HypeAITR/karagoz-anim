"""RM-2: shoulder pivots match the golden deltoid json, the experiment, and synthetic checks."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rig_golden

from karagoz.engine.rig import joints, pivot
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
    out = tmp_path_factory.mktemp("rm2pivot")
    ctx = _xy_context(out)
    d = rig_golden.r2() / "d"
    motion = json.loads((d / "hareket_xy.json").read_text(encoding="utf-8"))
    library = json.loads((rig_golden.r2() / "out" / motion["karakter"] / "iskelet.json").read_text(encoding="utf-8"))
    skeleton = joints.build_skeleton(ctx, d / "cache" / "sam", library, motion["karakter"], out / "bardak_katman.png")
    deltoid = pivot.measure_pivots(ctx, skeleton, d / "cache" / "sam")
    return ctx, skeleton, deltoid, out


@pytest.fixture(scope="module")
def exp_deltoid():
    rig_golden.fixture()
    before_path = list(sys.path)
    before_env = dict(os.environ)
    try:
        os.environ["RIG_ARALIK"] = "xy"  # zincir reads the motion file at import
        os.environ["RIG_CIZGI"] = str(XY_LINE_WIDTH)
        sys.path.insert(0, str(rig_golden.fixture()))
        sys.path.insert(0, str(rig_golden.r2()))
        sys.path.insert(0, str(rig_golden.r2() / "plan"))
        import deltoid

        return deltoid
    finally:
        os.environ.clear()
        os.environ.update(before_env)
        sys.path[:] = before_path


def _golden(name: str) -> dict:
    return json.loads((rig_golden.fixture() / "_altin_r13" / name).read_text(encoding="utf-8"))


@pytest.mark.rig_golden
def test_chained_skeleton_matches_golden(chained) -> None:
    _ctx, skeleton, _deltoid, _out = chained
    golden = _golden("iskelet_xy.json")
    assert set(skeleton) == set(golden)
    for key in golden:
        if key == "bardak_katman":
            continue
        _assert_json_close(skeleton[key], golden[key], key)
    for key in golden["bardak_katman"]:
        if key == "dosya":
            continue
        _assert_json_close(skeleton["bardak_katman"][key], golden["bardak_katman"][key], f"bardak_katman.{key}")


@pytest.mark.rig_golden
def test_deltoid_matches_golden_structure_and_numbers(chained) -> None:
    ctx, _skeleton, deltoid, _out = chained
    on_disk = json.loads((ctx.output_dir / "deltoid.json").read_text(encoding="utf-8"))
    assert on_disk == deltoid  # the file carries the returned dict
    _assert_json_close(on_disk, _golden("xy/deltoid.json"))


@pytest.mark.rig_golden
def test_dome_radii_match_golden_windows(chained) -> None:
    _ctx, _skeleton, deltoid, _out = chained
    golden = _golden("xy/deltoid.json")
    assert deltoid["kubbe_yaricap"] == pytest.approx(golden["kubbe_yaricap"], rel=0, abs=1e-9)
    for name in ("poz_X", "poz_Y"):
        assert deltoid[name]["gecerli"] is True
        _assert_json_close(deltoid["kubbe"][name], golden["kubbe"][name], f"kubbe.{name}")


@pytest.mark.rig_golden
def test_kasa_circle_matches_experiment(exp_deltoid, tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    angles = np.linspace(0.0, 2 * math.pi, 60, endpoint=False)
    xs = 320.0 + 75.0 * np.cos(angles)
    ys = 510.0 + 75.0 * np.sin(angles)
    assert pivot.kasa_circle(ctx, xs, ys) == pytest.approx(exp_deltoid.kasa(xs, ys), rel=0, abs=1e-9)
    wobble = xs + np.sin(3 * angles) * 2.0
    assert pivot.kasa_circle(ctx, wobble, ys) == pytest.approx(exp_deltoid.kasa(wobble, ys), rel=0, abs=1e-9)


@pytest.mark.rig_golden
def test_constrained_fit_matches_experiment(chained, exp_deltoid) -> None:
    ctx, _skeleton, _deltoid, _out = chained
    golden = _golden("xy/deltoid.json")
    d = rig_golden.r2() / "d"
    top = golden["kubbe"][ctx.key_a]["tepe"]
    guess = (top[0], top[1] + golden["kubbe_yaricap"])
    got = pivot.measure_constrained(ctx, ctx.key_a, ctx.image_a, d / "cache" / "sam", guess, golden["kubbe_yaricap"])
    exp = exp_deltoid.sinirli_olc(ctx.key_a, ctx.image_a, guess, golden["kubbe_yaricap"])
    exp.pop("_pts")  # plot-only fitted points; the port does not return them
    assert set(got) == set(exp)
    _assert_json_close(got, exp)


def test_kasa_circle_recovers_a_drawn_circle(tmp_path: Path) -> None:
    ctx = _dummy_ctx(tmp_path)
    angles = np.linspace(0.0, 2 * math.pi, 100, endpoint=False)
    cx, cy, r = pivot.kasa_circle(ctx, 40.0 + 25.0 * np.cos(angles), 50.0 + 25.0 * np.sin(angles))
    assert (cx, cy, r) == pytest.approx((40.0, 50.0, 25.0), rel=0, abs=1e-9)


def test_measure_pivots_rejects_a_stale_skeleton(tmp_path: Path) -> None:
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
        output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="stale"):
        pivot.measure_pivots(ctx, {"kaynak_ozetleri": {"a": "0" * 64, "b": "0" * 64}}, tmp_path / "cache")
