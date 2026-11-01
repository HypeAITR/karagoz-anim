"""RM-1: one interval's context matches the experiment's scale maths, plus synthetic checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import rig_golden

from karagoz.engine.rig.interval import LINE_REF, IntervalContext, file_digest

XY_LINE_WIDTH = 2.8  # panel line width p50 the experiment measured (plan/aralik_kur.py CIZGI)


def _experiment_zincir(monkeypatch: pytest.MonkeyPatch):
    rig_golden.fixture()
    monkeypatch.setenv("RIG_ARALIK", "xy")
    monkeypatch.setenv("RIG_CIZGI", str(XY_LINE_WIDTH))
    sys.path.insert(0, str(rig_golden.r2() / "plan"))
    import zincir

    assert os.environ["RIG_ARALIK"] == "xy"  # the import reads these; same values every run
    return zincir


def _xy_context(tmp_path: Path) -> IntervalContext:
    d = rig_golden.r2() / "d"
    motion_path = d / "hareket_xy.json"
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    return IntervalContext(
        key_a=motion["poz_a"],
        key_b=motion["poz_b"],
        image_a=d / "girdi" / f"{motion['poz_a']}.png",
        image_b=d / "girdi" / f"{motion['poz_b']}.png",
        skeleton_path=d / "iskelet_xy.json",
        motion_path=motion_path,
        output_dir=tmp_path,
        line_width=XY_LINE_WIDTH,
    )


@pytest.mark.rig_golden
def test_scale_maths_match_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zincir = _experiment_zincir(monkeypatch)
    ctx = _xy_context(tmp_path)
    assert ctx.scale == pytest.approx(zincir.OLCEK)
    for n in (0.0, 0.5, 1.0, 4.39, 18.0, 100.0):
        assert ctx.px(n) == zincir.px(n)
        assert ctx.odd(n) == zincir.tek(n)
        assert ctx.area(n) == zincir.alan(n)


@pytest.mark.rig_golden
def test_file_digest_matches_experiment_on_both_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zincir = _experiment_zincir(monkeypatch)
    ctx = _xy_context(tmp_path)
    assert file_digest(ctx.image_a) == zincir.ozet(ctx.image_a)
    assert file_digest(ctx.image_b) == zincir.ozet(ctx.image_b)


@pytest.mark.rig_golden
def test_source_digests_match_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zincir = _experiment_zincir(monkeypatch)
    assert _xy_context(tmp_path).source_digests() == zincir.kaynak_ozetleri()


@pytest.mark.rig_golden
def test_verify_matches_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zincir = _experiment_zincir(monkeypatch)
    ctx = _xy_context(tmp_path)
    fresh = {"kaynak_ozetleri": ctx.source_digests()}
    zincir.dogrula(fresh, "test")  # both accept fresh derivatives
    ctx.verify(fresh, "test")
    stale = {"kaynak_ozetleri": {"poz_X": "0" * 64, "poz_Y": "0" * 64}}
    with pytest.raises(RuntimeError):
        zincir.dogrula(stale, "test")
    with pytest.raises(RuntimeError, match="stale"):
        ctx.verify(stale, "test")


def test_reference_width_gives_unit_scale(tmp_path: Path) -> None:
    ctx = IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
    )
    assert ctx.line_width == LINE_REF
    assert ctx.scale == 1.0
    assert ctx.px(5) == 5
    assert ctx.odd(4) == 5
    assert ctx.area(9) == 9


def test_px_clamps_to_one_and_odd_stays_odd(tmp_path: Path) -> None:
    ctx = IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
        line_width=LINE_REF / 10,
    )
    assert ctx.px(0) == 1
    assert ctx.px(1) == 1
    for n in (0.0, 1.0, 2.0, 7.0, 20.0):
        assert ctx.odd(n) % 2 == 1


def test_area_uses_the_square_of_the_scale(tmp_path: Path) -> None:
    ctx = IntervalContext(
        key_a="a",
        key_b="b",
        image_a=tmp_path / "a.png",
        image_b=tmp_path / "b.png",
        skeleton_path=tmp_path / "s.json",
        motion_path=tmp_path / "m.json",
        output_dir=tmp_path,
        line_width=2 * LINE_REF,
    )
    assert ctx.scale == 2.0
    assert ctx.area(3) == 12


def test_file_digest_of_known_bytes(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert file_digest(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_verify_rejects_stale_derivatives(tmp_path: Path) -> None:
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
    ctx.verify({"kaynak_ozetleri": ctx.source_digests()}, "here")
    b.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="here is stale"):
        ctx.verify({"kaynak_ozetleri": {"a": file_digest(a), "b": "0" * 64}}, "here")
