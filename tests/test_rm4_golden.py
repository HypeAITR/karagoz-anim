"""RM-4: canonical sources and frames match the golden X->Y run pixel by pixel.

One module-scoped run of ``frames.rig_interval`` on the golden fixture (line width 2.8, the
experiment's ``aralik_kur.CIZGI``); every file of ``_altin_r13/xy/kanonik/`` the setup writes and the
eight frames ``k00..k07`` must equal the golden ones. The fixture's cached ``muse_kolsuz_*.png`` are
pre-seeded as inputs, so no model is ever called (``fill=None``).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import frames
from karagoz.engine.rig.interval import IntervalContext

XY_LINE_WIDTH = 2.8  # panel line width p50 the experiment measured (plan/aralik_kur.py CIZGI)
MUSE_PREFIX = "muse_kolsuz_"


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


def _golden_xy() -> Path:
    return rig_golden.fixture() / "_altin_r13" / "xy"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("rm4golden") / "xy"
    ctx = _xy_context(out)
    # The fixture's cached Muse fills are inputs: pre-seed them so the file cache hits, no model runs.
    seeded = out / "kanonik"
    seeded.mkdir(parents=True, exist_ok=True)
    for cached in sorted((_golden_xy() / "kanonik").glob(f"{MUSE_PREFIX}*.png")):
        shutil.copyfile(cached, seeded / cached.name)
    paths = frames.rig_interval(ctx, None, out / "rig_AB")
    return ctx, paths, out


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


def _read_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


@pytest.mark.rig_golden
def test_canonical_writes_every_golden_file(built) -> None:
    _ctx, _paths, out = built
    golden = {p.name for p in (_golden_xy() / "kanonik").iterdir() if not p.name.startswith(MUSE_PREFIX)}
    got = {p.name for p in (out / "kanonik").iterdir() if not p.name.startswith(MUSE_PREFIX)}
    assert got == golden, f"canonical file set differs by {got ^ golden}"


@pytest.mark.rig_golden
def test_canonical_pngs_are_pixel_identical(built) -> None:
    _ctx, _paths, out = built
    for golden_path in sorted((_golden_xy() / "kanonik").glob("*.png")):
        if golden_path.name.startswith(MUSE_PREFIX):
            continue
        got = _read_png(out / "kanonik" / golden_path.name)
        exp = _read_png(golden_path)
        assert got.shape == exp.shape, golden_path.name
        diff = np.abs(got.astype(np.int32) - exp.astype(np.int32)).max(axis=2)
        assert int((diff > 0).sum()) == 0, f"{golden_path.name}: {(diff > 0).sum()} differing pixels"
        assert np.array_equal(got, exp), golden_path.name


@pytest.mark.rig_golden
def test_canonical_jsons_match_structure_and_numbers(built) -> None:
    _ctx, _paths, out = built
    for golden_path in sorted((_golden_xy() / "kanonik").glob("*.json")):
        got = json.loads((out / "kanonik" / golden_path.name).read_text(encoding="utf-8"))
        exp = json.loads(golden_path.read_text(encoding="utf-8"))
        _assert_json_close(got, exp, golden_path.name)


@pytest.mark.rig_golden
def test_frames_are_pixel_identical(built) -> None:
    _ctx, paths, _out = built
    assert [p.name for p in paths] == [f"k{i:02d}.png" for i in range(8)]
    for i, path in enumerate(paths):
        got = _read_png(path)
        exp = _read_png(_golden_xy() / "rig_AB" / f"k{i:02d}.png")
        assert got.shape == exp.shape, f"k{i:02d}"
        diff = np.abs(got.astype(np.int32) - exp.astype(np.int32))
        assert int((diff.max(axis=2) > 0).sum()) == 0, (
            f"k{i:02d}: {(diff.max(axis=2) > 0).sum()} differing pixels, max abs {int(diff.max())}"
        )
        assert np.array_equal(got, exp), f"k{i:02d}"


@pytest.mark.rig_golden
def test_frame_report_and_ground_match_golden(built) -> None:
    _ctx, _paths, out = built
    got_report = json.loads((out / "rig_AB" / "rapor.json").read_text(encoding="utf-8"))
    exp_report = json.loads((_golden_xy() / "rig_AB" / "rapor.json").read_text(encoding="utf-8"))
    _assert_json_close(got_report, exp_report, "rapor.json")
    assert np.array_equal(
        _read_png(out / "rig_AB" / "govde_kolsuz.png"), _read_png(_golden_xy() / "rig_AB" / "govde_kolsuz.png")
    )


@pytest.mark.rig_golden
def test_cut_choice_and_post_cut_plan_match_golden(built) -> None:
    _ctx, _paths, out = built
    got_cut = json.loads((out / "r7" / "kesme.json").read_text(encoding="utf-8"))
    exp_cut = json.loads((_golden_xy() / "r7" / "kesme.json").read_text(encoding="utf-8"))
    _assert_json_close(got_cut, exp_cut, "kesme.json")
    assert got_cut["kesme"] == [6]
    got_plan = json.loads((out / "kare_plani_AB.json").read_text(encoding="utf-8"))
    exp_plan = json.loads((_golden_xy() / "kare_plani_AB.json").read_text(encoding="utf-8"))
    _assert_json_close(got_plan, exp_plan, "kare_plani_AB.json")


@pytest.mark.rig_golden
def test_r9_head_and_region_intermediates_match_golden(built) -> None:
    _ctx, _paths, out = built
    for name in ("kafa.json", "alan_noktalari.json", "degisen.json"):
        got = json.loads((out / "r9" / name).read_text(encoding="utf-8"))
        exp = json.loads((_golden_xy() / "r9" / name).read_text(encoding="utf-8"))
        _assert_json_close(got, exp, name)
    assert np.array_equal(_read_png(out / "r9" / "degisen.png"), _read_png(_golden_xy() / "r9" / "degisen.png"))
