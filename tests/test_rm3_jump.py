"""RM-3: jump measure matches sicrama on the golden frames, plus synthetic checks."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import jump


def _experiment():
    root = rig_golden.fixture()
    for p in ("deneme/r2/plan", "deneme/r2", "deneme/r2/d", ""):
        sys.path.insert(0, str(root / p) if p else str(root))
    import sicrama

    return sicrama


@pytest.fixture(scope="module")
def exp():
    return _experiment()


@pytest.fixture(scope="module")
def frames():
    folder = rig_golden.fixture() / "_altin_r13" / "xy" / "rig_AB"
    return [np.array(Image.open(folder / f"k{i:02d}.png").convert("RGB")) for i in range(8)]


@pytest.fixture(scope="module")
def plan():
    return json.loads((rig_golden.fixture() / "_altin_r13" / "xy" / "kare_plani_AB.json").read_text(encoding="utf-8"))


@pytest.mark.rig_golden
def test_measure_matches_experiment_on_golden_frames(exp, frames, plan) -> None:
    got = jump.measure(frames, plan)
    want = exp.olc(frames, plan)
    assert got == want
    assert set(got) == {"omuz", "dirsek", "bilek", "bardak"}
    assert all(len(got[r]["degisim"]) == 7 for r in got)


@pytest.mark.rig_golden
def test_tea_center_matches_experiment_on_golden_frames(exp, frames) -> None:
    for frame in frames:
        assert np.array_equal(jump.tea_center(frame), exp.cay_merkezi(frame))


@pytest.mark.rig_golden
def test_flag_steps_matches_experiment_inner_and_end(exp) -> None:
    d, areas, paths = [10, 100, 10], [2000, 2000, 2000], [5.0, 5.0, 5.0]
    got = jump.flag_steps(d, areas, paths)
    assert got == exp.isaretle(d, areas, paths)
    assert got == [{"adim": "k01->k02", "tur": "ic", "degisim": 100, "beklenen": 10.0}]
    d, areas, paths = [100, 10], [2000, 2000], [5.0, 5.0]
    got = jump.flag_steps(d, areas, paths)
    assert got == exp.isaretle(d, areas, paths)
    assert got == [{"adim": "k00->k01", "tur": "uc", "degisim": 100, "beklenen": 10.0}]
    assert jump.flag_steps([1, 2, 1], [2000, 2000, 2000], [5.0, 5.0, 5.0]) == []
    assert exp.isaretle([1, 2, 1], [2000, 2000, 2000], [5.0, 5.0, 5.0]) == []


def test_tea_center_without_tea_returns_canvas_centre() -> None:
    rgb = np.full((100, 80, 3), 255, np.uint8)
    assert np.array_equal(jump.tea_center(rgb), [40.0, 50.0])


def test_tea_center_finds_red_blob() -> None:
    rgb = np.full((100, 100, 3), 255, np.uint8)
    rgb[20:40, 60:80] = (150, 30, 30)
    centre = jump.tea_center(rgb)
    assert abs(centre[0] - 69.5) < 1.0
    assert abs(centre[1] - 29.5) < 1.0


def test_measure_two_identical_frames_flags_nothing() -> None:
    frame = np.full((100, 100, 3), 255, np.uint8)
    frame[20:40, 60:80] = (150, 30, 30)
    joints = {"shoulder_L": [50, 50], "elbow_L": [50, 60], "wrist_L": [50, 70]}
    plan = {"kareler": [{"hedef_eklemler": joints}, {"hedef_eklemler": joints}]}
    got = jump.measure([frame, frame.copy()], plan)
    for region in ("omuz", "dirsek", "bilek", "bardak"):
        assert got[region]["degisim"] == [0]
        assert got[region]["isaretli"] == []
