"""RM-1: rig shape measures match the experiment, plus synthetic checks."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import rig_golden
from PIL import Image

from karagoz.engine.rig import measure


def _experiment_olcu():
    root = rig_golden.fixture()
    sys.path.insert(0, str(root))
    from deneme.faz1c import olcu

    return olcu


def _pose_x_rgb() -> np.ndarray:
    img = rig_golden.r2() / "d" / "girdi" / "poz_X.png"
    return np.array(Image.open(img).convert("RGB"))


@pytest.mark.rig_golden
def test_foreground_matches_experiment_on_pose_x() -> None:
    olcu = _experiment_olcu()
    rgb = _pose_x_rgb()
    assert np.array_equal(measure.foreground(rgb), olcu.foreground(rgb))


@pytest.mark.rig_golden
def test_ink_mask_matches_experiment_on_pose_x() -> None:
    olcu = _experiment_olcu()
    rgb = _pose_x_rgb()
    assert np.array_equal(measure.ink_mask(rgb), olcu.ink_mask(rgb))


@pytest.mark.rig_golden
def test_nearest_indices_matches_experiment_on_real_crop() -> None:
    olcu = _experiment_olcu()
    seeds = measure.foreground(_pose_x_rgb())[600:800, 300:500]
    dist, iy, ix = measure.nearest_indices(seeds)
    exp_dist, exp_iy, exp_ix = olcu.nearest_indices(seeds)
    assert np.array_equal(dist, exp_dist)
    assert np.array_equal(iy, exp_iy)
    assert np.array_equal(ix, exp_ix)


@pytest.mark.rig_golden
def test_disk_matches_experiment() -> None:
    olcu = _experiment_olcu()
    assert np.array_equal(
        measure.disk((1440, 960), 347.68, 534.58, 51.46),
        olcu.disk((1440, 960), 347.68, 534.58, 51.46),
    )


def test_foreground_edge_touching_white_is_background() -> None:
    rgb = np.full((60, 60, 3), 255, np.uint8)
    rgb[20:40, 20:40] = (240, 200, 160)  # skin square floating in white
    fg = measure.foreground(rgb)
    assert fg.sum() == 400
    assert fg[30, 30] and not fg[0, 0]


def test_foreground_studio_green_is_background() -> None:
    rgb = np.full((60, 60, 3), 255, np.uint8)
    rgb[:, :10] = (150, 190, 150)  # studio-green strip touching the edge
    rgb[20:40, 20:40] = (240, 200, 160)
    fg = measure.foreground(rgb)
    assert fg.sum() == 400
    assert not fg[30, 5]


def test_ink_mask_finds_black_line_not_mid_grey() -> None:
    rgb = np.full((20, 20, 3), 255, np.uint8)
    rgb[5, 5:15] = (0, 0, 0)
    rgb[10, 5:15] = (130, 130, 130)
    ink = measure.ink_mask(rgb)
    assert ink[5, 5:15].all()
    assert not ink[10, 5:15].any()
    assert ink.sum() == 10


def test_disk_is_a_filled_circle() -> None:
    d = measure.disk((11, 11), 5, 5, 2)
    assert d.sum() == 13
    assert d[5, 5] and d[5, 7] and not d[0, 0]


def test_nearest_indices_on_tiny_seeds() -> None:
    seeds = np.zeros((5, 5), bool)
    seeds[2, 2] = True
    dist, iy, ix = measure.nearest_indices(seeds)
    assert dist[2, 2] == pytest.approx(0.0)
    assert dist[0, 0] == pytest.approx(2.8, abs=0.05)  # chamfer approximation of hypot(2, 2)
    assert (iy == 2).all() and (ix == 2).all()
