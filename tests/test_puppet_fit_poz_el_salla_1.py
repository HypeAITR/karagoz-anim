"""Pose el_salla_1: hanging chain must prefer unexplained pixels over a dense distractor.

Failing-first for the K-1-es1 fix: ``explained_mask`` must cover the
line-width neighbourhood of explained lines, not just exact pixels, so a
fold lying on (or next to) dense already-explained lines decays while a
straight hang on unexplained pixels keeps its score.
"""

from __future__ import annotations

import numpy as np
from test_puppet_fit_common import build_synth_puppet

from karagoz.engine.puppet.fit import PartSamples, explained_mask
from karagoz.engine.puppet.pose import Pose


def _line_only_samples(xy: np.ndarray) -> PartSamples:
    return PartSamples(
        line_xy=np.asarray(xy, dtype=np.float64),
        flat_xy=np.zeros((0, 2), np.float64),
        flat_rgb=np.zeros((0, 3), np.uint8),
        cover_xy=np.zeros((0, 2), np.float64),
        line_cover_xy=np.asarray(xy, dtype=np.float64),
    )


def test_explained_mask_covers_line_neighbourhood() -> None:
    """A pixel next to an explained line counts as explained (line-width radius)."""
    puppet, _images = build_synth_puppet()
    # One explained line pixel at (50, 50) in reference pixels; identity pose.
    samples = {
        name: _line_only_samples(np.array([[50.0, 50.0]])) if name == "arm" else _line_only_samples(np.zeros((0, 2)))
        for name in puppet.parts
    }
    pose = Pose(view="synth", canvas=[160, 160])
    mask = explained_mask(puppet, pose, samples, (160, 160), skip={"hand"})
    assert bool(mask[50, 50]), "the explained line pixel itself"
    assert not bool(mask[50, 52]), "pixel-exact hole stays exact for fill_last"
    grown = explained_mask(puppet, pose, samples, (160, 160), skip={"hand"}, line_radius=2)
    # Two pixels away is inside the line-width neighbourhood (synth
    # line_width 2.0 -> radius 2) and must decay as explained.
    assert bool(grown[50, 52]), "neighbour of an explained line is explained"
