"""K-1 shoulder slack + never fold (failing-first).

A straight hanging arm can leave the shoulder dome from a point offset
inside the slack (the animator nudges the shoulder): the chain fit must
recover the offset together with the angles. And a chain candidate with
a member stacked on the same chain past overlap is infeasible, whatever
its score: a distractor that rewards folding never wins over a feasible
straight chain.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_puppet_fit_common import TRUE_ROOT, W, build_synth_puppet, draw_pose, white

from karagoz.engine.puppet.fit import (
    _ang_dist,
    chain_feasible,
    drawing_features,
    extract_part_samples,
    fit_chain,
)
from karagoz.engine.puppet.pose import Pose, render_pose

STRAIGHT = {"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
OFFSET = [7.0, 5.0]
SLACK = 12.0


def _slack_puppet():
    puppet, images = build_synth_puppet()
    puppet.parts["arm"].slack_px = float(SLACK)
    return puppet, images


def _draw_with_offset(puppet, images, angles, root, offset):
    pose = Pose(
        view="synth",
        canvas=[W, W],
        angles=dict(angles),
        tx=root["tx"],
        ty=root["ty"],
        rotation=root.get("rotation", 0.0),
        scale=root.get("scale", 1.0),
        offset=dict(offset),
    )
    return white(render_pose(puppet, pose, images))


def _pose_at(angles):
    pose = Pose(view="synth", canvas=[W, W], angles=dict(angles))
    pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    return pose


def test_slack_offset_recovered() -> None:
    puppet, images = _slack_puppet()
    drawing = _draw_with_offset(puppet, images, dict(STRAIGHT), dict(TRUE_ROOT), {"arm": list(OFFSET)})
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)
    start = _pose_at(dict(STRAIGHT))
    fitted = fit_chain(puppet, start, "hand", drawing, samples, feat)
    assert _ang_dist(fitted.angles.get("arm", 0.0), 0.0) <= 2.0
    assert _ang_dist(fitted.angles.get("fore", 0.0), 0.0) <= 2.0
    got = fitted.offset.get("arm", [0.0, 0.0])
    assert float(np.hypot(got[0] - OFFSET[0], got[1] - OFFSET[1])) <= 2.0


def test_slack_zero_leaves_no_offset() -> None:
    puppet, images = build_synth_puppet()
    drawing = draw_pose(puppet, images, dict(STRAIGHT), dict(TRUE_ROOT))
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)
    start = _pose_at(dict(STRAIGHT))
    fitted = fit_chain(puppet, start, "hand", drawing, samples, feat)
    assert dict(fitted.offset) == {}
    assert _ang_dist(fitted.angles.get("arm", 0.0), 0.0) <= 2.0
    assert _ang_dist(fitted.angles.get("fore", 0.0), 0.0) <= 2.0


@pytest.fixture(scope="module")
def fold_setup() -> tuple:
    from test_puppet_fit_poz_kollar_asagi_2 import redrawn_images

    puppet, images = build_synth_puppet()
    drawn = redrawn_images(images)
    drawing = draw_pose(
        puppet, drawn, {"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}, dict(TRUE_ROOT)
    )
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)
    return puppet, drawing, feat, samples


def test_fold_on_own_tube_is_infeasible(fold_setup: tuple) -> None:
    puppet, drawing, feat, samples = fold_setup
    fold = _pose_at({"arm": 0.0, "fore": 180.0, "hand": 0.0, "leg": 10.0, "foot": 0.0})
    straight = _pose_at({"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0})
    feasible_fold, _ = chain_feasible(puppet, fold, ["arm", "fore", "hand"], samples, feat.line.shape)
    feasible_straight, _ = chain_feasible(puppet, straight, ["arm", "fore", "hand"], samples, feat.line.shape)
    assert feasible_straight and not feasible_fold


def test_distractor_fold_never_beats_feasible_straight(fold_setup: tuple) -> None:
    puppet, drawing, feat, samples = fold_setup
    start = _pose_at({"arm": 0.0, "fore": 180.0, "hand": 0.0, "leg": 10.0, "foot": 0.0})
    fitted = fit_chain(puppet, start, "hand", drawing, samples, feat)
    assert _ang_dist(fitted.angles.get("arm", 0.0), 0.0) <= 5.0
    assert _ang_dist(fitted.angles.get("fore", 0.0), 0.0) <= 5.0
