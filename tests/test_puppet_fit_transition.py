"""K-1 transition reach rules (failing-first).

``ret`` names a chain that cannot reach Y (scores / feasibility), never a
low-confidence part; low confidence belongs in the transition report's
``needs_human``. Near-identical pairs go ``rig``, large clean moves ``split``,
a real shape change still ``ret``.
"""

from __future__ import annotations

from test_puppet_fit_common import TRUE_ROOT, W, build_synth_puppet, draw_pose

from karagoz.engine.puppet.fit import (
    LoadedView,
    _part_confidence,
    _tau_for,
    chain_feasible,
    drawing_features,
    extract_part_samples,
    fit_chain,
    transition_from_poses,
)
from karagoz.engine.puppet.pose import Pose

CHAIN = ["arm", "fore", "hand"]
ZERO = {"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
# A real shape change: the forearm bends away and the hand is gone from Y.
# Every candidate then seats the hand on coincidence below SCORE_MIN.
YBENT_NOHAND = {"arm": 0.0, "fore": 90.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}


def _setup():
    puppet, images = build_synth_puppet()
    loaded = LoadedView(name="synth", puppet=puppet, images=images)
    samples = extract_part_samples(puppet, images)
    return puppet, images, loaded, samples


def _pose_at(angles):
    pose = Pose(view="synth", canvas=[W, W], angles=dict(angles))
    pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    return pose


def _fit_on(puppet, images, samples, angles):
    drawing = draw_pose(puppet, images, dict(angles), dict(TRUE_ROOT))
    feat = drawing_features(drawing)
    return drawing, feat, fit_chain(puppet, _pose_at(ZERO), "hand", drawing, samples, feat)


def test_near_identical_pair_goes_rig() -> None:
    puppet, images, loaded, samples = _setup()
    rgb_x, _, pose_x = _fit_on(puppet, images, samples, dict(ZERO))
    near = dict(ZERO)
    near["arm"] = 10.0
    rgb_y, _, pose_y = _fit_on(puppet, images, samples, near)
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)
    assert result.decision == "rig", result.reason


def test_large_clean_move_goes_split_not_ret() -> None:
    puppet, images, loaded, samples = _setup()
    rgb_x, _, pose_x = _fit_on(puppet, images, samples, dict(ZERO))
    far = dict(ZERO)
    far["arm"] = -80.0
    rgb_y, _, pose_y = _fit_on(puppet, images, samples, far)
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)
    assert result.decision == "split", result.reason


def test_real_shape_change_stays_ret() -> None:
    puppet, images, loaded, samples = _setup()
    rgb_x, _, pose_x = _fit_on(puppet, images, samples, dict(ZERO))
    rgb_y = draw_pose(puppet, images, dict(YBENT_NOHAND), dict(TRUE_ROOT), skip=("hand",))
    feat_y = drawing_features(rgb_y)
    pose_y = fit_chain(puppet, _pose_at(ZERO), "hand", rgb_y, samples, feat_y)
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)
    assert result.decision == "ret", result.reason


def test_confidence_measured_inside_limits() -> None:
    from karagoz.engine.puppet.fit import NO_AMBIGUITY_GAP

    puppet, images, _, samples = _setup()
    puppet.parts["fore"].limits = [-10.0, 10.0]
    drawing = draw_pose(puppet, images, dict(ZERO), dict(TRUE_ROOT))
    feat = drawing_features(drawing)
    pose = _pose_at(ZERO)
    tau = _tau_for(puppet, pose)
    _, limited_gap = _part_confidence(puppet, pose, "fore", samples, feat, tau)
    assert limited_gap == NO_AMBIGUITY_GAP
    _, open_gap = _part_confidence(puppet, pose, "arm", samples, feat, tau)
    assert 0.0 <= open_gap < NO_AMBIGUITY_GAP


def test_folded_fresh_loses_to_pose_y() -> None:
    from karagoz.engine.puppet.fit import _pick_chain_winner

    puppet, images, _, samples = _setup()
    drawing = draw_pose(puppet, images, dict(ZERO), dict(TRUE_ROOT))
    feat = drawing_features(drawing)
    tau = _tau_for(puppet, _pose_at(ZERO))
    center = {m: 0.0 for m in CHAIN}
    stuck = _pose_at({**ZERO, "arm": 30.0})
    folded = _pose_at({**ZERO, "fore": 180.0})
    seated = _pose_at(dict(ZERO))
    feas, _ = chain_feasible(puppet, folded, CHAIN, samples, feat.line.shape)
    assert not feas
    winner, total = _pick_chain_winner(puppet, CHAIN, [stuck, folded, seated], samples, feat, tau, center, None)
    assert dict(winner.angles) == dict(seated.angles)
    assert total > 0.0
    all_folded = _pick_chain_winner(
        puppet, CHAIN, [stuck, folded, _pose_at({**ZERO, "fore": -180.0})], samples, feat, tau, center, None
    )
    assert dict(all_folded[0].angles) == dict(stuck.angles)


def test_transition_report_carries_needs_human() -> None:
    puppet, images, loaded, samples = _setup()
    rgb_x, _, pose_x = _fit_on(puppet, images, samples, dict(ZERO))
    near = dict(ZERO)
    near["arm"] = 10.0
    rgb_y, _, pose_y = _fit_on(puppet, images, samples, near)
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)
    from karagoz.engine.puppet.fit import needs_human, transition_dict

    doc = transition_dict(result)
    assert doc["needs_human"] == needs_human(result.pose_y, loaded.puppet)
    assert set(doc["confidences"]) >= set(doc["needs_human"])
