"""K-1 hanging basin: a hang that needs its slack offset (failing-first).

A straight hanging arm slid inside its slack (the animator nudged the
shoulder) is only the best placement when the angle is ranked WITH a
searched offset. Ranked at fixed offsets, a folded distractor across dense
lines wins the coarse stages and the later offset descent only polishes it.
"""

from __future__ import annotations

from test_puppet_fit_common import TRUE_ROOT, W, build_synth_puppet, white

from karagoz.engine.puppet.fit import (
    _ang_dist,
    _offset_value,
    chain_feasible,
    context_blocked,
    drawing_features,
    explained_mask,
    extract_part_samples,
    fit_chain,
)
from karagoz.engine.puppet.pose import Pose, render_pose

CHAIN = ["arm", "fore", "hand"]
ZERO = {"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
SLACK = 12.0
HANG_DEG = -70.0
HANG_OFF = [8.0, 7.0]


def _slack_puppet():
    puppet, images = build_synth_puppet()
    puppet.parts["arm"].slack_px = float(SLACK)
    return puppet, images


def _start():
    pose = Pose(view="synth", canvas=[W, W], angles=dict(ZERO))
    pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    return pose


def test_hang_with_offset_beats_fixed_offset_distractor() -> None:
    puppet, images = _slack_puppet()
    drawn = Pose(
        view="synth",
        canvas=[W, W],
        angles={**ZERO, "arm": HANG_DEG},
        tx=TRUE_ROOT["tx"],
        ty=TRUE_ROOT["ty"],
        offset={"arm": list(HANG_OFF)},
    )
    drawing = white(render_pose(puppet, drawn, images))
    samples = extract_part_samples(puppet, images)
    feat = drawing_features(drawing)
    got = fit_chain(puppet, _start(), "hand", drawing, samples, feat)
    feas, _ = chain_feasible(puppet, got, CHAIN, samples, feat.line.shape)
    assert feas
    # The pipeline resolves ridges, not points: an offset error trades for an
    # angle error along the hang (production shows the same: the reviewer's
    # (12.7, 19.0) vs the refit (13.4, 21.8)). The windows below still exclude
    # the distractor (-90 with a folded fore) by a wide margin.
    assert abs(_ang_dist(float(got.angles["arm"]), HANG_DEG)) <= 7.0, dict(got.angles)
    assert abs(_ang_dist(float(got.angles["fore"]), 0.0)) <= 8.0, dict(got.angles)
    off = got.offset.get("arm", [0.0, 0.0])
    assert abs(off[0] - HANG_OFF[0]) <= 3.5 and abs(off[1] - HANG_OFF[1]) <= 3.5, got.offset
    # Mechanism: under the fit's own context the seat is on the truth's ridge,
    # far above the fixed-offset distractor level (~1.6) the old code kept.
    center = {m: 0.0 for m in CHAIN}
    blocked = context_blocked(puppet, got, CHAIN, samples, feat.line.shape)
    expl = explained_mask(puppet, got, samples, feat.line.shape, skip=set(CHAIN), line_radius=2.0)
    truth = _start()
    truth.angles["arm"] = HANG_DEG
    truth.offset["arm"] = list(HANG_OFF)
    v_got = _offset_value(puppet, got, CHAIN, samples, feat, 2.0, center, blocked, expl)
    v_truth = _offset_value(puppet, truth, CHAIN, samples, feat, 2.0, center, blocked, expl)
    assert abs(v_got - v_truth) < 0.5, (v_got, v_truth)
    assert v_got > 2.5, v_got
