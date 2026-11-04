"""K-0r round 2: the from-scratch chain greedy must not park a narrow first joint blind.

Synthetic puppet only. The greedy decides each joint with its downstream at
rest; a narrow first joint (a cap with a +-10 degree limit) decided against
the rest limb lands at a limit and the child compensates, a basin a
one-joint descent cannot leave. The beam reruns the greedy with that joint
pinned to each candidate and keeps the best joint objective.
"""

from __future__ import annotations

import numpy as np
from test_puppet_fit_common import TRUE_ROOT, build_synth_puppet, draw_pose

from karagoz.engine.puppet import fit as F
from karagoz.engine.puppet.pose import Pose, chains


def _setup(limits: list[float] | None, truth: dict[str, float]) -> tuple:
    puppet, images = build_synth_puppet()
    lim = puppet.model_copy(deep=True)
    lim.parts["arm"].limits = limits
    samples = F.extract_part_samples(lim, images)
    chain = chains(lim)["hand"]
    drawing = draw_pose(lim, images, {**truth, "leg": 10.0, "foot": -8.0}, dict(TRUE_ROOT))
    feat = F.drawing_features(drawing)
    work = Pose(view="synth", canvas=[160, 160])
    work.tx, work.ty, work.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    tau = F._tau_for(lim, work)
    center = {m: 0.0 for m in chain}
    blocked = F.context_blocked(lim, work, chain, samples, feat.line.shape)
    explained = F.explained_mask(
        lim, work, samples, feat.line.shape, skip=set(chain), line_radius=F._explain_line_radius(lim, work)
    )
    return lim, samples, chain, feat, work, tau, center, blocked, explained


def _value(ctx: tuple, angles: dict[str, float], offsets: dict[str, list[float]]) -> float:
    lim, samples, chain, feat, work, tau, center, blocked, explained = ctx
    trial = work.model_copy(update={"angles": dict(angles), "offset": dict(offsets)})
    return float(F._offset_value(lim, trial, chain, samples, feat, tau, center, blocked, explained))


def test_narrow_first_joint_greedy_beam_finds_true_basin() -> None:
    """Truth arm 0 / fore -90: plain greedy parks the arm at -10; the beam returns 0 / -90."""
    ctx = _setup([-10.0, 10.0], {"arm": 0.0, "fore": -90.0, "hand": 0.0})
    lim, samples, chain, feat, work, tau, center, blocked, explained = ctx
    rest = {m: 0.0 for m in chain}
    plain, off_plain, _ = F._greedy_from(
        lim,
        work,
        chain,
        rest,
        samples,
        feat,
        tau,
        center,
        None,
        blocked,
        pair_grid=True,
        frozen={},
        explained=explained,
    )
    assert plain["arm"] == -10.0, "scene no longer shows the blind-first-joint trap"
    beam, off_beam, _pa, _po = F._greedy_beam(
        lim, work, chain, rest, samples, feat, tau, center, blocked, True, {}, explained
    )
    assert beam["arm"] == 0.0
    assert abs(beam["fore"] - -90.0) <= 1.0
    assert _value(ctx, beam, off_beam) > _value(ctx, plain, off_plain) + 1.0


def test_greedy_beam_never_worse_than_plain() -> None:
    """The unpinned run competes, so the beam's objective never drops below the plain greedy."""
    for truth in ({"arm": 5.0, "fore": 45.0, "hand": 0.0}, {"arm": -10.0, "fore": 45.0, "hand": 0.0}):
        ctx = _setup([-10.0, 10.0], truth)
        lim, samples, chain, feat, work, tau, center, blocked, explained = ctx
        rest = {m: 0.0 for m in chain}
        plain, off_plain, _ = F._greedy_from(
            lim,
            work,
            chain,
            rest,
            samples,
            feat,
            tau,
            center,
            None,
            blocked,
            pair_grid=True,
            frozen={},
            explained=explained,
        )
        beam, off_beam, _pa, _po = F._greedy_beam(
            lim, work, chain, rest, samples, feat, tau, center, blocked, True, {}, explained
        )
        assert _value(ctx, beam, off_beam) >= _value(ctx, plain, off_plain) - 1e-9


def test_wide_first_joint_takes_no_beam() -> None:
    """Without a narrow limit the beam is the plain greedy exactly."""
    ctx = _setup(None, {"arm": 40.0, "fore": -25.0, "hand": 15.0})
    lim, samples, chain, feat, work, tau, center, blocked, explained = ctx
    rest = {m: 0.0 for m in chain}
    plain, off_plain, _ = F._greedy_from(
        lim,
        work,
        chain,
        rest,
        samples,
        feat,
        tau,
        center,
        None,
        blocked,
        pair_grid=True,
        frozen={},
        explained=explained,
    )
    beam, off_beam, _pa, _po = F._greedy_beam(
        lim, work, chain, rest, samples, feat, tau, center, blocked, True, {}, explained
    )
    assert beam == plain
    assert {k: list(v) for k, v in off_beam.items()} == {k: list(v) for k, v in off_plain.items()}
    assert np.isfinite(_value(ctx, beam, off_beam))
