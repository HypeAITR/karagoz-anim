"""K-2a timing and angle interpolation (interp.py): easing, shortest-path angles."""

from __future__ import annotations

import pytest
from test_puppet_fit_common import build_synth_puppet

from karagoz.engine.puppet.interp import ease, frame_times, interpolate, interpolate_sequence
from karagoz.engine.puppet.pose import Pose


def _pose(
    angles: dict[str, float] | None = None,
    *,
    view: str = "synth",
    tx: float = 0.0,
    ty: float = 0.0,
    rotation: float = 0.0,
    scale: float = 1.0,
    z_override: dict[str, int] | None = None,
    canvas: list[int] | None = None,
) -> Pose:
    return Pose(
        view=view,
        angles=dict(angles or {}),
        tx=tx,
        ty=ty,
        rotation=rotation,
        scale=scale,
        z_override=dict(z_override or {}),
        canvas=list(canvas if canvas is not None else [160, 160]),
    )


def _with_offsets(pose: Pose, offsets: dict[str, tuple[float, float]]) -> Pose:
    out = pose.model_copy()
    object.__setattr__(out, "offsets", dict(offsets))
    return out


def test_angle_travels_shortest_path_through_180() -> None:
    mid = interpolate(_pose({"arm": 170.0}), _pose({"arm": -170.0}), 0.5)
    assert mid.angles["arm"] == 180.0  # +20 through 180, not -340
    back = interpolate(_pose({"arm": -170.0}), _pose({"arm": 170.0}), 0.5)
    assert back.angles["arm"] == -180.0
    small = interpolate(_pose({"arm": 10.0}), _pose({"arm": 350.0}), 0.5)
    assert small.angles["arm"] == pytest.approx(0.0)


def test_wrap_lands_exactly_on_y_at_t1() -> None:
    end = interpolate(_pose({"arm": 170.0}), _pose({"arm": -170.0}), 1.0)
    assert end.angles["arm"] == -170.0
    start = interpolate(_pose({"arm": 170.0}), _pose({"arm": -170.0}), 0.0)
    assert start.angles["arm"] == 170.0


def test_root_moves_linear_and_scale_geometric() -> None:
    x = _pose({}, tx=0.0, ty=0.0, rotation=0.0, scale=1.0)
    y = _pose({}, tx=10.0, ty=-4.0, rotation=90.0, scale=4.0)
    mid = interpolate(x, y, 0.5)
    assert (mid.tx, mid.ty, mid.rotation) == (5.0, -2.0, 45.0)
    assert mid.scale == 2.0


def test_ease_endpoints_exact_monotone_and_rejects_unknown() -> None:
    for kind in ("inout", "linear", "in", "out"):
        assert ease(0.0, kind) == 0.0
        assert ease(1.0, kind) == 1.0
        xs = [ease(i / 100.0, kind) for i in range(101)]
        assert all(b >= a for a, b in zip(xs, xs[1:], strict=False)), kind
    with pytest.raises(ValueError):
        ease(0.5, "smooth")


def test_ease_inout_has_zero_slope_at_both_ends() -> None:
    h = 1e-4
    assert ease(h, "inout") / h < 1e-2
    assert (1.0 - ease(1.0 - h, "inout")) / h < 1e-2
    assert ease(h, "in") / h < 1e-2  # ease-in rests at the start
    assert (1.0 - ease(1.0 - h, "out")) / h < 1e-2  # ease-out rests at the end
    assert ease(h, "linear") / h == pytest.approx(1.0)


def test_frame_times_endpoints_count_and_rejects_short() -> None:
    assert frame_times(5, "linear") == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert frame_times(2) == [0.0, 1.0]
    times = frame_times(12)
    assert len(times) == 12 and times[0] == 0.0 and times[-1] == 1.0
    with pytest.raises(ValueError):
        frame_times(1)


def test_sequence_first_equals_x_and_last_equals_y() -> None:
    puppet, _ = build_synth_puppet()
    x = _pose({"arm": 10.0, "fore": -5.0}, tx=2.0)
    y = _pose({"arm": 40.0, "fore": 20.0}, tx=6.0)
    seq = interpolate_sequence(puppet, x, y, 7)
    assert len(seq) == 7
    assert seq[0] == x
    assert seq[0].angles == x.angles
    assert seq[-1] == y
    assert seq[-1].angles == y.angles


def test_locked_parts_hold_x_value_in_every_frame() -> None:
    puppet, _ = build_synth_puppet()
    x = _pose({"arm": 10.0, "fore": -5.0, "hand": 3.0})
    y = _pose({"arm": 50.0, "fore": 25.0, "hand": -10.0})
    seq = interpolate_sequence(puppet, x, y, 9, kind="linear", locked=("fore", "hand"))
    assert len(seq) == 9
    for frame in seq:
        assert frame.angles["fore"] == -5.0
        assert frame.angles["hand"] == 3.0
    assert seq[4].angles["arm"] == 30.0  # unlocked joints still travel


def test_follow_through_child_lags_parent_but_meets_at_ends() -> None:
    puppet, _ = build_synth_puppet()  # arm chain: arm -> fore -> hand
    x = _pose({"arm": 0.0, "fore": 0.0, "hand": 0.0})
    y = _pose({"arm": 30.0, "fore": 30.0, "hand": 0.0})
    plain = interpolate_sequence(puppet, x, y, 9, kind="linear")
    lagged = interpolate_sequence(puppet, x, y, 9, kind="linear", follow_through=1)
    assert lagged[4].angles["arm"] == plain[4].angles["arm"] == 15.0  # first mover keeps the clock
    assert lagged[4].angles["fore"] < plain[4].angles["fore"] == 15.0  # child breaks later
    assert lagged[0].angles == x.angles
    assert lagged[-1].angles == y.angles
    for frame in lagged:
        assert 0.0 <= frame.angles["arm"] <= 30.0  # no overshoot
        assert 0.0 <= frame.angles["fore"] <= 30.0
        assert frame.angles["hand"] == 0.0


def test_offsets_interpolate_linearly_and_absent_offsets_are_ignored() -> None:
    x = _with_offsets(_pose({"arm": 0.0}), {"arm": (0.0, 0.0)})
    y = _with_offsets(_pose({"arm": 30.0}), {"arm": (10.0, -4.0), "fore": (8.0, 0.0)})
    mid = interpolate(x, y, 0.5)
    assert mid.offsets["arm"] == (5.0, -2.0)  # type: ignore[attr-defined]
    assert mid.offsets["fore"] == (4.0, 0.0)  # type: ignore[attr-defined]  # missing side is (0, 0)
    held = interpolate(x, y, 0.5, locked=("arm",))
    assert held.offsets["arm"] == (0.0, 0.0)  # type: ignore[attr-defined]
    assert held.angles["arm"] == 0.0
    plain = interpolate(_pose({"arm": 0.0}), _pose({"arm": 30.0}), 0.5)  # no offsets anywhere
    assert plain.angles["arm"] == 15.0
    puppet, _ = build_synth_puppet()
    seq = interpolate_sequence(puppet, _pose({"arm": 0.0}), _pose({"arm": 30.0}), 3)
    assert [f.angles["arm"] for f in seq] == [0.0, 15.0, 30.0]


def test_carrier_fields_come_from_x_until_final_frame() -> None:
    x = _pose({"arm": 0.0}, view="a", z_override={"arm": 3}, canvas=[160, 160])
    y = _pose({"arm": 30.0}, view="b", z_override={}, canvas=[80, 80])
    mid = interpolate(x, y, 0.5)
    assert (mid.view, mid.canvas, mid.z_override) == ("a", [160, 160], {"arm": 3})
    end = interpolate(x, y, 1.0)
    assert (end.view, end.canvas, end.z_override) == ("b", [80, 80], {})


def test_locked_root_missing_from_both_dicts_stays_missing() -> None:
    puppet, _ = build_synth_puppet()
    x = _pose({"arm": 10.0})
    y = _pose({"arm": 40.0})
    seq = interpolate_sequence(puppet, x, y, 5, locked=("torso",))
    assert seq[0] == x
    assert seq[-1] == y
    assert "torso" not in seq[2].angles


def test_sequence_validates_inputs() -> None:
    puppet, _ = build_synth_puppet()
    x = _pose({"arm": 0.0})
    y = _pose({"arm": 30.0})
    with pytest.raises(ValueError):
        interpolate_sequence(puppet, x, y, 1)
    with pytest.raises(ValueError):
        interpolate_sequence(puppet, x, y, 5, follow_through=-1)
