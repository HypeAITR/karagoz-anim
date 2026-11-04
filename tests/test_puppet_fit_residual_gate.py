"""Fit-quality gate (4 Nov): a key the puppet cannot represent makes the interval ``ret``, never ``split``/``rig``."""

from __future__ import annotations

import numpy as np
from test_puppet_fit_chain import TRUE_ROOT, ZERO_ANGLES, built, draw_pose, images, puppet, synth_view  # noqa: F401  (pytest fixtures)

from karagoz.engine.puppet.fit import FIT_RESIDUAL_MAX, residual_share, transition_from_poses
from karagoz.engine.puppet.pose import Pose


def _poses(angles_x: dict[str, float], angles_y: dict[str, float]) -> tuple[Pose, Pose]:
    px = Pose(view="synth", canvas=[160, 160], angles=dict(angles_x), **TRUE_ROOT)
    py = Pose(view="synth", canvas=[160, 160], angles=dict(angles_y), **TRUE_ROOT)
    return px, py


def test_drawing_the_puppet_can_draw_has_no_residual(puppet: object, images: dict) -> None:  # noqa: F811
    rgb = draw_pose(puppet, images, dict(ZERO_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]
    px, _ = _poses(ZERO_ANGLES, ZERO_ANGLES)
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    assert residual_share(loaded, px, rgb) <= 0.005  # type: ignore[arg-type]


def test_unrepresentable_key_rets(puppet: object, images: dict) -> None:  # noqa: F811
    rgb_x = draw_pose(puppet, images, dict(ZERO_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]
    rgb_y = np.array(rgb_x, copy=True)
    # A shape no part has (a new limb, a foreshortened arm): a dark block in an empty corner of Y.
    rgb_y[4:44, 4:44] = (60, 30, 30)
    px, py = _poses(ZERO_ANGLES, ZERO_ANGLES)
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    assert residual_share(loaded, py, rgb_y) > FIT_RESIDUAL_MAX  # type: ignore[arg-type]
    result = transition_from_poses(loaded, px, py, rgb_x, rgb_y)  # type: ignore[arg-type]
    assert result.decision == "ret"
    assert "cannot represent" in result.reason and "Y " in result.reason
