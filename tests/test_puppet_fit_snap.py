"""K-1 single-part fitting: root similarity, snap, fill-last, human handoff."""

from __future__ import annotations

import numpy as np
import pytest
from test_puppet_fit_common import TRUE_ANGLES, TRUE_ROOT, build_synth_puppet, draw_pose, lengthen, shift, synth_view

from karagoz.engine.puppet.fit import (
    LoadedView,
    _ang_dist,
    drawing_features,
    extract_part_samples,
    fill_last,
    fit_view_pose,
    needs_human,
    snap_part,
)
from karagoz.engine.puppet.pose import Pose


@pytest.fixture(scope="module")
def built() -> tuple[object, dict]:
    return build_synth_puppet()  # type: ignore[return-value]


@pytest.fixture(scope="module")
def puppet(built: object) -> object:
    return built[0]


@pytest.fixture(scope="module")
def images(built: object) -> dict:
    return built[1]  # type: ignore[index]


@pytest.fixture(scope="module")
def view(puppet: object, images: dict) -> LoadedView:
    return synth_view(puppet, images)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def drawing(puppet: object, images: dict) -> np.ndarray:
    return draw_pose(puppet, images, dict(TRUE_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]


def test_known_angles_recovered_within_1deg(view: LoadedView, drawing: np.ndarray) -> None:
    pose = fit_view_pose(view, drawing)
    assert abs(pose.tx - TRUE_ROOT["tx"]) <= 1.5
    assert abs(pose.ty - TRUE_ROOT["ty"]) <= 1.5
    assert abs(pose.rotation) <= 1.0
    assert abs(pose.scale - TRUE_ROOT["scale"]) <= 0.01
    for name, want in TRUE_ANGLES.items():
        assert _ang_dist(pose.angles.get(name, 0.0), want) <= 1.0, name


def test_redrawn_longer_and_shifted_recovered_within_3deg(puppet: object, images: dict, view: LoadedView) -> None:
    long_images = lengthen(images, "fore", (31.0, 92.0), 1.08)
    drawing = draw_pose(puppet, long_images, dict(TRUE_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]
    pose = fit_view_pose(view, shift(drawing, 2, 0), second_pass=False)
    for name, want in TRUE_ANGLES.items():
        assert _ang_dist(pose.angles.get(name, 0.0), want) <= 3.0, name


def test_hidden_part_goes_to_needs_human(puppet: object, images: dict, view: LoadedView) -> None:
    # The arm points right, the forearm hangs down from its end, and the hand
    # rides under the torso: the drawing never shows it, so it must go to a
    # human even though the visible limb fits cleanly.
    drawing = draw_pose(
        puppet,  # type: ignore[arg-type]
        images,
        {"arm": 180.0, "fore": 90.0, "hand": 0.0, "leg": 0.0, "foot": 0.0},
        {"tx": 0.0, "ty": 0.0, "rotation": 0.0, "scale": 1.0},
        z_override={"hand": -5},
    )
    pose = fit_view_pose(view, drawing)
    need = needs_human(pose, puppet)  # type: ignore[arg-type]
    assert "hand" in need
    assert _ang_dist(pose.angles.get("arm", 0.0), 180.0) <= 3.0
    assert _ang_dist(pose.angles.get("fore", 0.0), 90.0) <= 3.0
    assert "fore" not in need
    assert "arm" not in need


def test_fill_last_places_removed_part_within_1deg(puppet: object, images: dict) -> None:
    # The drawing shows every part; the pose has everything placed except the
    # hand, so the hand must land in the unexplained hole.
    drawing = draw_pose(puppet, images, dict(TRUE_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    pose = Pose(view="synth", canvas=[160, 160], angles={k: v for k, v in TRUE_ANGLES.items() if k != "hand"})
    pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    result = fill_last(puppet, pose, "hand", drawing, samples, feat)  # type: ignore[arg-type]
    assert _ang_dist(result.angle, TRUE_ANGLES["hand"]) <= 1.0


def test_hint_from_wrong_side_converges(puppet: object, images: dict, view: LoadedView, drawing: np.ndarray) -> None:
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    pose = Pose(view="synth", canvas=[160, 160], angles={"arm": -60.0})
    pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    result = snap_part(puppet, pose, "arm", drawing, samples, feat, hint_deg=0.0, window_deg=180.0)  # type: ignore[arg-type]
    assert _ang_dist(result.angle, TRUE_ANGLES["arm"]) <= 1.0
