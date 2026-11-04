"""K-1 limb fitting and X->Y transitions on the synthetic puppet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_puppet_fit_common import TRUE_ROOT, build_synth_puppet, draw_pose, lengthen, synth_view

from karagoz.engine.puppet.fit import (
    LoadedView,
    TransitionResult,
    _ang_dist,
    _warp_points,
    chain_objective,
    context_blocked,
    difference_locked,
    drawing_features,
    extract_part_samples,
    fit_chain,
    fit_view_pose,
    gate_split_needed,
    snap_part,
    transition_from_poses,
    write_transition_outputs,
)
from karagoz.engine.puppet.pose import Pose, part_matrix3
from karagoz.engine.rig.plan import MAX_EXTRA_KEYS


@pytest.fixture(scope="module")
def built() -> tuple:
    puppet, images = build_synth_puppet()
    return puppet, images


@pytest.fixture(scope="module")
def puppet(built: tuple) -> object:
    return built[0]


@pytest.fixture(scope="module")
def images(built: tuple) -> dict:
    return built[1]


@pytest.fixture(scope="module")
def view(puppet: object, images: dict) -> LoadedView:
    return synth_view(puppet, images)  # type: ignore[arg-type]


def test_chain_angles_recovered_within_2deg_with_longer_middle(puppet: object, images: dict, view: LoadedView) -> None:
    long_images = lengthen(images, "fore", (31.0, 92.0), 1.08)
    angles = {"arm": 40.0, "fore": -25.0, "hand": 15.0, "leg": 10.0, "foot": -8.0}
    drawing = draw_pose(puppet, long_images, angles, dict(TRUE_ROOT))  # type: ignore[arg-type]
    pose = fit_view_pose(view, drawing)
    for name in ("arm", "fore", "hand"):
        assert _ang_dist(pose.angles.get(name, 0.0), angles[name]) <= 2.0, name


def test_chain_fit_never_worse_than_part_by_part(puppet: object, images: dict) -> None:
    angles = {"arm": 40.0, "fore": -25.0, "hand": 15.0, "leg": 10.0, "foot": -8.0}
    drawing = draw_pose(puppet, images, angles, dict(TRUE_ROOT))  # type: ignore[arg-type]
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    start = Pose(view="synth", canvas=[160, 160])
    start.tx, start.ty, start.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    partwise = start.model_copy(deep=True)
    for name in ("arm", "fore", "hand"):
        result = snap_part(puppet, partwise, name, drawing, samples, feat)  # type: ignore[arg-type]
        partwise.angles[name] = result.angle
    chained = fit_chain(puppet, start, "hand", drawing, samples, feat)  # type: ignore[arg-type]
    chain = ["arm", "fore", "hand"]
    tau = 2.0
    # Both sides use the joint objective the chain fit maximizes (exclusive
    # pixels with the rest tie-break, same context); the chain fit keeps the
    # better of the two.
    blocked = context_blocked(puppet, start, chain, samples, feat.line.shape)  # type: ignore[arg-type]
    assert chain_objective(puppet, chained, chain, samples, feat, tau, None, blocked) >= (  # type: ignore[arg-type]
        chain_objective(puppet, partwise, chain, samples, feat, tau, None, blocked) - 1e-9
    )


ZERO_ANGLES = {"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 0.0, "foot": 0.0}


def _moved_pair(puppet: object, images: dict) -> tuple:
    angles_x = dict(ZERO_ANGLES)
    angles_y = {"arm": 20.0, "fore": 10.0, "hand": 5.0, "leg": 0.0, "foot": 0.0}
    rgb_x = draw_pose(puppet, images, angles_x, dict(TRUE_ROOT))  # type: ignore[arg-type]
    rgb_y = draw_pose(puppet, images, angles_y, dict(TRUE_ROOT))  # type: ignore[arg-type]
    pose_x = Pose(view="synth", canvas=[160, 160], angles=dict(angles_x), **TRUE_ROOT)
    pose_y = Pose(view="synth", canvas=[160, 160], angles=dict(angles_y), **TRUE_ROOT)
    return rgb_x, rgb_y, pose_x, pose_y, angles_y


@pytest.fixture(scope="module")
def moved(puppet: object, images: dict) -> TransitionResult:
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    rgb_x, rgb_y, pose_x, pose_y, _ = _moved_pair(puppet, images)
    return transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)  # type: ignore[arg-type]


def test_xy_single_arm_move_locks_rest_and_recovers_arm(moved: TransitionResult) -> None:
    assert {"torso", "leg", "foot"} <= set(moved.locked)
    assert not ({"arm", "fore", "hand"} & set(moved.locked))
    for name in moved.locked:
        assert moved.pose_y.angles.get(name, 0.0) == moved.pose_x.angles.get(name, 0.0), name
    assert _ang_dist(moved.pose_y.angles.get("arm", 0.0), 20.0) <= 2.0
    assert _ang_dist(moved.pose_y.angles.get("fore", 0.0), 10.0) <= 2.0
    assert _ang_dist(moved.pose_y.angles.get("hand", 0.0), 5.0) <= 2.0
    assert moved.decision == "rig"


def test_xy_shape_change_gets_ret_with_reason(puppet: object, images: dict) -> None:
    from PIL import Image, ImageDraw

    rgb_x = draw_pose(puppet, images, dict(ZERO_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]
    other = dict(images)
    ring = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ring)
    draw.ellipse([-4, 82, 16, 102], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
    other["hand"] = np.asarray(ring, dtype=np.uint8)
    rgb_y = draw_pose(puppet, other, dict(ZERO_ANGLES), dict(TRUE_ROOT))  # type: ignore[arg-type]
    pose_x = Pose(view="synth", canvas=[160, 160], **TRUE_ROOT)
    pose_y = Pose(view="synth", canvas=[160, 160], **TRUE_ROOT)
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)  # type: ignore[arg-type]
    assert "torso" in result.locked
    assert "hand" not in result.locked
    assert result.decision == "ret"
    assert "hand" in result.reason


def test_gate_split_proposes_intermediates() -> None:
    assert gate_split_needed(10.0) == (False, 0)
    split, need = gate_split_needed(60.0)
    assert split and need >= 1
    _split, capped = gate_split_needed(1000.0)
    assert capped <= MAX_EXTRA_KEYS


def test_transition_sheet_and_json_written(
    tmp_path: Path, puppet: object, images: dict, moved: TransitionResult
) -> None:
    rgb_x, rgb_y, _, _, _ = _moved_pair(puppet, images)
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    out = tmp_path / "xy"
    doc = write_transition_outputs(loaded, moved, rgb_x, rgb_y, out, "X.png", "Y.png")  # type: ignore[arg-type]
    assert set(doc) == {
        "pose_X",
        "pose_Y",
        "locked",
        "changed_chains",
        "decision",
        "reason",
        "scores",
        "confidences",
        "needs_human",
    }
    assert (out / "transition.json").is_file()
    assert (out / "pose_X.json").is_file()
    assert (out / "pose_Y.json").is_file()
    assert (out / "transition_sheet.png").is_file()
    assert json.loads((out / "transition.json").read_text(encoding="utf-8"))["decision"] == "rig"


def test_limits_keep_chain_search_inside(puppet: object, images: dict) -> None:
    limited = puppet.model_copy(deep=True)  # type: ignore[union-attr]
    limited.parts["arm"].limits = [-10.0, 10.0]
    angles = {"arm": 40.0, "fore": -25.0, "hand": 15.0, "leg": 10.0, "foot": -8.0}
    drawing = draw_pose(limited, images, angles, dict(TRUE_ROOT))  # type: ignore[arg-type]
    feat = drawing_features(drawing)
    samples = extract_part_samples(limited, images)  # type: ignore[arg-type]
    start = Pose(view="synth", canvas=[160, 160])
    start.tx, start.ty, start.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    fitted = fit_chain(limited, start, "hand", drawing, samples, feat)  # type: ignore[arg-type]
    assert -10.0 <= float(fitted.angles.get("arm", 999.0)) <= 10.0


def test_explained_pixels_decay_chain_scores(puppet: object, images: dict) -> None:
    import cv2

    true = {"arm": -80.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
    rigid = {"arm": 140.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
    drawing = draw_pose(puppet, images, true, dict(TRUE_ROOT))  # type: ignore[arg-type]
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    height, width = feat.line.shape

    def pose_at(angles: dict[str, float]) -> Pose:
        pose = Pose(view="synth", canvas=[160, 160], angles=dict(angles))
        pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
        return pose

    def cover_mask(angles: dict[str, float]) -> np.ndarray:
        mask = np.zeros((height, width), np.bool_)
        pose = pose_at(angles)
        for name in ("arm", "fore", "hand"):
            mat = np.asarray(part_matrix3(puppet, pose, name)[:2], dtype=np.float64)  # type: ignore[arg-type]
            for pts in (samples[name].line_xy, samples[name].flat_xy):  # type: ignore[union-attr]
                if not len(pts):
                    continue
                warped = np.round(_warp_points(np.asarray(pts, dtype=np.float64), mat)).astype(np.int64)
                valid = (warped[:, 0] >= 0) & (warped[:, 1] >= 0) & (warped[:, 0] < width) & (warped[:, 1] < height)
                mask[warped[valid, 1], warped[valid, 0]] = True
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return np.asarray(grown, dtype=np.bool_)

    chain = ["arm", "fore", "hand"]
    tau = 4.0
    true_pose, rigid_pose = pose_at(true), pose_at(rigid)
    plain_true = chain_objective(puppet, true_pose, chain, samples, feat, tau)  # type: ignore[arg-type]
    plain_rigid = chain_objective(puppet, rigid_pose, chain, samples, feat, tau)  # type: ignore[arg-type]
    assert plain_true > plain_rigid
    explained = cover_mask(rigid)
    decayed_true = chain_objective(puppet, true_pose, chain, samples, feat, tau, None, None, explained)  # type: ignore[arg-type]
    decayed_rigid = chain_objective(puppet, rigid_pose, chain, samples, feat, tau, None, None, explained)  # type: ignore[arg-type]
    # Agreement on explained pixels decays: the rigid chain lying on explained
    # pixels loses most of its score, the true chain on unexplained pixels
    # keeps its own, so unexplained placement wins by a wider margin.
    assert decayed_rigid < plain_rigid - 0.5
    # The true hang shares only the shoulder pivot region with the mask, so it
    # keeps nearly all of its score (a broken decay would cost most of it).
    assert abs(decayed_true - plain_true) < 0.25
    assert decayed_true > decayed_rigid


def test_limits_do_not_harm_inside_recovery(puppet: object, images: dict) -> None:
    limited = puppet.model_copy(deep=True)  # type: ignore[union-attr]
    limited.parts["arm"].limits = [-10.0, 10.0]
    angles = {"arm": 5.0, "fore": -25.0, "hand": 15.0, "leg": 10.0, "foot": -8.0}
    drawing = draw_pose(limited, images, angles, dict(TRUE_ROOT))  # type: ignore[arg-type]
    feat = drawing_features(drawing)
    samples = extract_part_samples(limited, images)  # type: ignore[arg-type]
    start = Pose(view="synth", canvas=[160, 160])
    start.tx, start.ty, start.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    fitted = fit_chain(limited, start, "hand", drawing, samples, feat)  # type: ignore[arg-type]
    assert _ang_dist(fitted.angles.get("arm", 0.0), 5.0) <= 2.0


def test_xy_far_move_pulled_from_scratch_start(puppet: object, images: dict) -> None:
    angles_x = dict(ZERO_ANGLES)
    angles_y = {"arm": 150.0, "fore": 0.0, "hand": 0.0, "leg": 0.0, "foot": 0.0}
    rgb_x = draw_pose(puppet, images, angles_x, dict(TRUE_ROOT))  # type: ignore[arg-type]
    rgb_y = draw_pose(puppet, images, angles_y, dict(TRUE_ROOT))  # type: ignore[arg-type]
    pose_x = Pose(view="synth", canvas=[160, 160], angles=dict(angles_x), **TRUE_ROOT)
    pose_y = Pose(view="synth", canvas=[160, 160], angles=dict(angles_y), **TRUE_ROOT)
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)  # type: ignore[arg-type]
    # A 150 deg move exceeds the X-anchored pull window: the pull must also
    # try the chain's from-scratch fit on Y and keep the best, so the move
    # is reached (a justified split) instead of ret.
    assert _ang_dist(result.pose_y.angles.get("arm", 0.0), 150.0) <= 2.0
    assert result.decision == "split"


def test_difference_map_marks_only_the_moved_chain(puppet: object, images: dict) -> None:
    rgb_x, rgb_y, pose_x, _, _ = _moved_pair(puppet, images)
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    locked, _metrics = difference_locked(puppet, pose_x, rgb_x, rgb_y, samples)  # type: ignore[arg-type]
    assert {"torso", "leg", "foot"} <= set(locked)
    assert not ({"arm", "fore", "hand"} & set(locked))


def test_fit_chain_frozen_member_keeps_angle(puppet: object, images: dict) -> None:
    angles = {"arm": 40.0, "fore": -25.0, "hand": 15.0, "leg": 10.0, "foot": -8.0}
    drawing = draw_pose(puppet, images, angles, dict(TRUE_ROOT))  # type: ignore[arg-type]
    feat = drawing_features(drawing)
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    start = Pose(view="synth", canvas=[160, 160])
    start.tx, start.ty, start.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    fitted = fit_chain(puppet, start, "hand", drawing, samples, feat, frozen={"fore": -25.0})  # type: ignore[arg-type]
    assert fitted.angles.get("fore", 999.0) == -25.0
    assert _ang_dist(fitted.angles.get("arm", 0.0), 40.0) <= 2.0
    assert _ang_dist(fitted.angles.get("hand", 0.0), 15.0) <= 2.0


def test_xy_hand_only_move_keeps_locked_chain_members(puppet: object, images: dict) -> None:
    angles_x = dict(ZERO_ANGLES)
    angles_y = {"arm": 0.0, "fore": 0.0, "hand": 15.0, "leg": 0.0, "foot": 0.0}
    rgb_x = draw_pose(puppet, images, angles_x, dict(TRUE_ROOT))  # type: ignore[arg-type]
    rgb_y = draw_pose(puppet, images, angles_y, dict(TRUE_ROOT))  # type: ignore[arg-type]
    pose_x = Pose(view="synth", canvas=[160, 160], angles=dict(angles_x), **TRUE_ROOT)
    pose_y = Pose(view="synth", canvas=[160, 160], angles=dict(angles_y), **TRUE_ROOT)
    loaded = synth_view(puppet, images)  # type: ignore[arg-type]
    result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)  # type: ignore[arg-type]
    assert {"arm", "fore"} <= set(result.locked)
    assert "hand" not in result.locked
    # Locked members of the changed chain keep X's angles exactly: the pull
    # must not move what the difference map locked (pixel-identical, no
    # flicker by construction).
    for name in ("arm", "fore"):
        assert result.pose_y.angles.get(name, 999.0) == pose_x.angles.get(name, 0.0), name
    assert _ang_dist(result.pose_y.angles.get("hand", 0.0), 15.0) <= 2.0
    assert result.decision == "rig"
