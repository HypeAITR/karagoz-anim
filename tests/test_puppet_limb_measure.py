"""Tests for limb-by-limb measurement (`limb_measure.py`) and engine rules (K-2e)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from karagoz.engine.puppet.fit import (
    LoadedView,
    load_drawing,
    load_view,
    transition_from_poses,
)
from karagoz.engine.puppet.limb_measure import (
    compare_pose_limbs,
    compare_transition_limbs,
    detect_unchanged_chain_members,
    format_pose_limb_table,
    format_transition_limb_table,
)
from karagoz.engine.puppet.model import Puppet, PuppetPart
from karagoz.engine.puppet.pose import Pose, render_pose


def _build_general_puppet(view_name: str = "front") -> tuple[Puppet, dict[str, np.ndarray]]:
    """Build a synthetic puppet in a given view with distinct landmark caps and paired lobes."""
    parts = {
        "torso": PuppetPart(
            parent=None,
            pivot=[120.0, 110.0],
            z=0,
            file="torso.png",
        ),
        "hip_cap": PuppetPart(
            parent="torso",
            pivot=[120.0, 145.0],
            z=1,
            file="hip_cap.png",
            limits=[-45.0, 45.0],
        ),
        "thigh": PuppetPart(
            parent="hip_cap",
            pivot=[120.0, 148.0],
            z=2,
            file="thigh.png",
            limits=[-60.0, 60.0],
        ),
        "knee_ring": PuppetPart(
            parent="thigh",
            pivot=[120.0, 200.0],
            z=3,
            file="knee_ring.png",
            limits=[-45.0, 45.0],
        ),
        "shin": PuppetPart(
            parent="knee_ring",
            pivot=[120.0, 203.0],
            z=4,
            file="shin.png",
            limits=[-60.0, 60.0],
        ),
        "foot_claw": PuppetPart(
            parent="shin",
            pivot=[120.0, 250.0],
            z=5,
            file="foot_claw.png",
            limits=[-45.0, 45.0],
        ),
    }
    puppet = Puppet(
        version=1,
        reference=f"{view_name}.png",
        canvas=[240, 320],
        line_width_px=2.0,
        root="torso",
        parts=parts,
    )

    def _blank() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        im = Image.new("RGBA", (240, 320), (0, 0, 0, 0))
        return im, ImageDraw.Draw(im)

    im_torso, d = _blank()
    d.rectangle([85, 65, 155, 148], fill=(180, 190, 210, 255), outline=(10, 10, 10, 255), width=2)

    im_hip, d = _blank()
    d.ellipse([106, 133, 134, 159], fill=(165, 45, 55, 255), outline=(10, 10, 10, 255), width=2)

    im_thigh, d = _blank()
    d.rectangle([109, 148, 131, 202], fill=(130, 180, 225, 255), outline=(10, 10, 10, 255), width=2)

    im_knee, d = _blank()
    d.ellipse([107, 189, 133, 213], fill=(165, 45, 55, 255), outline=(10, 10, 10, 255), width=2)

    im_shin, d = _blank()
    d.rectangle([110, 203, 130, 252], fill=(130, 180, 225, 255), outline=(10, 10, 10, 255), width=2)

    im_foot, d = _blank()
    # Two paired distal lobes (left claw and right claw) in maroon.
    d.polygon(
        [(102, 248), (116, 248), (114, 282), (100, 282)],
        fill=(165, 45, 55, 255),
        outline=(10, 10, 10, 255),
    )
    d.polygon(
        [(124, 248), (138, 248), (140, 282), (126, 282)],
        fill=(165, 45, 55, 255),
        outline=(10, 10, 10, 255),
    )

    images = {
        "torso": np.asarray(im_torso, dtype=np.uint8),
        "hip_cap": np.asarray(im_hip, dtype=np.uint8),
        "thigh": np.asarray(im_thigh, dtype=np.uint8),
        "knee_ring": np.asarray(im_knee, dtype=np.uint8),
        "shin": np.asarray(im_shin, dtype=np.uint8),
        "foot_claw": np.asarray(im_foot, dtype=np.uint8),
    }
    return puppet, images


def _render_rgb(puppet: Puppet, pose: Pose, images: dict[str, np.ndarray]) -> np.ndarray:
    rgba = render_pose(puppet, pose, images)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def test_identity_self_comparison_gives_zero_diff_and_ok() -> None:
    """Counter-metric (3): Comparing a pose with itself yields 0 diff and 'ok' for every limb."""
    puppet, images = _build_general_puppet("front")
    pose = Pose(view="front", canvas=[240, 320], angles={"thigh": 5.0, "shin": -3.0, "foot_claw": 4.0})
    rgb = _render_rgb(puppet, pose, images)

    report = compare_pose_limbs(puppet, images, pose, rgb, rgb, pose_name="identity_front")
    assert report.counts["motor_error"] == 0
    assert report.counts["drawing_diff"] == 0
    assert report.counts["ok"] == len(report.limbs) > 0
    for limb in report.limbs:
        assert limb.status == "ok"
        assert limb.length_diff_px == 0.0
        assert limb.width_diff_px == 0.0
        assert limb.joint_shift_px == 0.0
        assert limb.abs_angle_diff_deg == 0.0

    trans = compare_transition_limbs(
        puppet, images, pose, rgb, rgb, pose, rgb, rgb, pose_a_name="id1", pose_b_name="id2"
    )
    assert trans.counts["motor_error"] == 0
    for tl in trans.limbs:
        assert tl.status == "ok"
        assert tl.drawing_angle_change_deg == 0.0
        assert tl.puppet_angle_change_deg == 0.0
        assert not tl.changed_in_drawing


def test_pose_exam_separates_drawing_diff_from_motor_error() -> None:
    """Pose exam classifies stretched/thickened bone as drawing_diff and rotated leaf as motor_error."""
    puppet, images = _build_general_puppet("profile_right")
    pose_draw = Pose(view="profile_right", canvas=[240, 320], angles={})

    # 1. Create a drawing where `shin` is 12% longer (drawing_diff), while puppet is at rest.
    images_stretched = dict(images)
    im_shin = Image.new("RGBA", (240, 320), (0, 0, 0, 0))
    d = ImageDraw.Draw(im_shin)
    d.rectangle([110, 203, 130, 259], fill=(130, 180, 225, 255), outline=(10, 10, 10, 255), width=2)
    im_foot = Image.new("RGBA", (240, 320), (0, 0, 0, 0))
    df = ImageDraw.Draw(im_foot)
    df.polygon(
        [(102, 255), (116, 255), (114, 289), (100, 289)],
        fill=(165, 45, 55, 255),
        outline=(10, 10, 10, 255),
    )
    df.polygon(
        [(124, 255), (138, 255), (140, 289), (126, 289)],
        fill=(165, 45, 55, 255),
        outline=(10, 10, 10, 255),
    )
    images_stretched["shin"] = np.asarray(im_shin, dtype=np.uint8)
    images_stretched["foot_claw"] = np.asarray(im_foot, dtype=np.uint8)

    rgb_draw_diff = _render_rgb(puppet, pose_draw, images_stretched)
    rgb_pup_rest = _render_rgb(puppet, pose_draw, images)

    rep_diff = compare_pose_limbs(puppet, images, pose_draw, rgb_draw_diff, rgb_pup_rest, pose_name="stretch_case")
    by_name = {e.name: e for e in rep_diff.limbs}
    assert by_name["shin"].status == "drawing_diff"
    assert rep_diff.counts["motor_error"] == 0

    # 2. Create a motor_error case: drawing has foot_claw at 0 deg, puppet rotated it to +16 deg.
    pose_bad_motor = Pose(view="profile_right", canvas=[240, 320], angles={"foot_claw": 16.0})
    rgb_pup_motor = _render_rgb(puppet, pose_bad_motor, images)
    rep_motor = compare_pose_limbs(puppet, images, pose_bad_motor, rgb_pup_rest, rgb_pup_motor, pose_name="motor_case")
    by_name_m = {e.name: e for e in rep_motor.limbs}
    assert by_name_m["foot_claw"].status == "motor_error"
    assert by_name_m["foot_claw"].abs_angle_diff_deg > 10.0
    assert "| `foot_claw` |" in format_pose_limb_table(rep_motor)


def test_transition_exam_flags_wobble_and_locks_unchanged_limb() -> None:
    """Transition exam catches leaf wobble (d_k >> d_c) and transition_from_poses locks still limbs."""
    puppet, images = _build_general_puppet("front")
    pose_a_draw = Pose(view="front", canvas=[240, 320], angles={"foot_claw": 0.0})
    pose_b_draw = Pose(view="front", canvas=[240, 320], angles={"foot_claw": 1.0})
    rgb_a = _render_rgb(puppet, pose_a_draw, images)
    rgb_b = _render_rgb(puppet, pose_b_draw, images)

    # Simulate an engine error where the puppet swings foot_claw from -12 deg to +11 deg.
    pose_a_bad = Pose(view="front", canvas=[240, 320], angles={"foot_claw": -12.0})
    pose_b_bad = Pose(view="front", canvas=[240, 320], angles={"foot_claw": 11.0})
    pup_a_bad = _render_rgb(puppet, pose_a_bad, images)
    pup_b_bad = _render_rgb(puppet, pose_b_bad, images)

    trans_bad = compare_transition_limbs(
        puppet,
        images,
        pose_a_bad,
        rgb_a,
        pup_a_bad,
        pose_b_bad,
        rgb_b,
        pup_b_bad,
        pose_a_name="a",
        pose_b_name="b",
    )
    assert "foot_claw" in trans_bad.motor_error_limbs
    assert "foot_claw" in trans_bad.unchanged_limbs
    assert "motor_error" in format_transition_limb_table(trans_bad)

    # Verify Rule 2(b): transition_from_poses locks the unchanged chain so foot_claw does not wobble.
    unchanged = detect_unchanged_chain_members(puppet, images, pose_a_bad, rgb_a, pose_b_bad, rgb_b)
    assert "foot_claw" in unchanged

    loaded = LoadedView(
        name="front",
        puppet=puppet,
        images=images,
    )
    res = transition_from_poses(loaded, pose_a_bad, pose_b_bad, rgb_a, rgb_b)
    assert "foot_claw" in res.locked
    assert res.pose_y.angles.get("foot_claw") == pose_a_bad.angles.get("foot_claw")


def test_baseline_kabul_k0i_caught_as_motor_error_and_fixed() -> None:
    """Verify on real workspace data (`_wp/out/kabul_k0i`) that `hand_l` wobble is caught as motor_error."""
    set_dir = Path("_wp/out/k0i_set")
    base_dir = Path("_wp/out/kabul_k0i")
    if not set_dir.exists() or not base_dir.exists():
        pytest.skip("reviewer data (_wp/out/k0i_set, kabul_k0i) not in this checkout")

    loaded = load_view(set_dir, "3q_right")
    rgb_1 = load_drawing(Path("_wp/pozlar/kollar_asagi_1.png"))
    rgb_2 = load_drawing(Path("_wp/pozlar/kollar_asagi_2.png"))
    pup_1 = load_drawing(base_dir / "kollar_asagi_1/render.png")
    pup_2 = load_drawing(base_dir / "kollar_asagi_2/render.png")
    pose_1 = Pose.model_validate_json((base_dir / "kollar_asagi_1/pose.json").read_text(encoding="utf-8"))
    pose_2 = Pose.model_validate_json((base_dir / "kollar_asagi_2/pose.json").read_text(encoding="utf-8"))

    # Identity check on real pose:
    id_rep = compare_pose_limbs(loaded.puppet, loaded.images, pose_1, rgb_1, rgb_1, pose_name="kollar_asagi_1_id")
    assert id_rep.counts["motor_error"] == 0
    assert id_rep.counts["drawing_diff"] == 0

    # Baseline transition exam must catch `hand_l` as `motor_error`:
    trans_before = compare_transition_limbs(
        loaded.puppet,
        loaded.images,
        pose_1,
        rgb_1,
        pup_1,
        pose_2,
        rgb_2,
        pup_2,
        pose_a_name="kollar_asagi_1",
        pose_b_name="kollar_asagi_2",
    )
    assert "hand_l" in trans_before.motor_error_limbs
    by_name = {e.name: e for e in trans_before.limbs}
    assert by_name["hand_l"].drawing_angle_change_deg < 3.0
    assert by_name["hand_l"].puppet_angle_change_deg > 10.0

    # After the rule: the drawing's far hand did not change, so the transition locks it at X.
    res = transition_from_poses(loaded, pose_1, pose_2, rgb_1, rgb_2)
    assert "hand_l" in res.locked
    assert res.pose_y.angles.get("hand_l") == pose_1.angles.get("hand_l")


def test_next_interval_starts_from_the_held_value() -> None:
    """A member locked in X->Y keeps X's value at Y, so Y->Z starts from it (no jump at the boundary)."""
    from karagoz.engine.puppet.fit import carry_locked

    own_fit_y = {"view": "front", "angles": {"foot_claw": 11.0, "leg": 5.0}, "offset": {"foot_claw": [2.0, 1.0]}}
    prev = {
        "locked": ["foot_claw", "torso"],
        "pose_Y": {"view": "front", "angles": {"foot_claw": -12.0, "leg": 4.0}, "offset": {}},
    }
    got = carry_locked(own_fit_y, prev)
    assert got["angles"]["foot_claw"] == -12.0  # held value, not Y's own fit
    assert got["angles"]["leg"] == 5.0  # not locked: Y's own fit stays
    assert "foot_claw" not in got["offset"]  # held offset (none) carried too
    assert "torso" not in got["angles"]
    assert own_fit_y["angles"]["foot_claw"] == 11.0  # input untouched
