"""K-2b fast anti-aliased renderer: determinism, identity band, AA edges, crops."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from test_puppet_fit_common import TRUE_ANGLES, build_synth_puppet

from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.pose import Pose, render_pose
from karagoz.engine.puppet.render_aa import PreparedPart, PreparedParts, part_masks, prepare_parts, render_frame


@pytest.fixture()
def built() -> tuple[Puppet, dict[str, NDArray[np.uint8]], PreparedParts]:
    puppet, images = build_synth_puppet()
    return puppet, images, prepare_parts(puppet, images)


def test_same_pose_renders_identical_pixels(built: tuple[Puppet, dict[str, NDArray[np.uint8]], PreparedParts]) -> None:
    puppet, _, prepared = built
    pose = Pose(view="synth", canvas=[160, 160], angles=dict(TRUE_ANGLES), tx=5.0, ty=-3.0)
    first = render_frame(puppet, prepared, pose)
    second = render_frame(puppet, prepared, pose)
    assert np.array_equal(first, second)


def test_identity_pose_matches_render_pose_outside_2px_edge_band(
    built: tuple[Puppet, dict[str, NDArray[np.uint8]], PreparedParts],
) -> None:
    puppet, images, prepared = built
    pose = Pose(view="synth", canvas=[160, 160])
    ref = render_pose(puppet, pose, images)
    aa = render_frame(puppet, prepared, pose)
    assert ref.shape == aa.shape
    cover = (ref[..., 3] > 127).astype(np.uint8) * np.uint8(255)
    kernel = np.ones((5, 5), np.uint8)
    band = cv2.dilate(cover, kernel) != cv2.erode(cover, kernel)
    diff = np.any(ref != aa, axis=2)
    assert int(np.count_nonzero(diff & ~band)) == 0


def test_rotated_edge_has_intermediate_alpha(
    built: tuple[Puppet, dict[str, NDArray[np.uint8]], PreparedParts],
) -> None:
    puppet, images, prepared = built
    pose = Pose(view="synth", canvas=[160, 160], angles={"arm": 30.0, "fore": -20.0})
    aa = render_frame(puppet, prepared, pose)
    assert int(np.count_nonzero((aa[..., 3] > 0) & (aa[..., 3] < 255))) > 0
    ref = render_pose(puppet, pose, images)
    assert int(np.count_nonzero((ref[..., 3] > 0) & (ref[..., 3] < 255))) == 0


def _premultiply_full(img: NDArray[np.uint8]) -> PreparedPart:
    flat = np.asarray(img, dtype=np.float32) / np.float32(255.0)
    alpha = flat[..., 3:4]
    pre = np.concatenate([flat[..., :3] * alpha, alpha], axis=2)
    return PreparedPart(
        image=np.ascontiguousarray(pre, dtype=np.float32),
        offset=(0, 0),
        full_shape=(int(img.shape[0]), int(img.shape[1])),
    )


def test_cropping_changes_nothing_max_abs_diff_1(
    built: tuple[Puppet, dict[str, NDArray[np.uint8]], PreparedParts],
) -> None:
    puppet, images, prepared = built
    full = PreparedParts(parts={name: _premultiply_full(img) for name, img in images.items()})
    pose = Pose(view="synth", canvas=[160, 160], angles={"arm": 30.0, "fore": -20.0, "hand": 10.0})
    cropped = render_frame(puppet, prepared, pose)
    uncropped = render_frame(puppet, full, pose)
    assert int(np.abs(cropped.astype(np.int16) - uncropped.astype(np.int16)).max()) <= 1


def test_part_masks_match_render_pose_cover_at_identity(
    built: tuple[Puppet, dict[str, NDArray[np.uint8]], PreparedParts],
) -> None:
    puppet, images, prepared = built
    pose = Pose(view="synth", canvas=[160, 160])
    ref = render_pose(puppet, pose, images)
    masks = part_masks(puppet, prepared, pose)
    assert set(masks) == {name for name, part in puppet.parts.items() if not part.hidden and name in images}
    for _name, mask in masks.items():
        assert mask.shape == (160, 160)
        assert mask.dtype == np.bool_
    union = np.zeros((160, 160), dtype=np.bool_)
    for mask in masks.values():
        union |= mask
    assert np.array_equal(union, ref[..., 3] > 127)
