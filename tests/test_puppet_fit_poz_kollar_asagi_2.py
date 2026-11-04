"""K-1-ka2: a straight chain must not fold back onto its own dense tube.

On ``kollar_asagi_2`` both forearms folded onto their own upper-arm tubes:
the straight continuation's tubes came back weak (narrow, flats spilling
onto paper) while the own tube carried dense panel lines in the same
colour. Samples stacked on the chain's own tube keep full agreement there
(``explained_mask`` skips the whole chain, and ``STACK_PENALTY`` fires
only at zero owned pixels), so the fold beats the true straight chain
beside it. The chain objective discounts agreement on pixels another
member of the same chain covers (sibling-stack decay), so the straight
chain on unexplained drawing pixels wins.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from test_puppet_fit_common import TRUE_ROOT, build_synth_puppet, draw_pose

from karagoz.engine.puppet.fit import (
    _ang_dist,
    chain_objective,
    drawing_features,
    extract_part_samples,
    fit_chain,
)
from karagoz.engine.puppet.pose import Pose

STRAIGHT = {"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
FOLD = {"arm": 0.0, "fore": 180.0, "hand": 0.0, "leg": 10.0, "foot": 0.0}
CHAIN = ["arm", "fore", "hand"]
TAU = 4.0


def redrawn_images(images: dict) -> dict:
    """How the model redrew the design (ka2): the arm tube carries dense
    panel lines (extra black stripes, same fill) while the forearm and
    hand tubes come back narrower, so a fold stacked on the arm tube
    outscores the true straight chain beside it."""
    out = dict(images)
    arm = Image.fromarray(np.asarray(images["arm"], dtype=np.uint8))
    draw = ImageDraw.Draw(arm)
    for y in range(78, 106, 2):
        draw.line([(30, y), (72, y)], fill=(0, 0, 0, 255), width=1)
    out["arm"] = np.asarray(arm, dtype=np.uint8)
    tube = (150, 70, 70, 255)
    dark = (50, 50, 50, 255)
    black = (0, 0, 0, 255)
    narrow_fore = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(narrow_fore)
    draw.rectangle([8, 89, 32, 95], fill=tube, outline=black, width=2)
    draw.rectangle([18, 95, 30, 99], fill=tube, outline=black, width=2)
    out["fore"] = np.asarray(narrow_fore, dtype=np.uint8)
    narrow_hand = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(narrow_hand)
    draw.rectangle([0, 87, 12, 97], fill=dark, outline=black, width=2)
    out["hand"] = np.asarray(narrow_hand, dtype=np.uint8)
    return out


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
def drawing(puppet: object, images: dict) -> object:
    drawn = redrawn_images(images)  # type: ignore[arg-type]
    return draw_pose(puppet, drawn, dict(STRAIGHT), dict(TRUE_ROOT))  # type: ignore[arg-type]


def _pose_at(angles: dict[str, float]) -> Pose:
    pose = Pose(view="synth", canvas=[160, 160], angles=dict(angles))
    pose.tx, pose.ty, pose.scale = TRUE_ROOT["tx"], TRUE_ROOT["ty"], TRUE_ROOT["scale"]
    return pose


def _striped_images(images: dict) -> dict:
    """Copy of ``images`` whose arm tube carries dense panel lines."""
    out = dict(images)
    arm = Image.fromarray(np.asarray(images["arm"], dtype=np.uint8))
    draw = ImageDraw.Draw(arm)
    for y in range(78, 106, 2):
        draw.line([(30, y), (72, y)], fill=(0, 0, 0, 255), width=1)
    out["arm"] = np.asarray(arm, dtype=np.uint8)
    return out


def test_explained_mask_covers_dense_lines() -> None:
    from karagoz.engine.puppet.fit import LINE_LUMA, explained_mask, extract_part_samples
    from karagoz.engine.puppet.parts import line_mask

    _puppet, images = build_synth_puppet()
    striped = _striped_images(images)
    # Sparsely strided scoring samples (as in production): the explanation
    # cover must still follow every dense panel line, not the stride gaps.
    samples = extract_part_samples(_puppet, striped, max_line=50)  # type: ignore[arg-type]
    assert len(samples["arm"].line_xy) <= 50
    # Identity root: reference pixels land on themselves, so the arm layer's
    # own ink pixels must be explained by the arm's cover.
    from karagoz.engine.puppet.pose import Pose

    pose = Pose(view="synth", canvas=[160, 160], angles={"arm": 0.0, "fore": 0.0, "hand": 0.0, "leg": 0.0, "foot": 0.0})
    mask = explained_mask(_puppet, pose, samples, (160, 160), skip="fore")  # type: ignore[arg-type]
    layer = np.asarray(striped["arm"], dtype=np.uint8)
    ink = line_mask(layer[..., :3], LINE_LUMA) & (layer[..., 3] > 127)
    assert ink.sum() > 500
    assert float(mask[ink].mean()) >= 0.9


def test_straight_beats_own_tube_fold(puppet: object, images: dict, drawing: object) -> None:
    feat = drawing_features(drawing)  # type: ignore[arg-type]
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]

    def objective(angles: dict[str, float]) -> float:
        return chain_objective(puppet, _pose_at(angles), CHAIN, samples, feat, TAU)  # type: ignore[arg-type]

    assert objective(dict(STRAIGHT)) > objective(dict(FOLD))


def test_straight_chain_stays_straight_not_folds(puppet: object, images: dict, drawing: object) -> None:
    feat = drawing_features(drawing)  # type: ignore[arg-type]
    samples = extract_part_samples(puppet, images)  # type: ignore[arg-type]
    start = _pose_at(dict(FOLD))
    fitted = fit_chain(puppet, start, "hand", drawing, samples, feat)  # type: ignore[arg-type]
    assert _ang_dist(fitted.angles.get("arm", 0.0), 0.0) <= 5.0
    assert _ang_dist(fitted.angles.get("fore", 0.0), 0.0) <= 5.0
    # The redrawn hand tube is narrow, so the fit parks the hand within half
    # a right angle; what must not happen is the fold-back (~180).
    assert _ang_dist(fitted.angles.get("hand", 0.0), 0.0) <= 45.0
