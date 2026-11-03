"""K-2d inbetween run: poses, holds, z switch, measures, files (inbetween.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from test_puppet_fit_common import build_synth_puppet, draw_pose

from karagoz.engine.puppet.inbetween import FRAME_KEYS, REPORT_KEYS, inbetween
from karagoz.engine.puppet.interp import interpolate, interpolate_sequence
from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.pose import Pose, z_order
from karagoz.engine.puppet.render_aa import part_masks, prepare_parts

LOCKED = ["torso", "arm", "leg", "foot"]


def _pose(angles: dict[str, float], z_override: dict[str, int] | None = None) -> Pose:
    return Pose(view="synth", angles=dict(angles), z_override=dict(z_override or {}), canvas=[160, 160])


def _write_synth_set(tmp_path: Path, puppet: Puppet, images: dict[str, np.ndarray]) -> Path:
    """A one-view puppet set on disk (what ``load_view`` reads)."""
    set_dir = tmp_path / "set"
    view_dir = set_dir / "views" / "synth"
    (view_dir / "parts").mkdir(parents=True)
    parts = {}
    for name, part in puppet.parts.items():
        Image.fromarray(np.asarray(images[name])).save(view_dir / "parts" / f"{name}.png")
        doc = part.model_dump()
        doc["file"] = f"parts/{name}.png"
        parts[name] = doc
    doc = puppet.model_dump()
    doc["parts"] = parts
    (view_dir / "puppet.json").write_text(json.dumps(Puppet.model_validate(doc).model_dump()) + "\n")
    (set_dir / "set.json").write_text(
        json.dumps(
            {
                "version": 1,
                "character": "synth",
                "views": {"synth": {"yaw_deg": 0.0, "dir": "views/synth"}},
                "default_view": "synth",
                "head_part": "torso",
            }
        )
        + "\n"
    )
    return set_dir


def _write_transition(
    tmp_path: Path,
    puppet: Puppet,
    images: dict[str, np.ndarray],
    pose_x: Pose,
    pose_y: Pose,
    *,
    decision: str = "rig",
    x_z: dict[str, int] | None = None,
    y_z: dict[str, int] | None = None,
) -> Path:
    """Keys as white drawings plus the K-1 transition document."""
    tdir = tmp_path / "t"
    tdir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_pose(puppet, images, pose_x.angles, z_override=x_z)).save(tdir / "X.png")
    Image.fromarray(draw_pose(puppet, images, pose_y.angles, z_override=y_z)).save(tdir / "Y.png")
    doc = {
        "x": "X.png",
        "y": "Y.png",
        "pose_X": pose_x.model_dump(mode="json"),
        "pose_Y": pose_y.model_dump(mode="json"),
        "locked": LOCKED,
        "decision": decision,
        "reason": "synthetic K-2d interval",
    }
    (tdir / "transition.json").write_text(json.dumps(doc) + "\n")
    return tdir / "transition.json"


def _rgba(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGBA"))


def test_ret_decision_writes_nothing_and_ok_false(tmp_path: Path) -> None:
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    tj = _write_transition(tmp_path, puppet, images, _pose({"fore": 0.0}), _pose({"fore": 30.0}), decision="ret")
    out = tmp_path / "out"
    res = inbetween(set_dir, tj, 3, out)
    assert res.ok is False
    assert res.frames == []
    assert set(res.report) == {"decision", "reason", "ok"}
    assert res.report["decision"] == "ret"
    assert not out.exists()  # a non-rig decision writes nothing


def test_key_holds_first_and_last_equal_key_drawings(tmp_path: Path) -> None:
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    pose_x, pose_y = _pose({"arm": 10.0, "fore": 0.0}), _pose({"arm": 10.0, "fore": 30.0})
    tj = _write_transition(tmp_path, puppet, images, pose_x, pose_y)
    out = tmp_path / "out"
    res = inbetween(set_dir, tj, 3, out)
    assert set(res.report) == set(REPORT_KEYS)
    assert all(set(row) == set(FRAME_KEYS) for row in res.report["frames"])
    assert len(res.frames) == 3 == res.report["n_frames"]
    key_x = np.asarray(Image.open(tj.parent / "X.png").convert("RGB"))
    key_y = np.asarray(Image.open(tj.parent / "Y.png").convert("RGB"))
    assert np.array_equal(_rgba(res.frames[0])[..., :3], key_x)  # hold: frame 0 is X.png
    assert np.array_equal(_rgba(res.frames[-1])[..., :3], key_y)  # hold: last frame is Y.png
    assert (out / "inbetween.json").is_file()
    assert (out / "preview.mp4").is_file() and (out / "preview.mp4").stat().st_size > 0
    assert (out / "inbetween_sheet.png").is_file()
    assert [row["t"] for row in res.report["frames"]] == [0.0, 0.5, 1.0]
    assert res.report["cut"]["start"].keys() == {"silhouette_px", "line_shift_px"}
    assert res.report["cut"]["end"].keys() == {"silhouette_px", "line_shift_px"}


def test_locked_parts_pixel_identical_across_frames(tmp_path: Path) -> None:
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    tj = _write_transition(
        tmp_path, puppet, images, _pose({"arm": 10.0, "fore": 0.0}), _pose({"arm": 10.0, "fore": 30.0})
    )
    res = inbetween(set_dir, tj, 3, tmp_path / "out", hold_from_puppet=True)
    assert len(res.frames) == 3
    assert [row["locked_change_px"] for row in res.report["frames"]] == [0, 0, 0]
    assert any(not np.array_equal(_rgba(path), _rgba(res.frames[0])) for path in res.frames[1:])


def test_z_switch_lands_on_first_non_overlap_frame(tmp_path: Path) -> None:
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    # Only (arm, fore) flips: the z=1 tie breaks arm-first at X, fore drops to z=0 at Y.
    pose_x = _pose({"arm": 10.0, "fore": 0.0})
    pose_y = _pose({"arm": 10.0, "fore": 40.0}, {"fore": 0})
    tj = _write_transition(tmp_path, puppet, images, pose_x, pose_y, x_z={}, y_z={"fore": 0})
    res = inbetween(set_dir, tj, 5, tmp_path / "out")
    assert len(res.report["z_switch_frames"]) == 1
    switch = res.report["z_switch_frames"][0]
    # The only order-differing pair, found again from the public z_order:
    rank_x = {n: i for i, n in enumerate(z_order(puppet, pose_x))}
    rank_y = {n: i for i, n in enumerate(z_order(puppet, pose_y))}
    flipped = [
        (a, b)
        for a in puppet.parts
        for b in puppet.parts
        if a < b and (rank_x[a] < rank_x[b]) != (rank_y[a] < rank_y[b])
    ]
    assert flipped == [("arm", "fore")]
    seq = interpolate_sequence(puppet, pose_x, pose_y, 5, locked=LOCKED)
    prepared = prepare_parts(puppet, images)
    counts = []
    for pose in seq:
        masks = part_masks(puppet, prepared, pose)
        counts.append(int(np.logical_and(masks["arm"], masks["fore"]).sum()))
    first_open = next((i for i, c in enumerate(counts) if c == 0), None)
    expected = first_open if first_open is not None else int(np.argmin(np.asarray(counts)))
    assert switch == expected


def test_future_part_field_does_not_break_the_run(tmp_path: Path) -> None:
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    # K-1's in-flight `slack_px` on parts: the run strips what its model
    # does not know yet instead of failing on the reviewer's sets.
    doc = json.loads((set_dir / "views" / "synth" / "puppet.json").read_text())
    doc["parts"]["arm"]["slack_px"] = 45.5
    (set_dir / "views" / "synth" / "puppet.json").write_text(json.dumps(doc))
    tj = _write_transition(
        tmp_path, puppet, images, _pose({"arm": 10.0, "fore": 0.0}), _pose({"arm": 10.0, "fore": 30.0})
    )
    res = inbetween(set_dir, tj, 3, tmp_path / "out")
    assert len(res.frames) == 3
    assert [row["locked_change_px"] for row in res.report["frames"]] == [0, 0, 0]


def test_offset_name_read_and_written_back() -> None:
    def with_offset(pose: Pose, offsets: dict[str, tuple[float, float]]) -> Pose:
        out = pose.model_copy()
        object.__setattr__(out, "offset", dict(offsets))
        return out

    x = with_offset(_pose({"arm": 0.0}), {"arm": (0.0, 0.0)})
    y = with_offset(_pose({"arm": 30.0}), {"arm": (10.0, -4.0)})
    mid = interpolate(x, y, 0.5)
    assert mid.offset == {"arm": (5.0, -2.0)}  # type: ignore[attr-defined]
    end = interpolate(x, y, 1.0)
    assert end.offset == {"arm": (10.0, -4.0)}  # type: ignore[attr-defined]
    held = interpolate(x, y, 0.5, locked=("arm",))
    assert held.offset == {"arm": (0.0, 0.0)}  # type: ignore[attr-defined]
