"""K-2d inbetween run: poses, holds, z switch, measures, files (inbetween.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from test_puppet_fit_common import build_synth_puppet, draw_pose

from karagoz.engine.puppet.inbetween import FRAME_KEYS, REPORT_KEYS, inbetween
from karagoz.engine.puppet.interp import interpolate
from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.pose import Pose

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
    locked: list[str] | None = None,
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
        "locked": LOCKED if locked is None else locked,
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
    assert set(res.report) == set(REPORT_KEYS)
    assert res.report["frames"][0]["locked_change_px"] >= 0


def test_offset_name_read_and_written_back() -> None:
    """After the K-1 merge ``Pose.offset`` is a real field: interpolation reads and writes it."""
    x = _pose({"arm": 0.0}).model_copy(update={"offset": {"arm": [0.0, 0.0]}})
    y = _pose({"arm": 30.0}).model_copy(update={"offset": {"arm": [10.0, -4.0]}})
    mid = interpolate(x, y, 0.5)
    assert mid.offset == {"arm": [5.0, -2.0]}
    end = interpolate(x, y, 1.0)
    assert end.offset == {"arm": [10.0, -4.0]}
    held = interpolate(x, y, 0.5, locked=("arm",))
    assert held.offset == {"arm": [0.0, 0.0]}


def _box(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def test_hole_region_counts_transient_slit_and_ignores_strip() -> None:
    """Review item 1: a slit open now but covered at both ends counts; a strip does not."""
    from karagoz.engine.puppet.inbetween import _hole_region

    h = w = 60
    collar = _box(h, w, 20, 40, 10, 20)
    closed = _box(h, w, 20, 40, 15, 40)
    shifted = _box(h, w, 20, 40, 23, 48)
    masks_prev = {"cap": collar, "arm": closed}
    masks_cur = {"cap": collar, "arm": shifted}
    both_closed = (collar | closed) & (collar | closed)
    region = _hole_region(masks_prev, masks_cur, both_closed, "cap", "arm")
    assert bool(region[30, 21])  # slit: covered at both ends, paper now, inside swept
    assert not bool(region[5, 5])  # outside the swept neighbourhood
    far = _box(h, w, 20, 40, 30, 55)  # Y far away: slit is a trailing strip, not a hole
    end_far = (collar | closed) & (collar | far)
    assert not bool(_hole_region(masks_prev, masks_cur, end_far, "cap", "arm")[30, 21])


def test_paper_tone_and_over_bg() -> None:
    """Review item 2: flat paper tone from the keys; alpha-preserving composite."""
    from karagoz.engine.puppet.inbetween import _on_tone, _over_bg, _paper_tone

    key = np.full((10, 10, 3), 250, np.uint8)
    key[0, 0] = (10, 10, 10)
    assert _paper_tone([key]) == (250, 250, 250)
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., :3] = (1, 2, 3)
    rgba[..., 3] = 0
    rgba[0, 0] = (9, 9, 9, 255)
    rgb = _over_bg(rgba, (250, 251, 252))
    assert rgb.shape == (4, 4, 3)
    assert tuple(rgb[1, 1]) == (250, 251, 252)  # transparent shows the tone
    out = _on_tone(rgba, (250, 251, 252))
    assert tuple(out[1, 1]) == (250, 251, 252, 0)  # alpha kept
    assert tuple(out[0, 0]) == (9, 9, 9, 255)  # ink untouched


def test_locked_change_tol_ignores_lsb_noise() -> None:
    """Review item 3: ±1 LSB churn is silent, a real change still counts."""
    from karagoz.engine.puppet.frame_checks import locked_change_px

    a = np.full((8, 8, 4), 100, np.uint8)
    b = a.copy()
    b[2:4, 2:4, 0] += 1
    mask = np.ones((8, 8), dtype=bool)
    assert locked_change_px(a, b, mask) == 4
    assert locked_change_px(a, b, mask, tol=8) == 0
    c = a.copy()
    c[2:4, 2:4] = 200
    assert locked_change_px(a, c, mask, tol=8) == 4
    from karagoz.engine.puppet.inbetween import LOCKED_TOL_PX

    assert locked_change_px(a, b, mask, tol=LOCKED_TOL_PX) == 0
    assert locked_change_px(a, c, mask, tol=LOCKED_TOL_PX) == 4


def test_cut_rows_measure_key_vs_puppet(tmp_path: Path) -> None:
    """Review item 3: the f0 cut row reports the key/puppet difference, not 0."""
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    tj = _write_transition(
        tmp_path, puppet, images, _pose({"arm": 10.0, "fore": 0.0}), _pose({"arm": 10.0, "fore": 30.0})
    )
    with Image.open(tj.parent / "X.png") as opened:
        key = np.asarray(opened.convert("RGB")).copy()
    key[80:90, 80:90] = (0, 0, 0)  # extra ink inside the locked torso
    Image.fromarray(key).save(tj.parent / "X.png")
    res = inbetween(set_dir, tj, 3, tmp_path / "out")
    assert res.report["frames"][0]["locked_change_px"] > 0


def test_z_pair_without_pixel_support_does_not_switch(tmp_path: Path) -> None:
    """Review item 4: a z-rank flip with no pixel support keeps X order (no switch)."""
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    # Same setup as the old timing test, but arm/fore share one fill colour:
    # the Y overlap can never name a winner, so nothing switches.
    pose_x = _pose({"arm": 10.0, "fore": 0.0})
    pose_y = _pose({"arm": 10.0, "fore": 40.0}, {"fore": 0})
    tj = _write_transition(tmp_path, puppet, images, pose_x, pose_y, x_z={}, y_z={"fore": 0})
    res = inbetween(set_dir, tj, 5, tmp_path / "out")
    assert res.report["z_switch_frames"] == []
    assert res.report["z_switches"] == []


def test_z_switches_name_front_and_back(tmp_path: Path) -> None:
    """Review item 4: the report names the switching pair (front on top)."""
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    # Leg swings over the torso; the Y key is drawn leg-on-top while X order
    # keeps leg below: the drawing supports the pass.
    pose_x = _pose({"leg": 0.0})
    pose_y = _pose({"leg": 45.0})
    tj = _write_transition(
        tmp_path,
        puppet,
        images,
        pose_x,
        pose_y,
        x_z={},
        y_z={"leg": 5},
        locked=["torso", "arm", "hand", "foot"],
    )
    res = inbetween(set_dir, tj, 5, tmp_path / "out")
    named = {(s["front"], s["back"]) for s in res.report["z_switches"]}
    assert ("leg", "torso") in named
    assert all(s["front"] == "leg" for s in res.report["z_switches"])
    assert sorted(res.report["z_switch_frames"]) == sorted(s["frame"] for s in res.report["z_switches"])


def test_workers_one_runs(tmp_path: Path) -> None:
    """Review item 5: the run works with a single render worker (no nested pool)."""
    puppet, images = build_synth_puppet()
    set_dir = _write_synth_set(tmp_path, puppet, images)
    tj = _write_transition(
        tmp_path, puppet, images, _pose({"arm": 10.0, "fore": 0.0}), _pose({"arm": 10.0, "fore": 30.0})
    )
    res = inbetween(set_dir, tj, 3, tmp_path / "out", workers=1)
    assert len(res.frames) == 3
