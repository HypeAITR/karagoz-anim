"""WP-51 extra tests: episode checks for poses, sequences and loop beats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from test_draw import EP, prod  # noqa: F401 (prod is a pytest fixture)

from karagoz.spec import check_episode


def _load_poses(prod: Path) -> tuple[Path, dict[str, Any]]:  # noqa: F811
    path = prod / EP / "shots" / "sh01" / "poses.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return path, data


def _save(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_duplicate_pose_ids_are_errors_naming_the_id(prod: Path) -> None:  # noqa: F811
    path, data = _load_poses(prod)
    p2 = dict(data["characters"]["ali"]["poses"][1])
    data["characters"]["ali"]["poses"].append(p2)
    _save(path, data)
    errors = [f for f in check_episode(prod / EP) if f.level == "error" and "p2" in f.message]
    assert errors, "duplicate pose id p2 should be an error naming p2"


def test_transition_from_undefined_pose_is_error(prod: Path) -> None:  # noqa: F811
    path, data = _load_poses(prod)
    data["characters"]["ali"]["transitions"][0]["from"] = "p7"
    _save(path, data)
    assert any(f.level == "error" and "p7" in f.message for f in check_episode(prod / EP))


def test_transition_to_undefined_pose_is_error(prod: Path) -> None:  # noqa: F811
    path, data = _load_poses(prod)
    data["characters"]["ali"]["transitions"][0]["to"] = "p8"
    _save(path, data)
    assert any(f.level == "error" and "p8" in f.message for f in check_episode(prod / EP))


def test_sequence_to_undefined_pose_is_error(prod: Path) -> None:  # noqa: F811
    path, data = _load_poses(prod)
    data["characters"]["ali"]["sequences"] = [
        {
            "id": "s1",
            "from": "p1",
            "to": "p8",
            "drawings": ["Pouring tea slowly now.", "Tea stream falling down."],
        }
    ]
    _save(path, data)
    assert any(f.level == "error" and "p8" in f.message for f in check_episode(prod / EP))


def test_loop_step_unknown_beat_is_error(prod: Path) -> None:  # noqa: F811
    xs_path = prod / EP / "shots" / "sh01" / "xsheet.yaml"
    data = yaml.safe_load(xs_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    data["tracks"]["ali.body"].append(
        {"at": "L02.start", "loop": ["p1", "p2"], "step": "beat.bogus", "until": "shot.out"}
    )
    _save(xs_path, data)
    assert any(f.level == "error" and "bogus" in f.message for f in check_episode(prod / EP))
