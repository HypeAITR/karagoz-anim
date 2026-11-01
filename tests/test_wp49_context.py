"""WP-49 item 4: timer sit-down/stand-up follow transitions, not pose list order."""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.context import load_sections, select_sections

EP = "ep1"
GUIDE = Path(__file__).parent.parent / "guide"

ROUND_TRIP_POSES = """\
version: 1
shot: sh01
characters:
  ali:
    poses:
      - id: p1
        view: three_quarter_right
        expression: neutral
        support: stand
        acting: "Standing by the sofa, arms relaxed at his sides."
      - id: p2
        view: three_quarter_right
        expression: neutral
        support: sit
        acting: "Sitting on the sofa, hands resting on his knees."
    transitions:
      - from: p1
        to: p2
        inbetweens:
          - "Knees bending, body lowering toward the sofa seat."
      - from: p2
        to: p1
        inbetweens:
          - "Pushing up from the sofa, body rising to stand."
"""

STAND_UP_ONLY_POSES = """\
version: 1
shot: sh01
characters:
  ali:
    poses:
      - id: p1
        view: three_quarter_right
        expression: neutral
        support: sit
        acting: "Sitting on the sofa, hands resting on his knees."
      - id: p2
        view: three_quarter_right
        expression: neutral
        support: sit
        acting: "Still sitting, leaning forward with arms outstretched."
    transitions:
      - from: p1
        to: p2
        inbetweens:
          - "Leaning forward while staying seated on the sofa."
"""


def _prod_with_poses(tmp_path: Path, poses_text: str) -> Path:
    prod = tmp_path / "prod"
    poses_path = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    poses_path.write_text(poses_text, encoding="utf-8")
    return prod


def test_wp49_timer_round_trip_gives_both_sections(tmp_path: Path) -> None:
    prod = _prod_with_poses(tmp_path, ROUND_TRIP_POSES)
    sections = load_sections(GUIDE)
    sel = select_sections("timer", prod, EP, "sh01", sections)
    assert not sel.fallback
    assert "sit-down" in sel.given
    assert "stand-up" in sel.given


def test_wp49_timer_transition_direction_beats_list_order(tmp_path: Path) -> None:
    # Pose list is sit, sit (no list-adjacent stand-up); transitions also sit->sit:
    # neither section should appear from support alone.
    prod = _prod_with_poses(tmp_path, STAND_UP_ONLY_POSES)
    sections = load_sections(GUIDE)
    sel = select_sections("timer", prod, EP, "sh01", sections)
    assert not sel.fallback
    assert "sit-down" not in sel.given
    assert "stand-up" not in sel.given


def test_wp49_timer_without_transitions_falls_back_to_pose_order(tmp_path: Path) -> None:
    poses_text = """\
version: 1
shot: sh01
characters:
  ali:
    poses:
      - id: p1
        view: three_quarter_right
        expression: neutral
        support: sit
        acting: "Sitting on the sofa, hands resting on his knees."
      - id: p2
        view: three_quarter_right
        expression: neutral
        support: stand
        acting: "Standing by the sofa, arms relaxed at his sides."
"""
    prod = _prod_with_poses(tmp_path, poses_text)
    sections = load_sections(GUIDE)
    sel = select_sections("timer", prod, EP, "sh01", sections)
    assert not sel.fallback
    assert "stand-up" in sel.given
