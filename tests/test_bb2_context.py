"""BB-2 bug hunt: per-task context section routing.

Every test in this file FAILS on today's code. No paid API calls, no network.
"""

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


def test_timer_sections_follow_transition_direction(tmp_path: Path) -> None:
    """sit-down/stand-up routing must follow transitions, not pose list order.

    The rules say "a pose goes from sit to stand" (a directed move), but the
    code compares list-adjacent poses. On a sit-down-then-stand-up round
    trip the return move is missed: the timer gets sit-down but not
    stand-up for a shot that does both.
    """
    prod = tmp_path / "prod"
    poses_path = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses_path.parent.mkdir(parents=True, exist_ok=True)
    poses_path.write_text(ROUND_TRIP_POSES, encoding="utf-8")
    sections = load_sections(GUIDE)
    sel = select_sections("timer", prod, EP, "sh01", sections)
    assert not sel.fallback, "poses did not classify; fixture invalid?"
    assert "sit-down" in sel.given, f"sit-down missing: {sel.given}"
    assert "stand-up" in sel.given, f"return move missed, only: {sel.given}"
