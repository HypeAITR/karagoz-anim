"""BB-1 bug hunt: mouth track bugs.

Each test fails on today's code and proves one user-facing bug. Mini episode and
style helpers come from test_mouth.py (same pattern as test_mouth_short.py).
"""

from __future__ import annotations

import pytest
from test_mouth import track

from karagoz.spec import Xsheet


def at(frames: list[int]) -> list[float]:
    return [f / 24 for f in frames]


def test_bb1_mouth_edit_outside_shot_is_rejected() -> None:
    """A mouth edit that resolves outside the shot must fail loudly, not vanish.

    Today an edit anchored past the shot end (a timer typo) finds no cue and is
    skipped with `continue`: the xsheet asks for a correction, the track ignores
    it, and nothing tells the timer. Unknown shots and missing audio raise; a
    dangling edit should raise too.
    """
    xs = Xsheet.model_validate(
        {
            "version": 1,
            "shot": "sh01",
            "tracks": {},
            "mouth": {"ali": {"source": "auto", "edits": [{"at": "L01.end + 100f", "shape": "E"}]}},
        }
    )
    with pytest.raises(ValueError, match="edit"):
        track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], xs=xs)


def test_bb1_absorbed_closed_cue_keeps_min_hold() -> None:
    """A closed-lips cue that absorbs its neighbour must still keep min_hold (2).

    "amie" piles m@5f, i@6f, e@7f so A(28..28) absorbs B(28..29) and stays
    1 frame long: _apply_min_hold moves past it without rechecking, breaking the
    min_hold guarantee the suite asserts for the other WP-29 cases (a 1-frame
    closed-lips flash between two vowels).
    """
    cues = track("amie", at([0, 5, 6, 7]), at([5, 6, 7, 8]))
    shaped = [c for c in cues[1:-1] if c[2] != "X"]
    assert shaped, "expected shaped cues between the rests"
    assert all(end - start >= 2 for start, end, _, _ in shaped), f"sub-min_hold cue in {shaped}"
