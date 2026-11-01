"""Extra tests for WP-47 (mouth edit outside shot, min_hold after absorb)."""

from __future__ import annotations

import pytest
from test_mouth import track

from karagoz.engine.mouth import _apply_min_hold, _Cue
from karagoz.spec import Xsheet


def test_mouth_edit_before_shot_start_is_rejected() -> None:
    xs = Xsheet.model_validate(
        {
            "version": 1,
            "shot": "sh01",
            "tracks": {},
            "mouth": {"ali": {"source": "auto", "edits": [{"at": "L01.start - 100f", "shape": "E"}]}},
        }
    )
    with pytest.raises(ValueError, match="edit"):
        track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], xs=xs)


def test_apply_min_hold_rechecks_absorbed_closed_cue() -> None:
    cues = [
        _Cue(start=20, end=28, shape="D", letters="a"),
        _Cue(start=28, end=28, shape="A", letters="m"),
        _Cue(start=28, end=29, shape="B", letters="i"),
        _Cue(start=29, end=36, shape="C", letters="e"),
    ]
    out = _apply_min_hold(cues, 2)
    shaped = [c for c in out if c.shape != "X"]
    assert shaped
    assert all(c.end - c.start >= 2 for c in shaped), f"sub-min_hold cue in {shaped}"
