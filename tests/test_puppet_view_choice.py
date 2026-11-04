"""View choice (5 Nov demo): the fitted view explaining more of the drawing wins; the part total only breaks ties."""

from __future__ import annotations

from karagoz.engine.puppet.fit import VIEW_SHARE_TIE, view_beats


def test_lower_unexplained_share_wins_over_a_higher_total() -> None:
    # Measured: Spark p1, 3q_right 2.0% unexplained / total 16.35 vs back 6.8% / 16.94.
    assert view_beats(0.020, 16.35, 0.068, 16.94)
    assert not view_beats(0.068, 16.94, 0.020, 16.35)


def test_total_breaks_a_tie_in_the_share() -> None:
    assert view_beats(0.010, 20.0, 0.010 + VIEW_SHARE_TIE / 2, 19.0)
    assert not view_beats(0.010, 19.0, 0.010 + VIEW_SHARE_TIE / 2, 20.0)
