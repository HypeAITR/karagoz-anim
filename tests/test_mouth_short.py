"""Acceptance tests for WP-29 (a short vowel stays visible). Written by the maintainer; do not edit.

Same mini episode as test_mouth.py (line at frame 24, lead 2, first shape lead 4, min_hold 2, end hold 4).
Letter times are whole frames (f / 24 s), so a letter at frame f makes its shape at 24 + f - 2.
"""

from __future__ import annotations

from test_mouth import track

from karagoz.engine.mouth import OPENNESS


def at(frames: list[int]) -> list[float]:
    return [f / 24 for f in frames]


def body(cues: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str]]:
    """Cues between the leading and the trailing rest."""
    return [c[:3] for c in cues[1:-1]]


def test_openness_order() -> None:
    assert OPENNESS["X"] == OPENNESS["A"] == 0
    assert OPENNESS["B"] == OPENNESS["F"] == OPENNESS["G"] == 1
    assert OPENNESS["C"] == OPENNESS["E"] == OPENNESS["H"] == 2
    assert OPENNESS["D"] == 3


def test_wider_short_vowel_takes_frames_from_the_previous() -> None:
    # e -> C 20; a -> D 27 (1 frame); i -> B 28..30 (2, cannot give); a -> D 30..37.
    # The next cannot give; C (7 frames) can, and D is wider: C 20..26, D 26..28.
    cues = track("eaia", at([0, 5, 6, 8]), at([5, 6, 8, 9]))
    assert body(cues) == [(20, 26, "C"), (26, 28, "D"), (28, 30, "B"), (30, 37, "D")]


def test_narrower_short_vowel_is_dropped() -> None:
    # a -> D 20; i -> B 27 (1 frame); a -> D 28..30 (cannot give); e -> C 30.
    # B is narrower than D: dropped, the two D's join.
    cues = track("aiae", at([0, 5, 6, 8]), at([5, 6, 8, 9]))
    assert body(cues) == [(20, 30, "D"), (30, 37, "C")]


def test_wider_short_vowel_replaces_a_neighbour_that_cannot_give() -> None:
    # i -> B 20; e -> C 27..29 (2); a -> D 29 (1 frame); e -> C 30..32 (2); i -> B 32..39.
    # Neither neighbour can give a frame; D is wider than the previous C: that C is dropped, D 27..30.
    cues = track("ieaei", at([0, 5, 7, 8, 10]), at([5, 7, 8, 10, 11]))
    assert body(cues) == [(20, 27, "B"), (27, 30, "D"), (30, 32, "C"), (32, 39, "B")]


def test_closed_lips_are_never_replaced() -> None:
    # a -> D 20; m -> A 24+5-1=28..30 (2, close_lead 1); a -> D 30 (1 frame); e -> C 31..33 (2); i -> B 33.
    # D cannot take from C (2) nor from A (2), and A (closed lips) is never dropped for a vowel: D is dropped.
    cues = track("amaei", at([0, 5, 8, 9, 11]), at([5, 8, 9, 11, 12]))
    shapes = [c[2] for c in body(cues)]
    assert "A" in shapes
    assert all(e - s >= 2 for s, e, _ in body(cues))


def test_every_cue_keeps_min_hold_and_the_track_is_continuous() -> None:
    for text, starts, ends in (
        ("eaia", at([0, 5, 6, 8]), at([5, 6, 8, 9])),
        ("aiae", at([0, 5, 6, 8]), at([5, 6, 8, 9])),
        ("ieaei", at([0, 5, 7, 8, 10]), at([5, 7, 8, 10, 11])),
        ("aea", [0.0, 0.05, 0.1], [0.05, 0.1, 0.15]),
    ):
        cues = track(text, starts, ends)
        assert all(a[1] == b[0] for a, b in zip(cues, cues[1:], strict=False))
        assert all(e - s >= 2 for s, e, _ in body(cues))
        assert all(a[2] != b[2] for a, b in zip(cues, cues[1:], strict=False))
