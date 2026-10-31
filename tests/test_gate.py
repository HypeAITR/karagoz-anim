"""Acceptance tests for WP-25 (mechanical gate: flags, never rejects; silenced after a look). Written by the maintainer.

Do not edit. Synthetic landmarks/placements; the last tests use the mini production of the WP-09a tests
(`test_draw.py`): ali p1 -> p2 with two inbetweens; continuity has one prop (the glass).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.engine.gate import (
    KINDS,
    Flag,
    acknowledge,
    exposure,
    gate_flags,
    gate_shot,
    select_flags,
    sequence,
)
from karagoz.spec import Poses

SCALE = 0.2  # every synthetic drawing is placed at (0, 0) with this scale; head box 300 px tall -> 60 px on screen


def lm(cy: float, *, head_h: float = 300, cups: int = 0, weak_cups: int = 0, sha: str = "") -> dict[str, Any]:
    """Landmarks with the head centred at drawing-y `cy` (screen y = cy * SCALE)."""
    box = [400, cy - head_h / 2, 600, cy + head_h / 2]
    return {"version": 1, "size": [1000, 1500], "sha256": sha or f"sha-{cy}-{head_h}-{cups}",
            "head": {"box": box, "confidence": 0.9}, "face": None,
            "cups": [{"box": [1, 1, 2, 2], "confidence": 0.9}] * cups
            + [{"box": [1, 1, 2, 2], "confidence": 0.4}] * weak_cups}  # fmt: skip


def placed(names: dict[str, list[str]], warnings: list[str] | None = None) -> dict[str, Any]:
    chars = {c: {d: {"scale": SCALE, "x": 0, "y": 0, "mirror": False} for d in ds} for c, ds in names.items()}
    return {"characters": chars, "warnings": warnings or []}


def poses_of(supports: tuple[str, str], n: int) -> Poses:
    base = {"view": "front", "expression": "neutral"}  # expression is required unless faceless
    data = {"version": 1, "shot": "sh01", "characters": {"ali": {
        "poses": [{"id": "p1", **base, "acting": "Oturuyor, bekliyor.", "support": supports[0]},
                  {"id": "p2", **base, "acting": "Ayakta duruyor, bakıyor.", "support": supports[1]}],
        "transitions": [{"from": "p1", "to": "p2", "inbetweens": [f"Ara çizim numara {k}." for k in range(1, n + 1)]}],
    }}}  # fmt: skip
    return Poses.model_validate(data)


def kinds(flags: list[Flag]) -> list[tuple[str, tuple[str, ...]]]:
    return [(f.kind, f.drawings) for f in flags]


# --- basics --------------------------------------------------------------------------------------------------------


def test_sequence_and_ids() -> None:
    assert sequence([None, "p1", "p1", "p1-p2.1", None, "p1-p2.1", "p2"]) == [
        ("p1", 1, 3), ("p1-p2.1", 3, 4), ("p1-p2.1", 5, 6), ("p2", 6, 7)]  # fmt: skip
    f = Flag("jump", "ali", ("p1", "p1-p2.1"), 1.0, "m", ("a", "b"))
    assert f.id == "jump:ali:p1+p1-p2.1" and f.frames is None and not f.silenced
    assert Flag("prop_count", None, ("ali/p1",), 1, "m", ("a",), (0, 2)).id == "prop_count:-:ali/p1"
    assert KINDS == ("prop_count", "reverse", "jump", "scale_jump", "placement", "world")  # WP-38


def test_smooth_rise_is_not_flagged() -> None:
    names = ["p1", "p1-p2.1", "p1-p2.2", "p1-p2.3", "p2"]
    ys = [1000, 900, 800, 700, 600]  # screen: 200 -> 120, 20 px (1/3 head) per drawing
    cells = {"ali": [n for n in names for _ in range(2)]}
    lms = {"ali": dict(zip(names, (lm(y) for y in ys), strict=True))}
    assert gate_flags(cells, placed({"ali": names}), lms, poses_of(("sit", "stand"), 3), 0) == []


# --- rules ---------------------------------------------------------------------------------------------------------


def test_jump() -> None:
    cells = {"ali": ["p1", "p1", "p2", "p2"]}
    lms = {"ali": {"p1": lm(1000), "p2": lm(1300)}}  # 60 px = 1 head
    flags = gate_flags(cells, placed({"ali": ["p1", "p2"]}), lms, None, 0)
    assert kinds(flags) == [("jump", ("p1", "p2"))]
    assert flags[0].size == pytest.approx(1.0, abs=0.01) and flags[0].character == "ali"
    assert flags[0].shas == ("sha-1000-300-0", "sha-1300-300-0")
    assert "the head moves 1.00 head heights" in flags[0].message


def test_reverse_inside_a_rise() -> None:
    names = ["p1", "p1-p2.1", "p1-p2.2", "p1-p2.3", "p2"]
    ys = [1000, 875, 975, 700, 600]  # .2 falls back towards the seat
    cells = {"ali": names}
    lms = {"ali": dict(zip(names, (lm(y) for y in ys), strict=True))}
    flags = gate_flags(cells, placed({"ali": names}), lms, poses_of(("sit", "stand"), 3), 0)
    rev = [f for f in flags if f.kind == "reverse"]
    assert kinds(rev) == [("reverse", ("p1-p2.2",))]
    assert rev[0].size == pytest.approx(20 / 60, abs=0.01)
    assert "goes back between p1 and p2" in rev[0].message
    assert [f.kind for f in flags] == sorted((f.kind for f in flags), key=KINDS.index)


def test_down_first_then_up() -> None:
    names = ["p1", "p1-p2.1", "p2"]
    ys = [1000, 1060, 600]  # first goes 12 px lower than the seat
    lms = {"ali": dict(zip(names, (lm(y) for y in ys), strict=True))}
    flags = gate_flags({"ali": names}, placed({"ali": names}), lms, poses_of(("sit", "stand"), 1), 0)
    assert ("reverse", ("p1-p2.1",)) in kinds(flags)


def test_standing_up_but_going_down() -> None:
    lms = {"ali": {"p1": lm(1000), "p2": lm(1150)}}  # 30 px lower: half a head
    flags = gate_flags({"ali": ["p1", "p2"]}, placed({"ali": ["p1", "p2"]}), lms, poses_of(("sit", "stand"), 1), 0)
    rev = [f for f in flags if f.kind == "reverse"]
    assert kinds(rev) == [("reverse", ("p2",))]
    assert "sit to stand but the head goes down" in rev[0].message
    lms_ok = {"ali": {"p1": lm(1000), "p2": lm(1150)}}
    sitting = gate_flags({"ali": ["p1", "p2"]}, placed({"ali": ["p1", "p2"]}), lms_ok, poses_of(("stand", "sit"), 1), 0)
    assert not [f for f in sitting if f.kind == "reverse"]


def test_scale_jump() -> None:
    lms = {"ali": {"p1": lm(1000, head_h=300), "p2": lm(1000, head_h=400)}}
    flags = gate_flags({"ali": ["p1", "p2"]}, placed({"ali": ["p1", "p2"]}), lms, None, 0)
    assert kinds(flags) == [("scale_jump", ("p1", "p2"))]
    assert flags[0].size == pytest.approx(1 / 3, abs=0.01) and "33%" in flags[0].message


def test_prop_count() -> None:
    cells = {"ali": ["p1", "p1", "p2", "p2"], "ayse": ["q", "q", "q", "q"]}
    lms = {"ali": {"p1": lm(1000, cups=1), "p2": lm(1000, weak_cups=1)}, "ayse": {"q": lm(1000, cups=1)}}
    pl = placed({"ali": ["p1", "p2"], "ayse": ["q"]})
    flags = gate_flags(cells, pl, lms, None, 1)
    assert kinds(flags) == [("prop_count", ("ali/p1", "ayse/q"))]
    f = flags[0]
    assert f.frames == (0, 2) and f.size == 1 and f.character is None
    assert f.message == "frames 0-1: 2 props seen, 1 expected (ali/p1, ayse/q)"
    assert gate_flags(cells, pl, lms, None, 0) == []  # no props in the shot: nothing to count


def test_placement_warnings() -> None:
    pl = placed({"ali": ["p1"]}, ["ali p1: feet 55 px above the floor while sitting on sofa_left"])
    flags = gate_flags({"ali": ["p1"]}, pl, {"ali": {"p1": lm(1000)}}, None, 0)
    assert kinds(flags) == [("placement", ("p1",))]
    assert flags[0].size == pytest.approx(55 / 60, abs=0.01)
    assert flags[0].message.startswith("ali p1: feet 55 px")


def test_mirror_and_missing_data_are_handled() -> None:
    pl = placed({"ali": ["p1", "p2"]})
    pl["characters"]["ali"]["p2"]["mirror"] = True  # head box centred at x=500 of 1000: same place mirrored
    lms = {"ali": {"p1": lm(1000), "p2": lm(1000)}}
    assert gate_flags({"ali": ["p1", "p2", "p3"]}, pl, lms, None, 0) == []  # p3 has no data: skipped


def test_select_flags() -> None:
    def fl(kind: str, size: float, silenced: bool = False) -> Flag:
        return Flag(kind, "ali", (f"d{size}",), size, "m", ("s",), silenced=silenced)

    flags = [fl("prop_count", 1), fl("jump", 0.7), fl("jump", 2.0), fl("scale_jump", 0.3), fl("placement", 0.9),
             fl("jump", 5.0, silenced=True)]  # fmt: skip
    shown, silenced, dropped = select_flags(flags, limit=2)
    assert [f.size for f in shown] == [1, 2.0, 0.9]  # prop always + the two biggest, original order
    assert [f.size for f in silenced] == [5.0] and dropped == 2


# --- production ----------------------------------------------------------------------------------------------------


def setup_shot(prod: Path, cups_in_ib1: int = 2) -> Path:  # noqa: F811
    a = prod / EP / "assets" / "sh01"
    names = {"ali": ["p1", "p1-p2.1", "p1-p2.2", "p2"]}
    ys = {"p1": 1000, "p1-p2.1": 1000, "p1-p2.2": 1000, "p2": 1000}
    for d in names["ali"]:
        (a / "ali").mkdir(parents=True, exist_ok=True)
        data = lm(ys[d], cups=cups_in_ib1 if d == "p1-p2.1" else 1, sha=f"{d}-v1")
        (a / "ali" / f"{d}.landmarks.json").write_text(json.dumps(data), encoding="utf-8")
    (a / "placement.json").write_text(json.dumps(placed(names)), encoding="utf-8")
    return a


def test_exposure_matches_the_drawings(prod: Path) -> None:  # noqa: F811
    cells = exposure(prod, EP, "sh01")
    seen = {c for c in cells["ali"] if c is not None}
    assert seen == {"p1", "p1-p2.1", "p1-p2.2", "p2"}
    assert cells["ali"].index("p1-p2.1") < cells["ali"].index("p1-p2.2") < cells["ali"].index("p2")


def test_gate_shot_writes_flags_and_silences_after_a_look(prod: Path) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="no placement.json"):
        gate_shot(prod, EP, "sh01")
    a = setup_shot(prod)
    out = gate_shot(prod, EP, "sh01")
    path = prod / EP / "review" / "sh01" / "gate.json"
    assert json.loads(path.read_text(encoding="utf-8")) == out
    assert [f["kind"] for f in out["flags"]] == ["prop_count"]  # two cups in p1-p2.1, one glass in the shot
    f = out["flags"][0]
    assert f["drawings"] == ["ali/p1-p2.1"] and f["shas"] == ["p1-p2.1-v1"] and f["size"] == 1
    assert isinstance(f["frames"], list) and out["silenced"] == [] and out["dropped"] == 0
    flag = Flag(f["kind"], f["character"], tuple(f["drawings"]), f["size"], f["message"], tuple(f["shas"]),
                tuple(f["frames"]))  # type: ignore[arg-type]  # fmt: skip
    acknowledge(prod, EP, "sh01", flag, "Intended: the glass is reflected in the tray.")
    ack = yaml.safe_load((prod / EP / "review" / "sh01" / "gate-ack.json").read_text(encoding="utf-8"))
    assert ack[flag.id]["shas"] == ["p1-p2.1-v1"] and ack[flag.id]["note"].startswith("Intended")
    again = gate_shot(prod, EP, "sh01")
    assert again["flags"] == [] and [x["id"] for x in again["silenced"]] == [flag.id]
    redrawn = json.loads((a / "ali" / "p1-p2.1.landmarks.json").read_text(encoding="utf-8"))
    redrawn["sha256"] = "p1-p2.1-v2"  # a new drawing: the flag can come back
    (a / "ali" / "p1-p2.1.landmarks.json").write_text(json.dumps(redrawn), encoding="utf-8")
    assert [x["id"] for x in gate_shot(prod, EP, "sh01")["flags"]] == [flag.id]


def test_gate_shot_without_props_counts_nothing(prod: Path) -> None:  # noqa: F811
    setup_shot(prod)
    cont = prod / EP / "continuity.yaml"
    data = yaml.safe_load(cont.read_text(encoding="utf-8"))
    data.pop("props")
    cont.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert gate_shot(prod, EP, "sh01")["flags"] == []
