"""Acceptance tests for WP-37 (sequence and loop events). Written by the maintainer; do not edit.

Fixture timing (see WP-02/WP-03 tests): shot sh01 = [0, 100); L02 (ali) = [60, 88).
The tests replace ali.body with their own events; poses.yaml gets a sequence s1 (p1 -> p2, three drawings).
"""

from __future__ import annotations

import dataclasses
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw
from test_draw import STYLE_EXTRA  # transition, blink, hold and mouth values of the draw tests

from karagoz.engine.placement import CharacterInput, PoseInfo, solve
from karagoz.engine.timing import EpisodeTiming, resolve_episode
from karagoz.engine.xsheet import LOOP_STEP_DEFAULT, Exposure, expand, lint
from karagoz.spec import Align, Lines, Poses, Regions, Shots, Style, Xsheet
from karagoz.studio.draw import DrawJob, plan_draw

FIX = Path(__file__).parent / "fixtures" / "spec"
EP = "ep"

SEQ = {
    "id": "s1",
    "from": "p1",
    "to": "p2",
    "drawings": [
        "Sağ kol bardakla birlikte göğse kalkıyor, dirsek bükük.",
        "Kol omuz hizasında, bardak hâlâ dik, baş hafif öne.",
        "Kol Ayşe'ye doğru neredeyse tam uzanmış, bardak dik.",
    ],
}


def style_dict() -> dict[str, Any]:
    d = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    d["live"].update(STYLE_EXTRA)
    d["live"]["beats"]["action_step"] = {"value": 6, "source": "test"}
    return d


def style() -> Style:
    return Style.model_validate(style_dict())


def align(line_id: str, text: str) -> Align:
    n = len(text)
    return Align.model_validate(
        {
            "version": 1,
            "line": line_id,
            "text": text,
            "characters": list(text),
            "starts": [round(i * 0.1, 6) for i in range(n)],
            "ends": [round((i + 1) * 0.1, 6) for i in range(n)],
        }
    )


ALIGNS = {"L01": "Yarın pazar, Ali.", "L02": "Su ver bana."}


def timing() -> EpisodeTiming:
    lines = Lines.model_validate(yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8")))
    shots = Shots.model_validate(yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8")))
    return resolve_episode(lines, shots, {k: align(k, v) for k, v in ALIGNS.items()}, style())


def poses_dict(seq: dict[str, Any] | None = SEQ) -> dict[str, Any]:
    d = yaml.safe_load((FIX / "poses.yaml").read_text(encoding="utf-8"))
    if seq is not None:
        d["characters"]["ali"]["sequences"] = [seq]
    return d


def poses(seq: dict[str, Any] | None = SEQ) -> Poses:
    return Poses.model_validate(poses_dict(seq))


def xsheet_dict(body: list[dict[str, Any]], ali_eyes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    d = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    d["tracks"]["ali.body"] = body
    if ali_eyes is not None:
        d["tracks"]["ali.eyes"] = ali_eyes
    return d


def run(body: list[dict[str, Any]], seq: dict[str, Any] | None = SEQ, **kw: Any) -> tuple[Exposure, list[Any]]:
    xs = Xsheet.model_validate(xsheet_dict(body, **kw))
    exp = expand(xs, poses(seq), timing(), style())
    return exp, lint(exp, style())


def codes(findings: list[Any]) -> set[str]:
    return {f.code for f in findings}


HOLD_P1 = {"at": "shot.in", "pose": "p1"}


# --- sequences --------------------------------------------------------------------------------------------------------


def test_sequence_plays_its_drawings_then_holds_the_target_pose() -> None:
    exp, findings = run([HOLD_P1, {"at": "shot.in + 10f", "sequence": "s1", "cadence": "twos"}])
    cells = exp.body["ali"]
    assert cells[:10] == ["p1"] * 10
    assert cells[10:16] == ["s1.1", "s1.1", "s1.2", "s1.2", "s1.3", "s1.3"]
    assert cells[16:] == ["p2"] * (100 - 16)
    assert exp.unsupported == []
    assert "timing.unsupported" not in codes(findings)


def test_sequence_default_cadence_is_twos_and_threes_works() -> None:
    exp, _ = run([HOLD_P1, {"at": "shot.in + 10f", "sequence": "s1"}])
    assert exp.body["ali"][10:16] == ["s1.1", "s1.1", "s1.2", "s1.2", "s1.3", "s1.3"]
    exp, _ = run([HOLD_P1, {"at": "shot.in + 10f", "sequence": "s1", "cadence": "threes"}])
    assert exp.body["ali"][10:19] == ["s1.1"] * 3 + ["s1.2"] * 3 + ["s1.3"] * 3
    assert exp.body["ali"][19] == "p2"


def test_sequence_without_target_holds_its_last_drawing_and_a_later_transition_snaps() -> None:
    seq = {k: v for k, v in SEQ.items() if k != "to"}
    exp, _ = run(
        [
            HOLD_P1,
            {"at": "shot.in + 10f", "sequence": "s1", "cadence": "ones"},
            {"at": "shot.in + 40f", "to": "p2", "via": "inbetween", "cadence": "twos"},
        ],
        seq=seq,
    )
    cells = exp.body["ali"]
    assert cells[10:13] == ["s1.1", "s1.2", "s1.3"]
    assert cells[13:40] == ["s1.3"] * 27  # held, and no "s1.3-p2.k" inbetweens: there is no transition from it
    assert cells[40:] == ["p2"] * 60


def test_unknown_sequence_is_reported_and_skipped() -> None:
    exp, findings = run([HOLD_P1, {"at": "shot.in + 10f", "sequence": "s9"}])
    assert exp.body["ali"] == ["p1"] * 100
    assert "timing.sequence_unknown" in codes(findings)


def test_sequence_that_does_not_start_from_the_current_pose_is_warned() -> None:
    seq = dict(SEQ, **{"from": "p2"})
    _, findings = run([HOLD_P1, {"at": "shot.in + 10f", "sequence": "s1"}], seq=seq)
    assert "timing.sequence_from" in codes(findings)
    _, findings = run([HOLD_P1, {"at": "shot.in + 10f", "sequence": "s1"}])
    assert "timing.sequence_from" not in codes(findings)


# --- loops ------------------------------------------------------------------------------------------------------------


def test_loop_cycles_poses_with_inbetweens_until_the_end_anchor() -> None:
    assert LOOP_STEP_DEFAULT == 8
    loop = {"at": "shot.in + 10f", "loop": ["p1", "p2"], "cadence": "twos", "until": "shot.in + 40f"}
    exp, findings = run([HOLD_P1, loop])
    cells = exp.body["ali"]
    # step 8 frames: p1 holds 8 - 2 inbetweens x 2 = 4 frames, then p1-p2.1, p1-p2.2; p2 -> p1 has no transition: snap
    one_cycle = ["p1"] * 4 + ["p1-p2.1"] * 2 + ["p1-p2.2"] * 2 + ["p2"] * 8
    assert cells[10:40] == (one_cycle + one_cycle)[:30]
    assert cells[40:] == ["p2"] * 60  # after `until` the last drawing shown is held
    assert cells[:10] == ["p1"] * 10
    assert exp.unsupported == [] and "timing.unsupported" not in codes(findings)


def test_loop_step_comes_from_the_named_beat() -> None:
    loop = {
        "at": "shot.in + 10f",
        "loop": ["p1", "p2"],
        "step": "beat.action_step",
        "cadence": "twos",
        "until": "shot.in + 34f",
    }
    exp, _ = run([HOLD_P1, loop])
    one_cycle = ["p1"] * 2 + ["p1-p2.1"] * 2 + ["p1-p2.2"] * 2 + ["p2"] * 6  # step 6
    assert exp.body["ali"][10:34] == one_cycle + one_cycle


def test_loop_can_contain_a_sequence() -> None:
    loop = {"at": "shot.in + 10f", "loop": ["s1", "p1"], "cadence": "ones", "until": "shot.in + 32f"}
    exp, _ = run([HOLD_P1, loop])
    # s1 plays its three drawings on ones (its own length, step ignored); p1 is held one step (8): 11 frames a cycle
    one_cycle = ["s1.1", "s1.2", "s1.3"] + ["p1"] * 8
    assert exp.body["ali"][10:32] == one_cycle + one_cycle


def test_empty_or_reversed_loop_is_an_error() -> None:
    loop = {"at": "shot.in + 40f", "loop": ["p1", "p2"], "until": "shot.in + 40f"}
    exp, findings = run([HOLD_P1, loop])
    bad = [f for f in findings if f.code == "timing.loop_empty"]
    assert bad and bad[0].level == "error"
    assert exp.body["ali"] == ["p1"] * 100


def test_loop_with_an_unknown_item_is_reported() -> None:
    loop = {"at": "shot.in + 10f", "loop": ["p1", "s7"], "until": "shot.in + 40f"}
    _, findings = run([HOLD_P1, loop])
    assert "timing.loop_unknown" in codes(findings)


def test_sequence_on_an_eyes_track_is_still_unsupported() -> None:
    exp, findings = run([HOLD_P1], ali_eyes=[{"at": "shot.in + 10f", "sequence": "s1"}])
    assert exp.unsupported == [("ali.eyes", 0)]
    assert "timing.unsupported" in codes(findings)


# --- drawing jobs -----------------------------------------------------------------------------------------------------


def person(colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (1024, 1536), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((350, 200, 650, 560), fill=colour, outline=(0, 0, 0), width=6)
    return img


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    ep = p / EP
    (ep / "shots" / "sh01").mkdir(parents=True)
    (ep / "audio").mkdir()
    (p / "referans").mkdir()
    (p / "style.yaml").write_text(yaml.safe_dump(style_dict(), allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    shot_dir = ep / "shots" / "sh01"
    (shot_dir / "poses.yaml").write_text(yaml.safe_dump(poses_dict(), allow_unicode=True), encoding="utf-8")
    body = [HOLD_P1, {"at": "shot.in + 10f", "sequence": "s1", "cadence": "twos"}]
    (shot_dir / "xsheet.yaml").write_text(yaml.safe_dump(xsheet_dict(body), allow_unicode=True), encoding="utf-8")
    for lid, text in ALIGNS.items():
        a = align(lid, text)
        (ep / "audio" / f"{lid}.align.json").write_text(json.dumps(a.model_dump(mode="json")), encoding="utf-8")
    buf = io.BytesIO()
    person((240, 200, 160)).save(buf, "PNG")
    (p / "referans" / "ali.png").write_bytes(buf.getvalue())
    (p / "referans" / "ayse.png").write_bytes(buf.getvalue())
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: false }\n",
        encoding="utf-8",
    )
    (p / "locations.yaml").write_text(
        "version: 1\nlocations:\n  salon: { description: 'a small test living room' }\n", encoding="utf-8"
    )
    return p


def by_kind(jobs: list[DrawJob]) -> dict[str, list[DrawJob]]:
    out: dict[str, list[DrawJob]] = {}
    for j in jobs:
        out.setdefault(j.kind, []).append(j)
    return out


def test_sequence_drawings_are_planned_in_order_each_from_the_previous(prod: Path) -> None:
    jobs = plan_draw(prod, EP, "sh01")
    k = by_kind(jobs)
    assets = prod / EP / "assets" / "sh01" / "ali"
    ref = prod / "referans" / "ali.png"
    assert [(j.character, j.drawing) for j in k["pose"]] == [("ali", "p1"), ("ali", "p2")]  # from and to are keys
    seq = k["sequence"]
    assert [j.drawing for j in seq] == ["s1.1", "s1.2", "s1.3"]
    assert [j.refs for j in seq] == [(ref, assets / "p1.png"), (ref, assets / "s1.1.png"), (ref, assets / "s1.2.png")]
    assert [j.out for j in seq] == [assets / "s1.1.png", assets / "s1.2.png", assets / "s1.3.png"]
    after_poses = max(j.wave for j in k["pose"]) + 1
    assert [j.wave for j in seq] == [after_poses, after_poses + 1, after_poses + 2]
    assert "Kol omuz hizasında" in seq[1].prompt
    assert "Image 2 is the previous pose" in seq[1].prompt  # the key_pose_next template, no new template
    assert all(j.size == "1024x1536" and j.reasoning == "high" for j in seq)
    assert {j.drawing for j in k.get("mouth", []) + k.get("eyes", [])} <= {"p1", "p2"}  # patches only on keys


def test_job_order_puts_sequences_after_inbetweens(prod: Path) -> None:
    kinds = [j.kind for j in plan_draw(prod, EP, "sh01")]
    order = ["background", "pose", "inbetween", "sequence", "mouth", "eyes"]
    assert kinds == sorted(kinds, key=order.index)


# --- placement --------------------------------------------------------------------------------------------------------

REGIONS = Regions.model_validate(
    {
        "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
        "scale": {"px_per_cm": 3.211, "source": "sofa", "known_cm": 45},
        "objects": [
            {"name": "sofa", "kind": "sofa", "box": [536, 462, 1427, 847], "floor_y": 848, "confidence": 0.96,
             "mask": "masks/sofa.png", "front": False},
        ],
        "places": [
            {"name": "sofa_left", "object": "sofa", "kind": "seat", "seat": [802, 704], "floor": [802, 874]},
        ],
    }
)  # fmt: skip


def lm(head: list[int] | None = None, hip: list[float] | None = None, foot: list[float] | None = None) -> dict:  # type: ignore[type-arg]
    box = lambda b: None if b is None else {"box": b, "confidence": 0.9}  # noqa: E731
    return {"version": 1, "image": "x.png", "size": [1280, 1920], "facing": "right", "head": box(head),
            "face": None, "feet": [], "feet_source": "foot", "foot_low": foot, "pants": None, "hip": hip,
            "hands": [], "cups": []}  # fmt: skip


BETWEEN = lm(head=[420, 200, 700, 700], hip=[550, 1600], foot=[700, 1840])
NAIME = CharacterInput(
    name="naime",
    target_head_px=118.0,
    mirror=False,
    place="sofa_left",
    poses={"p1": PoseInfo(support="sit"), "p2": PoseInfo(support="sit")},
    landmarks={
        "p1": lm(head=[400, 100, 700, 690], hip=[500, 1700], foot=[800, 1750]),
        "p2": lm(head=[500, 300, 760, 772], hip=[600, 1500], foot=[640, 1852]),
        "p1-p2.1": BETWEEN,
    },
)


def test_sequence_drawing_is_placed_like_an_inbetween_of_its_keys() -> None:
    char = dataclasses.replace(NAIME, landmarks={**NAIME.landmarks, "s1.1": BETWEEN}, sequences={"s1": ("p1", "p2")})
    placements, warnings = solve([char], REGIONS)
    ib, sq = placements["naime"]["p1-p2.1"], placements["naime"]["s1.1"]
    assert (sq.scale, sq.x, sq.y, sq.anchor, sq.depth, sq.occluders) == (
        ib.scale, ib.x, ib.y, ib.anchor, ib.depth, ib.occluders
    )  # fmt: skip
    assert not any("s1.1" in w and "unknown drawing name" in w for w in warnings)


def test_sequence_without_target_is_placed_from_its_start_key() -> None:
    char = dataclasses.replace(NAIME, landmarks={**NAIME.landmarks, "s2.1": BETWEEN}, sequences={"s2": ("p1", None)})
    placements, warnings = solve([char], REGIONS)
    assert placements["naime"]["s2.1"].anchor != "fallback" or placements["naime"]["p1"].anchor == "fallback"
    assert not any("s2.1" in w and "unknown drawing name" in w for w in warnings)


def test_character_input_sequences_default_to_empty() -> None:
    assert NAIME.sequences == {}
