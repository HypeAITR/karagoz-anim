"""Acceptance tests for WP-23 (inbetween hands from Layout or none, BODY in inbetweens, rising instead of squat).

Written by the maintainer; do not edit. Uses the mini production of the WP-09a tests (`test_draw.py`) and the spec
fixture of the WP-19 tests (`test_spec_v11.py`). The fixture's ali: p1 and p2 sit holding the glass in his right
hand; transition p1 -> p2 with two inbetweens; ayse has one pose.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_spec_v11 import codes, edit, ep  # noqa: F401  (ep is a pytest fixture)

import karagoz.studio.draw as draw
from karagoz.spec import PoseTransition, check_episode
from karagoz.studio.draw import DrawJob, plan_draw

SIT = (
    "BODY: sitting on an invisible seat at knee height: thighs horizontal, shins vertical, both feet flat on the "
    "ground below the knees; the hips are at knee height, not lower (not crouching, not squatting)."
)
CROUCH = (
    "BODY: rising out of the seat, half-way up: the hips just lifted off the invisible seat and still at about knee "
    "height, the upper body leaning forward over the knees, both feet flat on the ground and taking the weight; not "
    "squatting, not crouching low."
)
POSES = "shots/sh01/poses.yaml"


def ib_prompts(prod: Path, character: str = "ali") -> dict[str, str]:  # noqa: F811
    jobs = plan_draw(prod, EP, "sh01")
    return {j.drawing: j.prompt for j in jobs if j.character == character and j.kind in ("pose", "inbetween")}


def set_poses(prod: Path, mutate) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    p = prod / EP / POSES
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    mutate(data["characters"]["ali"])
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def props_file(prod: Path) -> None:  # noqa: F811
    (prod / "props.yaml").write_text(
        "props:\n  glass: { description: a small tulip-shaped Turkish tea glass }\n", encoding="utf-8"
    )


# --- templates -----------------------------------------------------------------------------------------------------


def test_templates_v5() -> None:
    data = yaml.safe_load(draw.TEMPLATES_PATH.read_text(encoding="utf-8"))
    assert data["version"] == 5
    assert data["support_text"]["crouch"] == CROUCH
    assert data["support_text"]["sit"] == SIT
    assert "CHANGE: {support}{description} Mouth closed." in data["inbetween"]
    assert set(data["support_words"]) == {"sit", "stand", "crouch", "lie"}
    assert {"{from_words}", "{to_words}", "{k}", "{n}", "{pct}", "{direction}"} <= {
        "{" + part.split("}")[0] + "}" for part in data["support_between"].split("{")[1:]
    }
    assert set(data["support_direction"]) == {"up", "down", "level"}


# --- hands ---------------------------------------------------------------------------------------------------------


def test_same_hands_in_both_keys_go_to_every_inbetween(prod: Path) -> None:  # noqa: F811
    j = ib_prompts(prod)
    for name in ("p1-p2.1", "p1-p2.2"):
        assert "HANDS: Right hand holds glass. Left hand is empty. Nothing else in the hands." in j[name]


def test_changing_hands_without_layout_words_get_no_hands_line(prod: Path) -> None:  # noqa: F811
    props_file(prod)
    set_poses(prod, lambda ali: ali["poses"][1].update(holds={"right": None, "left": None}))
    j = ib_prompts(prod)
    for name in ("p1-p2.1", "p1-p2.2"):
        assert "HANDS" not in j[name]
        assert "PROPS: glass is a small tulip-shaped Turkish tea glass." in j[name]
    assert "HANDS: Right hand is empty. Left hand is empty." in j["p2"]  # keys keep their own hands


def test_layout_writes_each_inbetweens_hands(prod: Path) -> None:  # noqa: F811
    def mutate(ali):  # type: ignore[no-untyped-def]
        ali["poses"][1]["holds"] = {"right": None, "left": None}
        ali["transitions"][0]["holds"] = [{"right": "glass", "left": None}, None]

    set_poses(prod, mutate)
    j = ib_prompts(prod)
    assert "HANDS: Right hand holds glass. Left hand is empty. Nothing else in the hands." in j["p1-p2.1"]
    assert "HANDS" not in j["p1-p2.2"]  # None: not said, keys differ -> no hands line


def test_three_inbetweens_middle_one_is_not_given_the_target_hands(prod: Path) -> None:  # noqa: F811
    def mutate(ali):  # type: ignore[no-untyped-def]
        ali["poses"][1]["holds"] = {"right": None, "left": None}
        ali["transitions"][0]["inbetweens"] = [
            "Kol göğüs hizasında, bardak dik duruyor.",
            "Kol yarıya kadar uzanmış, parmaklar açılıyor, bardak dik.",
            "Kol neredeyse tam uzanmış, bardak dik duruyor.",
        ]

    set_poses(prod, mutate)
    j = ib_prompts(prod)
    middle = [name for name in j if name.startswith("p1-p2.")]
    assert sorted(middle) == ["p1-p2.1", "p1-p2.2", "p1-p2.3"]
    for name in middle:
        assert "HANDS" not in j[name], name  # the old rule gave .2 and .3 the target's empty hands


# --- body ----------------------------------------------------------------------------------------------------------


def test_same_support_goes_to_inbetweens(prod: Path) -> None:  # noqa: F811
    j = ib_prompts(prod)
    assert f"CHANGE: {SIT} Kol göğüs hizasında" in j["p1-p2.1"]


def test_rising_inbetweens_get_the_way_up(prod: Path) -> None:  # noqa: F811
    set_poses(prod, lambda ali: ali["poses"][1].update(support="stand"))
    j = ib_prompts(prod)
    one, two = j["p1-p2.1"], j["p1-p2.2"]
    assert "BODY: on the way from sitting (image 1) to standing (image 2), drawing 1 of 2 between them" in one
    assert "about 33% of the way" in one and "about 67% of the way" in two
    assert "While rising the body never goes lower than in image 1." in one
    assert "Kol göğüs hizasında" in one and "  " not in one.split("CHANGE:")[1].split("Mouth closed")[0]


def test_going_down_and_crouch_words(prod: Path) -> None:  # noqa: F811
    def mutate(ali):  # type: ignore[no-untyped-def]
        ali["poses"][0]["support"] = "crouch"
        ali["poses"][1]["support"] = "sit"

    set_poses(prod, mutate)
    j = ib_prompts(prod)
    assert "BODY: on the way from half-way up out of the seat (image 1) to sitting (image 2)" in j["p1-p2.1"]
    assert "While going down the body never goes higher than in image 1." in j["p1-p2.1"]
    assert CROUCH in j["p1"]


def test_no_support_no_body(prod: Path) -> None:  # noqa: F811
    set_poses(prod, lambda ali: ali["poses"][0].pop("support"))
    assert "BODY:" not in ib_prompts(prod)["p1-p2.1"]


# --- spec and check ------------------------------------------------------------------------------------------------


def test_transition_holds_field() -> None:
    t = PoseTransition.model_validate(
        {"from": "p1", "to": "p2", "inbetweens": ["Kol göğüs hizasında."], "holds": [{"right": "glass", "left": None}]}
    )
    assert t.holds == [{"right": "glass", "left": None}]
    t2 = PoseTransition.model_validate({"from": "p1", "to": "p2", "inbetweens": ["Kol göğüs hizasında."]})
    assert t2.holds is None


def test_check_hands_change_without_inbetween_holds(ep: Path) -> None:  # noqa: F811
    assert codes(check_episode(ep)) == []  # fixture: same hands in both keys
    edit(ep / POSES, "rests_on: [sehpa]\n        holds: { right: glass, left: null }",
         "rests_on: [sehpa]\n        holds: { right: null, left: null }")  # fmt: skip
    found = [f for f in check_episode(ep) if f.code == "holds.inbetween_missing"]
    assert codes(found) == [("holds.inbetween_missing", POSES, "characters.ali.transitions[0].holds")]
    assert found[0].level == "warn" and "the hands change" in found[0].message


def test_check_inbetween_holds_count_and_props(ep: Path) -> None:  # noqa: F811
    edit(ep / POSES, "          - \"Kol yarıya kadar uzanmış, bardak dik duruyor.\"\n",
         "          - \"Kol yarıya kadar uzanmış, bardak dik duruyor.\"\n"
         "        holds:\n          - { right: glass, left: null }\n")  # fmt: skip
    errs = [f for f in check_episode(ep) if f.code == "holds.inbetween_count"]
    assert codes(errs, "error") == [("holds.inbetween_count", POSES, "characters.ali.transitions[0].holds")]
    one = "        holds:\n          - { right: glass, left: null }\n"
    edit(ep / POSES, one, one + "          - { right: cup, left: null }\n")
    assert [f for f in check_episode(ep) if f.code == "holds.inbetween_count"] == []
    unknown = [f for f in check_episode(ep) if f.code == "holds.unknown_prop"]
    assert codes(unknown) == [("holds.unknown_prop", POSES, "characters.ali.transitions[0].holds[1].right")]


def test_job_types_unchanged(prod: Path) -> None:  # noqa: F811
    jobs = plan_draw(prod, EP, "sh01")
    assert all(isinstance(j, DrawJob) for j in jobs)
    assert {j.kind for j in jobs} >= {"pose", "inbetween"}
