"""Acceptance tests for WP-19 (spec v1.1: pose support/on/rests_on/holds, regions.yaml). Do not edit."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from karagoz.spec import Place, Pose, RegionObject, Regions, RegionScale, SpecError, check_episode, kind_of, load

FIX = Path(__file__).parent / "fixtures" / "spec"

REGIONS = {
    "version": 1,
    "location": "salon",
    "image": "background.png",
    "size": [1920, 1080],
    "scale": {"px_per_cm": 3.21, "source": "koltuk", "known_cm": 45},
    "objects": [
        {"name": "koltuk", "kind": "sofa", "box": [536, 462, 1427, 848], "floor_y": 848, "confidence": 0.96,
         "mask": "masks/koltuk.png", "front": False},
        {"name": "sehpa", "kind": "coffee table", "box": [652, 782, 1296, 977], "floor_y": 977, "confidence": 0.93,
         "mask": "masks/sehpa.png", "front": True},
    ],
    "places": [
        {"name": "koltuk", "object": "koltuk", "kind": "seat", "seat": [980, 704], "floor": [980, 874]},
        {"name": "kapi", "object": "koltuk", "kind": "floor", "floor": [200, 900]},
    ],
}  # fmt: skip


@pytest.fixture
def ep(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    e = prod / "ep"
    (e / "shots" / "sh01").mkdir(parents=True)
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    for name in ["lines.yaml", "shots.yaml", "continuity.yaml"]:
        shutil.copy(FIX / name, e / name)
    for name in ["poses.yaml", "xsheet.yaml"]:
        shutil.copy(FIX / name, e / "shots" / "sh01" / name)
    return e


def write_regions(ep: Path, data: dict | None = None) -> Path:  # type: ignore[type-arg]
    d = ep.parent / "locations" / "salon"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "regions.yaml"
    p.write_text(yaml.safe_dump(data or REGIONS, allow_unicode=True), encoding="utf-8")
    return p


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def codes(findings: list, level: str | None = None) -> list[tuple[str, str, str]]:  # type: ignore[type-arg]
    """Errors and warnings (info findings such as timing.no_audio are not counted), or only `level`."""
    return [(f.code, f.file, f.path) for f in findings if f.level == level or (level is None and f.level != "info")]


# --- models --------------------------------------------------------------------------------------------------------


def test_pose_new_fields_are_optional_and_typed() -> None:
    base = {"id": "p1", "view": "front", "expression": "neutral", "acting": "Oturuyor, eller dizde."}
    p = Pose.model_validate(base)
    assert p.support is None and p.on is None and p.rests_on is None and p.holds is None
    p = Pose.model_validate({**base, "support": "sit", "on": "sofa_left", "rests_on": ["coffee_table"],
                             "holds": {"right": "cup", "left": None}})  # fmt: skip
    assert p.support == "sit" and p.on == "sofa_left" and p.rests_on == ["coffee_table"]
    assert p.holds == {"right": "cup", "left": None}
    for bad in ({"support": "kneel"}, {"on": "Sofa"}, {"holds": {"middle": "cup"}}, {"rests_on": ["Masa"]}):
        with pytest.raises(ValidationError):
            Pose.model_validate({**base, **bad})


def test_regions_model() -> None:
    r = Regions.model_validate(REGIONS)
    assert isinstance(r.scale, RegionScale) and r.scale.px_per_cm == pytest.approx(3.21)
    assert isinstance(r.objects[0], RegionObject) and r.objects[1].front is True and r.objects[1].kind == "coffee table"
    assert isinstance(r.places[0], Place) and r.places[0].seat == (980, 704) and r.places[1].seat is None
    assert Regions.model_validate({**REGIONS, "scale": None}).scale is None


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda d: d["objects"][0].update(box=[536, 462, 500, 848]), "regions_box"),
        (lambda d: d["places"][0].pop("seat"), "regions_seat"),
        (lambda d: d["objects"].append(dict(d["objects"][0])), "regions_names"),
        (lambda d: d["places"].append(dict(d["places"][1])), "regions_names"),
        (lambda d: d["places"][1].update(object="kapi_objesi"), "regions_names"),
    ],
)
def test_regions_model_rules(change, code) -> None:  # type: ignore[no-untyped-def]
    import copy

    d = copy.deepcopy(REGIONS)
    change(d)
    with pytest.raises(ValidationError) as exc:
        Regions.model_validate(d)
    assert code in str(exc.value)


def test_regions_kind_and_load(tmp_path: Path) -> None:
    assert kind_of(Path("x/locations/salon/regions.yaml")) == "regions"
    p = tmp_path / "regions.yaml"
    p.write_text(yaml.safe_dump(REGIONS), encoding="utf-8")
    assert isinstance(load(p), Regions)
    p.write_text("version: 1\nlocation: salon\n", encoding="utf-8")
    with pytest.raises(SpecError):
        load(p)


# --- check_episode -------------------------------------------------------------------------------------------------


def test_clean_with_and_without_regions(ep: Path) -> None:
    assert codes(check_episode(ep)) == []  # the fixture's poses carry support / holds
    write_regions(ep)
    assert codes(check_episode(ep)) == []


def test_missing_support_warns(ep: Path) -> None:
    held = "        holds: { right: glass, left: null }\n      - id: p2"
    edit(ep / "shots/sh01/poses.yaml", "        support: sit\n" + held, held)
    assert codes(check_episode(ep)) == [("support.missing", "shots/sh01/poses.yaml", "characters.ali.poses[0]")]
    assert next(f for f in check_episode(ep) if f.code == "support.missing").level == "warn"


def test_unknown_place_in_shots(ep: Path) -> None:
    write_regions(ep)
    edit(ep / "shots.yaml", "{ name: ayse, place: koltuk", "{ name: ayse, place: balkon")
    found = [f for f in check_episode(ep) if f.code == "place.unknown"]
    assert codes(found) == [("place.unknown", "shots.yaml", "shots[0].characters[1].place")]
    assert found[0].level == "error" and found[0].hint is not None and "kapi, koltuk" in found[0].hint


def test_unknown_place_and_object_in_poses(ep: Path) -> None:
    write_regions(ep)
    edit(ep / "shots/sh01/poses.yaml", "rests_on: [sehpa]", "rests_on: [masa]\n        on: balkon")
    got = sorted(codes(check_episode(ep), "error"))
    assert got == [
        ("place.unknown", "shots/sh01/poses.yaml", "characters.ali.poses[1].on"),
        ("rests_on.unknown", "shots/sh01/poses.yaml", "characters.ali.poses[1].rests_on[0]"),
    ]


def test_sitting_on_a_floor_place(ep: Path) -> None:
    write_regions(ep)
    edit(ep / "shots/sh01/poses.yaml", "rests_on: [sehpa]", "rests_on: [sehpa]\n        on: kapi")
    assert codes(check_episode(ep), "error") == [
        ("support.not_seat", "shots/sh01/poses.yaml", "characters.ali.poses[1].support")
    ]
    edit(ep / "shots/sh01/poses.yaml", "        support: sit\n        rests_on: [sehpa]\n        on: kapi",
         "        support: stand\n        rests_on: [sehpa]\n        on: kapi")  # fmt: skip
    assert codes(check_episode(ep), "error") == []
    edit(ep / "shots.yaml", "{ name: ali, place: koltuk", "{ name: ali, place: kapi")  # shot place is a floor
    assert ("support.not_seat", "shots/sh01/poses.yaml", "characters.ali.poses[0].support") in codes(check_episode(ep))


def test_held_prop_must_be_known(ep: Path) -> None:
    edit(ep / "shots/sh01/poses.yaml", "holds: { right: glass, left: null }\n      - id: p2",
         "holds: { right: cup, left: null }\n      - id: p2")  # fmt: skip
    found = [f for f in check_episode(ep) if f.code == "holds.unknown_prop"]
    assert codes(found) == [("holds.unknown_prop", "shots/sh01/poses.yaml", "characters.ali.poses[0].holds.right")]
    assert found[0].level == "warn" and found[0].hint == "glass"


def test_broken_regions_file_is_reported(ep: Path) -> None:
    p = write_regions(ep)
    p.write_text("version: 1\nlocation: salon\nimage: x.png\n", encoding="utf-8")
    files = {f.file for f in check_episode(ep) if f.level == "error"}
    assert files == {"../locations/salon/regions.yaml"}
