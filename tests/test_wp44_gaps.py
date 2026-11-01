"""WP-44 items 2-4: draw extra steps, gate without placement, doctor optional key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz import cli


def test_draw_calls_link_retire_measure_in_order(
    prod: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import karagoz.engine.landmarks as landmarks_module
    import karagoz.studio.draw as draw_module
    from karagoz.studio.draw import DrawReport

    order: list[str] = []

    def fake_link(p: Path, ep: str, shot: str) -> None:
        order.append("link")

    def fake_retire(jobs: list[Any], assets: Path) -> list[Path]:
        order.append("retire")
        return []

    def fake_run(jobs: list[Any], image: Any, **kw: Any) -> DrawReport:
        order.append("run")
        return DrawReport(done=[], cached=[], failed=[], calls=0)

    def fake_measure(p: Path, ep: str, shot: str, sam: Any, **kw: Any) -> list[Path]:
        order.append("measure")
        return []

    monkeypatch.setattr(draw_module, "link_location_background", fake_link)
    monkeypatch.setattr(draw_module, "retire_stale", fake_retire)
    monkeypatch.setattr(draw_module, "run_draw", fake_run)
    monkeypatch.setattr(landmarks_module, "measure_shot", fake_measure)

    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    regions = {"version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080], "scale": None,
               "objects": [], "places": []}  # fmt: skip
    (loc / "regions.yaml").write_text(yaml.safe_dump(regions), encoding="utf-8")

    assert cli.main(["draw", str(prod), EP, "sh01"]) == 0
    assert order == ["link", "retire", "run", "measure"]


def test_gate_without_placement_writes_noted_file_and_exits_zero(
    prod: Path,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["gate", str(prod), EP, "sh01"]) == 0
    out = capsys.readouterr().out
    assert "no placement.json" in out
    assert out.strip().count("\n") == 0  # one line
    data = json.loads((prod / EP / "review" / "sh01" / "gate.json").read_text(encoding="utf-8"))
    assert data["shot"] == "sh01"
    assert data["flags"] == [] and data["silenced"] == [] and data["dropped"] == 0
    assert "no placement.json" in data["note"]


def test_gate_help_says_what_the_gate_needs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["gate", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "placement.json" in out or "region map" in out


def test_doctor_lists_optional_typesafe_key_without_its_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    from karagoz import config

    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret-value")
    monkeypatch.setattr(config, "has_secret", lambda name: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "TYPESAFE_API_KEY" in out
    assert "typesafe-secret-value" not in out


def test_doctor_missing_optional_key_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    from karagoz import config

    def fake_has(name: str) -> bool:
        return name != "TYPESAFE_API_KEY"

    monkeypatch.setattr(config, "has_secret", fake_has)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "TYPESAFE_API_KEY" in out
