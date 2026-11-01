"""BB-2 #1 (= AUD-2 #9): each scene's continuity task owns its output file (frozen acceptance test, WP-52)."""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.orchestrator import plan_tasks

EP = "ep1"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


TWO_SCENE_SHOTS = """\
version: 1
episode: ep1
shots:
  - id: sh01
    scene: 1
    location: salon
    framing: wide
    lines: [L01]
    in: start
    out: L01.end
  - id: sh02
    scene: 2
    location: salon
    framing: medium
    lines: [L02]
    in: L01.end
    out: L02.end
"""


def test_two_scenes_get_distinct_continuity_outputs(tmp_path: Path) -> None:
    """Each scene's continuity task must own its output file.

    continuity.yaml holds a single `scene:` number, so two scene tasks sharing
    one path cannot both be kept: the second run overwrites the first scene.
    """
    prod = tmp_path / "prod"
    _write(prod / EP / "shots.yaml", TWO_SCENE_SHOTS)
    tasks = {t.id: t for t in plan_tasks(prod, EP)}
    cont = {tid: t.files_out for tid, t in tasks.items() if tid.endswith("/continuity")}
    assert set(cont) == {"scene1/continuity", "scene2/continuity"}
    outs = [o for fos in cont.values() for o in fos]
    assert len(set(outs)) == len(outs), f"continuity tasks share output files: {cont}"
