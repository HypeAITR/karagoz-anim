"""WP-52: continuity is one file per scene, chained; the plan wires the chain."""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.briefs import make_brief
from karagoz.studio.orchestrator import inputs_digest, plan_tasks
from karagoz.studio.tasks import Board, State

EP = "ep1"

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

SCENE1 = """\
version: 1
scene: 1
states:
  - shot: sh01
    character: ali
    in: "Sitting on the sofa, holding a full glass of tea."
    out: "Sitting on the sofa, holding an empty glass of tea."
prop_states:
  glass: [full, half, empty]
props:
  - shot: sh01
    prop: glass
    in: { holder: ali, state: full }
    out: { holder: ali, state: empty }
"""

SCENE2 = """\
version: 1
scene: 2
states:
  - shot: sh02
    character: ali
    in: "Sitting on the sofa, holding an empty glass of tea."
    out: "Standing by the sofa, the empty glass on the table."
prop_states:
  glass: [full, half, empty]
props:
  - shot: sh02
    prop: glass
    in: { holder: ali, state: empty }
    out: { holder: null, place: table, state: empty }
"""

BASE_IN = [f"script/{EP}.txt", f"{EP}/lines.yaml", f"{EP}/shots.yaml"]


def _prod(tmp_path: Path, *, review: bool = False) -> Path:
    prod = tmp_path / "prod"
    (prod / "script").mkdir(parents=True)
    (prod / EP).mkdir(parents=True)
    (prod / "script" / f"{EP}.txt").write_text("ALI: Su ver bana.\n", encoding="utf-8")
    (prod / EP / "shots.yaml").write_text(TWO_SCENE_SHOTS, encoding="utf-8")
    if review:
        (prod / EP / "studio.yaml").write_text("version: 1\nreview: on\n", encoding="utf-8")
    return prod


def test_continuity_tasks_own_per_scene_files(tmp_path: Path) -> None:
    tasks = {t.id: t for t in plan_tasks(_prod(tmp_path), EP)}
    assert tasks["scene1/continuity"].files_out == [f"{EP}/continuity/scene1.yaml"]
    assert tasks["scene2/continuity"].files_out == [f"{EP}/continuity/scene2.yaml"]


def test_scene2_continuity_is_chained_on_scene1(tmp_path: Path) -> None:
    tasks = {t.id: t for t in plan_tasks(_prod(tmp_path), EP)}
    first = tasks["scene1/continuity"]
    assert first.inputs == ["ep/director"]
    assert first.files_in == BASE_IN
    assert first.notes == []
    second = tasks["scene2/continuity"]
    assert second.inputs == ["ep/director", "scene1/continuity"]
    assert second.files_in == [*BASE_IN, f"{EP}/continuity/scene1.yaml"]


def test_chained_continuity_names_the_previous_file_in_one_sentence(tmp_path: Path) -> None:
    prod = _prod(tmp_path)
    task = {t.id: t for t in plan_tasks(prod, EP)}["scene2/continuity"]
    assert task.notes == [
        f"Start from where scene 1 ended (see {EP}/continuity/scene1.yaml for who holds which prop and who sits where)."
    ]
    brief = make_brief(task, prod, EP, tmp_path / "work", tmp_path / "skills", context="CTX")
    assert task.notes[0] in brief


def test_layout_reads_own_and_previous_scene_files(tmp_path: Path) -> None:
    tasks = {t.id: t for t in plan_tasks(_prod(tmp_path), EP)}
    assert tasks["sh01/layout"].inputs == ["scene1/continuity"]
    assert tasks["sh01/layout"].files_in == [*BASE_IN, f"{EP}/continuity/scene1.yaml"]
    assert tasks["sh02/layout"].inputs == ["scene2/continuity"]
    assert tasks["sh02/layout"].files_in == [*BASE_IN, f"{EP}/continuity/scene2.yaml", f"{EP}/continuity/scene1.yaml"]


def test_timer_and_gate_read_only_the_own_scene_file(tmp_path: Path) -> None:
    tasks = {t.id: t for t in plan_tasks(_prod(tmp_path, review=True), EP)}
    timer_in = tasks["sh02/timer"].files_in
    assert f"{EP}/continuity/scene2.yaml" in timer_in
    assert f"{EP}/continuity/scene1.yaml" not in timer_in
    gate_in = tasks["sh02/gate"].files_in
    assert f"{EP}/continuity/scene2.yaml" in gate_in
    assert f"{EP}/continuity/scene1.yaml" not in gate_in


def test_no_task_uses_the_old_single_continuity_path(tmp_path: Path) -> None:
    paths = [f for t in plan_tasks(_prod(tmp_path, review=True), EP) for f in (*t.files_in, *t.files_out)]
    assert f"{EP}/continuity.yaml" not in paths


def test_rewritten_scene1_stales_scene2_through_inputs_digest(tmp_path: Path) -> None:
    prod = _prod(tmp_path)
    cont = prod / EP / "continuity"
    cont.mkdir(parents=True)
    (cont / "scene1.yaml").write_text(SCENE1, encoding="utf-8")
    (cont / "scene2.yaml").write_text(SCENE2, encoding="utf-8")
    board = Board(plan_tasks(prod, EP))
    for task in board.tasks.values():
        task.state = State.DONE
        task.input_digest = inputs_digest(prod, task)
    (cont / "scene1.yaml").write_text(SCENE1.replace("holding an empty glass", "holding the empty glass"))
    changes = board.refresh(lambda _rels: "", task_digest=lambda t: inputs_digest(prod, t))
    assert any(c.task == "scene2/continuity" and c.old is State.DONE and c.new is State.STALE for c in changes)
    assert not any(c.task == "scene1/continuity" and c.new is State.STALE for c in changes)
