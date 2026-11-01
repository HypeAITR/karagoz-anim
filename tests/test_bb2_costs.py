"""BB-2 bug hunt: uncounted spend (claims Spark calls, interview cost events).

Every test in this file FAILS on today's code. No paid API calls, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from karagoz.studio import claims as claims_module
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.orchestrator import Studio

EP = "ep1"


def test_make_claims_reports_its_spend(tmp_path: Path) -> None:
    """make_claims uses paid Spark calls but reports no cost anywhere.

    Every other paid step reports its spend (jev.json cost_usd, check parts
    cost_usd, RunResult.cost_usd); claims.json has no cost channel, and the
    draw branch never charges it, so budget and regress miss this spend.
    """
    fix = Path(__file__).parent / "fixtures"
    prod = tmp_path / "prod"
    poses_dst = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses_dst.parent.mkdir(parents=True)
    poses_dst.write_bytes((fix / "poses.yaml").read_bytes())
    calls: list[str] = []

    def fake_model(prompt: str) -> str:
        calls.append(prompt)
        return '{"drawings": {"ali/p1": ["a hand holds a cup"]}, "pairs": {}}'

    old = claims_module.text_model
    claims_module.text_model = fake_model
    try:
        out = claims_module.make_claims(prod, EP, "sh01", parallel=1)
    finally:
        claims_module.text_model = old
    assert calls, "stub model was never called; fixture has no transitions?"
    assert "cost_usd" in out, f"claims result carries no cost: {sorted(out)}"


class _InterviewAgent:
    """Answers the follow-up interview with a fixed paid cost."""

    name = "muse-exec"

    def run(self, brief: Brief, on_event: Any = None) -> RunResult:
        assert "unclear" in brief.prompt or "retake" in brief.prompt.lower() or brief.prompt
        return RunResult(ok=True, final_text="the note was vague", cost_usd=5.0)


def test_interview_spend_is_a_cost_event(tmp_path: Path) -> None:
    """Interview USD is charged to the budget but logged as kind=interview.

    Cost aggregation (regress.department_summary) sums only kind=cost, so
    the budget total and the event-log total diverge after any interview.
    """
    prod = tmp_path / "prod"
    (prod / EP).mkdir(parents=True)
    (prod / EP / "studio.yaml").write_text(
        "version: 1\nhuman: full\ninterview: on\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n",
        encoding="utf-8",
    )
    (prod / EP / "run").mkdir(parents=True, exist_ok=True)
    (prod / EP / "run" / "tasks.json").write_text(
        json.dumps(
            {
                "max_attempts": 3,
                "max_retakes": 3,
                "tasks": [
                    {
                        "id": "sh01/layout",
                        "inputs": [],
                        "files_in": [],
                        "files_out": [f"{EP}/shots/sh01/poses.yaml"],
                        "needs_approval": False,
                        "state": "running",
                        "attempts": 1,
                        "retakes": 1,
                        "notes": [],
                        "input_digest": None,
                        "input_files": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "skills" / "layout").mkdir(parents=True)
    (tmp_path / "skills" / "layout" / "SKILL.md").write_text("# layout\n", encoding="utf-8")
    studio = Studio(
        prod=prod,
        episode=EP,
        carriers={},  # type: ignore[arg-type]
        skills_dir=tmp_path / "skills",
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio.load()
    task = studio.board.tasks["sh01/layout"]
    workdir = studio.human_workdir(task.id)
    out_file = workdir / f"{EP}/shots/sh01/poses.yaml"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(b"poses\n")
    studio._complete_agent_run(
        task,
        workdir,
        RunResult(ok=True, session_id="s", cost_usd=1.0),
        [],
        _InterviewAgent(),  # type: ignore[arg-type]
    )
    assert studio.budget.total() == pytest.approx(6.0)
    cost_sum = sum(float(e.get("usd", 0.0)) for e in studio.log.read() if e.get("kind") == "cost")
    assert cost_sum == pytest.approx(studio.budget.total()), (
        f"event log counts ${cost_sum} of ${studio.budget.total()} charged (interview usd is kind=interview)"
    )
