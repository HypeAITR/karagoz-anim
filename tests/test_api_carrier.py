"""Acceptance tests for WP-F12 (ApiCarrier: one model request instead of an agent loop).

Written by the maintainer; do not edit. The Meta API is faked with a scripted transport; the check command with a
scripted runner. No network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from karagoz.meta.client import MetaClient
from karagoz.studio.api_carrier import API_MODEL, ApiCarrier
from karagoz.studio.carriers import PRICE_PER_MTOK, Brief

CHECK = '"py" -m karagoz check ep'
OUT = "ep/shots/sh01/poses.yaml"

BRIEF = f"""# Task: sh01/layout

Department: layout

# Your department guide

## Inputs

- guide/example-only.yaml

The guide's own example list above is not this task's input list.

## Inputs

- ep/lines.yaml
- ep/missing.yaml
- ref.png

## Output

Write only these files:

- {OUT}

## Questions

Decide yourself.

# Check your work

Run `{CHECK}` and fix every error. Read the warnings.
"""


def files(*pairs: tuple[str, str]) -> str:
    return "".join(f"=== FILE: {path} ===\n{body}\n=== END ===\n" for path, body in pairs)


class Api:
    """Scripted chat completions: each call pops the next answer; records every request body."""

    def __init__(self, answers: list[str | tuple[str, str]]) -> None:
        self.answers = list(answers)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> tuple[int, dict]:
        assert url.endswith("/chat/completions")
        self.bodies.append(json.loads(json.dumps(body)))
        item = self.answers.pop(0)
        text, finish = (item, "stop") if isinstance(item, str) else item
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 100}}
        return 200, {"choices": [{"message": {"content": text}, "finish_reason": finish}], "usage": usage}


class Runner:
    def __init__(self, reports: list[str]) -> None:
        self.reports = list(reports)
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, command: str, workdir: Path) -> str:
        self.calls.append((command, workdir))
        return self.reports.pop(0) if self.reports else "0 error(s), 0 warning(s)"


def make(tmp_path: Path, api: Api, runner: Runner | None = None, **kw: Any) -> ApiCarrier:
    client = MetaClient(api_key="k", transport=api, sleep=lambda s: None)
    return ApiCarrier(client=client, run_command=runner or Runner([]), **kw)


@pytest.fixture
def wd(tmp_path: Path) -> Path:
    w = tmp_path / "wd"
    (w / "ep").mkdir(parents=True)
    (w / "ep" / "lines.yaml").write_text("version: 1\nlines: [L01 Çay koy]\n", encoding="utf-8")
    Image.new("RGB", (8, 8), (255, 0, 0)).save(w / "ref.png")
    (w / "notes" / "layout").mkdir(parents=True)
    (w / "notes" / "layout" / "DECISIONS.md").write_text("# Decisions\n- old line\n", encoding="utf-8")
    (w / "BRIEF.md").write_text(BRIEF, encoding="utf-8")
    return w


def brief(wd: Path, **kw: Any) -> Brief:
    kw.setdefault("prompt", "Read BRIEF.md in the working directory.")
    return Brief(task_id="sh01/layout", workdir=wd, **kw)


def text_of(body: dict[str, Any], i: int = 0) -> str:
    content = body["messages"][i]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(p["text"] for p in content if p.get("type") == "text")


# --- the first request ---------------------------------------------------------------------------------------------


def test_one_round_writes_the_file_and_runs_the_check(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "version: 1\nshot: sh01"))])
    runner = Runner(["info timing.no_audio x\n0 error(s), 0 warning(s)"])
    events: list[dict[str, Any]] = []
    r = make(tmp_path, api, runner).run(brief(wd), on_event=events.append)
    assert r.ok and r.error is None
    assert (wd / OUT).read_text(encoding="utf-8").strip() == "version: 1\nshot: sh01"
    assert runner.calls == [(CHECK, wd)]
    assert len(api.bodies) == 1 and api.bodies[0]["model"] == API_MODEL == "muse-spark-1.3-contributor"
    assert r.usage.calls == 1 and r.usage.input_tokens == 1000 and r.usage.output_tokens == 200
    assert r.usage.cached_tokens == 100
    assert r.cost_usd == pytest.approx((1000 * PRICE_PER_MTOK["input"] + 200 * PRICE_PER_MTOK["output"]) / 1e6)
    assert isinstance(r.session_id, str) and r.session_id
    usage = [e for e in events if e["payload_type"] == "run.usage"]
    assert usage and usage[-1]["payload"]["models"][0]["calls"] == 1
    notes = [e["payload"]["text"] for e in events if e["payload_type"] == "code.note"]
    assert any(n.startswith("api call 1:") for n in notes) and any(n.startswith("api check:") for n in notes)


def test_request_holds_brief_inputs_notes_and_images(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "x: 1"))])
    make(tmp_path, api).run(brief(wd))
    body = api.bodies[0]
    assert len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
    text = text_of(body)
    assert "Department: layout" in text  # the brief
    assert "=== ep/lines.yaml ===" in text and "L01 Çay koy" in text  # an input, inlined
    assert "=== ep/missing.yaml (missing) ===" in text
    assert "guide/example-only.yaml ===" not in text  # the guide's list is not the input list
    assert "ref.png (image 1, attached)" in text
    assert "- old line" in text  # the notes folder
    assert "=== FILE:" in text and "READ:" in text  # the protocol
    parts = body["messages"][0]["content"]
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 1 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_previous_output_is_shown(tmp_path: Path, wd: Path) -> None:
    (wd / OUT).parent.mkdir(parents=True)
    (wd / OUT).write_text("previous: take\n", encoding="utf-8")
    api = Api([files((OUT, "x: 1"))])
    make(tmp_path, api).run(brief(wd))
    assert "previous: take" in text_of(api.bodies[0])


def test_model_from_brief_and_contributor_only(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "x: 1"))])
    assert make(tmp_path, api).run(brief(wd, model="muse-spark-1.2-contributor")).ok
    assert api.bodies[0]["model"] == "muse-spark-1.2-contributor"
    api2 = Api([])
    r = make(tmp_path, api2, model="muse-spark-1.3").run(brief(wd))
    assert not r.ok and "only Spark Contributor" in (r.error or "") and api2.bodies == []


# --- the loop ------------------------------------------------------------------------------------------------------


def test_check_errors_go_back_and_get_fixed(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "bad: 1")), files((OUT, "good: 1"))])
    runner = Runner(["error support.x poses.yaml:p1 — broken\n1 error(s), 0 warning(s)", "0 error(s), 0 warning(s)"])
    r = make(tmp_path, api, runner).run(brief(wd))
    assert r.ok and r.usage.calls == 2 and len(runner.calls) == 2
    second = api.bodies[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "error support.x" in text_of(api.bodies[1], 2)
    assert (wd / OUT).read_text(encoding="utf-8").strip() == "good: 1"


def test_errors_after_the_last_round_fail(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "bad: 1"))] * 2)
    runner = Runner(["error a.b f:p — still broken\n1 error(s), 0 warning(s)"] * 2)
    r = make(tmp_path, api, runner, max_rounds=2).run(brief(wd))
    assert not r.ok and (r.error or "").startswith("check errors after the last round")
    assert "still broken" in (r.error or "") and len(api.bodies) == 2


def test_warnings_are_shown_once_then_done(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "first: 1")), "DONE"])
    runner = Runner(["warn holds.x poses.yaml:p — maybe\n0 error(s), 1 warning(s)"])
    r = make(tmp_path, api, runner).run(brief(wd))
    assert r.ok and len(api.bodies) == 2
    assert "only warnings" in text_of(api.bodies[1], 2) and "warn holds.x" in text_of(api.bodies[1], 2)
    assert (wd / OUT).read_text(encoding="utf-8").strip() == "first: 1"


def test_read_request_gets_the_file(tmp_path: Path, wd: Path) -> None:
    (wd / "sections").mkdir()
    (wd / "sections" / "handover.md").write_text("# Handover\nboth hands on the glass", encoding="utf-8")
    api = Api(["Let me read.READ: sections/handover.md\nREAD: ../secret.txt", files((OUT, "x: 1"))])
    r = make(tmp_path, api).run(brief(wd))
    assert r.ok and len(api.bodies) == 2
    reply = text_of(api.bodies[1], 2)
    assert "both hands on the glass" in reply
    assert "../secret.txt (not allowed)" in reply


def test_reads_are_limited(tmp_path: Path, wd: Path) -> None:
    api = Api(["READ: a.md", "READ: b.md", "READ: c.md", files((OUT, "x: 1"))])
    r = make(tmp_path, api, max_reads=2).run(brief(wd))
    assert r.ok and len(api.bodies) == 4
    assert "no file blocks" in text_of(api.bodies[3], 6)  # the third READ counted as a round without files


def test_files_outside_the_outputs_are_refused(tmp_path: Path, wd: Path) -> None:
    api = Api([
        files((OUT, "x: 1"), ("src/evil.py", "boom"), ("../out.txt", "boom"), ("notes/layout/LESSONS.md", "1. ok")),
        files((OUT, "x: 1")),
    ])  # fmt: skip
    runner = Runner(["0 error(s), 0 warning(s)", "0 error(s), 0 warning(s)"])
    r = make(tmp_path, api, runner).run(brief(wd))
    assert r.ok
    assert not (wd / "src" / "evil.py").exists() and not (wd.parent / "out.txt").exists()
    assert (wd / "notes" / "layout" / "LESSONS.md").read_text(encoding="utf-8").strip() == "1. ok"
    msg = text_of(api.bodies[1], 2)
    assert "file.not_yours" in msg and "src/evil.py" in msg


def test_no_files_then_gives_up(tmp_path: Path, wd: Path) -> None:
    api = Api(["I think it is fine.", "Still fine."])
    r = make(tmp_path, api, max_rounds=2).run(brief(wd))
    assert not r.ok and r.error == "no files in answer"
    assert "no file blocks" in text_of(api.bodies[1], 2)


def test_question_stops_without_check(tmp_path: Path, wd: Path) -> None:
    api = Api([files(("QUESTION.md", "topic: script\nWhich hand?"))])
    runner = Runner([])
    r = make(tmp_path, api, runner).run(brief(wd))
    assert r.ok and (wd / "QUESTION.md").is_file() and runner.calls == []


def test_cut_off_answer_is_asked_again(tmp_path: Path, wd: Path) -> None:
    api = Api([(files((OUT, "half")), "length"), files((OUT, "whole: 1"))])
    r = make(tmp_path, api).run(brief(wd))
    assert r.ok and "cut off" in text_of(api.bodies[1], 2)
    assert (wd / OUT).read_text(encoding="utf-8").strip() == "whole: 1"


def test_api_error_fails_the_run(tmp_path: Path, wd: Path) -> None:
    def broken(url: str, body: dict, headers: dict, timeout: float) -> tuple[int, dict]:
        return 400, {"error": {"message": "bad request"}}

    carrier = ApiCarrier(client=MetaClient(api_key="k", transport=broken, sleep=lambda s: None), run_command=Runner([]))
    r = carrier.run(brief(wd))
    assert not r.ok and "MetaApiError" in (r.error or "")
    assert (wd / "API_TRACE.json").is_file()


# --- sessions ------------------------------------------------------------------------------------------------------


def test_a_session_continues_with_the_prompt(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "x: 1")), "The handover section was clear; the timing one was not needed."])
    carrier = make(tmp_path, api)
    first = carrier.run(brief(wd))
    r = carrier.run(brief(wd, session_id=first.session_id, prompt="What was unclear?"))
    assert r.ok and r.final_text.startswith("The handover section")
    msgs = api.bodies[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"] and text_of(api.bodies[1], 2) == (
        "What was unclear?"
    )


def test_a_session_can_write_files_again(tmp_path: Path, wd: Path) -> None:
    api = Api(["no files here", files(("findings.yaml", "part: x"))])
    carrier = make(tmp_path, api, max_rounds=1)
    first = carrier.run(brief(wd))
    assert not first.ok
    (wd / "BRIEF.md").write_text(BRIEF.replace(f"- {OUT}", "- findings.yaml"), encoding="utf-8")
    r = carrier.run(brief(wd, session_id=first.session_id, prompt="findings.yaml is missing. Write it again."))
    assert r.ok and (wd / "findings.yaml").read_text(encoding="utf-8").strip() == "part: x"


def test_unknown_session(tmp_path: Path, wd: Path) -> None:
    api = Api([])
    r = make(tmp_path, api).run(brief(wd, session_id="nope", prompt="hi"))
    assert not r.ok and r.error == "unknown session nope" and api.bodies == []


# --- trace ---------------------------------------------------------------------------------------------------------


def test_trace_has_no_image_data(tmp_path: Path, wd: Path) -> None:
    api = Api([files((OUT, "x: 1"))])
    make(tmp_path, api).run(brief(wd))
    trace = json.loads((wd / "API_TRACE.json").read_text(encoding="utf-8"))
    assert trace["model"] == API_MODEL and len(trace["calls"]) == 1
    raw = json.dumps(trace)
    assert "base64" not in raw and '"path": "ref.png"' in raw
