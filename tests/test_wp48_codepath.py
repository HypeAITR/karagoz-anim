"""WP-48 extra tests: the draw branch books claims spend (#3), one spend counter on the
code-task path (#10: budget total == sum of kind=cost events, no double counting)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw
from test_draw import EP, FakeImage, person, png, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.meta.client import IMAGE_USD_PER_IMAGE, MetaClient
from karagoz.studio import claims as claims_module
from karagoz.studio.draw import plan_draw
from karagoz.studio.orchestrator import Studio, code_work
from karagoz.studio.tasks import State, Task


def _review_on(prod_dir: Path) -> None:
    (prod_dir / EP / "studio.yaml").write_text(
        "version: 1\nreview: on\njev: on\nbudget: { unit_usd: 100.0, episode_usd: 1000.0 }\n",
        encoding="utf-8",
    )


def test_draw_branch_books_the_claims_spend(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """The draw branch adds the claims result's cost_usd to the returned cost."""
    _review_on(prod)
    seen: list[str] = []

    def priced_make(p: Path, ep: str, shot: str, **kw: Any) -> dict[str, Any]:
        seen.append(shot)
        return {"drawings": {"ali/p1": ["x"]}, "pairs": {}, "errors": [], "seconds": 1.0, "cost_usd": 0.007}

    monkeypatch.setattr(claims_module, "make_claims", priced_make)
    result = code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage())
    assert result.ok and seen == ["sh01"]
    expected = len(plan_draw(prod, EP, "sh01")) * IMAGE_USD_PER_IMAGE + 0.007
    assert result.cost_usd == pytest.approx(expected)


def test_draw_branch_books_claims_spend_when_drawing_fails(
    prod: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claims ran (and spent) even when every drawing failed: the failure still carries its cost."""
    _review_on(prod)

    def priced_make(p: Path, ep: str, shot: str, **kw: Any) -> dict[str, Any]:
        return {"drawings": {}, "pairs": {}, "errors": [], "seconds": 1.0, "cost_usd": 0.007}

    monkeypatch.setattr(claims_module, "make_claims", priced_make)
    result = code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(refuse_all=True))
    assert not result.ok
    # Only the background generate succeeds (refusals hit edits); its image plus the claims spend.
    assert result.cost_usd == pytest.approx(IMAGE_USD_PER_IMAGE + 0.007)


def test_code_task_path_counts_hooked_spend_once(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """KK-2 #9: through _run_code_task with a hooked fake transport, budget total == cost events."""
    bodies: list[dict[str, Any]] = []
    drawing = base64.b64encode(png(person((200, 150, 100)))).decode()

    def fake_transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        bodies.append(body)
        prompt = body.get("prompt", "")
        refs = body.get("images", [])
        if isinstance(prompt, str) and refs and ("Change ONLY the mouth" in prompt or "Change ONLY the eyes" in prompt):
            raw = refs[0]["image_url"].split(",", 1)[1]
            base = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            marker = ImageDraw.Draw(base)
            if "Change ONLY the mouth" in prompt:
                marker.rectangle((470, 440, 530, 480), fill=(110, 30, 30))
            else:
                marker.rectangle((420, 330, 580, 350), fill=(0, 0, 0))
            buf = io.BytesIO()
            base.save(buf, "PNG")
            return 200, {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
        return 200, {"data": [{"b64_json": drawing}]}

    real_client = MetaClient
    monkeypatch.setattr(
        "karagoz.meta.client.MetaClient",
        lambda *a, **k: real_client(api_key="k", transport=fake_transport, sleep=lambda s: None),
    )
    (prod / EP / "studio.yaml").write_text(
        "version: 1\nhuman: full\nbudget: { unit_usd: 100.0, episode_usd: 1000.0 }\n",
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
                        "id": "sh01/draw",
                        "inputs": [],
                        "files_in": [],
                        "files_out": [f"{EP}/assets/sh01/manifest.json"],
                        "needs_approval": False,
                        "state": "ready",
                        "attempts": 0,
                        "retakes": 0,
                        "notes": [],
                        "input_digest": None,
                        "input_files": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    studio = Studio(
        prod=prod,
        episode=EP,
        carriers={},  # type: ignore[arg-type]
        skills_dir=prod / "skills",
        work_root=prod / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio.load()
    task = studio.board.tasks["sh01/draw"]
    studio._run_code_task(task, [])
    assert task.state is State.DONE
    assert len(bodies) == len(plan_draw(prod, EP, "sh01")) > 0
    expected = sum((b["n"] if isinstance(b.get("n"), int) else 1) * IMAGE_USD_PER_IMAGE for b in bodies)
    assert studio.budget.total() == pytest.approx(expected)
    cost_sum = sum(float(e.get("usd", 0.0)) for e in studio.log.read() if e.get("kind") == "cost")
    assert cost_sum == pytest.approx(expected)
    assert cost_sum == pytest.approx(studio.budget.total())
