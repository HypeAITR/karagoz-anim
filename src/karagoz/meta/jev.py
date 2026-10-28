"""Jev (TypeSafe System One): typed yes/no questions about images with calibrated probabilities."""

from __future__ import annotations

import base64
import copy
import io
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from karagoz import config
from karagoz.meta.client import Transport, urllib_transport

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
PRICE_IN_PER_MTOK = 0.042
RETRYABLE = {0, 429, 500, 502, 503, 504}


class JevError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Jev API returned HTTP {status}: {detail[:500]}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class JevAnswer:
    answers: dict[str, dict[str, Any]]
    input_tokens: int
    cost_usd: float
    seconds: float


def fit_png(path: Path) -> bytes:
    """RGB PNG bytes for Jev: longest side at most 1568, every side at least 280."""
    with Image.open(path) as opened:
        img = opened.convert("RGBA") if "A" in opened.getbands() else opened.convert("RGB")
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, img).convert("RGB")
        else:
            rgb = img
    w, h = rgb.size
    longest = max(w, h)
    if longest > 1568:
        scale = 1568 / longest
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        w, h = rgb.size
    padded_w, padded_h = max(w, 280), max(h, 280)
    if (padded_w, padded_h) != (w, h):
        canvas = Image.new("RGB", (padded_w, padded_h), (255, 255, 255))
        canvas.paste(rgb, (0, 0))
        rgb = canvas
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


def _uses_placeholder(state: dict[str, Any], questions: dict[str, Any], placeholder: str) -> bool:
    for value in state.values():
        if isinstance(value, str) and placeholder in value:
            return True
    for question in questions.values():
        if not isinstance(question, dict):
            continue
        instructions = question.get("instructions")
        if isinstance(instructions, str) and placeholder in instructions:
            return True
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            for criterion in criteria.values():
                if isinstance(criterion, str) and placeholder in criterion:
                    return True
    return False


def _replace_placeholder(state: dict[str, Any], questions: dict[str, Any], placeholder: str, ref: str) -> None:
    for key, value in list(state.items()):
        if isinstance(value, str) and placeholder in value:
            state[key] = value.replace(placeholder, ref)
    for question in questions.values():
        if not isinstance(question, dict):
            continue
        instructions = question.get("instructions")
        if isinstance(instructions, str) and placeholder in instructions:
            question["instructions"] = instructions.replace(placeholder, ref)
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            for criterion_key, criterion_value in list(criteria.items()):
                if isinstance(criterion_value, str) and placeholder in criterion_value:
                    criteria[criterion_key] = criterion_value.replace(placeholder, ref)


@dataclass
class Jev:
    api_key: str | None = None
    timeout: float = 120.0
    max_attempts: int = 3
    transport: Transport = urllib_transport
    sleep: Callable[[float], None] = time.sleep

    def _key(self) -> str:
        return self.api_key or config.secret("TYPESAFE_API_KEY")

    def ask(self, state: dict[str, Any], questions: dict[str, Any], images: dict[str, Path]) -> JevAnswer:
        new_state = copy.deepcopy(state)
        new_questions = copy.deepcopy(questions)
        assets: dict[str, dict[str, str]] = {}
        for name, path in images.items():
            placeholder = "{" + name + "}"
            if not _uses_placeholder(new_state, new_questions, placeholder):
                continue
            ref = f"img-{uuid.uuid4().hex}"
            _replace_placeholder(new_state, new_questions, placeholder, ref)
            data = base64.b64encode(fit_png(path)).decode()
            assets[ref] = {"type": "image", "format": "png", "data_base64": data}
        body: dict[str, Any] = {"model": MODEL, "state": new_state, "questions": new_questions, "assets": assets}
        headers = {
            "Authorization": "Bearer " + self._key(),
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        }
        start = time.monotonic()
        status, result = 0, dict[str, Any]()
        for attempt in range(1, self.max_attempts + 1):
            try:
                status, result = self.transport(URL, body, headers, self.timeout)
            except (OSError, TimeoutError) as exc:
                status, result = 0, {"error": {"message": f"transport error: {exc}"}}
            if status not in RETRYABLE or attempt == self.max_attempts:
                break
            self.sleep(min(30, 2**attempt))
        seconds = round(time.monotonic() - start, 2)
        if status != 200:
            raise JevError(status, json.dumps(result))
        answers = result.get("answers")
        if not isinstance(answers, dict):
            answers = {}
        usage = result.get("usage")
        input_tokens = 0
        if isinstance(usage, dict):
            raw_tokens = usage.get("input_tokens", 0)
            if isinstance(raw_tokens, bool):
                input_tokens = 0
            elif isinstance(raw_tokens, (int, float)):
                input_tokens = int(raw_tokens)
        cost_usd = input_tokens * PRICE_IN_PER_MTOK / 1e6
        typed_answers: dict[str, dict[str, Any]] = {
            str(qid): dict(answer) if isinstance(answer, dict) else {"value": answer} for qid, answer in answers.items()
        }
        return JevAnswer(answers=typed_answers, input_tokens=input_tokens, cost_usd=cost_usd, seconds=seconds)
