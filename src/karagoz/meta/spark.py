"""Muse Spark (Contributor mode only) for vision checks such as reading a numbered contact sheet.

Standard-mode Spark and Spark 1.1 are many times more expensive, so they are refused here.
Contributor mode must never receive secrets or personal data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from karagoz.meta.client import MetaClient, data_url

CHECK_MODEL = "muse-spark-1.2-contributor"  # faster
CAREFUL_MODEL = "muse-spark-1.3-contributor"  # better on subtle errors, ~3x slower


def extract_json(text: str) -> Any:
    """The first JSON object in a model answer (answers may wrap it in prose or code fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in answer")
    return json.loads(text[start : end + 1])


@dataclass
class Spark:
    client: MetaClient
    model: str = CHECK_MODEL

    def __post_init__(self) -> None:
        if not self.model.endswith("-contributor"):
            raise ValueError(f"{self.model}: only Spark Contributor models are allowed")

    def ask(self, prompt: str, images: list[Path | bytes] | None = None, *, tag: str = "") -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": data_url(img)}} for img in images or []]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        result = self.client.post("/chat/completions", body, tag)
        answer: str = result["choices"][0]["message"]["content"]
        return answer

    def ask_json(self, prompt: str, images: list[Path | bytes] | None = None, *, tag: str = "") -> Any:
        return extract_json(self.ask(prompt, images, tag=tag))
