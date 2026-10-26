"""Muse Image: generate and edit drawings.

Measured behaviour this module is built around:
- ``size`` only sets the aspect ratio; returned pixels differ (1024x1536 in -> 1280x1920 out). Callers re-align.
- Output is not deterministic (no seed): generate once, store, reuse.
- ``reasoning_strength="low"`` cuts tokens ~4x and time ~25-30% with similar quality on small edits; the engine
  chooses it per task weight, agents do not.
- Web/image search and shell tools are switched off: drawings must come only from our references.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from karagoz.meta.client import MetaClient, data_url

MODEL = "muse-image-1.0"
Reasoning = Literal["low", "high"]
TOOLS_OFF = {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}


@dataclass
class MuseImage:
    client: MetaClient
    output_format: Literal["png", "jpeg", "webp"] = "png"

    def _body(self, prompt: str, n: int, reasoning: Reasoning, size: str | None) -> dict[str, Any]:
        if not 1 <= n <= 10:
            raise ValueError("n must be 1..10")
        body: dict[str, Any] = {
            "model": MODEL,
            "prompt": prompt,
            "n": n,
            "output_format": self.output_format,
            "reasoning_strength": reasoning,
            "tool_enablement": TOOLS_OFF,
        }
        if size:
            body["size"] = size
        return body

    @staticmethod
    def _images(result: dict[str, Any]) -> list[bytes]:
        return [base64.b64decode(item["b64_json"]) for item in result.get("data", [])]

    def generate(
        self, prompt: str, *, n: int = 1, reasoning: Reasoning = "high", size: str | None = None, tag: str = ""
    ) -> list[bytes]:
        result = self.client.post("/images/generations", self._body(prompt, n, reasoning, size), tag)
        return self._images(result)

    def edit(
        self,
        images: list[Path | bytes],
        prompt: str,
        *,
        n: int = 1,
        reasoning: Reasoning = "high",
        size: str | None = None,
        tag: str = "",
    ) -> list[bytes]:
        if not 1 <= len(images) <= 10:
            raise ValueError("edit takes 1..10 input images")
        body = self._body(prompt, n, reasoning, size)
        body["images"] = [{"image_url": data_url(img)} for img in images]
        return self._images(self.client.post("/images/edits", body, tag))
