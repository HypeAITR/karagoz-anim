"""Thin JSON client for https://api.meta.ai/v1 with retries and a call log that never contains the key."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from karagoz import config

BASE_URL = "https://api.meta.ai/v1"
RETRYABLE = {0, 429, 500, 502, 503, 504}  # 0 = transport error, no response

Transport = Callable[[str, dict[str, Any], dict[str, str], float], tuple[int, dict[str, Any]]]


class MetaApiError(RuntimeError):
    def __init__(self, status: int, endpoint: str, detail: str) -> None:
        super().__init__(f"Meta API {endpoint} returned HTTP {status}: {detail[:500]}")
        self.status = status
        self.endpoint = endpoint


@dataclass(frozen=True)
class CallRecord:
    """One API call for the event log / budget keeper. Holds no request body and no key."""

    tag: str
    endpoint: str
    status: int
    seconds: float
    attempts: int
    usage: dict[str, Any] | None = None


def urllib_transport(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        text = err.read().decode(errors="replace")
        try:
            return err.code, json.loads(text)
        except json.JSONDecodeError:
            return err.code, {"error": {"message": text}}


def data_url(image: Path | bytes, mime: str = "image/png") -> str:
    """Encode an image file or bytes as a base64 data URL (the API takes images inline)."""
    if isinstance(image, Path):
        mime = mimetypes.guess_type(image.name)[0] or mime
        image = image.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(image).decode()


@dataclass
class MetaClient:
    api_key: str | None = None
    base_url: str = BASE_URL
    timeout: float = 600.0
    max_attempts: int = 4
    transport: Transport = urllib_transport
    on_call: Callable[[CallRecord], None] | None = None
    sleep: Callable[[float], None] = time.sleep
    records: list[CallRecord] = field(default_factory=list)

    def _key(self) -> str:
        return self.api_key or config.secret("META_API_KEY")

    def post(self, endpoint: str, body: dict[str, Any], tag: str = "") -> dict[str, Any]:
        headers = {"Authorization": "Bearer " + self._key(), "Content-Type": "application/json"}
        start = time.monotonic()
        status, result = 0, dict[str, Any]()
        for attempt in range(1, self.max_attempts + 1):
            try:
                status, result = self.transport(self.base_url + endpoint, body, headers, self.timeout)
            except (OSError, TimeoutError) as exc:  # timeouts, SSL EOF, resets (URLError is an OSError)
                status, result = 0, {"error": {"message": f"transport error: {exc}"}}
            if status not in RETRYABLE or attempt == self.max_attempts:
                break
            self.sleep(min(30.0, 2.0**attempt))
        record = CallRecord(tag, endpoint, status, round(time.monotonic() - start, 2), attempt, result.get("usage"))
        self.records.append(record)
        if self.on_call:
            self.on_call(record)
        if status != 200:
            raise MetaApiError(status, endpoint, json.dumps(result.get("error", result)))
        return result
