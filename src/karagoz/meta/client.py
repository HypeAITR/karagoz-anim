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

# One cost counter (WP-44): every paid Meta call is priced here, once.
# Mirrors studio.draw.USD_PER_IMAGE, meta.sam.USD_PER_CALL, studio.carriers.PRICE_PER_MTOK.
IMAGE_USD_PER_IMAGE = 0.01
SAM_USD_PER_CALL = 0.0025
SPARK_USD_PER_MTOK_IN = 0.10
SPARK_USD_PER_MTOK_OUT = 0.20

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
    model: str | None = None
    units: int = 1
    cost_usd: float = 0.0


def _tokens(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and value > 0:
            return int(value)
    return 0


def _call_cost(endpoint: str, body: dict[str, Any], result: dict[str, Any]) -> tuple[str | None, int, float]:
    """The (model, billable units, USD) of one Meta call. Failed calls cost what the manual count charged."""
    raw_model = body.get("model")
    model = raw_model if isinstance(raw_model, str) else None
    if endpoint in ("/images/generations", "/images/edits"):
        raw_n = body.get("n", 1)
        n = raw_n if isinstance(raw_n, int) and not isinstance(raw_n, bool) and raw_n >= 1 else 1
        return (model or "muse-image-1.0", n, round(n * IMAGE_USD_PER_IMAGE, 6))
    if model == "sam-3.1":
        return (model, 1, SAM_USD_PER_CALL)
    if endpoint == "/chat/completions":
        usage = result.get("usage")
        tokens_in = tokens_out = 0
        if isinstance(usage, dict):
            tokens_in = _tokens(usage, "prompt_tokens", "input_tokens")
            tokens_out = _tokens(usage, "completion_tokens", "output_tokens")
        cost = round((tokens_in * SPARK_USD_PER_MTOK_IN + tokens_out * SPARK_USD_PER_MTOK_OUT) / 1e6, 6)
        return (model, tokens_in + tokens_out, cost)
    return (model, 1, 0.0)


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
        model, units, cost_usd = _call_cost(endpoint, body, result)
        record = CallRecord(
            tag,
            endpoint,
            status,
            round(time.monotonic() - start, 2),
            attempt,
            result.get("usage"),
            model,
            units,
            cost_usd,
        )
        self.records.append(record)
        if self.on_call:
            self.on_call(record)
        if status != 200:
            raise MetaApiError(status, endpoint, json.dumps(result.get("error", result)))
        return result
