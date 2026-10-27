import base64
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster

from karagoz.meta import MetaApiError, MetaClient, MuseImage, Sam, Spark
from karagoz.meta.sam import parse
from karagoz.meta.spark import extract_json

KEY = "test-key-not-real"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeTransport:
    def __init__(self, *responses: tuple[int, dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, body, headers))
        return self.responses.pop(0)


def client(*responses: tuple[int, dict[str, Any]]) -> tuple[MetaClient, FakeTransport]:
    t = FakeTransport(*responses)
    return MetaClient(api_key=KEY, transport=t, sleep=lambda s: None), t


def test_retries_then_records_without_key() -> None:
    c, t = client(
        (503, {}), (429, {}), (200, {"data": [{"b64_json": base64.b64encode(PNG).decode()}], "usage": {"x": 1}})
    )
    out = MuseImage(c).generate("apple", tag="t1")
    assert out == [PNG]
    assert len(t.calls) == 3
    rec = c.records[-1]
    assert (rec.status, rec.attempts, rec.tag, rec.usage) == (200, 3, "t1", {"x": 1})
    assert KEY not in repr(c.records)
    assert t.calls[0][2]["Authorization"] == "Bearer " + KEY


def test_error_raises_with_status() -> None:
    c, _ = client((400, {"error": {"message": "content_policy_violation"}}))
    with pytest.raises(MetaApiError) as err:
        MuseImage(c).generate("x")
    assert err.value.status == 400
    assert "content_policy_violation" in str(err.value)


def test_edit_body_has_tools_off_and_reasoning() -> None:
    c, t = client((200, {"data": []}))
    MuseImage(c).edit([PNG], "open the mouth", reasoning="low", n=2)
    body = t.calls[0][1]
    assert body["reasoning_strength"] == "low"
    assert body["tool_enablement"] == {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}
    assert body["images"][0]["image_url"].startswith("data:image/png;base64,")
    assert body["n"] == 2


def test_sam_parse_places_box_local_mask() -> None:
    local = np.zeros((3, 4), np.uint8)
    local[1, 1:3] = 1
    payload = _encode_raster(bytes(local.ravel()), 4, 3)
    text = f"<0f>0<|box;x1=5;y1=2;x2=8;y2=4;w=10;h=6;c=0.9|><|mask;x=0;y=0;c=0.9;data=3,4,{payload}|>"
    (seg,) = parse(text)
    assert seg.box == (5, 2, 8, 4) and seg.confidence == 0.9
    full = seg.full_mask()
    assert full.shape == (6, 10)
    assert full[3, 6] == 255 and full[3, 7] == 255 and full.sum() == 2 * 255


def test_sam_request_shape() -> None:
    c, t = client((200, {"output": [{"type": "message", "content": [{"text": ""}]}]}))
    assert Sam(c).segment(PNG, "mouth") == []
    body = t.calls[0][1]
    assert body["model"] == "sam-3.1"
    assert body["input"][0]["content"][0] == {"type": "input_text", "text": "mouth"}


def test_spark_refuses_standard_models() -> None:
    c, _ = client()
    with pytest.raises(ValueError):
        Spark(c, model="muse-spark-1.3")
    with pytest.raises(ValueError):
        Spark(c, model="muse-spark-1.1")


def test_extract_json() -> None:
    assert extract_json('Sure:\n```json\n{"tiles": [1]}\n```') == {"tiles": [1]}


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("KARAGOZ_LIVE") != "1", reason="paid API; set KARAGOZ_LIVE=1")
def test_live_generate_and_segment(tmp_path: Path) -> None:
    c = MetaClient()
    (img,) = MuseImage(c).generate(
        "A flat-colour cartoon red apple with a thick black outline on a plain white background.",
        reasoning="low",
        size="1024x1024",
    )
    segs = Sam(c).segment(img, "apple")
    assert segs and segs[0].full_mask().max() > 0


def test_transport_errors_are_retried() -> None:
    """Regression (27 Oct): timeouts and SSL EOF raised out of the client instead of being retried like 5xx."""
    from karagoz.meta.client import MetaClient

    calls = {"n": 0}

    def flaky(url: str, body: dict, headers: dict, timeout: float) -> tuple[int, dict]:  # type: ignore[type-arg]
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("The read operation timed out")
        return 200, {"ok": True}

    c = MetaClient(api_key="k", transport=flaky, sleep=lambda s: None)
    assert c.post("/x", {}) == {"ok": True}
    assert calls["n"] == 3 and c.records[-1].attempts == 3
