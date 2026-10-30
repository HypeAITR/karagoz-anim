"""SAM 3.1: text-prompted boxes and masks.

The API answers with special-token text, one record per match:
``<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;x=0;y=0;c=..;data=H,W,<enc>payload|>``.
Masks are box-local rasters (top-left at x1, y1). The default ``lossless`` encoding carries soft coverage 0..255,
which we keep as an alpha edge; ``one_bit`` masks become 0/255.
Decoding uses Meta's ``meta-sam-parser`` (SAM License); the soft decoder is a private function, so the version is
pinned and covered by a test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from meta_sam_parser import decode_mask_to_raster
from meta_sam_parser._mask_codec import _decode_lossless_raster
from meta_sam_parser._types import SegmentationMask

from karagoz.meta.client import MetaClient, data_url

MODEL = "sam-3.1"
# estimate from D24 (36 calls ≈ 0.09 $), not a price list.
USD_PER_CALL = 0.0025
RECORD = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
    r"<\|mask;x=0;y=0(?:;c=[\d.]+)?;data=(\d+),(\d+),([~!])([^|]*)\|>"
)


@dataclass(frozen=True)
class Segment:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in source pixels
    frame: tuple[int, int]  # source width, height
    confidence: float | None
    local_mask: np.ndarray  # uint8 0..255, box-local

    def full_mask(self) -> np.ndarray:
        """The mask placed into a full-frame uint8 array (height x width)."""
        w, h = self.frame
        x1, y1 = self.box[0], self.box[1]
        out = np.zeros((h, w), np.uint8)
        mh, mw = self.local_mask.shape
        out[y1 : min(h, y1 + mh), x1 : min(w, x1 + mw)] = self.local_mask[: h - y1, : w - x1]
        return out


def decode(encoding: str, payload: str, width: int, height: int) -> np.ndarray:
    if encoding == "~":
        soft = bytes(_decode_lossless_raster(payload, width, height))
        return np.frombuffer(soft, np.uint8).reshape(height, width).copy()
    raster = decode_mask_to_raster(SegmentationMask("one_bit", payload, width, height))
    return (np.frombuffer(raster, np.uint8).reshape(height, width) * 255).astype(np.uint8)


def parse(text: str) -> list[Segment]:
    segments = []
    for m in RECORD.finditer(text):
        x1, y1, x2, y2, w, h = (int(v) for v in m.groups()[:6])
        conf = float(m.group(7)) if m.group(7) else None
        mh, mw, enc = int(m.group(8)), int(m.group(9)), m.group(10)
        mask = decode(enc, enc + m.group(11), mw, mh)
        segments.append(Segment((x1, y1, x2, y2), (w, h), conf, mask))
    return segments


def output_text(result: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in result.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


@dataclass
class Sam:
    client: MetaClient

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        """All matches of a short noun phrase ("mouth", "arm", "sofa")."""
        body = {
            "model": MODEL,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": concept},
                        {"type": "input_image", "image_url": data_url(image)},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true"},
        }
        return parse(output_text(self.client.post("/responses", body, tag)))
