"""Stage 2 of plan interpretation: a vision-capable model reads the rendered sheet.

The model only proposes values. Its output is merged with the text-layer
candidates from :mod:`gv.ingest.pdf`; when both agree the confidence goes up,
when they disagree the item is flagged for review. Pipe positions, surfaces
and quantities are still computed by the geometry engine, never by the model.

Requires ``ANTHROPIC_API_KEY``. Set ``GV_VISION_MODEL`` to choose the model.
"""
from __future__ import annotations

import base64
import json
import os

PROMPT = """You are reading one sheet of a civil construction plan set.
Return ONLY a JSON object with this shape:
{"sheet_id": str|null, "category": str, "structures": [{"id": str, "rim": number|null,
  "inverts": [{"dir": "IN"|"OUT"|null, "elev": number}], "type": str|null}],
 "pipes": [{"from": str|null, "to": str|null, "dia_in": number|null, "material": str|null, "slope_pct": number|null}]}
Copy numbers exactly as printed. If a value is not legible or not shown, use null.
Never estimate or calculate a value that is not printed on the sheet."""


def vision_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def read_sheet_image(png: bytes) -> dict:
    import anthropic  # optional dependency

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("GV_VISION_MODEL", "claude-sonnet-5-5"),
        max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start >= 0 else {}


def merge(text_candidates: list[dict], vision: dict) -> list[dict]:
    """Raise confidence where text layer and vision agree, flag disagreements."""
    by_id = {v.get("id"): v for v in vision.get("structures", []) if v.get("id")}
    for c in text_candidates:
        if c["kind"] != "structure":
            continue
        v = by_id.get(c["label"])
        if not v:
            continue
        if v.get("rim") is not None and c["values"].get("rim") is not None:
            agree = abs(v["rim"] - c["values"]["rim"]) < 0.005
            c["values"]["rim_confidence"] = min(0.98, c["values"].get("rim_confidence", 0.5) + 0.2) if agree else 0.4
            c["values"]["rim_vision"] = v["rim"]
        c["values"]["vision_type"] = v.get("type")
    return text_candidates
