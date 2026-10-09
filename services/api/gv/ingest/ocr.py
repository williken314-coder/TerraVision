"""Read text that a PDF draws as line strokes instead of characters.

AutoCAD's SHX fonts (simplex, romans ...) plot every letter as tiny line
segments, so a PDF can show ``1" = 50'`` under a view title with nothing in
its text layer. Drawing scales are the most important case: without them no
geometry on the sheet can be used.

Only small, targeted regions are read (the line under each view title), never
whole sheets. Readings are parsed tolerantly but accepted only when they give a
standard engineering scale, and they are reported as OCR readings for a person
to confirm. OCR is optional: without ``rapidocr-onnxruntime`` installed,
nothing here runs and those sheets are reported as having no readable scale.
"""
from __future__ import annotations

import re

try:
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    OCR = True
except ImportError:  # pragma: no cover
    OCR = False

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

STANDARD_SCALES = {1, 2, 3, 4, 5, 8, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 2000}
VIEW_WORDS = re.compile(r"\b(PLAN|PROFILE|VIEW|SECTION|SECTIONS|LAYOUT|ELEVATION|SITE|AREA|DETAIL)\b", re.I)
_engine = None


def _ocr():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def read_region(page, rect: list[float], dpi: int = 400) -> list[str]:
    """Text in a page region, read from its rendering. Thin plotted strokes are thickened first."""
    import pymupdf

    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*rect), colorspace=pymupdf.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
    if cv2 is not None:
        g = cv2.erode(g, np.ones((3, 3), np.uint8))
    g = np.pad(g, 40, constant_values=255)
    res, _ = _ocr()(np.repeat(g[:, :, None], 3, axis=2))
    return [r[1] for r in (res or [])]


def parse_scale_reading(parts: list[str]) -> float | None:
    """Feet per inch from an OCR reading of a scale line.

    OCR often drops or mangles the inch and equals signs (``H: 11 20， V: 1 10"``), so: take the horizontal part,
    drop a leading 1 (the inch), and accept the next number only if it is a standard scale.
    """
    text = " ".join(parts).upper().replace("，", ",")
    if re.search(r"\bV\s*[:;]", text):
        text = re.split(r"\bV\s*[:;]", text)[0]
    m = re.search(r"1\s*:\s*(\d{2,5})\b", text)  # metric ratio 1:500
    if m and int(m.group(1)) % 50 == 0:
        return int(m.group(1)) / 12.0
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if nums and nums[0] in (1, 11, 111) and len(nums) > 1:
        nums = nums[1:]
    for n in nums:
        if n in STANDARD_SCALES and n != 1:
            return n
    return None


def view_scales(page, spans, w: float, h: float) -> list[dict]:
    """Scales printed (as strokes) under each view title in the drawing area."""
    if not OCR or not spans:
        return []
    sizes = sorted(s.size for s in spans)
    med = sizes[len(sizes) // 2]
    titles = [s for s in spans if s.size >= med * 1.6 and s.x < w * 0.86 and VIEW_WORDS.search(s.text) and len(s.text.strip()) >= 4
              and (s.bbox[2] - s.bbox[0]) > (s.bbox[3] - s.bbox[1])]
    out = []
    for t in titles[:8]:
        x0, y0, x1, y1 = t.bbox
        region = [x0 - t.size, y1, x1 + t.size, min(h, y1 + t.size * 2.2)]
        try:
            parts = read_region(page, region)
        except Exception:  # a rendering or OCR failure on one view must not stop the sheet
            continue
        ft = parse_scale_reading(parts)
        if ft:
            out.append(dict(title=t.text.strip(), ft_per_in=ft, text=" ".join(parts), bbox=[round(v, 1) for v in region]))
    return out


def pick_sheet_scale(views: list[dict]) -> dict | None:
    """The scale the sheet's plan geometry is drawn at: a plan view's, else the most common one."""
    if not views:
        return None
    plan = [v for v in views if re.search(r"\bPLAN\b", v["title"], re.I) and not re.search(r"PROFILE", v["title"], re.I)]
    if plan:
        return plan[0]
    counts = {}
    for v in views:
        counts[v["ft_per_in"]] = counts.get(v["ft_per_in"], 0) + 1
    best = max(counts, key=counts.get)
    return next(v for v in views if v["ft_per_in"] == best)
