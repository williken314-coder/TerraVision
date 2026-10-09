"""Read a civil plan set PDF with PyMuPDF.

Stage 1 of plan interpretation, fully deterministic:

* classify each sheet from its title block and find the drawing scale
* pull vector geometry: line segments (with width / dash style), small closed
  symbols (structures), and polylines (contours, alignments)
* pull labelled values from the text layer: structure IDs, rims, inverts,
  pipe callouts, spot elevations, contour labels, stations
* group labels with the nearest structure and the nearest symbol

Everything found here becomes an ``ai`` candidate with a confidence and an
exact page location. Nothing is promoted to "verified" without a person.
Scanned sheets (no text layer) are detected and handed to the raster/vision
path in :mod:`gv.ingest.raster` and :mod:`gv.ingest.vision`.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    try:
        import fitz  # older PyMuPDF
    except ImportError:  # lets the pure-text helpers be tested without PyMuPDF
        fitz = None

SHEET_NO = re.compile(r"\b([CEALSUG])-?\s?(\d{1,2}\.\d{1,2})\b")
SCALE_IMP = re.compile(r"1\s*(?:\"|”|IN\.?|INCH)\s*=\s*(\d+(?:\.\d+)?)\s*(?:'|’|FT\.?|FEET)", re.I)
SCALE_RATIO = re.compile(r"\bSCALE\s*:?\s*1\s*:\s*(\d{2,5})\b", re.I)
STRUCT_ID = re.compile(r"\b(CB|SMH|STMH|MH|HW|FES|FH|INL|DI|WV|TX)\s?-?\s?(\d{1,3}[A-Z]?)\b")
RIM = re.compile(r"\b(?:RIM|TOP|TG|GRATE|T\.?G\.?)\s*(?:EL\.?|ELEV\.?)?\s*[=:]?\s*(\d{3,4}\.\d{1,3})", re.I)
INV = re.compile(r"\bI(?:NV|E)\.?\s*(IN|OUT)?\s*(?:\(?\b([NSEW]{1,2})\b\)?)?\s*(?:EL\.?)?\s*[=:]?\s*(\d{3,4}\.\d{1,3})", re.I)
PIPE = re.compile(r"(?:(\d+(?:\.\d+)?)\s*(?:LF|L\.F\.)\s*(?:OF\s*)?)?(\d{1,2})\s*(?:\"|”|IN\.?|-INCH)\s*(RCP|PVC|DIP|DI|HDPE|CMP|PP|CPEP)\b[^@\n]{0,40}?@\s*(-?\d+\.\d+)\s*%", re.I)
SPOT = re.compile(r"\b(FG|TC|BC|FFE|FF|EG|TW|BW|HP|LP|FL|PAD)\s*[=:]?\s*(\d{3,4}\.\d{1,2})\b|\b(\d{3,4}\.\d{1,2})\s*(FG|TC|BC|FFE|FF|EG|TW|BW|FL|PAD)\b", re.I)
CONTOUR_LABEL = re.compile(r"^\s*(\d{3,4})(?:\.0)?\s*$")
STATION = re.compile(r"\bSTA\.?\s*(\d{1,4})\+(\d{2}(?:\.\d+)?)", re.I)

CATEGORIES = [
    ("existing", ["EXISTING CONDITIONS", "TOPOGRAPHIC", "SURVEY", "EXISTING SITE"]),
    ("demolition", ["DEMOLITION", "EROSION", "SEDIMENT", "SWPPP"]),
    ("storm_profile", ["STORM PROFILE", "STORM SEWER PROFILE", "DRAINAGE PROFILE", "STORM DRAIN PROFILE"]),
    ("sanitary_profile", ["SANITARY PROFILE", "SEWER PROFILE"]),
    ("cross_section", ["CROSS SECTION", "CROSS-SECTION"]),
    ("grading", ["GRADING"]),
    ("storm", ["STORM", "DRAINAGE"]),
    ("sanitary", ["SANITARY", "SEWER"]),
    ("water", ["WATER"]),
    ("utility", ["UTILITY", "UTILITIES"]),
    ("roadway", ["ROADWAY", "PLAN & PROFILE", "PLAN AND PROFILE", "PAVING", "STREET"]),
    ("site", ["SITE PLAN", "SITE DEVELOPMENT", "LAYOUT", "DIMENSION"]),
    ("details", ["DETAIL"]),
    ("electrical", ["ELECTRICAL", "LIGHTING"]),
    ("landscape", ["LANDSCAPE", "PLANTING"]),
    ("cover", ["COVER", "INDEX", "TITLE SHEET"]),
]
PLAN_CATEGORIES = {"grading", "storm", "sanitary", "water", "utility", "roadway", "site", "existing"}
PROFILE_CATEGORIES = {"storm_profile", "sanitary_profile"}


@dataclass
class TextSpan:
    text: str
    x: float
    y: float
    size: float
    bbox: tuple


@dataclass
class Seg:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    dashed: bool

    @property
    def length(self):
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class Symbol:
    kind: str  # rect | circle
    x: float
    y: float
    size: float


@dataclass
class Polyline:
    pts: list
    width: float
    dashed: bool

    @property
    def length(self):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.pts, self.pts[1:]))


@dataclass
class Candidate:
    kind: str  # structure | pipe_callout | spot | contour | station
    label: str
    values: dict
    page: int
    x: float
    y: float
    bbox: list
    text: str
    confidence: float


@dataclass
class SheetReport:
    page: int
    sheet_id: str | None
    title: str
    category: str
    scale_ft_per_in: float | None
    scale_text: str | None
    has_text_layer: bool
    width_pt: float
    height_pt: float
    line_segments: int
    text_spans: int
    candidates: list[Candidate] = field(default_factory=list)
    pipe_segments: list[list[float]] = field(default_factory=list)  # thick plan lines [x0,y0,x1,y1]
    symbols: list[list[float]] = field(default_factory=list)  # [x, y, size, kind(0 rect/1 circle)]
    notes: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SheetReport":
        d = dict(d)
        d["candidates"] = [Candidate(**c) for c in d.get("candidates", [])]
        return cls(**d)


# ----- low-level extraction ----------------------------------------------

def _spans(page) -> list[TextSpan]:
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            size = max(s["size"] for s in line["spans"])
            out.append(TextSpan(text, (x0 + x1) / 2, (y0 + y1) / 2, size, (x0, y0, x1, y1)))
    return out


def _is_dashed(d: dict) -> bool:
    dashes = (d.get("dashes") or "").strip()
    return bool(dashes) and not dashes.startswith("[]") and dashes != "[] 0"


def _geometry(page):
    segs: list[Seg] = []
    symbols: list[Symbol] = []
    polys: list[Polyline] = []
    for d in page.get_drawings():
        width = d.get("width") or 0.0
        dashed = _is_dashed(d)
        r = d.get("rect")
        items = d["items"]
        kinds = {it[0] for it in items}
        if r is not None and 2.5 <= max(r.width, r.height) <= 22 and abs(r.width - r.height) <= max(r.width, r.height) * 0.6:
            if "c" in kinds and "l" not in kinds:
                symbols.append(Symbol("circle", (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, max(r.width, r.height)))
                continue
            if kinds <= {"re", "qu"} or (kinds == {"l"} and len(items) == 4 and d.get("closePath", True)):
                symbols.append(Symbol("rect", (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, max(r.width, r.height)))
                continue
        pts = []
        for it in items:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                segs.append(Seg(p1.x, p1.y, p2.x, p2.y, width, dashed))
                if not pts or (abs(pts[-1][0] - p1.x) > 0.01 or abs(pts[-1][1] - p1.y) > 0.01):
                    if len(pts) >= 2:
                        polys.append(Polyline(pts, width, dashed))
                    pts = [(p1.x, p1.y)]
                pts.append((p2.x, p2.y))
            elif it[0] == "c":
                p1, p4 = it[1], it[4]
                segs.append(Seg(p1.x, p1.y, p4.x, p4.y, width, dashed))
                if not pts:
                    pts = [(p1.x, p1.y)]
                pts.append((p4.x, p4.y))
        if len(pts) >= 2:
            polys.append(Polyline(pts, width, dashed))
    return segs, symbols, polys


def classify(text: str) -> str:
    up = text.upper()
    for cat, keys in CATEGORIES:
        if any(k in up for k in keys):
            return cat
    return "unknown"


def parse_scale(text: str) -> tuple[float | None, str | None]:
    """Return feet per inch of paper and the matched text."""
    m = SCALE_IMP.search(text)
    if m:
        return float(m.group(1)), m.group(0)
    m = SCALE_RATIO.search(text)
    if m:
        return float(m.group(1)) / 12.0, m.group(0)  # 1:N means N inches per inch -> N/12 ft per inch
    return None, None


def _dist_point_poly(x, y, pts):
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        vx, vy = bx - ax, by - ay
        l2 = vx * vx + vy * vy or 1e-9
        u = max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / l2))
        best = min(best, math.hypot(x - ax - vx * u, y - ay - vy * u))
    return best


# ----- page reading ---------------------------------------------------------

def read_page(page, index: int) -> SheetReport:
    w, h = page.rect.width, page.rect.height
    segs, symbols, polys = _geometry(page)
    return build_report(index, w, h, _spans(page), segs, symbols, polys)


def build_report(index: int, w: float, h: float, spans: list[TextSpan], segs: list[Seg], symbols: list[Symbol], polys: list[Polyline]) -> SheetReport:
    """Everything after raw extraction. Separate from PyMuPDF so it can be tested on recorded layouts."""
    tb = [s for s in spans if s.x > w * 0.6 and s.y > h * 0.7] or spans
    tb_text = "\n".join(s.text for s in tb)
    m = SHEET_NO.search(tb_text) or SHEET_NO.search("\n".join(s.text for s in spans))
    sheet_id = f"{m.group(1)}-{m.group(2)}" if m else None
    title_span = max((s for s in tb if not SHEET_NO.fullmatch(s.text.strip()) and not SCALE_IMP.search(s.text)),
                     key=lambda s: (s.size, len(s.text)), default=None)
    title = title_span.text if title_span else ""
    category = classify(title + "\n" + tb_text)
    scale, scale_text = parse_scale("\n".join(s.text for s in spans))

    report = SheetReport(page=index, sheet_id=sheet_id, title=title, category=category, scale_ft_per_in=scale,
                         scale_text=scale_text, has_text_layer=bool(spans), width_pt=w, height_pt=h,
                         line_segments=len(segs), text_spans=len(spans))
    if not spans:
        report.notes.append("No text layer: scanned sheet. Needs OCR or the vision reader.")
        return report

    size_ref = max(12.0, min(w, h) * 0.02)
    report.candidates = extract_candidates(spans, symbols, polys, index, size_ref, category)
    # thick, solid lines on plan sheets are pipe candidates
    if category in PLAN_CATEGORIES:
        widths = sorted(s.width for s in segs if s.length > 6)
        if widths:
            thick = max(1.2, widths[int(len(widths) * 0.9)] * 0.95)
            report.pipe_segments = [[s.x0, s.y0, s.x1, s.y1] for s in segs if s.width >= thick and not s.dashed and s.length > 8]
    report.symbols = [[s.x, s.y, s.size, 1 if s.kind == "circle" else 0] for s in symbols]
    if category in PLAN_CATEGORIES and not scale:
        report.notes.append("Plan sheet without a readable scale. Set the scale before using its geometry.")
    return report


def extract_candidates(spans, symbols, polys, page, size_ref, category) -> list[Candidate]:
    def bb(s):
        return [round(v, 2) for v in s.bbox]

    structs: list[Candidate] = []
    for s in spans:
        for m in STRUCT_ID.finditer(s.text.upper()):
            sid = f"{m.group(1)}-{m.group(2)}"
            if any(c.label == sid and math.hypot(c.x - s.x, c.y - s.y) < size_ref for c in structs):
                continue
            structs.append(Candidate("structure", sid, {"inverts": [], "label_xy": [s.x, s.y]}, page, s.x, s.y, bb(s), s.text, 0.0))

    def nearest(x, y):
        best, bd = None, float("inf")
        for c in structs:
            d = math.hypot(c.x - x, c.y - y)
            if d < bd:
                best, bd = c, d
        return best, bd

    def conf_from(dist):  # vector text close to its label is trustworthy; far is a guess
        return round(max(0.35, 0.95 - 0.55 * min(1.0, dist / (size_ref * 1.5))), 2)

    for s in spans:
        for m in RIM.finditer(s.text):
            c, d = nearest(s.x, s.y)
            if c and d < size_ref * 2:
                c.values["rim"] = float(m.group(1))
                c.values["rim_confidence"] = conf_from(d)
                c.values["rim_text"] = s.text
                c.values["rim_bbox"] = bb(s)
        for m in INV.finditer(s.text):
            c, d = nearest(s.x, s.y)
            if c and d < size_ref * 2:
                c.values["inverts"].append({"dir": (m.group(1) or "").upper() or None, "side": (m.group(2) or "").upper() or None,
                                            "elev": float(m.group(3)), "confidence": conf_from(d), "text": s.text, "bbox": bb(s)})

    # snap structure labels to symbols (the structure itself): one symbol per label, closest pairs first,
    # preferring the symbol shape the structure type is normally drawn with
    round_ids = ("SMH", "MH", "STMH", "FH", "WV")
    pairs = []
    for ci, c in enumerate(structs):
        want = "circle" if c.label.split("-")[0] in round_ids else "rect"
        for si, sym in enumerate(symbols):
            d = math.hypot(sym.x - c.x, sym.y - c.y)
            if d < size_ref * 2.5:
                pairs.append((d * (1.0 if sym.kind == want else 1.8), d, ci, si))
    used_c, used_s = set(), set()
    for _, d, ci, si in sorted(pairs):
        if ci in used_c or si in used_s:
            continue
        used_c.add(ci)
        used_s.add(si)
        sym, c = symbols[si], structs[ci]
        c.values["symbol_xy"] = [sym.x, sym.y]
        c.values["position_confidence"] = round(max(0.5, 0.95 - 0.4 * d / (size_ref * 2.5)), 2)
    for c in structs:
        found = [v for k, v in c.values.items() if k.endswith("confidence")] + [i["confidence"] for i in c.values["inverts"]]
        c.confidence = round(min(found), 2) if found else 0.3

    out: list[Candidate] = list(structs)
    for s in spans:
        for m in PIPE.finditer(s.text):
            out.append(Candidate("pipe_callout", m.group(0).strip(), {
                "length_lf": float(m.group(1)) if m.group(1) else None, "dia_in": float(m.group(2)),
                "material": m.group(3).upper(), "slope_pct": float(m.group(4))}, page, s.x, s.y, bb(s), s.text, 0.85))
        for m in SPOT.finditer(s.text):
            code = (m.group(1) or m.group(4)).upper()
            elev = float(m.group(2) or m.group(3))
            out.append(Candidate("spot", code, {"elev": elev}, page, s.x, s.y, bb(s), s.text, 0.9))
        for m in STATION.finditer(s.text):
            out.append(Candidate("station", m.group(0), {"station": int(m.group(1)) * 100 + float(m.group(2))}, page, s.x, s.y, bb(s), s.text, 0.9))

    # contour labels: a bare whole-number elevation sitting on a polyline
    if category in ("grading", "existing", "site", "roadway"):
        long_polys = [p for p in polys if p.length > size_ref * 3 and len(p.pts) >= 3]
        for s in spans:
            m = CONTOUR_LABEL.match(s.text)
            if not m or not long_polys:
                continue
            best, bd = None, float("inf")
            for p in long_polys:
                d = _dist_point_poly(s.x, s.y, p.pts)
                if d < bd:
                    best, bd = p, d
            if best and bd < s.size * 1.2:
                out.append(Candidate("contour", m.group(1), {
                    "elev": float(m.group(1)), "dashed": best.dashed,
                    "pts": [[round(x, 2), round(y, 2)] for x, y in best.pts]}, page, s.x, s.y, bb(s), s.text,
                    0.8 if best.dashed or category == "grading" else 0.7))
    return out


def read_plan_set(data: bytes) -> list[SheetReport]:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return [read_page(p, i) for i, p in enumerate(doc)]
    finally:
        doc.close()


def page_count(data: bytes) -> int:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


def render_page_png(data: bytes, page_index: int, dpi: int = 110, clip: list[float] | None = None) -> bytes:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc[page_index]
        kw = {"dpi": dpi}
        if clip:
            kw["clip"] = fitz.Rect(*clip)
        return page.get_pixmap(**kw).tobytes("png")
    finally:
        doc.close()
