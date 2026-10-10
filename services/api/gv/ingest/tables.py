"""Structure tables: one row per structure with its type, station, rim and pipe inverts.

    NAME   TYPE                    STATION           RIM            PIPES IN
    CB #6  CB TYPE 2 48 IN. DIAM.  25+41.5A, 6.5 L   ELEV. 876.20   I.E. 869.76, W
                                                                    I.E. 869.76, N
                                                                    I.E. 869.76, SE

Plans often keep rims and inverts here rather than beside each structure, and
a cell can span several lines. Rows are anchored on the structure IDs in the
name column; every other span in the table goes to the nearest row. A table
row says nothing about where the structure is on the plan, so it gives values,
never a location (the station and offset are kept for when the alignment is
known).
"""
from __future__ import annotations

import re

HEAD_RIM = re.compile(r"^\s*(RIM|TOP|GRATE|T\.?G\.?)(\s*(ELEV\.?|EL\.?))?\s*$", re.I)
HEAD_NAME = re.compile(r"^\s*(NAME|STRUCTURE|STRUCT\.?|ID|NO\.?|STR\.?\s*NO\.?|LABEL)\s*$", re.I)
HEAD_INV = re.compile(r"PIPES?|INV|I\.\s?E\.|INVERT", re.I)
HEAD_STA = re.compile(r"^\s*(STATION|STA\.?)(\s*/\s*OFFSET)?\s*$", re.I)
HEAD_TYPE = re.compile(r"^\s*(TYPE|DESCRIPTION|DESC\.?)\s*$", re.I)
NUM = re.compile(r"(\d{3,4}\.\d{1,3})")
INVERT = re.compile(r"(?:I\.?\s?E\.?|INV\.?)\s*(IN|OUT)?\s*[=:]?\s*(\d{3,4}\.\d{1,3})'?\s*,?\s*\(?\s*([NSEW]{1,2})?\b", re.I)
STATION = re.compile(r"(\d{1,4})\+(\d{2}(?:\.\d+)?)\s*([A-Z][A-Z0-9]{0,3})?\s*,?\s*(\d+(?:\.\d+)?)'?\s*(LT|RT|L|R)\b", re.I)


def _row(a, b, tol):
    return abs(a.y - b.y) <= tol


def structure_tables(spans) -> tuple[list[dict], set[int]]:
    """Rows of every structure table on the sheet, and the ids of the spans the tables use."""
    from .pdf import STRUCT_ID

    rows_out, used = [], set()
    for rim_h in [s for s in spans if HEAD_RIM.match(s.text)]:
        lh = rim_h.size * 1.6
        header = [s for s in spans if _row(s, rim_h, rim_h.size * 0.6)]
        name_h = next((s for s in sorted(header, key=lambda s: s.x) if HEAD_NAME.match(s.text) and s.x < rim_h.x), None)
        inv_h = [s for s in header if HEAD_INV.search(s.text) and s.x > rim_h.x - 5]
        if not inv_h:
            continue  # a rim column without inverts is something else (a note, a legend)
        left = (name_h.bbox[0] - 10) if name_h else min(s.bbox[0] for s in header) - 10
        right = max(s.bbox[2] for s in header) + max(60.0, rim_h.size * 8)
        anchors = sorted((s for s in spans if s.y > rim_h.y and left <= s.bbox[0] <= (name_h.bbox[2] + 20 if name_h else rim_h.bbox[0])
                          and STRUCT_ID.fullmatch(s.text.strip().upper().replace("  ", " "))), key=lambda s: s.y)
        # the table ends at the first gap of several lines after a row
        kept = []
        for a in anchors:
            if kept and a.y - kept[-1].y > lh * 6:
                break
            if not kept and a.y - rim_h.y > lh * 6:
                break
            kept.append(a)
        if not kept:
            continue
        bottom = kept[-1].y + lh * 3
        cells = [s for s in spans if left <= s.x <= right and rim_h.y + rim_h.size * 0.6 < s.y <= bottom and s not in kept]
        sta_h = next((s for s in header if HEAD_STA.match(s.text)), None)
        type_h = next((s for s in header if HEAD_TYPE.match(s.text)), None)
        ids = {id(s) for s in header} | {id(s) for s in kept}
        for i, a in enumerate(kept):
            lo = (kept[i - 1].y + a.y) / 2 if i else rim_h.y
            hi = (a.y + kept[i + 1].y) / 2 if i + 1 < len(kept) else bottom
            mine = [s for s in cells if lo < s.y <= hi]
            ids |= {id(s) for s in mine}
            m = STRUCT_ID.search(a.text.upper())
            row = dict(id=f"{m.group(1)}-{m.group(2)}", text=a.text.strip(), bbox=[round(v, 2) for v in a.bbox], inverts=[])
            rim = min((s for s in mine if NUM.search(s.text) and not INVERT.search(s.text)), key=lambda s: abs(s.x - rim_h.x), default=None)
            if rim is not None and abs(rim.x - rim_h.x) < rim_h.size * 6:
                row.update(rim=float(NUM.search(rim.text).group(1)), rim_text=rim.text.strip(), rim_bbox=[round(v, 2) for v in rim.bbox])
            for s in sorted(mine, key=lambda s: s.y):
                for mi in INVERT.finditer(s.text):
                    row["inverts"].append(dict(dir=(mi.group(1) or "").upper() or None, side=(mi.group(3) or "").upper() or None,
                                               elev=float(mi.group(2)), text=s.text.strip(), bbox=[round(v, 2) for v in s.bbox]))
            if sta_h:
                st = next((STATION.search(s.text) for s in mine if STATION.search(s.text)), None)
                if st:
                    row["station"] = dict(station=int(st.group(1)) * 100 + float(st.group(2)), alignment=st.group(3),
                                          offset=float(st.group(4)) * (-1 if st.group(5).upper().startswith("L") else 1))
            if type_h:
                t = min((s for s in mine if abs(s.x - type_h.x) < type_h.size * 6 and not NUM.fullmatch(s.text.strip())),
                        key=lambda s: (abs(s.y - a.y), abs(s.x - type_h.x)), default=None)
                if t is not None:
                    row["type"] = t.text.strip()
            if "rim" in row or row["inverts"]:
                rows_out.append(row)
        used |= ids
    return rows_out, used
