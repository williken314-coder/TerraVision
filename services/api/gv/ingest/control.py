"""Survey coordinates printed on the plans, as control points.

Alignment control sheets, structure callouts and survey control often print
real coordinates next to a point:

    STA. 10+00.00P2,
    CONSTRUCTION ALIGNMENT
    BEGINNING
    N. 130437.12
    E. 1791742.12

with a leader from the text to the point. The text sits wherever it fits, so
the point is where the leader ends, not where the text is. The leader is
followed as a chain of connected line segments from the edge of the text block
to its far end. A callout without a leader that can be followed is not used.

Two or more of these on a sheet place it in real coordinates on its own.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

NORTH = re.compile(r"^\s*N\.?\s*[:=]?\s*(\d{5,8}\.\d{1,4})\s*'?\s*$", re.I)
EAST = re.compile(r"^\s*E\.?\s*[:=]?\s*(\d{5,8}\.\d{1,4})\s*'?\s*$", re.I)
NE_ONE_LINE = re.compile(r"\bN\.?\s*[:=]?\s*(\d{5,8}\.\d{1,4})'?\s*,?\s*E\.?\s*[:=]?\s*(\d{5,8}\.\d{1,4})", re.I)


def callout_blocks(spans) -> list[dict]:
    """N/E coordinate pairs with the text block around them: {n, e, bbox, text, size}."""
    out = []
    used = set()
    for s in spans:
        m = NE_ONE_LINE.search(s.text)
        if m:
            out.append(dict(n=float(m.group(1)), e=float(m.group(2)), bbox=list(s.bbox), text=s.text, size=s.size))
            continue
        mn = NORTH.match(s.text)
        if not mn or id(s) in used:
            continue
        e = min((o for o in spans if EAST.match(o.text) and abs(o.bbox[0] - s.bbox[0]) <= max(3.0, s.size * 0.4)
                 and 0 < o.y - s.y <= s.size * 1.8), key=lambda o: o.y - s.y, default=None)
        if e is None:
            continue
        used.add(id(e))
        lines = [s, e]
        top = s
        while True:  # the lines above, in the same column
            up = min((o for o in spans if o not in lines and abs(o.bbox[0] - top.bbox[0]) <= max(3.0, top.size * 0.4)
                      and 0 < top.y - o.y <= top.size * 1.8), key=lambda o: top.y - o.y, default=None)
            if up is None or len(lines) >= 7:
                break
            lines.append(up)
            top = up
        x0 = min(o.bbox[0] for o in lines)
        y0 = min(o.bbox[1] for o in lines)
        x1 = max(o.bbox[2] for o in lines)
        y1 = max(o.bbox[3] for o in lines)
        out.append(dict(n=float(mn.group(1)), e=float(EAST.match(e.text).group(1)), bbox=[x0, y0, x1, y1],
                        text=" / ".join(o.text.strip() for o in sorted(lines, key=lambda o: o.y)), size=s.size))
    return out


class _Ends:
    """Segment endpoints on a grid, to follow connected line work."""

    def __init__(self, segs, cell: float = 4.0):
        self.segs = segs
        self.cell = cell
        self.grid = defaultdict(list)
        for i, g in enumerate(segs):
            for k, (x, y) in enumerate(((g.x0, g.y0), (g.x1, g.y1))):
                self.grid[(int(x // cell), int(y // cell))].append((i, k))

    def at(self, x, y, tol):
        cx, cy = int(x // self.cell), int(y // self.cell)
        r = int(math.ceil(tol / self.cell))
        for gx in range(cx - r, cx + r + 1):
            for gy in range(cy - r, cy + r + 1):
                for i, k in self.grid.get((gx, gy), ()):
                    g = self.segs[i]
                    px, py = (g.x0, g.y0) if k == 0 else (g.x1, g.y1)
                    if math.hypot(px - x, py - y) <= tol:
                        yield i, k


def follow_leader(block: dict, ends: _Ends) -> tuple[float, float] | None:
    """Where the leader from a text block ends: the far end of the longest chain of connected segments
    that starts at the block's edge and leaves it."""
    x0, y0, x1, y1 = block["bbox"]
    pad = max(4.0, block["size"] * 1.2)

    def inside(x, y, m=0.0):
        return x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m

    best = None
    starts = set()
    for gx in range(int((x0 - pad) // ends.cell), int((x1 + pad) // ends.cell) + 1):
        for gy in range(int((y0 - pad) // ends.cell), int((y1 + pad) // ends.cell) + 1):
            for i, k in ends.grid.get((gx, gy), ()):
                starts.add((i, k))
    for i, k in starts:
        g = ends.segs[i]
        sx, sy = (g.x0, g.y0) if k == 0 else (g.x1, g.y1)
        fx, fy = (g.x1, g.y1) if k == 0 else (g.x0, g.y0)
        if not inside(sx, sy, pad) or inside(fx, fy, 1.0) or inside(sx, sy, -1.0):
            continue  # must start at the edge (not inside the text: letters drawn as strokes) and leave it
        used, length, (cx, cy) = {i}, g.length, (fx, fy)
        while True:
            # leader runs are long; an arrowhead's strokes are short and start at the tip, so stop there
            nxt = [(j, kk) for j, kk in ends.at(cx, cy, 0.8) if j not in used and ends.segs[j].length >= block["size"] * 0.8]
            if not nxt or len(used) > 12:
                break
            j, kk = max(nxt, key=lambda t: ends.segs[t[0]].length)
            h = ends.segs[j]
            used.add(j)
            length += h.length
            cx, cy = (h.x1, h.y1) if kk == 0 else (h.x0, h.y0)
        reach = math.hypot(cx - (x0 + x1) / 2, cy - (y0 + y1) / 2)
        if length >= block["size"] * 1.5 and not inside(cx, cy, pad) and (best is None or reach > best[0]):
            best = (reach, cx, cy)
    return (best[1], best[2]) if best else None


def read_control(spans, segs) -> list[dict]:
    """Coordinate callouts with their leader targets: {n, e, x, y, text, bbox}."""
    blocks = callout_blocks(spans)
    if not blocks:
        return []
    ends = _Ends([g for g in segs if g.length > 0.2])
    out = []
    for b in blocks:
        t = follow_leader(b, ends)
        if t:
            out.append(dict(n=b["n"], e=b["e"], x=round(t[0], 2), y=round(t[1], 2), text=b["text"], bbox=[round(v, 1) for v in b["bbox"]]))
    return out
