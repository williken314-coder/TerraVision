"""Generate a sample civil plan set PDF from the Maple Ridge demo data.

The PDF looks like a (simplified) real plan set: title blocks, stated scale,
dashed existing contours, solid proposed contours, spot elevations, storm and
sanitary plans with structure symbols, rim/invert labels and pipe callouts,
and a storm profile sheet. Use it to exercise the upload pipeline end to end.

Also writes the exact layout that was drawn (text positions, symbols, lines)
to a JSON file so the extraction logic can be tested without PyMuPDF.

    python scripts/make_sample_plans.py  ->  samples/maple-ridge-sample-plans.pdf
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import contour as mcontour  # noqa: F401  (ensures matplotlib is present)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gv.demo import build_demo  # noqa: E402
from gv.geometry.surface import Grid  # noqa: E402
from gv.ingest.assemble import _compass  # noqa: E402

W, H = 24 * 72, 18 * 72
K = 40 / 72  # 1" = 40' -> feet per point
X0, Z0 = -300.0, -200.0
FONT = "Helvetica"


class Sheet:
    scale_ft = 40

    def __init__(self, c: canvas.Canvas, number: str, title: str, origin=(100, 120), scale=True):
        self.c, self.number, self.title, self.ox, self.oy = c, number, title, origin[0], origin[1]
        self.rec = {"w": W, "h": H, "spans": [], "segs": [], "symbols": [], "polys": []}
        self.title_block(scale)

    # coordinates: x/y in PDF points, origin top-left, y down (PyMuPDF convention)
    def P(self, x_ft, z_ft):
        return (x_ft - X0) / K + self.ox, (z_ft - Z0) / K + self.oy

    def text(self, x, y, s, size=7, center=False):
        w = stringWidth(s, FONT, size)
        if center:
            x -= w / 2
        self.c.setFont(FONT, size)
        self.c.drawString(x, H - y, s)
        self.rec["spans"].append({"text": s, "x": x + w / 2, "y": y - size * 0.3, "size": size, "bbox": [x, y - size * 0.8, x + w, y + size * 0.2]})

    def line(self, x0, y0, x1, y1, width=0.5, dash=None):
        self.c.setLineWidth(width)
        self.c.setDash(*dash) if dash else self.c.setDash()
        self.c.line(x0, H - y0, x1, H - y1)
        self.rec["segs"].append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "width": width, "dashed": bool(dash)})

    def poly(self, pts, width=0.5, dash=None):
        self.c.setLineWidth(width)
        self.c.setDash(*dash) if dash else self.c.setDash()
        p = self.c.beginPath()
        p.moveTo(pts[0][0], H - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, H - y)
        self.c.drawPath(p, stroke=1, fill=0)
        self.rec["polys"].append({"pts": [list(q) for q in pts], "width": width, "dashed": bool(dash)})
        for (a, b), (c_, d) in zip(pts, pts[1:]):
            self.rec["segs"].append({"x0": a, "y0": b, "x1": c_, "y1": d, "width": width, "dashed": bool(dash)})

    def rect_symbol(self, x, y, size=8):
        self.c.setDash()
        self.c.setLineWidth(0.8)
        self.c.rect(x - size / 2, H - y - size / 2, size, size, stroke=1, fill=0)
        self.rec["symbols"].append({"kind": "rect", "x": x, "y": y, "size": size})

    def circle_symbol(self, x, y, r=5):
        self.c.setDash()
        self.c.setLineWidth(0.8)
        self.c.circle(x, H - y, r, stroke=1, fill=0)
        self.rec["symbols"].append({"kind": "circle", "x": x, "y": y, "size": 2 * r})

    def title_block(self, scale):
        c = self.c
        c.setDash()
        c.setLineWidth(1.5)
        c.rect(36, 36, W - 72, H - 72)
        bx, by = W - 520, H - 250
        c.rect(bx, 36, 484, 214)
        self.text(bx + 16, by + 40, "MAPLE RIDGE PHASE 1", 10)
        self.text(bx + 16, by + 60, "SAMPLE PLAN SET - GENERATED FROM DEMO DATA", 7)
        self.text(bx + 16, by + 110, self.title, 14)
        if scale:
            self.text(bx + 16, by + 135, f"SCALE: 1\" = {self.scale_ft:g}'", 9)
        self.text(bx + 16, by + 165, "VERTICAL DATUM: NAVD88", 7)
        self.text(bx + 380, by + 190, self.number, 22)


class RoadSheet(Sheet):
    """A roadway plan at its own scale, optionally rotated, drawn the way CAD exports it."""

    def __init__(self, c, number, title, scale_ft, center_ft, center_pt, rot_deg=0.0):
        self.scale_ft, self.k = scale_ft, scale_ft / 72
        self.cx, self.cz, self.px, self.py = *center_ft, *center_pt
        self.cos, self.sin = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        super().__init__(c, number, title)

    def P(self, x_ft, z_ft):
        dx, dz = x_ft - self.cx, z_ft - self.cz
        return self.px + (self.cos * dx - self.sin * dz) / self.k, self.py + (self.sin * dx + self.cos * dz) / self.k

    def dash_segments(self, pts, pattern, width):
        """A dashed linetype as separate short lines, the way AutoCAD plots one."""
        P = np.asarray(pts, float)
        T = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        t, i = 0.0, 0
        while t < T[-1]:
            on = pattern[i % len(pattern)]
            if i % 2 == 0:
                a, b = t, min(t + on, T[-1])
                if b - a > 0.2:
                    self.line(np.interp(a, T, P[:, 0]), np.interp(a, T, P[:, 1]), np.interp(b, T, P[:, 0]), np.interp(b, T, P[:, 1]), width)
            t += on
            i += 1

    def curve(self, pts, width):
        """A smooth line as Bezier curves (Catmull-Rom through the points)."""
        self.c.setLineWidth(width)
        self.c.setDash()
        P = [np.array(q, float) for q in pts]
        path = self.c.beginPath()
        path.moveTo(P[0][0], H - P[0][1])
        for i in range(len(P) - 1):
            p0, p1, p2, p3 = P[max(0, i - 1)], P[i], P[i + 1], P[min(len(P) - 1, i + 2)]
            c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
            path.curveTo(c1[0], H - c1[1], c2[0], H - c2[1], p2[0], H - p2[1])
        self.c.drawPath(path, stroke=1, fill=0)


def arc_stationing(al):
    """Demo centerline restationed by true arc length, as CAD stations an alignment."""
    p = np.linspace(al.s[0], al.s[-1], 4000)
    x, z = al.to_xz(p, 0.0)
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    return lambda st: (float(np.interp(st, arc, x)), float(np.interp(st, arc, z))), float(arc[-1])


def road_sheet(c, number, title, s0, s1, scale_ft, rot, al, S, structures, labels, match, section=None):
    at, _ = arc_stationing(al)
    mid = at((s0 + s1) / 2)
    sh = RoadSheet(c, number, title, scale_ft, mid, (620, 520), rot)
    ss = np.arange(s0, s1 + 0.01, 2.0)

    def off_pts(d):
        out = []
        for st in ss:
            (xa, za), (xb, zb) = at(st - 0.5), at(st + 0.5)
            tx, tz = xb - xa, zb - za
            n = math.hypot(tx, tz)
            x, z = at(st)
            out.append(sh.P(x - tz / n * d, z + tx / n * d))
        return out

    sh.dash_segments(off_pts(0), [24, 4, 4, 4], 0.35)  # center linetype, dash by dash
    for d in (-14, 14):
        sh.poly(off_pts(d)[::3], 0.6)
    for d in (-15, 15):
        sh.curve(off_pts(d)[::10], 0.6)
    for d in (-18, -23, 18, 23):
        sh.poly(off_pts(d)[::3], 0.45)
    for d in (-30, 30):
        sh.poly(off_pts(d)[::3], 0.35, (10, 3))  # phantom R/W line, a real PDF dash pattern
    for st in np.arange(math.ceil(s0 / 50) * 50, s1 + 0.01, 50):  # ticks and station labels
        p1, p2 = off_pts(-2.5)[int((st - s0) / 2)], off_pts(2.5)[int((st - s0) / 2)]
        sh.line(*p1, *p2, 0.35)
        lx, ly = off_pts(-6.5)[int((st - s0) / 2)]
        sh.text(lx, ly, ("STA " if st == 0 else "") + f"{int(st // 100)}+{int(st % 100):02d}", 6, center=True)
    for txt, d, st in labels:  # edge labels sit on their line
        x, y = off_pts(d)[int((st - s0) / 2)]
        sh.text(x, y + 2, txt, 5.5, center=True)
    for sid in structures:
        px, py = sh.P(S[sid].x, S[sid].z)
        sh.rect_symbol(px, py, 6)
        sh.text(px + 7, py - 6, sid, 6)
    ma, mb = off_pts(-34)[int((match[0] - s0) / 2)], off_pts(34)[int((match[0] - s0) / 2)]
    sh.line(*ma, *mb, 1.2, (12, 4))
    sh.text(mb[0] + 6, mb[1] + 8, f"MATCHLINE STA {int(match[0] // 100)}+{int(match[0] % 100):02d}", 7)
    sh.text(mb[0] + 6, mb[1] + 18, f"SEE SHEET {match[1]}", 7)
    if section:
        for i, t in enumerate(["PAVEMENT SECTION (TYPICAL)"] + section):
            sh.text(140, 1000 + i * 11, t, 7)
    return sh


def contour_paths(grid: Grid, levels):
    X, Z = grid.coords()
    cs = plt.contour(X, Z, grid.elev, levels=levels)
    out = []
    for lvl, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) >= 4:
                out.append((float(lvl), seg))
    plt.close("all")
    return out


def _demo_alignment():
    from gv.geometry.alignment import Alignment
    a = build_demo().alignment
    return Alignment([p.s for p in a.points], [p.x for p in a.points], [p.z for p in a.points])


def main(out_dir: Path):
    prj = build_demo()
    eg, fg = Grid.from_model(prj.existing), Grid.from_model(prj.proposed)
    S = {s.id: s for s in prj.structures}
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "maple-ridge-sample-plans.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=landscape((H, W)))
    layouts = []

    # C-0.0 cover ------------------------------------------------------------
    sh = Sheet(c, "C-0.0", "COVER SHEET AND SHEET INDEX", scale=False)
    sh.text(160, 200, "MAPLE RIDGE PHASE 1 - SITE IMPROVEMENT PLANS", 26)
    for i, (n, t) in enumerate([("C-0.0", "COVER SHEET"), ("C-3.0", "GRADING PLAN"), ("C-4.0", "STORM DRAINAGE PLAN"),
                                ("C-4.1", "STORM PROFILES"), ("C-5.0", "SANITARY SEWER PLAN"),
                                ("C-6.0", "ROADWAY PLAN STA 0+00 TO 3+00"), ("C-6.1", "ROADWAY PLAN STA 3+00 TO END")]):
        sh.text(160, 300 + i * 24, f"{n}   {t}", 12)
    layouts.append(sh.rec)
    c.showPage()

    storm_ids = [s.id for s in prj.structures if s.system == "storm"]

    # C-3.0 grading -------------------------------------------------------------
    sh = Sheet(c, "C-3.0", "GRADING PLAN", origin=(110, 130))
    for lvl, seg in contour_paths(eg, np.arange(1036, 1066, 2)):
        pts = [sh.P(x, z) for x, z in seg[:: max(1, len(seg) // 60)]]
        sh.poly(pts, 0.4, (4, 3))
        mx, my = pts[len(pts) // 2]
        sh.text(mx, my + 2, f"{lvl:.0f}", 5.5, center=True)
    for lvl, seg in contour_paths(fg, np.arange(1036, 1066, 2)):
        pts = [sh.P(x, z) for x, z in seg[:: max(1, len(seg) // 60)]]
        sh.poly(pts, 0.7)
        mx, my = pts[len(pts) // 3]
        sh.text(mx, my + 2, f"{lvl:.0f}", 5.5, center=True)
    rng = np.random.default_rng(3)
    for _ in range(24):
        x, z = rng.uniform(-280, 280), rng.uniform(-180, 180)
        e = float(fg.sample(x, z))
        px, py = sh.P(x, z)
        sh.text(px + 3, py, f"FG {e:.2f}", 5.5)
    for _ in range(16):
        x, z = rng.uniform(-290, 290), rng.uniform(-190, 190)
        px, py = sh.P(x, z)
        sh.text(px + 3, py, f"EG {float(eg.sample(x, z)):.2f}", 5.5)
    for sid in storm_ids:
        s = S[sid]
        px, py = sh.P(s.x, s.z)
        sh.rect_symbol(px, py)
        sh.text(px + 8, py - 8, sid, 6.5)
        if s.kind != "HW":
            sh.text(px + 8, py + 1, f"RIM {s.rim.value:.2f}", 6)
    layouts.append(sh.rec)
    c.showPage()

    # C-4.0 storm plan (different sheet origin, to exercise registration) ------
    sh = Sheet(c, "C-4.0", "STORM DRAINAGE PLAN", origin=(150, 170))
    al = prj.alignment.points
    for off in (-14, 14):
        pts = []
        for a, b in zip(al, al[1:]):
            tx, tz = b.x - a.x, b.z - a.z
            l = math.hypot(tx, tz)
            pts.append(sh.P(a.x - tz / l * off, a.z + tx / l * off))
        sh.poly(pts, 0.5)
    storm = [p for p in prj.pipes if p.system == "storm"]
    for p in storm:
        a, b = S[p.from_id], S[p.to_id]
        (x0, y0), (x1, y1) = sh.P(a.x, a.z), sh.P(b.x, b.z)
        sh.line(x0, y0, x1, y1, 2.6)
        if p.material.value:
            L_ = math.hypot(b.x - a.x, b.z - a.z)
            slope = (p.inv_up.value - p.inv_dn.value) / L_ * 100
            nx, ny = -(y1 - y0), (x1 - x0)
            n = math.hypot(nx, ny) or 1
            sh.text((x0 + x1) / 2 + nx / n * 12, (y0 + y1) / 2 + ny / n * 12,
                    f"{L_:.0f} LF {p.dia_in.value:.0f}\" RCP @ {slope:.2f}%", 6, center=True)
    for sid in storm_ids:
        s = S[sid]
        px, py = sh.P(s.x, s.z)
        sh.rect_symbol(px, py)
        sh.text(px + 9, py - 10, sid, 7)
        lines = [] if s.kind == "HW" else [f"RIM {s.rim.value:.2f}"]
        for p in storm:
            if p.to_id == sid:
                o = S[p.from_id]
                lines.append(f"INV IN ({_compass(o.x - s.x, o.z - s.z)}) {p.inv_dn.value:.2f}")
        for p in storm:
            if p.from_id == sid:
                o = S[p.to_id]
                lines.append(f"INV OUT ({_compass(o.x - s.x, o.z - s.z)}) {p.inv_up.value:.2f}")
        for i, t in enumerate(lines):
            sh.text(px + 9, py - 1 + i * 8, t, 6)
    layouts.append(sh.rec)
    c.showPage()

    # C-4.1 storm profiles -------------------------------------------------------
    sh = Sheet(c, "C-4.1", "STORM PROFILES", scale=False)
    sh.text(120, 110, "HORIZ: 1\" = 40'   VERT: 1\" = 4'", 8)
    chains = [["CB-1", "CB-2", "CB-3", "CB-5", "CB-6", "HW-1"], ["CB-4", "CB-5"]]
    by_pair = {(p.from_id, p.to_id): p for p in storm}
    for row, chain in enumerate(chains):
        base_y, x = 300 + row * 420, 160.0
        xs = {}
        for i, sid in enumerate(chain):
            if i:
                a, b = S[chain[i - 1]], S[sid]
                x += math.hypot(b.x - a.x, b.z - a.z) / K * 0.9
            xs[sid] = x
        for i, sid in enumerate(chain):
            s, px = S[sid], xs[sid]
            sh.line(px, base_y - 60, px, base_y + 60, 0.6)
            sh.text(px + 4, base_y - 70, sid, 7)
            lines = [] if s.kind == "HW" else [f"RIM {s.rim.value:.2f}"]
            if i and (chain[i - 1], sid) in by_pair:
                lines.append(f"INV IN {by_pair[(chain[i - 1], sid)].inv_dn.value:.2f}")
            if i + 1 < len(chain) and (sid, chain[i + 1]) in by_pair:
                lines.append(f"INV OUT {by_pair[(sid, chain[i + 1])].inv_up.value:.2f}")
            for j, t in enumerate(lines):
                sh.text(px + 4, base_y - 61 + j * 8, t, 6)
            if i + 1 < len(chain):
                p = by_pair.get((sid, chain[i + 1]))
                if p and p.material.value:
                    a, b = S[p.from_id], S[p.to_id]
                    L_ = math.hypot(b.x - a.x, b.z - a.z)
                    sh.text((px + xs[chain[i + 1]]) / 2, base_y + 90,
                            f"{L_:.0f} LF {p.dia_in.value:.0f}\" RCP @ {(p.inv_up.value - p.inv_dn.value) / L_ * 100:.2f}%", 6, center=True)
    layouts.append(sh.rec)
    c.showPage()

    # C-5.0 sanitary plan -----------------------------------------------------------
    sh = Sheet(c, "C-5.0", "SANITARY SEWER PLAN", origin=(130, 150))
    san = [p for p in prj.pipes if p.system == "san"]
    for p in san:
        a, b = S[p.from_id], S[p.to_id]
        (x0, y0), (x1, y1) = sh.P(a.x, a.z), sh.P(b.x, b.z)
        sh.line(x0, y0, x1, y1, 2.2)
        L_ = math.hypot(b.x - a.x, b.z - a.z)
        sh.text((x0 + x1) / 2, (y0 + y1) / 2 - 10, f"{L_:.0f} LF 8\" PVC @ {(p.inv_up.value - p.inv_dn.value) / L_ * 100:.2f}%", 6, center=True)
    for s in (s for s in prj.structures if s.system == "san"):
        px, py = sh.P(s.x, s.z)
        sh.circle_symbol(px, py)
        sh.text(px + 9, py + 12, s.id, 7)
        lines = [f"RIM {s.rim.value:.2f}"]
        lines += [f"INV IN {p.inv_dn.value:.2f}" for p in san if p.to_id == s.id]
        lines += [f"INV OUT {p.inv_up.value:.2f}" for p in san if p.from_id == s.id]
        for i, t in enumerate(lines):
            sh.text(px + 9, py + 21 + i * 8, t, 6)
    for sid in ("CB-3", "CB-4", "CB-5"):  # storm shown for reference
        s = S[sid]
        px, py = sh.P(s.x, s.z)
        sh.rect_symbol(px, py, 7)
        sh.text(px + 8, py - 8, sid, 6)
    a, b = S["CB-4"], S["CB-5"]
    (x0, y0), (x1, y1) = sh.P(a.x, a.z), sh.P(b.x, b.z)
    sh.line(x0, y0, x1, y1, 0.5, (2, 2))
    layouts.append(sh.rec)
    c.showPage()

    # C-6.0 / C-6.1 roadway plans at 1" = 20': the second rotated, with no structures and no edge labels,
    # so it can only be placed through its matchline and its edges only named by order -----------------
    al = _demo_alignment()
    _, total = arc_stationing(al)
    on_first = [sid for sid in storm_ids if S[sid].x < -275 + 300 - 10]
    sh = road_sheet(c, "C-6.0", "ROADWAY PLAN STA 0+00 TO 3+00", 0, 300, 20, 0.0, al, S, on_first,
                    [("EP", -14, 120), ("BC", 15, 210), ("5' SIDEWALK", -20.5, 170), ("R/W", 30, 90)], (300, "C-6.1"))
    layouts.append(sh.rec)
    c.showPage()
    sh = road_sheet(c, "C-6.1", "ROADWAY PLAN STA 3+00 TO END", 300, math.floor(total), 20, -25.0, al, S, [], [], (300, "C-6.0"),
                    section=['1-1/2" HMA SURFACE COURSE', '2-1/2" HMA BASE COURSE', '8" AGGREGATE BASE', "SUBGRADE COMPACTED TO 95%"])
    layouts.append(sh.rec)
    c.showPage()

    c.save()
    (out_dir / "maple-ridge-sample-layout.json").write_text(json.dumps(layouts))
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "samples")
