"""Roadway geometry from plan sheets: centerline, stationing, and pavement / curb / sidewalk / R/W edges.

Written for how CAD actually exports PDFs:

* Dashed and center linetypes usually arrive as hundreds of separate short
  segments, and arcs as Bezier curves. :func:`chain_segments` joins them back
  into continuous lines (bridging dash gaps only where the line continues
  straight on).
* The centerline is found from its stationing, not its line style: it is the
  line along which station labels (``STA 10+00`` or bare ``10+00``) fit one
  consistent stationing. The fit also measures the drawing scale, which is
  checked against the stated scale (or stands in for it when there is none).
* Edges are lines that run at a constant offset from the centerline over a
  real length of road. They are named from nearby labels (EP, BC, sidewalk,
  R/W ...). Where there are no labels the order outward from the centerline is
  used, at lower confidence, and flagged for review. Dashed lines (lane
  striping, right-of-way) are never named from order alone.

All page geometry here is in PDF points (y down). Offsets are in feet,
positive to the right of the direction of stationing.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from ..geometry.alignment import Alignment

# label text -> edge kind
EDGE_LABELS = [
    ("row", re.compile(r"R/W|\bR\.O\.W\b|\bROW\b|RIGHT[\s-]OF[\s-]WAY|\bP/L\b|PROPERTY\s+LINE", re.I)),
    ("walk", re.compile(r"SIDEWALK|\bS/W\b|\bSW\b|\bWALK\b", re.I)),
    ("curb_back", re.compile(r"\bBC\b|\bB\.C\.|\bBOC\b|BACK\s+OF\s+CURB", re.I)),
    ("curb_face", re.compile(r"\bFC\b|FACE\s+OF\s+CURB|\bTC\b|TOP\s+OF\s+CURB|\bFL\b|FLOW\s*LINE|GUTTER", re.I)),
    ("pavement", re.compile(r"\bE\.?P\.?\b|\bEOP\b|EDGE\s+OF\s+(PAVEMENT|PVMT|ASPHALT|TRAVEL\s*WAY)|\bLIP\b", re.I)),
]
ORDER = {"pavement": 0, "curb_face": 1, "curb_back": 2, "walk": 3, "row": 4}  # outward from the centerline
NOT_EDGE_LABEL = re.compile(r"\bSTA\b|\d\+\d\d|\bPC\b|\bPT\b|\bPI\b|CURVE|\bR\s*=|\bL\s*=|\bΔ", re.I)


# ----- linework --------------------------------------------------------------

def _rdp(pts: list, tol: float) -> list:
    """Douglas-Peucker simplification (iterative)."""
    if len(pts) < 3:
        return pts
    P = np.asarray(pts, float)
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        v = b - a
        n = math.hypot(*v) or 1e-12
        d = np.abs(v[0] * (P[i + 1:j, 1] - a[1]) - v[1] * (P[i + 1:j, 0] - a[0])) / n
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return [list(map(float, p)) for p in P[keep]]


def chain_segments(segs, gap_tol: float = 9.0, join_tol: float = 0.35, turn_deg: float = 32.0, gap_deg: float = 9.0,
                   min_len: float = 60.0, simplify: float = 0.25) -> list[dict]:
    """Join line segments into polylines.

    ``segs``: iterables of (x0, y0, x1, y1, width, dashed). Segments touching end to end join if the line turns by
    less than ``turn_deg``; segments separated by a gap up to ``gap_tol`` join only if the gap and the next segment
    continue straight on (dashed linetypes exported as separate dashes). Widths must match.
    Returns ``{"pts", "w", "dashed", "len"}`` for chains at least ``min_len`` long.
    """
    S = [tuple(map(float, s[:5])) + (bool(s[5]),) for s in segs if math.hypot(s[2] - s[0], s[3] - s[1]) > 0.05]
    n = len(S)
    cell = max(gap_tol, 1.0)
    grid: dict[tuple, list] = defaultdict(list)
    for i, (x0, y0, x1, y1, _, _) in enumerate(S):
        grid[(int(x0 // cell), int(y0 // cell))].append((i, 0))
        grid[(int(x1 // cell), int(y1 // cell))].append((i, 1))
    used = [False] * n
    cos_turn, cos_gap = math.cos(math.radians(turn_deg)), math.cos(math.radians(gap_deg))

    def unit(dx, dy):
        ln = math.hypot(dx, dy) or 1e-12
        return dx / ln, dy / ln

    def extend(pts, width):
        gaps = 0
        while True:
            px, py = pts[-1]
            k = len(pts) - 2
            while k > 0 and math.hypot(px - pts[k][0], py - pts[k][1]) < 1.0:
                k -= 1
            dx, dy = unit(px - pts[k][0], py - pts[k][1])
            best = None
            cx, cy = int(px // cell), int(py // cell)
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for j, e in grid.get((gx, gy), ()):
                        if used[j]:
                            continue
                        s = S[j]
                        if abs(s[4] - width) > max(0.15, 0.15 * width):
                            continue
                        ex, ey = (s[0], s[1]) if e == 0 else (s[2], s[3])
                        ox, oy = (s[2], s[3]) if e == 0 else (s[0], s[1])
                        g = math.hypot(ex - px, ey - py)
                        if g > gap_tol:
                            continue
                        sx, sy = unit(ox - ex, oy - ey)
                        c_seg = dx * sx + dy * sy
                        if g <= join_tol:
                            if c_seg < cos_turn:
                                continue
                            cost = g + (1 - c_seg)
                        else:
                            gx_, gy_ = unit(ex - px, ey - py)
                            if dx * gx_ + dy * gy_ < cos_gap or c_seg < cos_gap:
                                continue
                            cost = g + 10 * (1 - c_seg) + 1.0
                        if best is None or cost < best[0]:
                            best = (cost, j, (ex, ey), (ox, oy), g)
            if best is None:
                return pts, gaps
            _, j, (ex, ey), (ox, oy), g = best
            used[j] = True
            if g > join_tol:
                gaps += 1
                pts.append([ex, ey])
            pts.append([ox, oy])

    out = []
    for i in sorted(range(n), key=lambda i: -math.hypot(S[i][2] - S[i][0], S[i][3] - S[i][1])):
        if used[i]:
            continue
        used[i] = True
        x0, y0, x1, y1, w, dashed = S[i]
        pts, g1 = extend([[x0, y0], [x1, y1]], w)
        pts.reverse()
        pts, g2 = extend(pts, w)
        P = np.asarray(pts)
        ln = float(np.hypot(*np.diff(P, axis=0).T).sum())
        if ln >= min_len:
            out.append({"pts": [[round(x, 2), round(y, 2)] for x, y in _rdp(pts, simplify)], "w": round(w, 3),
                        "dashed": dashed or (g1 + g2) >= 2, "len": round(ln, 1)})
    return out


# ----- centerline from stationing ---------------------------------------------------

def _arclen(P: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])


def _project(P: np.ndarray, T: np.ndarray, q: np.ndarray):
    """Arc-length position and distance of points q (m, 2) on polyline P."""
    a, v = P[:-1][None], np.diff(P, axis=0)[None]
    l2 = (v ** 2).sum(-1)
    l2[l2 == 0] = 1e-12
    u = np.clip((((q[:, None] - a) * v).sum(-1)) / l2, 0, 1)
    d2 = ((q[:, None] - (a + v * u[..., None])) ** 2).sum(-1)
    i = np.argmin(d2, axis=1)
    r = np.arange(len(q))
    return T[i] + u[r, i] * np.sqrt(l2[0, i]), np.sqrt(d2[r, i])


def fit_stationing(P: np.ndarray, labels: list[tuple], max_dist: float):
    """Fit station = a + b * arc_length to the station labels near polyline P.

    ``labels``: (x, y, station_ft). Returns a dict (a, b, inliers, rms, mean_dist) or None.
    Robust to labels from other alignments: the best pair-wise fit by inlier count is refined on its inliers.
    The scale is measured, not assumed, so half-size prints and wrong scale notes still fit (and get reported).
    """
    if len(labels) < 2:
        return None
    T = _arclen(P)
    q = np.array([[l[0], l[1]] for l in labels], float)
    t, dist = _project(P, T, q)
    near = np.nonzero(dist <= max_dist)[0]
    if len(near) < 2:
        return None
    st = np.array([l[2] for l in labels], float)
    best = None
    for ii in range(len(near)):
        for jj in range(ii + 1, len(near)):
            i, j = near[ii], near[jj]
            if abs(t[j] - t[i]) < 1.0 or st[i] == st[j]:
                continue
            b = (st[j] - st[i]) / (t[j] - t[i])
            if not 0.05 <= abs(b) <= 60:  # 1" = 3.6' ... 1" = 4320'
                continue
            a = st[i] - b * t[i]
            res = np.abs(st[near] - (a + b * t[near]))
            inl = near[res <= max(1.0, 0.003 * (st[near].max() - st[near].min()))]
            key = (len(inl), -float(res[np.isin(near, inl)].mean()))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None or len(best[1]) < 2:
        return None
    inl = best[1]
    A = np.column_stack([np.ones(len(inl)), t[inl]])
    (a, b), *_ = np.linalg.lstsq(A, st[inl], rcond=None)
    rms = float(np.sqrt(((st[inl] - (a + b * t[inl])) ** 2).mean()))
    return dict(a=float(a), b=float(b), inliers=[int(i) for i in inl], rms=rms, mean_dist=float(dist[inl].mean()),
                t_range=[float(T[0]), float(T[-1])])


# ----- edges ----------------------------------------------------------------------------

def _densify(P: np.ndarray, step: float) -> np.ndarray:
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(math.hypot(*(b - a)) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


def centerline_alignment(P: np.ndarray, fit: dict, ft_per_pt: float) -> tuple[Alignment, np.ndarray]:
    """Alignment in sheet feet (x right, z down the page) with increasing stations, and its vertices in points."""
    T = _arclen(P)
    s = fit["a"] + fit["b"] * T
    if fit["b"] < 0:
        P, s = P[::-1], s[::-1]
    return Alignment(s, P[:, 0] * ft_per_pt, P[:, 1] * ft_per_pt), P


def find_edges(al: Alignment, lines: list[dict], ft_per_pt: float, skip: int | None = None) -> list[dict]:
    """Lines running at a constant offset from the alignment, merged by side and offset."""
    L = al.length
    found = []
    for idx, ln in enumerate(lines):
        if idx == skip:
            continue
        P = np.asarray(ln["pts"], float) * ft_per_pt
        if len(P) < 2:
            continue
        Q = _densify(P, 2.5)
        s, d = al.to_so(Q[:, 0], Q[:, 1])
        inside = (s > al.s[0] + 1) & (s < al.s[-1] - 1)
        if inside.sum() < 5:
            continue
        m = float(np.median(d[inside]))
        if not 1.5 <= abs(m) <= 250:
            continue
        ok = inside & (np.abs(d - m) < 0.4)
        # longest contiguous run of samples at the offset
        best, start = (0, 0, 0), None
        for k, v in enumerate(np.append(ok, False)):
            if v and start is None:
                start = k
            elif not v and start is not None:
                span = float(np.ptp(s[start:k])) if k - start > 1 else 0.0
                if span > best[0]:
                    best = (span, start, k)
                start = None
        span, i0, i1 = best
        if span < max(30.0, 0.1 * L):
            continue
        off = float(np.median(d[i0:i1]))
        found.append(dict(side="R" if off > 0 else "L", offset=abs(off), s0=float(s[i0:i1].min()), s1=float(s[i0:i1].max()),
                          coverage=span, dashed=bool(ln["dashed"]), lines=[idx]))
    merged: list[dict] = []
    for e in sorted(found, key=lambda e: (e["side"], e["offset"])):
        m = next((x for x in merged if x["side"] == e["side"] and abs(x["offset"] - e["offset"]) < 0.3 and x["dashed"] == e["dashed"]), None)
        if m:
            w = m["coverage"] + e["coverage"]
            m["offset"] = (m["offset"] * m["coverage"] + e["offset"] * e["coverage"]) / w
            m["coverage"], m["s0"], m["s1"] = w, min(m["s0"], e["s0"]), max(m["s1"], e["s1"])
            m["lines"] += e["lines"]
        else:
            merged.append(dict(e))
    return merged


def looks_like_grid(edges: list[dict]) -> bool:
    """Evenly spaced parallels (a profile or plan grid) rather than road edges, which come at irregular offsets."""
    for side in ("L", "R"):
        offs = sorted(e["offset"] for e in edges if e["side"] == side and not e["dashed"])
        gaps = np.diff([0.0] + offs)
        if len(gaps) >= 3 and gaps.std() < 0.08 * gaps.mean():
            return True
    return False


def edge_label_kind(text: str) -> str | None:
    if NOT_EDGE_LABEL.search(text):
        return None
    for kind, rx in EDGE_LABELS:
        if rx.search(text):
            return kind
    return None


def classify_edges(edges: list[dict], lines: list[dict], labels: list[dict]) -> None:
    """Name each edge in place: ``kind`` (pavement, curb_face, curb_back, walk_inner, walk_outer, row, other),
    ``named_by`` ("label" | "order") and ``label`` (the label candidate it was named from)."""
    # 1. labels vote for the nearest edge line
    for e in edges:
        e.update(kind="other", named_by=None, label=None, votes=defaultdict(int))
    for lab in labels:
        best, bd = None, float("inf")
        q = np.array([[lab["x"], lab["y"]]])
        for e in edges:
            for li in e["lines"]:
                P = np.asarray(lines[li]["pts"], float)
                _, d = _project(P, _arclen(P), q)
                if d[0] < bd:
                    best, bd = e, float(d[0])
        if best is not None and bd <= max(15.0, 2.5 * lab.get("size", 6)):
            best["votes"][lab["kind"]] += 1
            best.setdefault("_labels", []).append(lab)
    for e in edges:
        if e["votes"]:
            kind = max(e["votes"], key=lambda k: (e["votes"][k], -ORDER[k]))
            e.update(kind=kind, named_by="label", label=next(l for l in e["_labels"] if l["kind"] == kind))
    # two kinds voted onto one line (labels for lines a foot apart): split them with the nearest unnamed
    # neighbour so the inner line gets the inner kind (outward order: pavement, curb, walk, R/W)
    for e in edges:
        kinds = sorted(e["votes"], key=ORDER.get)
        if len(kinds) < 2:
            continue
        nb = [o for o in edges if o["side"] == e["side"] and o["kind"] == "other" and 0 < abs(o["offset"] - e["offset"]) <= 3.05]
        if not nb:
            continue
        o = min(nb, key=lambda o: abs(o["offset"] - e["offset"]))
        inner, outer = sorted([e, o], key=lambda x: x["offset"])
        lab = {l["kind"]: l for l in e["_labels"]}
        inner.update(kind=kinds[0], named_by="label", label=lab[kinds[0]])
        outer.update(kind=kinds[1], named_by="label", label=lab[kinds[1]])
    for e in edges:
        e.pop("_labels", None)
        e["votes"] = dict(e["votes"])

    for side in ("L", "R"):
        es = sorted((e for e in edges if e["side"] == side), key=lambda e: e["offset"])
        # a sidewalk label usually sits between its two lines: pair it with the neighbour 3-12 ft away
        for e in [e for e in es if e["kind"] == "walk"]:
            mates = [o for o in es if o is not e and o["kind"] in ("other", "walk") and 3 <= abs(o["offset"] - e["offset"]) <= 12 and not o["dashed"]]
            if mates and not any(o["kind"] == "walk" and o is not e for o in es):
                o = min(mates, key=lambda o: abs(o["offset"] - e["offset"]))
                o.update(kind="walk", named_by=o["named_by"] or "label", label=o["label"] or e["label"])
        walks = [e for e in es if e["kind"] == "walk"]
        if walks:
            walks[0]["kind"] = "walk_inner"
            if len(walks) > 1:
                walks[-1]["kind"] = "walk_outer"
            for w in walks[1:-1]:
                w["kind"] = "other"

        # 2. order outward from the centerline, solid lines only, for whatever no label named
        solid = [e for e in es if not e["dashed"]]
        kinds = {e["kind"] for e in es}
        pav = next((e for e in es if e["kind"] == "pavement"), None)
        if pav is None:
            pav = next((e for e in solid if e["kind"] == "other" and e["offset"] >= 5), None)
            if pav:
                pav.update(kind="pavement", named_by="order")
        if pav is None:
            continue
        if "curb_back" not in kinds:
            near = [e for e in solid if e["kind"] in ("other", "curb_face") and pav["offset"] < e["offset"] <= pav["offset"] + 3.0]
            if near:
                near[-1].update(kind="curb_back", named_by=near[-1]["named_by"] or "order")
        if not walks:
            back = next((e for e in es if e["kind"] == "curb_back"), pav)
            rest = [e for e in solid if e["kind"] == "other" and back["offset"] < e["offset"] <= back["offset"] + 20]
            for a, b in zip(rest, rest[1:]):
                if 3 <= b["offset"] - a["offset"] <= 10:
                    a.update(kind="walk_inner", named_by="order")
                    b.update(kind="walk_outer", named_by="order")
                    break


# ----- the whole sheet ---------------------------------------------------------------------

def read_roadway_sheet(lines: list[dict], stations: list[tuple], labels: list[dict], ft_per_pt: float | None, size_ref: float) -> dict | None:
    """Centerline, stationing and edges of one roadway sheet, in sheet feet.

    ``stations``: (x_pt, y_pt, station_ft, candidate_index). ``ft_per_pt``: the stated scale, if any; when missing,
    the scale measured from the stationing is used and reported.
    """
    fits = []
    for idx, ln in enumerate(lines):
        if ln["len"] < 4 * size_ref:
            continue
        P = np.asarray(ln["pts"], float)
        f = fit_stationing(P, stations, 2.2 * size_ref)
        if not f:
            continue
        # against a stated scale the stationing must agree, or be a half/double-size print; anything else
        # means this line isn't the one the stations belong to
        if ft_per_pt:
            ratio = abs(f["b"]) / ft_per_pt
            if not any(abs(ratio / k - 1) <= 0.03 for k in (1.0, 2.0, 0.5)):
                continue
        fits.append((idx, f))
    if not fits:
        return None
    top = max(len(f["inliers"]) for _, f in fits)
    contenders = []
    for idx, f in fits:
        if len(f["inliers"]) < max(2, top - 1):
            continue
        k = ft_per_pt or abs(f["b"])
        al, P = centerline_alignment(np.asarray(lines[idx]["pts"], float), f, k)
        edges = find_edges(al, lines, k, skip=idx)
        # a road centerline has edges on both sides, at irregular offsets. This rules out a profile grid's lines,
        # which fit stationing perfectly but have grid lines on one side, or evenly spaced on both.
        if not (any(e["side"] == "L" for e in edges) and any(e["side"] == "R" for e in edges)) or looks_like_grid(edges):
            continue
        # prefer more labels, then the center linetype, then labels closest to the line
        contenders.append(((len(f["inliers"]), lines[idx]["dashed"], -f["mean_dist"]), idx, f, al, P, edges, k))
    if not contenders:
        return None
    _, idx, f, al, P, edges, k = max(contenders, key=lambda c: c[0])
    classify_edges(edges, lines, labels)
    used = [stations[i] for i in f["inliers"]]
    return dict(line=idx, fit=f, ft_per_pt=k, scale_from_stationing=ft_per_pt is None, measured_ft_per_pt=abs(f["b"]),
                stations=[float(s) for s in al.s], x=[float(v) for v in al.x], z=[float(v) for v in al.z],
                pts=[[float(a), float(b)] for a, b in P], edges=edges, labels_used=[u[3] for u in used],
                s_range=[float(al.s[0]), float(al.s[-1])])


# ----- placing roadway sheets by their stationing ------------------------------------------

def link_by_stationing(u: dict, r: dict, r_to_project, matchlines: list[tuple[float, str | None, str]], u_sheet: str, r_sheet: str):
    """Place roadway sheet ``u`` next to already-placed sheet ``r`` through their shared stationing.

    Only done where the plans say the two meet: a matchline label at a station where both centerlines are,
    naming the other sheet if it names one (``matchlines``: (station, sheet_ref, on_sheet)). Stationing alone
    isn't enough, since two different streets can both run 0+00 to 5+00.
    Returns (Similarity sheet-feet -> project, station) or None.
    """
    from .calibration import fit_similarity

    U = Alignment(u["stations"], u["x"], u["z"])
    R = Alignment(r["stations"], r["x"], r["z"])
    lo, hi = max(U.s[0], R.s[0]), min(U.s[-1], R.s[-1])
    S = None
    for st, ref, on in matchlines:
        other = r_sheet if on == u_sheet else u_sheet if on == r_sheet else None
        if other and ref in (None, other) and lo - 5 <= st <= hi + 5:
            S = st
            break
    if S is None:
        return None
    if hi - lo >= 20:  # overlapping stationing: match points along the overlap
        ss = np.linspace(lo, hi, 7)
        pu = np.column_stack(U.to_xz(ss, 0.0))
        pr = np.column_stack(r_to_project.apply(*R.to_xz(ss, 0.0)))
    else:  # meeting at the matchline: same point, same direction
        pts = []
        for A in (U, R):
            x, z, tx, tz = _point_and_tangent(A, S)
            n = math.hypot(tx, tz) or 1.0
            pts.append(np.array([[x, z], [x + 50 * tx / n, z + 50 * tz / n]]))
        pu, pr = pts[0], np.column_stack(r_to_project.apply(pts[1][:, 0], pts[1][:, 1]))
    sim = fit_similarity(pu, pr, allow_scale=False)
    return (sim, S) if sim.rms <= 2.0 else None


def _point_and_tangent(A: Alignment, S: float, reach: float = 60.0):
    """Position and direction at station S from a quadratic through nearby vertices.

    A chord's direction is off the tangent on a curve, and a sheet-to-sheet rotation error grows with
    distance from the matchline, so the direction is fitted rather than read off the last chord.
    """
    near = np.abs(A.s - S) <= reach
    if near.sum() >= 3:
        t = A.s[near] - S
        V = np.column_stack([np.ones_like(t), t, t * t])
        cx, *_ = np.linalg.lstsq(V, A.x[near], rcond=None)
        cz, *_ = np.linalg.lstsq(V, A.z[near], rcond=None)
        return float(cx[0]), float(cz[0]), float(cx[1]), float(cz[1])
    x, z, _, _, tx, tz = A.frame(S)
    return float(x), float(z), float(tx), float(tz)
