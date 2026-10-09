"""Deterministic utility calculations: slopes, cover, depth, crossings, trenches.

These functions take plan values as given. Nothing here estimates a value
that should come from the plans.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

_trap = getattr(np, "trapezoid", None) or np.trapz

from ..models import LinearUtility, Pipe, Project, Structure
from .surface import Grid


@dataclass
class PipeCalc:
    length: float
    slope: float | None  # ft/ft; None when an invert is missing
    min_cover: float | None  # None without a finished-grade surface over the pipe
    id_ft: float
    od_ft: float

    def as_dict(self):
        return asdict(self)


def plan_len(a, b) -> float:
    return math.hypot(b.x - a.x, b.z - a.z)


def pipe_calc(p: Pipe, prj: Project, fg: Grid | None) -> PipeCalc:
    a, b = prj.structure(p.from_id), prj.structure(p.to_id)
    length = plan_len(a, b)
    idf = (p.dia_in.value or 0) / 12
    up, dn = p.inv_up.value, p.inv_dn.value
    if up is None or dn is None or not length:
        return PipeCalc(length, None, None, idf, idf + 2 * p.wall_ft)
    slope = (up - dn) / length
    min_cover = None
    if fg is not None and p.dia_in.value:
        t1 = 0.75 if b.kind == "HW" else 0.95  # outfalls daylight at the end
        t = np.linspace(0.05, t1, 21)
        x, z = a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t
        crown = up + (dn - up) * t + idf + p.wall_ft
        cov = fg.sample(x, z) - crown
        if np.isfinite(cov).any():
            min_cover = float(np.nanmin(cov))
    return PipeCalc(length, slope, min_cover, idf, idf + 2 * p.wall_ft)


def structure_depth(s: Structure, prj: Project) -> float | None:
    inv = [p.inv_dn.value for p in prj.pipes if p.to_id == s.id] + [p.inv_up.value for p in prj.pipes if p.from_id == s.id]
    inv = [v for v in inv if v is not None]
    if not inv or s.rim.value is None:
        return None
    return s.rim.value - min(inv)


def line_length(l: LinearUtility) -> float:
    return sum(plan_len(a, b) for a, b in zip(l.points, l.points[1:]))


def rebuild_line_profile(l: LinearUtility) -> None:
    """Re-derive the main's centerline from ground and design cover."""
    for p in l.points:
        p.center = p.ground - (l.cover.value or 0) - l.od_ft / 2


# ----- crossings ---------------------------------------------------------

def seg_intersect(ax, az, bx, bz, cx, cz, dx, dz):
    rx, rz, sx, sz = bx - ax, bz - az, dx - cx, dz - cz
    den = rx * sz - rz * sx
    if abs(den) < 1e-9:
        return None
    t = ((cx - ax) * sz - (cz - az) * sx) / den
    u = ((cx - ax) * rz - (cz - az) * rx) / den
    return (t, u) if 0 <= t <= 1 and 0 <= u <= 1 else None


@dataclass
class Segment:
    id: str
    system: str
    ax: float
    az: float
    bx: float
    bz: float
    ea: float  # centerline elevation at a
    eb: float
    r: float  # outside radius


def segments(prj: Project) -> list[Segment]:
    out = []
    for p in prj.pipes:
        if p.inv_up.value is None or p.inv_dn.value is None:
            continue
        a, b = prj.structure(p.from_id), prj.structure(p.to_id)
        idf = (p.dia_in.value or 0) / 12
        out.append(Segment(p.id, p.system, a.x, a.z, b.x, b.z, p.inv_up.value + idf / 2, p.inv_dn.value + idf / 2, idf / 2 + p.wall_ft))
    for l in prj.lines:
        for a, b in zip(l.points, l.points[1:]):
            out.append(Segment(l.id, l.system, a.x, a.z, b.x, b.z, a.center, b.center, l.od_ft / 2))
    return out


CONFLICT_FT = 1.0  # flag anything under 12 in
TIGHT_FT = 1.5  # warn under 18 in; actual requirement comes from the specs


def find_crossings(prj: Project, station_of) -> list[dict]:
    sg = segments(prj)
    out = []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            A, B = sg[i], sg[j]
            if A.system == B.system:
                continue
            h = seg_intersect(A.ax, A.az, A.bx, A.bz, B.ax, B.az, B.bx, B.bz)
            if not h:
                continue
            t, u = h
            e1, e2 = A.ea + (A.eb - A.ea) * t, B.ea + (B.eb - B.ea) * u
            clr = abs(e1 - e2) - A.r - B.r
            x, z = A.ax + (A.bx - A.ax) * t, A.az + (A.bz - A.az) * t
            s, d = station_of(x, z)
            upper, lower = (A, B) if e1 > e2 else (B, A)
            out.append(dict(upper=upper.id, lower=lower.id, x=x, z=z, station=s, offset=d, clearance_ft=clr,
                            elev_mid=(e1 + e2) / 2,
                            severity="conflict" if clr < CONFLICT_FT else "tight" if clr < TIGHT_FT else "clear"))
    out.sort(key=lambda c: c["clearance_ft"])
    for k, c in enumerate(out):
        c["id"] = f"X-{k + 1}"
    return out


# ----- trench excavation --------------------------------------------------

class NotEnoughData(ValueError):
    pass


def trench_quantities(p: Pipe, prj: Project, fg: Grid | None, bedding_ft=0.5, side_clear_ft=1.0) -> dict:
    """Vertical-wall trench (trench box where needed) from structure face to face."""
    missing = [n for n, v in (("upstream invert", p.inv_up.value), ("downstream invert", p.inv_dn.value), ("pipe diameter", p.dia_in.value)) if v is None]
    if fg is None:
        missing.append("finished-grade surface (grading plan)")
    if missing:
        raise NotEnoughData("Trench quantities need: " + ", ".join(missing))
    a, b = prj.structure(p.from_id), prj.structure(p.to_id)
    c = pipe_calc(p, prj, fg)
    length = c.length
    t0 = min(0.3, 2.5 / length)
    t1 = 1 - min(0.3, (1.0 if b.kind == "HW" else 2.5) / length)
    n = max(6, math.ceil(length * (t1 - t0) / 2))
    t = np.linspace(t0, t1, n + 1)
    x, z = a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t
    g = fg.sample(x, z)
    if not np.isfinite(g).all():
        raise NotEnoughData("The finished-grade surface does not cover this whole pipe run.")
    inv = p.inv_up.value + (p.inv_dn.value - p.inv_up.value) * t
    bottom = inv - p.wall_ft - bedding_ft
    spring = inv + c.id_ft / 2
    width = c.od_ft + 2 * side_clear_ft
    ds = length * (t1 - t0) / n
    depth = g - bottom
    exc = float(_trap(width * depth, dx=ds))
    r = c.od_ft / 2
    bed = float(_trap(width * (spring - bottom), dx=ds)) - math.pi * r * r / 2 * ds * n
    pipe_vol = math.pi * r * r * ds * n
    back = exc - bed - pipe_vol
    return dict(pipe=p.id, length_ft=length * (t1 - t0), width_ft=width, max_depth_ft=float(depth.max()),
                excavation_cy=exc / 27, bedding_cy=bed / 27, bedding_tons=bed / 27 * 1.4, backfill_cy=back / 27,
                protective_system_required=bool(depth.max() > 5.0),
                samples=[dict(x=float(xi), z=float(zi), ground=float(gi), bottom=float(bi), spring=float(si))
                         for xi, zi, gi, bi, si in zip(x, z, g, bottom, spring)])


def earthwork(eg: Grid | None, fg: Grid | None) -> dict:
    """Grid method on the finished-grade grid: average of the four corner differences per cell.

    Only cells where both surfaces have data are counted; the covered area is reported.
    """
    if eg is None or fg is None:
        need = [n for n, g in (("existing ground (survey / existing conditions sheet)", eg), ("finished grade (grading plan)", fg)) if g is None]
        return dict(available=False, reason="Earthwork needs both surfaces. Missing: " + "; ".join(need) + ".")
    X, Z = fg.coords()
    dz = fg.elev - eg.sample(X, Z)
    cell = (dz[:-1, :-1] + dz[1:, :-1] + dz[:-1, 1:] + dz[1:, 1:]) / 4
    ok = np.isfinite(cell)
    if not ok.any():
        return dict(available=False, reason="The existing and finished surfaces do not overlap. Check sheet registration.")
    area = fg.cell * fg.cell
    c = cell[ok]
    cut = float(-c[c < 0].sum() * area / 27)
    fill = float(c[c > 0].sum() * area / 27)
    return dict(available=True, cut_cy=cut, fill_cy=fill, net_cy=cut - fill, area_acres=float(ok.sum() * area / 43560),
                max_cut_ft=float(max(0.0, -c.min())), max_fill_ft=float(max(0.0, c.max())),
                method=f"{fg.cell:g}-ft grid, existing vs finished grade; excludes topsoil stripping, pavement section, and shrink/swell")
