"""Maple Ridge Phase 1: a built-in demo project.

Stands in for the output of plan interpretation until real plan sets are
wired through :mod:`gv.ingest`. Values flagged ``ai`` or ``missing`` here are
deliberate so the review workflow has something to show.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry.alignment import Alignment
from .geometry.surface import Grid
from .geometry.utilities import seg_intersect
from .models import (Alignment as AlignmentModel, AlignmentPoint, Building, CorridorTemplate, Datum, Fact, LinearUtility, LinePoint,
                     Pipe, Pond, Project, Sheet, Status, Structure, Tree)

L = 550.0
POND = dict(x=250.0, z=105.0, rx=38.0, rz=28.0, floor=1038.5, top=1045.0, pool=1040.2)
RCP_WALL = {15: 0.1875, 18: 0.2083, 24: 0.25}


def prof(s):
    c = np.clip(s, 0, L)
    return 1052.5 - 0.0135 * c - 2.2 * np.sin(c / L * np.pi)


def existing_ground(x, z):
    return 1056 - 0.02 * (x + 300) + 0.03 * z + 2.5 * np.sin(x / 70) * np.cos(z / 55) + 1.2 * np.sin((x + z) / 37)


def pond_q(x, z):
    return np.hypot((x - POND["x"]) / POND["rx"], (z - POND["z"]) / POND["rz"])


def road_zone(r, a):
    return np.where(a <= 14, r - 0.02 * a, np.where(a <= 24, r + 0.22 + 0.02 * (a - 14), r + 0.42 + 0.012 * (a - 24)))


def finished_ground(al: Alignment):
    def fn(x, z):
        shape = np.shape(x)
        x, z = np.ravel(x), np.ravel(z)
        s, d = al.to_so(x, z)
        ext = np.where(s < 0, -s, np.where(s > L, s - L, 0))
        a = np.hypot(d, ext)
        r = prof(s)
        p = road_zone(r, np.minimum(a, 90))
        edge, t, e = road_zone(r, 90.0), a - 90, existing_ground(x, z)
        dl = e - edge
        beyond = np.where(np.abs(dl) <= t / 3, e, edge + np.sign(dl) * t / 3)
        p = np.where(a <= 90, p, beyond)
        q = pond_q(x, z)
        pe = np.where(q <= 0.6, POND["floor"], np.where(q <= 1, POND["floor"] + (q - 0.6) / 0.4 * (POND["top"] - POND["floor"]), POND["top"]))
        w = np.clip((q - 1) / 0.45, 0, 1)
        p = np.where(q <= 1, pe, np.where(q < 1.45, pe * (1 - w) + p * w, p))
        return p.reshape(shape)
    return fn


def r2(v):
    return round(float(v), 2)


def build_demo() -> Project:
    st = np.arange(0, L + 1, 5.0)
    al = Alignment(st, -275 + st, 30 * np.sin(st / L * np.pi * 1.2))
    fg_fn = finished_ground(al)
    eg = Grid.from_function(existing_ground, -300, -200, 5, 120, 80)
    fg = Grid.from_function(fg_fn, -300, -200, 5, 120, 80)
    ground = lambda x, z: float(fg.sample(x, z))

    structures: dict[str, Structure] = {}

    def add_s(sid, system, kind, type_, sheet, s=None, d=None, x=None, z=None):
        if x is None:
            x, z = (float(v) for v in al.to_xz(s, d))
        ss, dd = al.to_so(x, z)
        st_ = Structure(id=sid, system=system, kind=kind, x=x, z=z, station=float(ss[0]), offset=float(dd[0]),
                        rim=Fact[float](value=r2(ground(x, z)), source=sheet),
                        type=Fact[str](value=type_, source=sheet, status=Status.verified if type_ else Status.missing),
                        sheet=sheet)
        structures[sid] = st_
        return st_

    cb = "Curb inlet, Type C"
    for sid, s in [("CB-1", 60), ("CB-2", 170), ("CB-3", 280), ("CB-5", 375), ("CB-6", 470)]:
        add_s(sid, "storm", "CB", cb, "C-4.0", s=s, d=15.5)
    add_s("CB-4", "storm", "CB", cb, "C-4.0", s=375, d=-15.5)
    add_s("HW-1", "storm", "HW", None, "C-4.0", x=POND["x"] - POND["rx"] * 0.7 * math.sqrt(0.5), z=POND["z"] - POND["rz"] * 0.7 * math.sqrt(0.5))
    structures["CB-6"].rim.status, structures["CB-6"].rim.confidence = Status.ai, 0.88
    for sid, s in [("SMH-1", 20), ("SMH-2", 200), ("SMH-3", 380), ("SMH-4", 540)]:
        add_s(sid, "san", "SMH", "Sanitary manhole, 48 in precast", "C-5.0", s=s, d=3)
    structures["SMH-3"].rim.status, structures["SMH-3"].rim.confidence = Status.ai, 0.69
    add_s("FH-1", "water", "FH", "Fire hydrant assembly, 6 in lead", "C-6.0", s=150, d=-24.5)
    add_s("FH-2", "water", "FH", "Fire hydrant assembly, 6 in lead", "C-6.0", s=420, d=-24.5)
    add_s("TX-1", "elec", "TX", "Pad-mount transformer, 25 kVA", "E-1.0", s=250, d=-32)

    pipes: list[Pipe] = []
    out_inv: dict[str, float] = {}

    def add_p(pid, system, a, b, dia, mat, wall, sheet, slope, max_dn):
        sa, sb = structures[a], structures[b]
        length = math.hypot(sb.x - sa.x, sb.z - sa.z)
        up = out_inv[a]
        dn = up - slope * length
        if max_dn is not None and dn > max_dn:
            dn = max_dn
        p = Pipe(id=pid, system=system, from_id=a, to_id=b, dia_in=Fact[float](value=dia, source=sheet),
                 material=Fact[str](value=mat, source=sheet, status=Status.verified if mat else Status.missing),
                 wall_ft=wall, inv_up=Fact[float](value=r2(up), source=sheet), inv_dn=Fact[float](value=r2(dn), source=sheet), sheet=sheet)
        pipes.append(p)
        return p

    cb_max = lambda sid, dia: structures[sid].rim.value - 3.9 - dia / 12 - RCP_WALL[dia]
    out_inv["CB-1"] = r2(structures["CB-1"].rim.value - 5.2)
    p = add_p("ST-10", "storm", "CB-1", "CB-2", 15, "15 in RCP, Class III", RCP_WALL[15], "C-4.1", 0.01, cb_max("CB-2", 15))
    out_inv["CB-2"] = r2(p.inv_dn.value - 3 / 12)
    p = add_p("ST-11", "storm", "CB-2", "CB-3", 18, "18 in RCP, Class III", RCP_WALL[18], "C-4.1", 0.01, cb_max("CB-3", 18))
    out_inv["CB-3"] = r2(p.inv_dn.value - 0.1)
    p = add_p("ST-13", "storm", "CB-3", "CB-5", 18, "18 in RCP, Class III", RCP_WALL[18], "C-4.1", 0.008, cb_max("CB-5", 18))
    out_inv["CB-5"] = r2(p.inv_dn.value - 6 / 12)
    p = add_p("ST-14", "storm", "CB-5", "CB-6", 24, "24 in RCP, Class III", RCP_WALL[24], "C-4.1", 0.008, cb_max("CB-6", 24))
    out_inv["CB-6"] = r2(p.inv_dn.value - 0.1)
    add_p("ST-15", "storm", "CB-6", "HW-1", 24, None, RCP_WALL[24], "C-4.1", 0.003, None)

    out_inv["SMH-1"] = r2(structures["SMH-1"].rim.value - 9.0)
    for pid, a, b in [("SA-1", "SMH-1", "SMH-2"), ("SA-2", "SMH-2", "SMH-3"), ("SA-3", "SMH-3", "SMH-4")]:
        p = add_p(pid, "san", a, b, 8, "8 in PVC, SDR-35", 0.02, "C-5.1", 0.006, structures[b].rim.value - 7.5)
        out_inv[b] = r2(p.inv_dn.value - 0.1)

    def line(lid, system, name, dia, mat, od, cover, d, sheet, cover_status=Status.verified, conf=None):
        pts = []
        for s in np.arange(0, L + 1, 25.0):
            x, z = (float(v) for v in al.to_xz(s, d))
            g = ground(x, z)
            pts.append(LinePoint(s=float(s), x=x, z=z, ground=g, center=g - cover - od / 2))
        return LinearUtility(id=lid, system=system, name=name, dia_in=Fact[float](value=dia, source=sheet),
                             material=Fact[str](value=mat, source=sheet), od_ft=od,
                             cover=Fact[float](value=cover, source=sheet, status=cover_status, confidence=conf),
                             offset=d, points=pts, sheet=sheet)

    water = line("W-1", "water", "Water main", 8, "8 in DIP, Class 350", 9.05 / 12, 4.5, -9, "C-6.0")
    elec = line("E-1", "elec", "Street lighting duct bank", 4, "2 x 2 in PVC Sch 40 conduit", 0.5, 3.0, -25, "E-1.0", Status.ai, 0.7)

    # Storm lateral ST-12 crosses the water main. Its inverts were read from a
    # crowded profile, so they carry lower confidence.
    a, b = structures["CB-4"], structures["CB-5"]
    length = math.hypot(b.x - a.x, b.z - a.z)
    wall = RCP_WALL[15]
    inv_x = None
    for p0, p1 in zip(water.points, water.points[1:]):
        h = seg_intersect(a.x, a.z, b.x, b.z, p0.x, p0.z, p1.x, p1.z)
        if h:
            t, u = h
            wc = p0.center + (p1.center - p0.center) * u
            inv_x = wc + water.od_ft / 2 + 5 / 12 + wall
            up, dn = inv_x + 0.01 * t * length, inv_x - 0.01 * (1 - t) * length
    assert inv_x is not None
    pipes.append(Pipe(id="ST-12", system="storm", from_id="CB-4", to_id="CB-5", dia_in=Fact[float](value=15, source="C-4.1"),
                      material=Fact[str](value="15 in RCP, Class III", source="C-4.1"), wall_ft=wall,
                      inv_up=Fact[float](value=r2(up), status=Status.ai, confidence=0.74, source="C-4.1"),
                      inv_dn=Fact[float](value=r2(dn), status=Status.ai, confidence=0.81, source="C-4.1"), sheet="C-4.1"))

    buildings = []
    for lot, s, d in [(1, 120, -62), (2, 240, -62), (3, 340, -62), (4, 150, 62), (5, 300, 62)]:
        x, z = (float(v) for v in al.to_xz(s, d))
        _, _, _, _, tx, tz = al.frame(s)
        pad = ground(x, z)
        buildings.append(Building(id=f"LOT-{lot}", lot=lot, station=s, offset=d, x=x, z=z, rotation=float(math.atan2(-tz, tx)),
                                  pad=r2(pad), ffe=Fact[float](value=r2(pad + 0.67), source="A-0.1")))

    rng = np.random.default_rng(11)
    trees = []
    for _ in range(200):
        x, z = float(rng.uniform(-295, 295)), float(rng.uniform(-195, 195))
        s, d = al.to_so(x, z)
        ext = -s[0] if s[0] < 0 else (s[0] - L if s[0] > L else 0)
        a_ = math.hypot(d[0], ext)
        e = float(existing_ground(x, z))
        p = float(fg_fn(np.array([x]), np.array([z]))[0])
        group = "clear" if (pond_q(x, z) < 1.5 or a_ < 100 or abs(p - e) > 0.3) else "keep"
        trees.append(Tree(x=x, z=z, y=e, h=float(rng.uniform(30, 54)), group=group))
    avoid = [60, 170, 280, 375, 470] + [b.station + 11 for b in buildings]
    for s in np.arange(30, 526, 45.0):
        for d in (-16.5, 16.5):
            if any(abs(v - s) < 9 for v in avoid):
                continue
            x, z = (float(v) for v in al.to_xz(s, d))
            trees.append(Tree(x=x, z=z, y=ground(x, z), h=float(rng.uniform(18, 25)), group="new"))

    def p3(s, d):
        x, z = (float(v) for v in al.to_xz(s, d))
        return [round(x, 2), round(ground(x, z) + 0.4, 2), round(z, 2)]

    boundaries = []
    lot_lines = {-1: [60, 180, 300, 420], 1: [75, 225, 375, 480]}
    for sg in (-1, 1):
        boundaries.append([p3(s, sg * 30) for s in np.arange(0, L + 1, 10.0)])
        boundaries.append([p3(s, sg * 95) for s in np.arange(0, L + 1, 10.0)])
        for s in lot_lines[sg]:
            boundaries.append([p3(s, sg * d) for d in np.arange(30, 96, 5.0)])

    sheets = [Sheet(id=i, title=t, notes=n) for i, t, n in [
        ("C-0.0", "Cover & sheet index", ""), ("C-1.0", "Existing conditions", "1 ft contours, 312 spot shots"),
        ("C-2.0", "Demolition & erosion control", "Limits of clearing"), ("C-3.0", "Grading plan", "2 ft contours, 64 spot elevations, pond"),
        ("C-4.0", "Storm drainage plan", "6 pipes, 7 structures"), ("C-4.1", "Storm profiles", "Inverts, slopes, crossings"),
        ("C-5.0", "Sanitary sewer plan", "3 pipes, 4 manholes"), ("C-5.1", "Sanitary profiles", "Inverts, slopes"),
        ("C-6.0", "Water plan", "550 LF main, 2 hydrants"), ("C-7.0", "Roadway plan & profile", "Sta 0+00 to 5+50"),
        ("C-8.0", "Details", "Pavement section, trench & bedding"), ("E-1.0", "Site electrical", "Duct bank, 5 lights, TX-1"),
        ("L-1.0", "Landscape plan", "Street trees"), ("A-0.1", "Plot plan", "5 lots, FFEs")]]

    return Project(
        id="maple-ridge", name="Maple Ridge · Phase 1 (demo)", demo=True, sheets=sheets, corridor=CorridorTemplate(),
        datum=Datum(horizontal="Demo site grid"),
        alignment=AlignmentModel(name="Maple Ridge Dr", points=[AlignmentPoint(s=float(s), x=float(x), z=float(z), elev=float(prof(s)))
                                                              for s, x, z in zip(al.s, al.x, al.z)]),
        existing=eg.to_model("Existing ground", "C-1.0"), proposed=fg.to_model("Finished grade", "C-3.0"),
        structures=list(structures.values()), pipes=pipes, lines=[water, elec], buildings=buildings, trees=trees,
        boundaries=boundaries, pond=Pond(x=POND["x"], z=POND["z"], rx=POND["rx"], rz=POND["rz"], pool=POND["pool"]),
    )
