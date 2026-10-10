"""Sheets placed against each other without anyone tapping points: shared contours, named-alignment stations."""
import math

import numpy as np
import pytest

from gv.ingest.assemble import _profile_rows, assemble
from gv.ingest.calibration import Similarity
from gv.ingest.linking import match_contours, sample_contours
from gv.ingest.pdf import Candidate, SheetReport


def hills():
    """Contours of two lumpy hills, in ground feet: [(elev, pts)]."""
    out = []
    for cx, cy, top, lobes in ((0.0, 0.0, 1060.0, (0.25, 3, 0.1, 5)), (420.0, 160.0, 1030.0, (0.2, 2, 0.15, 4))):
        for elev in range(int(top) - 30, int(top), 2):
            r0 = (top - elev) * 9.0
            t = np.linspace(0, 2 * math.pi, 120)
            r = r0 * (1 + lobes[0] * np.sin(lobes[1] * t + 0.7) + lobes[2] * np.cos(lobes[3] * t))
            out.append((float(elev), np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])))
    return out


def on_sheet(ground, sim: Similarity, ft_per_in):
    """Ground feet -> page points of a sheet whose page feet map to ground by `sim`."""
    k = ft_per_in / 72
    c, s = math.cos(sim.theta), math.sin(sim.theta)
    d = ground - [sim.tx, sim.tz]
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]) / k


def grading_sheet(page, sid, ft_per_in, sim, cut=None, extra=0):
    r = SheetReport(page=page, sheet_id=sid, title="GRADING PLAN", category="grading", scale_ft_per_in=ft_per_in, scale_text="",
                    has_text_layer=True, width_pt=2448, height_pt=1584, line_segments=0, text_spans=0)
    for elev, g in hills():
        if cut is not None:
            g = g[cut(g)]
            if len(g) < 3:
                continue
        p = on_sheet(g, sim, ft_per_in)
        r.candidates.append(Candidate("contour", f"{elev:g}", {"elev": elev, "dashed": False, "surface": None, "pts": p.round(2).tolist()},
                                      page, p[0][0], p[0][1], [], f"{elev:g}", 0.8))
    for i in range(extra):  # something to make it the primary sheet
        r.candidates.append(Candidate("station", f"{i}+00", {"station": i * 100.0}, page, 50.0 + i, 50.0, [], "", 0.9))
    return r


def lines(r):
    k = r.scale_ft_per_in / 72
    return sample_contours([(c.values["elev"], "proposed", np.array(c.values["pts"]) * k) for c in r.candidates if c.kind == "contour"])


def test_contours_place_a_sheet_drawn_at_another_scale_and_rotation():
    detail = grading_sheet(0, "C19", 20, Similarity(theta=0.0, tx=-300.0, tz=-250.0))
    overview = grading_sheet(1, "G04", 50, Similarity(theta=math.radians(17.0), tx=-700.0, tz=-500.0))
    got = match_contours(lines(overview), lines(detail))
    assert got is not None
    sim, stats = got
    assert stats["fraction"] > 0.9 and stats["median_ft"] < 0.05
    # overview page feet -> detail page feet: rotated 17 degrees, and both describe the same ground
    g = np.array([[120.0, 40.0], [430.0, 200.0]])
    a = on_sheet(g, Similarity(theta=math.radians(17.0), tx=-700.0, tz=-500.0), 50) * 50 / 72
    b = on_sheet(g, Similarity(tx=-300.0, tz=-250.0), 20) * 20 / 72
    assert np.allclose(np.column_stack(sim.apply(a[:, 0], a[:, 1])), b, atol=0.3)


def test_contours_of_another_place_dont_match():
    a = grading_sheet(0, "C14", 20, Similarity())
    # the same elevations, a different hill shape: a sheet of somewhere else
    b = SheetReport(page=1, sheet_id="C15", title="GRADING PLAN", category="grading", scale_ft_per_in=20, scale_text="",
                    has_text_layer=True, width_pt=2448, height_pt=1584, line_segments=0, text_spans=0)
    for elev, g in hills():
        t = np.linspace(0, 2 * math.pi, 120)
        r = (1060 - elev) * 6.0 * (1 + 0.3 * np.cos(2 * t))
        p = np.column_stack([800 + r * np.cos(t), 600 + r * np.sin(t)]) / (20 / 72)
        b.candidates.append(Candidate("contour", f"{elev:g}", {"elev": elev, "dashed": False, "surface": None, "pts": p.tolist()}, 1, 0, 0, [], "", 0.8))
    assert match_contours(lines(b), lines(a)) is None


def test_assembly_chains_sheets_by_contours():
    primary = grading_sheet(0, "C19", 20, Similarity(tx=-300.0, tz=-250.0), extra=5)
    overview = grading_sheet(1, "G04", 50, Similarity(theta=math.radians(-8.0), tx=-650.0, tz=-420.0))
    res = assemble([("d", primary), ("d", overview)])
    assert {s.id for s in res["sheets"] if s.registered} == {"C19", "G04"}
    assert any(i.code == "placed_by_contours" and i.sheet == "G04" for i in res["issues"])
    t = Similarity(**res["calibrations"]["d:1"]["transform"])
    g = np.array([[60.0, -30.0]])
    a = on_sheet(g, Similarity(theta=math.radians(-8.0), tx=-650.0, tz=-420.0), 50) * 50 / 72
    expect = on_sheet(g, Similarity(tx=-300.0, tz=-250.0), 20) * 20 / 72  # C19 is the frame
    assert np.allclose(np.column_stack(t.apply(a[:, 0], a[:, 1])), expect, atol=0.3)


def named(page, sid, labels, ft_per_in=20):
    r = SheetReport(page=page, sheet_id=sid, title="PLAN", category="grading", scale_ft_per_in=ft_per_in, scale_text="",
                    has_text_layer=True, width_pt=2448, height_pt=1584, line_segments=0, text_spans=0)
    for text, x, y in labels:
        r.candidates.append(Candidate("named_station", text, {"station": 0.0, "alignment": "P1"}, page, x, y, [], text, 0.85))
    return r


def test_named_alignment_stations_link_sheets():
    pts = {"0+00P1L": (400, 300), "1+00P1L": (760, 330), "1+69P1L": (1010, 420), "2+02P1R": (1100, 600)}
    a = named(0, "AC02", [(t, x, y) for t, (x, y) in pts.items()] + [("9+99X", 5, 5)] * 2)
    th = math.radians(30)
    rot = lambda x, y: (x * math.cos(th) - y * math.sin(th) + 900, x * math.sin(th) + y * math.cos(th) + 100)
    b = named(1, "C14", [(t, *rot(x, y)) for t, (x, y) in pts.items()])
    res = assemble([("d", a), ("d", b)])
    msg = next(i.message for i in res["issues"] if i.code == "placed_by_named_stations")
    assert "4 alignment station labels" in msg and "0+00P1L" in msg
    # three labels close together don't place a sheet
    c = named(1, "C14", [(t, *rot(x, y)) for t, (x, y) in list(pts.items())[:3]][:1] + [("1+00P1L", *rot(410, 305)), ("1+69P1L", *rot(420, 310))])
    assert not any(i.code == "placed_by_named_stations" for i in assemble([("d", a), ("d", c)])["issues"])


def test_profile_label_rows_are_not_plan_spots():
    row = [Candidate("spot", "FG", {"elev": 960 + i}, 0, 103 + 90 * i, 1449, [], "", 0.9) for i in range(6)]
    loose = [Candidate("spot", "FG", {"elev": 950.0}, 0, 500, 700, [], "", 0.9), Candidate("spot", "FG", {"elev": 951.0}, 0, 820, 720, [], "", 0.9)]
    out = _profile_rows(row + loose)
    assert out == {id(c) for c in row}
