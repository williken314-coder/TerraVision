"""Layouts and drafting habits seen on real plan sets, in small synthetic form.

A right-hand title strip with a rotated title, drawing numbers like G03 or RW02, project names repeated on every
sheet, scales drawn as strokes (read by OCR), structures labelled "CB #5", sheets that only share reference
marks, and finished-grade contours drawn only inside the work limits.
"""
import numpy as np
import pytest

from gv.geometry.surface import TIN
from gv.ingest.assemble import _furniture_marks, _same_view, assemble
from gv.ingest.ocr import parse_scale_reading, pick_sheet_scale
from gv.ingest.pdf import STRUCT_ID, Candidate, SheetReport, TextSpan, classify, common_text, title_block

W, H = 2448.0, 1584.0  # 34 x 22 in


def span(text, x, y, size=11.2, vertical=False):
    w = len(text) * size * 0.55
    bbox = (x - size / 2, y - w / 2, x + size / 2, y + w / 2) if vertical else (x - w / 2, y - size / 2, x + w / 2, y + size / 2)
    return TextSpan(text, x, y, size, bbox)


PROJECT = ["CHELAN EXAMPLE PORT AUTHORITY", "EXAMPLE WATER SYSTEM IMPROVEMENTS", "RESERVOIR EARTHWORK", "SCALE:", "DWG NO.:", "SHEET NO.:"]


def strip_sheet(dwg, title_lines, extra=()):
    sp = [span(t, 2256 + 21 * i, 740, 17.5, vertical=True) for i, t in enumerate(PROJECT[:3])]
    sp += [span(t, 2290, 1450 + 20 * i, 7.6) for i, t in enumerate(PROJECT[3:])]
    sp += [span(t, 2355 + 30 * i, 740, 25.0, vertical=True) for i, t in enumerate(title_lines)]
    sp += [span(dwg, 2278, 1537, 17.4), span("4", 2338, 1537, 17.4), span("MRESEW-G1-D-EARTHWORK.DWG", 2268, 1155, 9.0, vertical=True)]
    sp += [span(t, x, y) for t, x, y in extra]
    return sp


def test_title_strip_and_drawing_numbers():
    pages = [strip_sheet(f"C{i:02d}", [f"SHEET {i}"]) for i in range(1, 7)]
    pages[0] = strip_sheet("G03", ["EXISTING GRADE CONTOURS"], [("ELEV E-39.07", 900, 600)])
    pages[1] = strip_sheet("RW02", ["RETAINING WALL SECTIONS", "& DETAILS"])
    common = common_text(pages)
    assert "EXAMPLE WATER SYSTEM IMPROVEMENTS" in common and "G03" not in common
    sid, title, tb = title_block(pages[0], W, H, common)
    assert (sid, title) == ("G03", "EXISTING GRADE CONTOURS")  # not "E-39.07" from the drawing
    assert classify(title) == "existing" and "WATER" not in tb  # the project name doesn't make it a water sheet
    sid, title, _ = title_block(pages[1], W, H, common)
    assert (sid, title) == ("RW02", "RETAINING WALL SECTIONS & DETAILS")
    assert common_text(pages[:3]) == set()  # too few sheets to tell boilerplate


@pytest.mark.parametrize("title,cat", [("STORM DETAILS", "details"), ("IRRIGATION ROAD RETAINING WALL", "wall"), ("TESC PLAN I", "demolition"),
                                       ("FINISHED GRADE CONTOURS", "grading"), ("ACCESS ROAD PLAN & PROFILE II", "roadway"),
                                       ("ALIGNMENT CONTROL PLAN I", "roadway"), ("STORM POND GRADING PLAN", "grading"), ("ROAD CROSS SECTIONS I", "cross_section")])
def test_categories(title, cat):
    assert classify(title) == cat


@pytest.mark.parametrize("parts,ft", [(["50'"], 50), (["H: 11", "20，", "V: 1", "10\""], 20), (["H: 1\"", "66", "20'，", "V: 1", "10\""], 20),
                                      (["1\" = 40'"], 40), (["SCALE 1:500"], 500 / 12), (["NOT TO SCALE"], None), (["1\" = 37'"], None), (["DETAIL 7"], None)])
def test_ocr_scale_readings(parts, ft):
    got = parse_scale_reading(parts)
    assert got == (pytest.approx(ft) if ft else None)


def test_plan_view_scale_wins():
    views = [dict(title="WEST ACCESS ROAD CENTERLINE PROFILE I", ft_per_in=20, text=""), dict(title="PLAN VIEW", ft_per_in=30, text=""),
             dict(title="EOG LT PROFILE", ft_per_in=20, text="")]
    assert pick_sheet_scale(views)["ft_per_in"] == 30
    assert pick_sheet_scale([dict(title="SECTION A", ft_per_in=5, text=""), dict(title="SECTION B", ft_per_in=10, text=""),
                             dict(title="SECTION C", ft_per_in=10, text="")])["ft_per_in"] == 10
    assert pick_sheet_scale([]) is None


@pytest.mark.parametrize("text,sid", [("CB #5", "CB-5"), ("CB NO. 12", "CB-12"), ("CB#3", "CB-3"), ("CB-4", "CB-4"), ("SMH 2", "SMH-2")])
def test_structure_ids(text, sid):
    m = STRUCT_ID.search(text)
    assert f"{m.group(1)}-{m.group(2)}" == sid


def test_tin_does_not_bridge_gaps():
    # two graded pads 600 ft apart: without a limit the triangulation fills the space between them
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, (60, 2))
    b = rng.uniform(0, 100, (60, 2)) + [700, 0]
    P = np.vstack([a, b])
    e = np.r_[np.full(60, 100.0), np.full(60, 120.0)]
    assert np.isfinite(TIN(P[:, 0], P[:, 1], e).sample(400, 50))
    t = TIN(P[:, 0], P[:, 1], e, max_edge=250)
    assert np.isnan(t.sample(400, 50)) and t.sample(50, 50) == pytest.approx(100) and t.sample(750, 50) == pytest.approx(120)


# ----- sheets that share only reference marks --------------------------------------------------

def sheet(page, sid, title, cat, marks, contours=(), structures=()):
    r = SheetReport(page=page, sheet_id=sid, title=title, category=cat, scale_ft_per_in=50, scale_text="1\" = 50'", has_text_layer=True,
                    width_pt=W, height_pt=H, line_segments=0, text_spans=0, marks=[list(m) for m in marks])
    for elev, role, pts in contours:
        r.candidates.append(Candidate("contour", str(elev), {"elev": elev, "dashed": False, "surface": role, "pts": pts}, page, pts[0][0], pts[0][1], [], str(elev), 0.8))
    for name, x, y in structures:
        r.candidates.append(Candidate("structure", name, {"inverts": [], "label_xy": [x, y], "symbol_xy": [x, y], "position_confidence": 0.9},
                                      page, x, y, [], name, 0.9))
    return r


TARGETS = [(447.3, 109.0, 9.6), (2007.5, 1318.7, 9.6)]
NORTH_ARROW = (2100.0, 200.0, 12.0)


def ramp(elev, role, x0, y0, n=12, step=40):
    """Contour runs across a pad: (elevation, role, points)."""
    return [(elev + i, role, [[x0 + i * step, y0 + j * 30.0] for j in range(8)]) for i in range(n)]


def test_same_view_needs_two_spread_marks_not_furniture():
    eg = sheet(0, "G03", "EXISTING GRADE CONTOURS", "existing", TARGETS + [NORTH_ARROW])
    fg = sheet(1, "G04", "FINISHED GRADE CONTOURS", "grading", TARGETS + [NORTH_ARROW])
    other = [sheet(2 + i, f"C{i}", "PLAN", "grading", [NORTH_ARROW]) for i in range(4)]
    furn = _furniture_marks([eg, fg] + other)
    assert (2100, 200, 12) in furn
    assert _same_view(eg, fg, 1.0, 1.0, furn)
    assert not _same_view(eg, fg, 1.0, 50 / 20, furn)  # different scale
    one = sheet(9, "G05", "", "grading", [TARGETS[0], NORTH_ARROW])
    assert not _same_view(eg, one, 1.0, 1.0, furn)  # one mark isn't enough
    assert not _same_view(other[0], other[1], 1.0, 1.0, furn)  # the shared north arrow means nothing


def test_model_is_built_on_the_earthwork_group():
    eg = sheet(0, "G03", "EXISTING GRADE CONTOURS", "existing", TARGETS, ramp(900, "existing", 300, 300))
    fg = sheet(1, "G04", "FINISHED GRADE CONTOURS", "grading", TARGETS, ramp(905, "proposed", 320, 300))
    # a separate group: two storm sheets sharing structures, somewhere else entirely
    st1 = sheet(2, "C07", "STORM PLAN", "storm", [], structures=[("CB-1", 100, 100), ("CB-2", 400, 120), ("CB-3", 700, 160)])
    st2 = sheet(3, "C08", "STORM PLAN", "storm", [], structures=[("CB-2", 150, 500), ("CB-3", 450, 540)])
    res = assemble([("d", r) for r in (eg, fg, st1, st2)])
    placed = {s.id for s in res["sheets"] if s.registered}
    assert placed == {"G03", "G04"}
    assert res["existing"] is not None and res["proposed"] is not None
    codes = [(i.code, i.sheet) for i in res["issues"]]
    assert ("placed_by_marks", "G04") in codes
    msgs = [i.message for i in res["issues"] if i.code == "unregistered_sheet"]
    assert msgs == ["C07, C08 line up with each other but are not tied to the model's sheets (G03, G04): no shared structures, matchline "
                    "or reference marks. Add control points to place them."]


def test_matched_points_join_the_earthwork_to_real_coordinates():
    eg = sheet(0, "G03", "EXISTING GRADE CONTOURS", "existing", TARGETS, ramp(900, "existing", 300, 300))
    fg = sheet(1, "G04", "FINISHED GRADE CONTOURS", "grading", TARGETS, ramp(905, "proposed", 320, 300))
    ac = sheet(2, "AC01", "ALIGNMENT CONTROL PLAN I", "roadway", [])
    ac.scale_ft_per_in = 20
    k_ac, k_g = 20 / 72, 50 / 72
    E0, N0 = 1_791_000.0, 131_000.0  # AC01's sheet origin in real coordinates (north up)

    def real(x, y):  # an AC01 page point in real E, N
        return E0 + x * k_ac, N0 - y * k_ac
    ac.control = [dict(x=x, y=y, e=real(x, y)[0], n=real(x, y)[1], text="", bbox=[]) for x, y in [(300, 300), (1500, 400), (900, 1200)]]
    # without a match: the earthwork group is the model, AC01 stands apart
    res = assemble([("d", r) for r in (eg, fg, ac)])
    assert {s.id for s in res["sheets"] if s.registered} == {"G03", "G04"} and not res["real_frame"]
    # G04's page point (p) shows the same spot as AC01's page point (q): G04 sits 200 ft east, 100 ft south of AC01's origin
    g04_to_ac = lambda x, y: ((x * k_g + 200) / k_ac, (y * k_g + 100) / k_ac)
    pairs = [[x, y, *g04_to_ac(x, y)] for x, y in [(400, 400), (1200, 900)]]
    res = assemble([("d", r) for r in (eg, fg, ac)], {"d:1": {"match": {"ref": "d:2", "pairs": pairs}}}, false_origin=[E0, N0])
    assert {s.id for s in res["sheets"] if s.registered} == {"G03", "G04", "AC01"} and res["real_frame"]
    assert ("placed_by_match", "G04") in [(i.code, i.sheet) for i in res["issues"]]
    # G03 (same view as G04) lands where AC01's coordinates say: project x = E - E0, z = -(N - N0)
    g = res["existing"]
    assert g.x0 == pytest.approx(300 * k_g + 200, abs=15) and g.z0 == pytest.approx(300 * k_g + 100, abs=15)
