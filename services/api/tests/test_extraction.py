"""Extraction + assembly on the recorded layout of the sample plan set (no PyMuPDF needed)."""
import json
import math
from itertools import combinations
from pathlib import Path

from gv.analysis import analyze
from gv.demo import build_demo
from gv.ingest.assemble import assemble
from gv.ingest.pdf import Polyline, Seg, Symbol, TextSpan, build_report, parse_scale
from gv.models import Project

LAYOUT = Path(__file__).resolve().parents[1] / "samples" / "maple-ridge-sample-layout.json"


def reports():
    out = []
    for i, L in enumerate(json.loads(LAYOUT.read_text())):
        spans = [TextSpan(s["text"], s["x"], s["y"], s["size"], tuple(s["bbox"])) for s in L["spans"]]
        r = build_report(i, L["w"], L["h"], spans, [Seg(**s) for s in L["segs"]], [Symbol(**s) for s in L["symbols"]],
                         [Polyline(p["pts"], p["width"], p["dashed"]) for p in L["polys"]])
        out.append(("doc", r))
    return out


def test_scale_parsing():
    assert parse_scale("SCALE: 1\" = 40'")[0] == 40
    assert parse_scale("SCALE 1:500")[0] == 500 / 12
    assert parse_scale("NTS")[0] is None


def test_sheets_are_identified():
    got = [(r.sheet_id, r.category) for _, r in reports()]
    assert got == [("C-0.0", "cover"), ("C-3.0", "grading"), ("C-4.0", "storm"), ("C-4.1", "storm_profile"), ("C-5.0", "sanitary")]


def test_assembly_matches_the_drawn_plans():
    res = assemble(reports())
    truth = build_demo()
    T = {s.id: s for s in truth.structures}
    S = {s.id: s for s in res["structures"]}
    assert set(S) == {"CB-1", "CB-2", "CB-3", "CB-4", "CB-5", "CB-6", "HW-1", "SMH-1", "SMH-2", "SMH-3", "SMH-4"}
    # sheets drawn at different origins are registered into one frame
    for a, b in combinations(S, 2):
        d1 = math.hypot(S[a].x - S[b].x, S[a].z - S[b].z)
        d2 = math.hypot(T[a].x - T[b].x, T[a].z - T[b].z)
        assert abs(d1 - d2) < 0.5, (a, b)
    for sid, s in S.items():
        if sid == "HW-1":
            assert s.rim.value is None and s.rim.status.value == "missing"  # not on the plans, so not invented
        else:
            assert s.rim.value == T[sid].rim.value
    tp = {(p.from_id, p.to_id): p for p in truth.pipes}
    assert len(res["pipes"]) == 9
    for p in res["pipes"]:
        t = tp[(p.from_id, p.to_id)]
        assert (p.inv_up.value, p.inv_dn.value) == (t.inv_up.value, t.inv_dn.value)
        assert p.inv_up.source and p.inv_up.page is not None and p.inv_up.bbox  # traceable to the drawing
    hw = next(p for p in res["pipes"] if p.to_id == "HW-1")
    assert hw.dia_in.value is None and hw.dia_in.status.value == "missing"
    assert res["existing"] is not None and res["proposed"] is not None


def test_assembled_project_analysis():
    res = assemble(reports())
    prj = Project(id="t", name="t", sheets=res["sheets"], structures=res["structures"], pipes=res["pipes"],
                  existing=res["existing"], proposed=res["proposed"], issues=res["issues"])
    a = analyze(prj)
    assert a["earthwork"]["available"] and a["earthwork"]["cut_cy"] > a["earthwork"]["fill_cy"] > 0
    lat = next(p for p in prj.pipes if p.from_id == "CB-4")
    x = next(x for x in a["crossings"] if lat.id in (x["upper"], x["lower"]))
    assert round(x["clearance_ft"] * 12) == 30
