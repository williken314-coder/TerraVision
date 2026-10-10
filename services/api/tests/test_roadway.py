"""Roadway sheets: linework chaining, centerline from stationing, edges, sheet linking, and the API."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gv.ingest.pdf import PAVE_LAYER, _inches
from gv.ingest.roadway import chain_segments, edge_label_kind, fit_stationing, read_roadway_sheet

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "maple-ridge-sample-plans.pdf"
K = 40 / 72  # 1" = 40', ft per point
SIZE_REF = 26.0


# ----- a synthetic roadway sheet, drawn the way CAD plots one ----------------------------

def road(scale_k=K):
    st = np.arange(0, 551, 2.0)
    X, Z = -275 + st, 30 * np.sin(st / 550 * np.pi * 1.2)
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(X), np.diff(Z)))])

    def off(d):
        tx, tz = np.gradient(X), np.gradient(Z)
        n = np.hypot(tx, tz)
        return (X - tz / n * d) / scale_k + 700, (Z + tx / n * d) / scale_k + 600

    segs = []

    def solid(d, w=0.6):
        x, y = off(d)
        segs.extend((x[i], y[i], x[i + 1], y[i + 1], w, False) for i in range(len(x) - 1))

    def dashes(d, w=0.35, on=12.0, gap=3.0):
        x, y = off(d)
        T = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        t = 0.0
        while t < T[-1]:
            a, b = t, min(t + on, T[-1])
            segs.append((np.interp(a, T, x), np.interp(a, T, y), np.interp(b, T, x), np.interp(b, T, y), w, False))
            t += on + gap

    def station_at(s, d=6.0):
        x, y = off(d)
        return float(np.interp(s, arc, x)), float(np.interp(s, arc, y))

    return segs, solid, dashes, station_at, off, arc


def standard_sheet(scale_k=K, labels=True):
    segs, solid, dashes, station_at, off, arc = road(scale_k)
    dashes(0)  # centerline, one dash at a time
    for d in (-14, 14, -15, 15):
        solid(d)
    for d in (-18, 18, -23, 23):
        solid(d, 0.45)
    for d in (-30, 30):
        dashes(d, 0.35, 20, 4)
    rng = np.random.default_rng(0)
    for _ in range(200):  # hatching and symbol noise
        x0, y0 = rng.uniform(300, 1500), rng.uniform(300, 1000)
        segs.append((x0, y0, x0 + 5, y0 + 5, 0.2, False))
    stations = [(*station_at(s), float(s), i) for i, s in enumerate(range(0, 551, 50))]
    labs = []
    if labels:
        for txt, d, s in [("EP", -14, 100), ("BC", 15, 200), ("5' SIDEWALK", -20.5, 250), ("R/W", 30, 400)]:
            x, y = station_at(s, d)
            labs.append(dict(x=x, y=y + 1, kind=edge_label_kind(txt), size=6, text=txt))
    return chain_segments(segs), stations, labs


def by_kind(res):
    return {(e["side"], e["kind"]): round(e["offset"], 1) for e in res["edges"] if e["kind"] != "other"}


EXPECTED = {("L", "pavement"): 14.0, ("L", "curb_back"): 15.0, ("L", "walk_inner"): 18.0, ("L", "walk_outer"): 23.0,
            ("R", "pavement"): 14.0, ("R", "curb_back"): 15.0, ("R", "walk_inner"): 18.0, ("R", "walk_outer"): 23.0}


def test_dashes_rejoin_into_lines():
    lines, _, _ = standard_sheet()
    long_ = [l for l in lines if l["len"] > 900]
    assert len(long_) == 11  # centerline + 8 edges + 2 R/W, each one line again
    assert sum(l["dashed"] for l in long_) == 3
    assert all(l["len"] < 60 for l in lines if l not in long_) or len(lines) == 11


def test_chaining_keeps_parallel_and_different_lines_apart():
    # parallel hatch strokes, and a thin line continuing a thick one, must not join
    segs = [(0, y, 100, y, 0.3, False) for y in range(0, 50, 5)]
    segs += [(200, 0, 300, 0, 2.0, False), (300, 0, 400, 0, 0.3, False)]
    out = chain_segments(segs, min_len=50)
    assert len(out) == 12 and all(o["len"] == pytest.approx(100) for o in out)
    # a dashed line with a sharp turn across a gap is two lines, not one
    out = chain_segments([(0, 0, 80, 0, 0.3, False), (84, 0, 84, 80, 0.3, False)], min_len=50)
    assert len(out) == 2


def test_centerline_and_edges_with_labels():
    lines, stations, labs = standard_sheet()
    r = read_roadway_sheet(lines, stations, labs, K, SIZE_REF)
    assert lines[r["line"]]["dashed"]
    assert r["fit"]["rms"] < 0.1 and len(r["fit"]["inliers"]) == len(stations)
    assert r["measured_ft_per_pt"] == pytest.approx(K, rel=0.002)
    assert by_kind(r) == {**EXPECTED, ("R", "row"): 30.0}
    named = {(e["side"], e["kind"]): e["named_by"] for e in r["edges"]}
    assert named[("L", "pavement")] == "label" and named[("R", "pavement")] == "order"
    # two labels voted onto the BC line a foot from EP: split by outward order, not lost
    assert named[("R", "curb_back")] == "label"


def test_edges_by_order_without_labels():
    lines, stations, _ = standard_sheet()
    r = read_roadway_sheet(lines, stations, [], K, SIZE_REF)
    assert by_kind(r) == EXPECTED  # dashed R/W lines are never named from order alone
    assert all(e["named_by"] == "order" for e in r["edges"] if e["kind"] != "other")


def test_scale_measured_from_stationing():
    lines, stations, _ = standard_sheet()
    r = read_roadway_sheet(lines, stations, [], None, SIZE_REF)  # no scale on the sheet
    assert r["scale_from_stationing"] and r["measured_ft_per_pt"] == pytest.approx(K, rel=0.002)
    assert by_kind(r) == EXPECTED
    # a half-size print: drawn at 1" = 80' but the note still says 40'. The fit measures the real scale.
    lines, stations, _ = standard_sheet(scale_k=2 * K)
    r = read_roadway_sheet(lines, stations, [], K, SIZE_REF)
    assert r["measured_ft_per_pt"] == pytest.approx(2 * K, rel=0.002)


def test_profile_grid_is_not_a_centerline():
    # a straight station axis with labels every 100 ft and elevation grid lines above it only
    segs = [(100, 800, 1300, 800, 0.5, False)] + [(100, 800 - 30 * i, 1300, 800 - 30 * i, 0.25, False) for i in range(1, 8)]
    stations = [(100 + i * 180, 812, i * 100.0, i) for i in range(7)]
    assert fit_stationing(np.array([[100, 800], [1300, 800]]), stations, 2.2 * SIZE_REF) is not None  # it does fit stationing...
    assert read_roadway_sheet(chain_segments(segs), stations, [], K, SIZE_REF) is None  # ...but has no road on both sides


def test_labels_from_other_alignments_are_outliers():
    lines, stations, _ = standard_sheet()
    x, y = stations[3][0], stations[3][1]
    r = read_roadway_sheet(lines, stations + [(x + 5, y, 2250.0, 99)], [], K, SIZE_REF)
    assert 99 not in r["labels_used"] and r["fit"]["rms"] < 0.1


@pytest.mark.parametrize("text,kind", [("EP", "pavement"), ("EDGE OF PAVEMENT", "pavement"), ("BACK OF CURB", "curb_back"), ("BC", "curb_back"),
                                       ("5' SIDEWALK", "walk"), ("R/W", "row"), ("PROPERTY LINE", "row"), ("FL", "curb_face"),
                                       ("BC STA 2+45.10", None), ("PC STA 1+00", None), ("STORM DRAIN", None)])
def test_edge_labels(text, kind):
    assert edge_label_kind(text) == kind


@pytest.mark.parametrize("text,inches,material", [('1-1/2" HMA SURFACE COURSE', 1.5, "HMA SURFACE COURSE"), ('2 1/2" HMA BASE', 2.5, "HMA BASE"),
                                                  ('8" AGGREGATE BASE', 8.0, "AGGREGATE BASE"), ("6 IN. CLASS 2 AGGREGATE BASE", 6.0, "CLASS 2 AGGREGATE BASE"),
                                                  ('4" PCC', 4.0, "PCC"), ('1½" A.C.', 1.5, "A.C."), ('REMOVE 15" RCP', None, None)])
def test_pavement_layers(text, inches, material):
    m = PAVE_LAYER.search(text)
    if inches is None:
        assert m is None
    else:
        assert _inches(m.group(1)) == inches and m.group(2).upper() == material


# ----- the sample plan set: two roadway sheets joined at a matchline --------------------------

@pytest.fixture(scope="module")
def sample_reports():
    pytest.importorskip("pymupdf")
    from gv.ingest.pdf import read_plan_set

    return read_plan_set(SAMPLE.read_bytes())


def test_sample_roadway(sample_reports):
    from gv.ingest.assemble import assemble
    from gv.ingest.calibration import fit_similarity
    from gv.geometry.alignment import Alignment
    from gv.demo import build_demo

    res = assemble([("d", r) for r in sample_reports])
    al, cr = res["alignment"], res["corridor"]
    assert al.source == "C-6.0, C-6.1" and al.points[0].s == pytest.approx(0, abs=0.5) and al.points[-1].s > 550
    assert any(i.code == "placed_by_stationing" and i.sheet == "C-6.1" for i in res["issues"])
    assert not any(i.code in ("unregistered_sheet", "scale_vs_stationing") for i in res["issues"])
    got = {e.id: e.offset.value for e in cr.edges}
    assert got == pytest.approx({"EP-LT": 14, "EP-RT": 14, "BC-LT": 15, "BC-RT": 15, "SW-IN-LT": 18, "SW-OUT-LT": 23,
                                 "SW-IN-RT": 18, "SW-OUT-RT": 23, "RW-RT": 30}, abs=0.05)
    assert all(e.offset.status.value == "ai" and e.offset.source for e in cr.edges)
    assert cr.pavement_section.value == "1.5 in hma surface course / 2.5 in hma base course / 8 in aggregate base"
    assert cr.pavement_section.source == "C-6.1"
    # where the centerline really is: the demo centerline mapped through the structures
    demo = build_demo()
    S = {s.id: s for s in demo.structures}
    sim = fit_similarity(np.array([[S[s.id].x, S[s.id].z] for s in res["structures"]]), np.array([[s.x, s.z] for s in res["structures"]]))
    x, z = sim.apply([p.x for p in demo.alignment.points], [p.z for p in demo.alignment.points])
    _, off = Alignment([p.s for p in demo.alignment.points], x, z).to_so([p.x for p in al.points], [p.z for p in al.points])
    on_c60 = np.array([p.s for p in al.points]) <= 300
    assert np.abs(off[on_c60]).max() < 0.05  # placed by structures
    assert np.abs(off).max() < 0.75  # C-6.1 placed through the matchline only
    # profile grid labels never become contours on roadway sheets
    assert not any(c.kind == "contour" for r in sample_reports if r.category == "roadway" for c in r.candidates)


def test_no_matchline_no_link(sample_reports):
    from dataclasses import replace

    from gv.ingest.assemble import assemble

    no_ml = [replace(r, candidates=[c for c in r.candidates if c.kind != "matchline"]) for r in sample_reports]
    res = assemble([("d", r) for r in no_ml])
    assert any(i.code == "unregistered_sheet" and i.sheet == "C-6.1" for i in res["issues"])
    assert res["alignment"].source == "C-6.0"
    # a matchline naming some other sheet doesn't link these two either
    wrong = []
    for r in sample_reports:
        cs = [replace(c, values={**c.values, "sheet_ref": "C-9.9"}) if c.kind == "matchline" else c for c in r.candidates]
        wrong.append(replace(r, candidates=cs))
    res = assemble([("d", r) for r in wrong])
    assert res["alignment"].source == "C-6.0"


# ----- through the API --------------------------------------------------------------------------

@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("pymupdf")
    os.environ["GV_DATA_DIR"] = tempfile.mkdtemp()
    os.environ.pop("DATABASE_URL", None)
    for m in [m for m in sys.modules if m.startswith("gv.")]:
        del sys.modules[m]
    api = importlib.import_module("gv.api")
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:
        yield c


def test_roadway_through_api(client):
    pid = client.post("/api/projects", json={"name": "Road"}).json()["project"]["id"]
    with SAMPLE.open("rb") as f:
        client.post(f"/api/projects/{pid}/documents", files={"file": ("plans.pdf", f, "application/pdf")})
    data = client.get(f"/api/projects/{pid}").json()
    prj = data["project"]
    assert prj["alignment"]["profile_source"].startswith("Finished-grade surface")
    assert all(p["elev"] is not None for p in prj["alignment"]["points"])
    assert prj["corridor"]["crown_slope_assumed"] and prj["corridor"]["curb_width"] is None
    review = {(r["object_id"], r["field"]) for r in data["analysis"]["review"]}
    assert ("EP-RT", "offset") in review and ("SECTION", "pavement_section") in review
    # confirm one edge, correct another; both survive a rebuild
    client.patch(f"/api/projects/{pid}/fields", json={"object_id": "EP-RT", "field": "offset", "action": "confirm"})
    r = client.patch(f"/api/projects/{pid}/fields", json={"object_id": "SW-OUT-RT", "field": "offset", "action": "set", "value": 24})
    assert r.status_code == 200, r.text
    assert client.patch(f"/api/projects/{pid}/fields", json={"object_id": "SW-OUT-RT", "field": "offset", "action": "set", "value": -3}).status_code == 422
    r = client.put(f"/api/projects/{pid}/datum", json={"vertical": "NAVD88", "elevation_offset_ft": 0, "units": "ft"})
    edges = {e["id"]: e for e in r.json()["project"]["corridor"]["edges"]}
    assert edges["EP-RT"]["offset"]["status"] == "user" and edges["SW-OUT-RT"]["offset"]["value"] == 24
    ans = client.post(f"/api/projects/{pid}/ask", json={"q": "Show me the finished roadway"}).json()
    assert "14.0 ft LT" in ans["text"] and "hma" in ans["text"] and "C-6.0" in ans["sources"] and "C-7.0" not in ans["sources"]
