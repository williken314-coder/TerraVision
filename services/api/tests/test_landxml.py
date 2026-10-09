"""LandXML surface import: parser, TIN gridding, and the upload -> register -> cross-check flow."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gv.geometry.surface import Grid, TriMesh
from gv.ingest.landxml import LandXMLError, grid_surface, guess_role, is_landxml, parse_landxml

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
LANDXML = SAMPLES / "maple-ridge-surfaces.xml"
E0, N0 = 6_052_400.0, 2_118_600.0  # scripts/make_sample_landxml.py


def lx(surfaces: str, units: str = '<Imperial linearUnit="USSurveyFoot"/>') -> bytes:
    return (f'<?xml version="1.0"?><LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">'
            f'<Units>{units}</Units><Surfaces>{surfaces}</Surfaces></LandXML>').encode()


# an L-shaped surface: 3 unit squares, the top-right one missing. Re-triangulating would fill the notch.
L_SHAPE = ('<Surface name="Proposed pad"><Definition surfType="TIN"><Pnts>'
           '<P id="a">0 0 100</P><P id="b">0 10 100</P><P id="c">0 20 100</P>'
           '<P id="d">10 0 110</P><P id="e">10 10 110</P><P id="f">10 20 110</P>'
           '<P id="g">20 0 120</P><P id="h">20 10 120</P>'
           '</Pnts><Faces><F>a b e</F><F>a e d</F><F>b c f</F><F>b f e</F><F>d e h</F><F>d h g</F><F i="1">e f h</F></Faces>'
           '</Definition></Surface>')


def test_sample_file():
    f = parse_landxml(LANDXML.read_bytes())
    assert f.units == "USSurveyFoot" and f.epsg == 2227
    assert [(s.name, s.role_guess) for s in f.surfaces] == [("EG", "existing"), ("FG", "proposed")]
    eg = f.surfaces[0]
    assert len(eg.z) == 2501 and len(eg.faces) == 2 * 50 * 48
    # LandXML points are northing, easting, elevation
    assert eg.n.max() == pytest.approx(N0 + 200) and eg.e.min() == pytest.approx(E0 - 300)


def test_faces_are_honoured_not_retriangulated():
    s = parse_landxml(lx(L_SHAPE)).surfaces[0]
    assert len(s.faces) == 6 and s.dropped_faces == 1  # the invisible face is outside the boundary
    g = grid_surface(s, None, 0.0, "pad", max_cells=20)
    G = Grid.from_model(g)
    # x = easting, z = -northing; elevation rises 1 ft per ft of northing
    assert G.sample(5, -5) == pytest.approx(105)
    assert G.sample(15, -5) == pytest.approx(105)
    assert G.sample(5, -15) == pytest.approx(115)
    assert np.isnan(G.sample(15, -15))  # inside the notch: no data, not interpolated across


def test_trimesh_matches_plane():
    rng = np.random.default_rng(1)
    x, z = rng.uniform(0, 100, 400), rng.uniform(0, 100, 400)
    from scipy.spatial import Delaunay
    tri = Delaunay(np.column_stack([x, z]))
    g = TriMesh(x, z, 3 * x - 2 * z + 50, tri.simplices).to_grid(0, 0, 2, 50, 50, chunk=500)
    X, Z = g.coords()
    ok = np.isfinite(g.elev)
    assert ok.mean() > 0.9
    assert np.allclose(g.elev[ok], (3 * X - 2 * Z + 50)[ok])


def test_units_and_false_origin():
    s = parse_landxml(lx(L_SHAPE, '<Metric linearUnit="meter"/>')).surfaces[0]
    assert s.z.max() == pytest.approx(120 / 0.3048)
    g = grid_surface(parse_landxml(lx(L_SHAPE)).surfaces[0], [-1000.0, 500.0], 1.5, "pad", max_cells=20)
    assert g.x0 == 1000.0 and g.z0 == pytest.approx(-(20 - 500))  # x = E - E0, z = -(N - N0)
    assert max(v for v in g.elev if v is not None) == pytest.approx(121.5)  # datum offset applied


@pytest.mark.parametrize("name,desc,role", [
    ("EG", "", "existing"), ("Existing Ground 2025", "", "existing"), ("TOPO", "", "existing"), ("OG", "", "existing"),
    ("FG", "", "proposed"), ("Finished Grade", "", "proposed"), ("Proposed", "", "proposed"), ("Surface1", "design grading", "proposed"),
    ("Corridor - Datum", "", None), ("Subgrade", "", None), ("EG vs FG volume", "", None), ("Surface1", "", None), ("BEGIN", "", None),
])
def test_role_guess(name, desc, role):
    assert guess_role(name, desc) == role


def test_rejects_bad_files():
    assert is_landxml(LANDXML.read_bytes()[:4096]) and not is_landxml(b"%PDF-1.7")
    with pytest.raises(LandXMLError):
        parse_landxml(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><LandXML/>')
    with pytest.raises(LandXMLError):
        parse_landxml(b"<Other/>")
    with pytest.raises(LandXMLError, match="No TIN surfaces"):
        parse_landxml(lx(""))
    with pytest.raises(LandXMLError, match="unit"):
        parse_landxml(lx(L_SHAPE, '<Imperial linearUnit="cubit"/>'))


# ----- through the API -----------------------------------------------------------

@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("pymupdf")
    tmp = tempfile.mkdtemp()
    os.environ["GV_DATA_DIR"] = tmp
    os.environ.pop("DATABASE_URL", None)
    for m in [m for m in sys.modules if m.startswith("gv.")]:
        del sys.modules[m]
    api = importlib.import_module("gv.api")
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:
        yield c


def upload(client, pid, path, ctype):
    with path.open("rb") as f:
        r = client.post(f"/api/projects/{pid}/documents", files={"file": (path.name, f, ctype)})
    assert r.status_code == 200, r.text
    st = client.get(f"/api/documents/{r.json()['id']}").json()
    assert st["status"] == "done", st
    return st


def test_landxml_only_project(client):
    pid = client.post("/api/projects", json={"name": "Surfaces only"}).json()["project"]["id"]
    st = upload(client, pid, LANDXML, "application/xml")
    assert st["kind"] == "landxml" and [s["name"] for s in st["surfaces"]] == ["EG", "FG"]
    data = client.get(f"/api/projects/{pid}").json()
    prj = data["project"]
    assert prj["datum"]["false_origin"] == [6052000.0, 2118000.0] and prj["datum"]["epsg"] == 2227
    assert prj["existing"]["source"] == "LandXML: EG" and prj["proposed"]["source"] == "LandXML: FG"
    assert prj["existing"]["x0"] == pytest.approx(E0 - 300 - 6052000)  # small project coordinates
    assert data["analysis"]["earthwork"]["available"]
    assert not any(i["code"] == "frames_not_linked" for i in prj["issues"])  # nothing else to line up with
    assert client.get(f"/api/documents/{st['id']}/pages/0.png").status_code == 404
    # saving the datum form (which doesn't send false_origin) keeps the frame
    d = dict(prj["datum"], false_origin=None, vertical="NAVD88")
    assert client.put(f"/api/projects/{pid}/datum", json=d).json()["project"]["datum"]["false_origin"] == [6052000.0, 2118000.0]


def test_landxml_with_plans_and_control_points(client):
    from gv.demo import build_demo
    from gv.ingest.pdf import read_plan_set

    pid = client.post("/api/projects", json={"name": "Plans + surfaces"}).json()["project"]["id"]
    pdf = upload(client, pid, SAMPLES / "maple-ridge-sample-plans.pdf", "application/pdf")
    before = client.get(f"/api/projects/{pid}").json()["project"]
    eg_from_plans = before["existing"]
    upload(client, pid, LANDXML, "text/xml")
    prj = client.get(f"/api/projects/{pid}").json()["project"]
    assert prj["existing"]["source"] == "LandXML: EG"
    codes = {i["code"] for i in prj["issues"]}
    assert "frames_not_linked" in codes and "surface_superseded" in codes

    # register every plan sheet with control points at three structures' state-plane coordinates
    demo = {s.id: s for s in build_demo().structures}
    for rep in read_plan_set((SAMPLES / "maple-ridge-sample-plans.pdf").read_bytes()):
        sym = {c.label: c.values["symbol_xy"] for c in rep.candidates if c.kind == "structure" and c.values.get("symbol_xy")}
        pts = [[*sym[s], E0 + demo[s].x, -(N0 - demo[s].z)] for s in ("CB-1", "CB-4", "CB-6") if s in sym]
        if len(pts) >= 2:
            r = client.put(f"/api/projects/{pid}/calibration", json={"document_id": pdf["id"], "page": rep.page, "control_points": pts})
            assert r.status_code == 200, r.text
    data = client.get(f"/api/projects/{pid}").json()
    prj = data["project"]
    assert prj["datum"]["horizontal"] == "Registered to control points (EPSG:2227)"
    cb4 = next(s for s in prj["structures"] if s["id"] == "CB-4")
    assert cb4["x"] == pytest.approx(demo["CB-4"].x + E0 - 6052000, abs=1.0)
    assert cb4["z"] == pytest.approx(demo["CB-4"].z - (N0 - 2118000), abs=1.0)
    issues = {i["code"]: i["message"] for i in prj["issues"]}
    assert "frames_not_linked" not in issues
    # the plan contours and the LandXML surface describe the same ground
    assert "surface_agreement" in issues, issues
    assert data["analysis"]["earthwork"]["available"]
    assert eg_from_plans["source"] == "from uploaded plans"


def test_surface_roles(client):
    pid = client.post("/api/projects", json={"name": "Roles"}).json()["project"]["id"]
    body = lx(L_SHAPE.replace("Proposed pad", "Surface1")).decode()
    r = client.post(f"/api/projects/{pid}/documents", files={"file": ("s.xml", body.encode(), "application/xml")})
    did = r.json()["id"]
    prj = client.get(f"/api/projects/{pid}").json()["project"]
    assert prj["proposed"] is None and prj["imported_surfaces"][0]["role"] is None
    assert any(i["code"] == "surface_role_unknown" for i in prj["issues"])
    r = client.put(f"/api/projects/{pid}/surfaces", json={"document_id": did, "index": 0, "role": "proposed"})
    prj = r.json()["project"]
    assert prj["proposed"]["source"] == "LandXML: Surface1"
    s = prj["imported_surfaces"][0]
    assert s["role"] == "proposed" and s["role_source"] == "user" and s["used"]
    assert not any(i["code"] == "surface_role_unknown" for i in prj["issues"])
    r = client.put(f"/api/projects/{pid}/surfaces", json={"document_id": did, "index": 0, "role": "ignore"})
    assert r.json()["project"]["proposed"] is None
    assert client.put(f"/api/projects/{pid}/surfaces", json={"document_id": did, "index": 5, "role": "existing"}).status_code == 404


def test_bad_landxml_upload_fails_visibly(client):
    pid = client.post("/api/projects", json={"name": "Bad"}).json()["project"]["id"]
    r = client.post(f"/api/projects/{pid}/documents", files={"file": ("x.xml", lx(""), "application/xml")})
    st = client.get(f"/api/documents/{r.json()['id']}").json()
    assert st["status"] == "failed" and "No TIN surfaces" in st["error"]
