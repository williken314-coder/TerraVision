"""End-to-end: real PDF through PyMuPDF and the HTTP API. Needs the full requirements installed."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "maple-ridge-sample-plans.pdf"


def test_pdf_reader_on_sample_set():
    from gv.ingest.pdf import read_plan_set

    reps = read_plan_set(SAMPLE.read_bytes())
    assert [r.sheet_id for r in reps] == ["C-0.0", "C-3.0", "C-4.0", "C-4.1", "C-5.0"]
    storm = reps[2]
    assert storm.category == "storm" and storm.scale_ft_per_in == 40
    ids = {c.label for c in storm.candidates if c.kind == "structure"}
    assert {"CB-1", "CB-4", "CB-5", "HW-1"} <= ids
    assert len(storm.pipe_segments) >= 6
    assert sum(1 for c in reps[1].candidates if c.kind == "contour") > 20


@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    tmp = tempfile.mkdtemp()
    os.environ["GV_DATA_DIR"] = tmp
    os.environ.pop("DATABASE_URL", None)
    for m in [m for m in sys.modules if m.startswith("gv.")]:
        del sys.modules[m]
    api = importlib.import_module("gv.api")
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:
        yield c


def test_upload_and_interpret(client):
    assert client.get("/api/health").json()["ok"]
    demo = client.get("/api/projects/maple-ridge").json()
    assert demo["project"]["demo"] is True
    pid = client.post("/api/projects", json={"name": "Sample upload"}).json()["project"]["id"]
    with SAMPLE.open("rb") as f:
        r = client.post(f"/api/projects/{pid}/documents", files={"file": ("plans.pdf", f, "application/pdf")})
    assert r.status_code == 200, r.text
    did = r.json()["id"]
    st = client.get(f"/api/documents/{did}").json()
    assert st["status"] == "done", st
    data = client.get(f"/api/projects/{pid}").json()
    assert len(data["project"]["structures"]) == 11
    assert data["analysis"]["earthwork"]["available"]
    # headwalls have no rim: not an issue, not a review item
    assert not any(i["object_id"] == "HW-1" and i["code"] == "missing_value" for i in data["project"]["issues"])
    assert not any(r["object_id"] == "HW-1" and r["field"] == "rim" for r in data["analysis"]["review"])
    cb4 = next(s for s in data["project"]["structures"] if s["id"] == "CB-4")
    assert cb4["rim"]["value"] == 1045.76 and cb4["rim"]["source"]
    png = client.get(f"/api/documents/{did}/pages/2.png", params={"bbox": ",".join(map(str, cb4["rim"]["bbox"]))})
    assert png.headers["content-type"] == "image/png"
    item = data["analysis"]["review"][0]
    r = client.patch(f"/api/projects/{pid}/fields", json={"object_id": item["object_id"], "field": item["field"], "action": "confirm"})
    assert r.status_code in (200, 422)
    ans = client.post(f"/api/projects/{pid}/ask", json={"q": "What's the invert elevation of CB-4?"}).json()
    assert "1041.58" in ans["text"] and ans["sources"]


def test_rejects_non_pdf(client):
    pid = client.post("/api/projects", json={"name": "x"}).json()["project"]["id"]
    r = client.post(f"/api/projects/{pid}/documents", files={"file": ("a.dwg", b"AC1032", "application/octet-stream")})
    assert r.status_code == 415
