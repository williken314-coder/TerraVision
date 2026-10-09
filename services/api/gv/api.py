"""TerraVision API (FastAPI).

Run:  uvicorn gv.api:app --reload --port 8000
Then open http://localhost:8000 for the app.
"""
from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import db as dbm
from .analysis import analyze
from .assistant import answer
from .demo import build_demo
from .edits import EditError, FieldEdit, apply_edit
from .geometry.surface import Grid
from .geometry.utilities import NotEnoughData, trench_quantities
from .ingest import landxml, raster, vision
from .ingest.pdf import page_count, render_page_png
from .models import Datum, Project
from .processing import is_landxml_doc, process_document, rebuild_project

MAX_UPLOAD_MB = int(os.environ.get("GV_MAX_UPLOAD_MB", "200"))
def _web_dir() -> Path:
    if os.environ.get("GV_WEB_DIR"):
        return Path(os.environ["GV_WEB_DIR"])
    here = Path(__file__).resolve()
    return here.parents[3] / "apps" / "web" if len(here.parents) > 3 else here.parent / "web"


WEB_DIR = _web_dir()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    dbm.init_db()
    with dbm.SessionLocal() as db:
        if not dbm.get_project_row(db, "maple-ridge"):
            demo = build_demo()
            db.add(dbm.ProjectRow(id=demo.id, name=demo.name, model=demo.model_dump(mode="json"), overrides={}))
            db.commit()
    yield


app = FastAPI(title="TerraVision API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("GV_CORS", "*").split(","), allow_methods=["*"], allow_headers=["*"])


def get_db():
    with dbm.SessionLocal() as s:
        yield s


def _load(db: Session, pid: str) -> tuple[dbm.ProjectRow, Project]:
    row = dbm.get_project_row(db, pid)
    if not row:
        raise HTTPException(404, f"No project {pid}")
    return row, Project(**row.model)


def _payload(prj: Project) -> dict:
    return {"project": prj.model_dump(mode="json"), "analysis": analyze(prj)}


# ----- status --------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "vision_available": vision.vision_available(), "opencv_available": raster.OPENCV,
            "database": dbm.DATABASE_URL.split("://")[0], "max_upload_mb": MAX_UPLOAD_MB}


# ----- projects -------------------------------------------------------------

class NewProject(BaseModel):
    name: str


@app.get("/api/projects")
def list_projects(db: Session = Depends(get_db)):
    rows = db.query(dbm.ProjectRow).order_by(dbm.ProjectRow.created_at).all()
    return [{"id": r.id, "name": r.name, "demo": bool(r.model.get("demo")), "documents": len(dbm.documents_for(db, r.id)),
             "updated_at": r.updated_at.isoformat() if r.updated_at else None} for r in rows]


@app.post("/api/projects")
def create_project(body: NewProject, db: Session = Depends(get_db)):
    pid = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")[:40] or "project"
    pid = f"{pid}-{dbm.new_id('p')[2:8]}"
    prj = Project(id=pid, name=body.name.strip()[:200])
    db.add(dbm.ProjectRow(id=pid, name=prj.name, model=prj.model_dump(mode="json"), overrides={}))
    db.commit()
    return _payload(prj)


@app.get("/api/projects/{pid}")
def get_project(pid: str, db: Session = Depends(get_db)):
    _, prj = _load(db, pid)
    return _payload(prj)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str, db: Session = Depends(get_db)):
    row, prj = _load(db, pid)
    if prj.demo:
        raise HTTPException(400, "The demo project can't be deleted")
    for d in dbm.documents_for(db, pid):
        Path(d.path).unlink(missing_ok=True)
        db.delete(d)
    db.delete(row)
    db.commit()
    return {"deleted": pid}


# ----- documents --------------------------------------------------------------

@app.post("/api/projects/{pid}/documents")
async def upload(pid: str, background: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    _, prj = _load(db, pid)
    if prj.demo:
        raise HTTPException(400, "Create a new project to upload plans; the demo project is read-only.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
    if data.startswith(b"%PDF"):
        try:
            pages, ext = page_count(data), "pdf"
        except Exception as ex:
            raise HTTPException(422, f"Could not open the PDF: {ex}")
    elif landxml.is_landxml(data):
        pages, ext = 0, "xml"  # surfaces are counted when the file is read
    else:
        raise HTTPException(415, "Upload a PDF plan set or a LandXML surface file (.xml). DXF and IFC import are on the roadmap.")
    did = dbm.new_id("doc")
    path = dbm.DATA_DIR / "uploads" / f"{did}.{ext}"
    path.write_bytes(data)
    db.add(dbm.DocumentRow(id=did, project_id=pid, filename=Path(file.filename or f"upload.{ext}").name[:300], path=str(path), pages=pages, status="queued"))
    db.commit()
    background.add_task(process_document, did)
    return {"id": did, "status": "queued", "pages": pages}


@app.get("/api/documents/{did}")
def document_status(did: str, db: Session = Depends(get_db)):
    d = db.get(dbm.DocumentRow, did)
    if not d:
        raise HTTPException(404, "No such document")
    reps = db.query(dbm.SheetReportRow).filter_by(document_id=did).order_by(dbm.SheetReportRow.page).all()
    if is_landxml_doc(d):
        return {"id": d.id, "project_id": d.project_id, "filename": d.filename, "kind": "landxml", "status": d.status, "error": d.error,
                "surfaces": [{k: r.report.get(k) for k in ("index", "name", "desc", "points", "faces", "dropped_faces", "role_guess", "units", "epsg", "extent", "elev_range")}
                             for r in reps]}
    return {"id": d.id, "project_id": d.project_id, "filename": d.filename, "kind": "pdf", "pages": d.pages, "status": d.status, "error": d.error,
            "vision_used": d.vision_used,
            "sheets": [{"page": r.page, "sheet_id": r.sheet_id, "category": r.category, "title": r.report.get("title"),
                        "scale": r.report.get("scale_text"), "items": len(r.report.get("candidates", [])),
                        "has_text_layer": r.report.get("has_text_layer"), "notes": r.report.get("notes", [])} for r in reps]}


@app.get("/api/documents/{did}/pages/{page}.png")
def page_image(did: str, page: int, dpi: int = Query(72, ge=36, le=300), bbox: Optional[str] = None, pad: float = 40,
               db: Session = Depends(get_db)):
    """Render a sheet, or a crop around an extracted value, so every number can be checked against the drawing."""
    d = db.get(dbm.DocumentRow, did)
    if not d:
        raise HTTPException(404, "No such document")
    if is_landxml_doc(d):
        raise HTTPException(404, "LandXML documents have no sheet images")
    clip = None
    if bbox:
        x0, y0, x1, y1 = (float(v) for v in bbox.split(","))
        clip = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
    png = render_page_png(Path(d.path).read_bytes(), page, dpi=dpi, clip=clip)
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


# ----- corrections, calibration, datum -------------------------------------------

@app.patch("/api/projects/{pid}/fields")
def edit_field(pid: str, e: FieldEdit, db: Session = Depends(get_db)):
    row, prj = _load(db, pid)
    try:
        msg = apply_edit(prj, e)
    except EditError as ex:
        raise HTTPException(422, str(ex))
    if not prj.demo:
        db.add(dbm.EditRow(project_id=pid, edit=e.model_dump()))
    row.model = prj.model_dump(mode="json")
    db.commit()
    return {"message": msg, **_payload(prj)}


class Calibration(BaseModel):
    document_id: str
    page: int
    scale_ft_per_in: Optional[float] = None
    # [[x_pt, y_pt, X_ft, Z_ft], ...]: sheet point -> project/state-plane coordinate (Z = -northing)
    control_points: Optional[list[list[float]]] = None


@app.put("/api/projects/{pid}/calibration")
def set_calibration(pid: str, c: Calibration, db: Session = Depends(get_db)):
    row, prj = _load(db, pid)
    if prj.demo:
        raise HTTPException(400, "The demo project is read-only")
    ov = dict(row.overrides or {})
    sheets = dict(ov.get("sheets", {}))
    key = f"{c.document_id}:{c.page}"
    entry = dict(sheets.get(key, {}))
    if c.scale_ft_per_in is not None:
        if c.scale_ft_per_in <= 0:
            raise HTTPException(422, "Scale must be positive")
        entry["scale_ft_per_in"] = c.scale_ft_per_in
    if c.control_points is not None:
        if len(c.control_points) < 2 or any(len(p) != 4 for p in c.control_points):
            raise HTTPException(422, "Give at least two control points as [x_pt, y_pt, X_ft, Z_ft]")
        entry["control_points"] = c.control_points
    sheets[key] = entry
    ov["sheets"] = sheets
    row.overrides = ov
    db.commit()
    return _payload(rebuild_project(db, pid))


@app.put("/api/projects/{pid}/datum")
def set_datum(pid: str, d: Datum, db: Session = Depends(get_db)):
    row, prj = _load(db, pid)
    if prj.demo:
        raise HTTPException(400, "The demo project is read-only")
    if d.false_origin is None:  # the datum form doesn't send it; keep the frame imported data was placed in
        d.false_origin = prj.datum.false_origin
    prj.datum = d
    row.model = prj.model_dump(mode="json")
    db.commit()
    return _payload(rebuild_project(db, pid))


class SurfaceRole(BaseModel):
    document_id: str
    index: int
    role: Literal["existing", "proposed", "ignore"]


@app.put("/api/projects/{pid}/surfaces")
def set_surface_role(pid: str, body: SurfaceRole, db: Session = Depends(get_db)):
    """Say whether an imported LandXML surface is the existing ground, the finished grade, or not used."""
    row, prj = _load(db, pid)
    if not any(s.document_id == body.document_id and s.index == body.index for s in prj.imported_surfaces):
        raise HTTPException(404, "No such imported surface in this project")
    ov = dict(row.overrides or {})
    ov["surfaces"] = {**ov.get("surfaces", {}), f"{body.document_id}:{body.index}": body.role}
    row.overrides = ov
    db.commit()
    return _payload(rebuild_project(db, pid))


# ----- engineering queries ------------------------------------------------------

@app.get("/api/projects/{pid}/pipes/{pipe_id}/trench")
def trench(pid: str, pipe_id: str, db: Session = Depends(get_db)):
    _, prj = _load(db, pid)
    p = prj.pipe(pipe_id)
    if not p:
        raise HTTPException(404, f"No pipe {pipe_id}")
    try:
        return trench_quantities(p, prj, Grid.from_model(prj.proposed) if prj.proposed else None)
    except NotEnoughData as ex:
        raise HTTPException(422, str(ex))


class Question(BaseModel):
    q: str
    selected: Optional[str] = None


@app.post("/api/projects/{pid}/ask")
def ask(pid: str, body: Question, db: Session = Depends(get_db)):
    _, prj = _load(db, pid)
    return answer(prj, body.q[:500], body.selected)


# ----- web app -------------------------------------------------------------------

if WEB_DIR.exists():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
