"""Document processing pipeline and project re-assembly."""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from . import db as dbm
from .edits import EditError, FieldEdit, apply_edit
from .ingest import raster, vision
from .ingest.assemble import assemble
from .ingest.pdf import Candidate, SheetReport, read_plan_set, render_page_png
from .models import Document, Issue, Project

log = logging.getLogger("gv.processing")


def _vision_candidates(data: dict, page: int) -> list[Candidate]:
    out = []
    for s in data.get("structures", []) or []:
        if not s.get("id"):
            continue
        vals = {"inverts": [], "label_xy": None}
        if s.get("rim") is not None:
            vals.update(rim=float(s["rim"]), rim_confidence=0.6, rim_text="vision reading")
        for inv in s.get("inverts", []) or []:
            if inv.get("elev") is not None:
                vals["inverts"].append({"dir": inv.get("dir"), "side": None, "elev": float(inv["elev"]), "confidence": 0.6, "text": "vision reading", "bbox": None})
        out.append(Candidate("structure", s["id"].upper().replace(" ", ""), vals, page, 0.0, 0.0, [], "vision reading", 0.6))
    return out


def process_document(doc_id: str) -> None:
    """Extract every page of an uploaded PDF and rebuild its project. Runs in a background task."""
    with dbm.SessionLocal() as db:
        doc = db.get(dbm.DocumentRow, doc_id)
        if not doc:
            return
        doc.status = "processing"
        db.commit()
        try:
            data = Path(doc.path).read_bytes()
            reports = read_plan_set(data)
            use_vision = vision.vision_available()
            for r in reports:
                if not r.has_text_layer:
                    png = render_page_png(data, r.page, dpi=150)
                    ra = raster.analyze_png(png)
                    if ra.get("available"):
                        r.notes.append(f"Raster scan found {ra['line_count']} line runs and {ra['circle_count']} round symbols.")
                    if use_vision:
                        v = vision.read_sheet_image(png)
                        r.candidates = _vision_candidates(v, r.page)
                        r.sheet_id = r.sheet_id or v.get("sheet_id")
                        r.category = v.get("category") or r.category
                        r.notes.append("Values on this scanned sheet were read by the vision model; they have no plan location.")
                        doc.vision_used = True
                    else:
                        r.notes.append("Vision reader is off (set ANTHROPIC_API_KEY), so nothing was read from this scanned sheet.")
                elif use_vision and os.environ.get("GV_VISION_CROSSCHECK") == "1":
                    v = vision.read_sheet_image(render_page_png(data, r.page, dpi=110))
                    merged = vision.merge([c.__dict__ for c in r.candidates], v)
                    r.candidates = [Candidate(**c) for c in merged]
                    doc.vision_used = True
            db.execute(delete(dbm.SheetReportRow).where(dbm.SheetReportRow.document_id == doc_id))
            for r in reports:
                db.add(dbm.SheetReportRow(document_id=doc_id, page=r.page, sheet_id=r.sheet_id, category=r.category, report=r.as_dict()))
            doc.pages = len(reports)
            doc.status = "done"
            db.commit()
            rebuild_project(db, doc.project_id)
        except Exception as ex:  # report, never hide
            log.exception("processing failed")
            doc.status, doc.error = "failed", f"{type(ex).__name__}: {ex}"
            db.commit()
            traceback.print_exc()


def rebuild_project(db: Session, pid: str) -> Project:
    row = dbm.get_project_row(db, pid)
    prj = Project(**row.model)
    docs = dbm.documents_for(db, pid)
    prj.documents = [Document(id=d.id, filename=d.filename, pages=d.pages, status=d.status, error=d.error,
                              uploaded_at=d.created_at.isoformat(), vision_used=d.vision_used) for d in docs]
    if prj.demo:
        row.model = prj.model_dump(mode="json")
        db.commit()
        return prj
    overrides = row.overrides or {}
    reports = [(doc_id, SheetReport.from_dict(rep)) for doc_id, rep in dbm.reports_for(db, pid)]
    result = assemble(reports, overrides.get("sheets", {}), prj.datum.elevation_offset_ft)
    prj.sheets, prj.structures, prj.pipes = result["sheets"], result["structures"], result["pipes"]
    prj.existing, prj.proposed, prj.issues = result["existing"], result["proposed"], result["issues"]
    if any(c["registered"] and c["transform"] and c["transform"]["n"] >= 2 and c["transform"]["rms"] >= 0 and overrides.get("sheets", {}).get(k, {}).get("control_points")
           for k, c in result["calibrations"].items()):
        prj.datum.horizontal = f"Registered to control points{f' (EPSG:{prj.datum.epsg})' if prj.datum.epsg else ''}"
    for e in dbm.edits_for(db, pid):
        try:
            apply_edit(prj, FieldEdit(**e))
        except EditError as ex:
            prj.issues.append(Issue(severity="info", code="edit_not_reapplied", object_id=e.get("object_id"),
                                    message=f"An earlier correction to {e.get('object_id')} {e.get('field')} could not be reapplied: {ex}"))
    row.model = prj.model_dump(mode="json")
    row.overrides = overrides
    db.commit()
    return prj
