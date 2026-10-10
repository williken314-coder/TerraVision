"""Document processing pipeline and project re-assembly."""
from __future__ import annotations

import logging
import math
import os
import threading
import traceback
from pathlib import Path

import numpy as np
from sqlalchemy import delete
from sqlalchemy.orm import Session

from . import db as dbm
from .edits import EditError, FieldEdit, apply_edit
from .geometry.surface import Grid
from .ingest import landxml, raster, vision
from .ingest.assemble import assemble
from .ingest.pdf import Candidate, SheetReport, read_plan_set, render_page_png
from .models import Document, ImportedSurface, Issue, Project

log = logging.getLogger("gv.processing")
PROGRESS: dict[str, tuple[int, int]] = {}  # document id -> (sheets read, sheets), while it is being read


def resume_unfinished() -> list[str]:
    """Uploads left queued or half-read by a restart (a crash, a redeploy) are read again from the start."""
    with dbm.SessionLocal() as db:
        ids = [d.id for d in db.query(dbm.DocumentRow).filter(dbm.DocumentRow.status.in_(("queued", "processing"))).all()]
    for i in ids:
        threading.Thread(target=process_document, args=(i,), daemon=True, name=f"read-{i}").start()
    return ids


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
            if is_landxml_doc(doc):
                _process_landxml(db, doc, data)
                rebuild_project(db, doc.project_id)
                return
            reports = read_plan_set(data, on_page=lambda i, n: PROGRESS.__setitem__(doc_id, (i, n)))
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
            PROGRESS.pop(doc_id, None)
            rebuild_project(db, doc.project_id)
        except Exception as ex:  # report, never hide
            PROGRESS.pop(doc_id, None)
            log.exception("processing failed")
            doc.status, doc.error = "failed", f"{type(ex).__name__}: {ex}"
            db.commit()
            traceback.print_exc()


def is_landxml_doc(doc: dbm.DocumentRow) -> bool:
    return Path(doc.path).suffix.lower() == ".xml"


def _process_landxml(db: Session, doc: dbm.DocumentRow, data: bytes) -> None:
    lx = landxml.parse_landxml(data)
    db.execute(delete(dbm.SheetReportRow).where(dbm.SheetReportRow.document_id == doc.id))
    for srf in lx.surfaces:
        rep = dict(srf.summary(), units=lx.units, epsg=lx.epsg, notes=lx.notes)
        db.add(dbm.SheetReportRow(document_id=doc.id, page=srf.index, sheet_id=srf.name[:32], category=dbm.LANDXML_SURFACE, report=rep))
    doc.pages = len(lx.surfaces)
    doc.status = "done"
    db.commit()


def _ensure_false_origin(prj: Project, overrides: dict, surfaces: list[tuple[str, dict]], printed: list[tuple[float, float]] = ()) -> None:
    """Pick a false origin the first time real-world coordinates arrive, so project coordinates stay small.

    Only for coordinates that are actually large (state plane, UTM); small local site grids are left as they are.
    """
    if prj.datum.false_origin is not None:
        return
    pts = [(c[2], -c[3]) for sh in overrides.get("sheets", {}).values() for c in sh.get("control_points") or []]
    pts += [(rep["extent"][0], rep["extent"][1]) for _, rep in surfaces if rep.get("extent")]
    pts += list(printed)  # N/E coordinates printed on the sheets
    if pts and max(max(abs(e), abs(n)) for e, n in pts) > 20000:
        e, n = pts[0]
        prj.datum.false_origin = [math.floor(e / 1000) * 1000.0, math.floor(n / 1000) * 1000.0]


def _apply_landxml(db: Session, prj: Project, overrides: dict, surfaces: list[tuple[str, dict]], pdf_real_frame: bool, pdf_has_geometry: bool) -> None:
    """Imported TIN surfaces replace plan-derived ones for the same role, and are cross-checked against them."""
    roles = overrides.get("surfaces", {})
    prj.imported_surfaces = []
    for doc_id, rep in surfaces:
        user = roles.get(f"{doc_id}:{rep['index']}")
        prj.imported_surfaces.append(ImportedSurface(document_id=doc_id, index=rep["index"], name=rep["name"], desc=rep.get("desc", ""),
                                                     points=rep["points"], faces=rep["faces"], role=user or rep.get("role_guess"),
                                                     role_source="user" if user else ("name" if rep.get("role_guess") else None)))
    for s in prj.imported_surfaces:
        if s.role is None:
            prj.issues.append(Issue(severity="warning", code="surface_role_unknown", object_id=f"{s.document_id}:{s.index}",
                                    message=f"LandXML surface '{s.name}': can't tell from its name whether it is existing ground or finished grade. Choose its role in Plans."))
    docs = {d.id: d for d in dbm.documents_for(db, prj.id)}
    parsed: dict[str, landxml.LandXMLFile] = {}
    for role, label in (("existing", "Existing ground"), ("proposed", "Finished grade")):
        cands = [s for s in prj.imported_surfaces if s.role == role]
        if not cands:
            continue
        # a person's choice beats a name guess; otherwise the most recent upload wins
        pick = max(cands, key=lambda s: (s.role_source == "user", list(docs).index(s.document_id) if s.document_id in docs else -1, -s.index))
        if len(cands) > 1:
            prj.issues.append(Issue(severity="info", code="surface_choice", message=f"{len(cands)} imported surfaces could be the {label.lower()}; using '{pick.name}'. "
                                    f"Others: {', '.join(repr(s.name) for s in cands if s is not pick)}. Change roles in Plans."))
        doc = docs.get(pick.document_id)
        if not doc:
            continue
        if doc.id not in parsed:
            try:
                parsed[doc.id] = landxml.parse_landxml(Path(doc.path).read_bytes())
            except (OSError, landxml.LandXMLError) as ex:
                prj.issues.append(Issue(severity="blocker", code="landxml_unreadable", message=f"{doc.filename}: {ex}"))
                continue
        srf = parsed[doc.id].surfaces[pick.index]
        grid = landxml.grid_surface(srf, prj.datum.false_origin, prj.datum.elevation_offset_ft, label)
        plan = getattr(prj, role)
        if plan is not None and pdf_real_frame:
            prj.issues.append(_compare_surfaces(Grid.from_model(grid), Grid.from_model(plan), role, pick.name))
        elif plan is not None:
            prj.issues.append(Issue(severity="info", code="surface_superseded", message=f"The {label.lower()} read from plan contours is replaced by LandXML surface '{pick.name}'. "
                                    "They can't be compared until the plan sheets are registered with control points."))
        setattr(prj, role, grid)
        pick.used = True
        prj.issues = [i for i in prj.issues if i.code != f"no_{role}_surface"]
    if any(s.used for s in prj.imported_surfaces):
        epsgs = {f.epsg for f in parsed.values() if f.epsg}
        if len(epsgs) == 1 and prj.datum.epsg is None:
            prj.datum.epsg = epsgs.pop()
        elif epsgs and prj.datum.epsg not in epsgs:
            prj.issues.append(Issue(severity="warning", code="epsg_mismatch", message=f"LandXML coordinate system EPSG {', '.join(map(str, sorted(epsgs)))} differs from the project's EPSG {prj.datum.epsg}."))
        if pdf_has_geometry and not pdf_real_frame:
            prj.issues.append(Issue(severity="warning", code="frames_not_linked",
                                    message="LandXML surfaces are in real-world coordinates, but the plan sheets are still in sheet-local coordinates, so they won't line up. "
                                            "Add control points to the main plan sheet in Plans → Sheets & calibration."))


def _compare_surfaces(lx: Grid, plan: Grid, role: str, name: str) -> Issue:
    X, Z = plan.coords()
    d = lx.sample(X, Z) - plan.elev
    d = d[np.isfinite(d)]
    what = "existing ground" if role == "existing" else "finished grade"
    if len(d) < 10:
        return Issue(severity="warning", code="surface_no_overlap", message=f"LandXML surface '{name}' doesn't overlap the {what} read from the plans. Check the control points and coordinate system.")
    mean, rms = float(d.mean()), float(np.sqrt((d ** 2).mean()))
    msg = f"LandXML '{name}' vs {what} from plan contours: mean difference {mean:+.2f} ft, RMS {rms:.2f} ft over {len(d)} points. The LandXML surface is used."
    if abs(mean) > 0.5:
        return Issue(severity="warning", code="surface_datum_shift", message=msg + " A consistent offset usually means a vertical datum difference; check Datum & units.")
    if rms > 2.0:
        return Issue(severity="warning", code="surface_disagreement", message=msg + " Large scatter: check that both are the same surface and revision.")
    return Issue(severity="info", code="surface_agreement", message=msg)


def _centerline_profile(prj: Project) -> None:
    """Centerline elevations from the finished-grade surface (calculated), until profiles are linked by station."""
    if not prj.alignment:
        return
    if not prj.proposed:
        prj.alignment.profile_source = None
        prj.issues.append(Issue(severity="info", code="no_road_profile", message="The road centerline has no elevations yet: upload the grading plan or a finished-grade surface."))
        return
    pts = prj.alignment.points
    e = Grid.from_model(prj.proposed).sample([p.x for p in pts], [p.z for p in pts])
    ok = np.isfinite(e)
    if ok.sum() < 2:
        prj.issues.append(Issue(severity="warning", code="road_off_surface", message="The road centerline doesn't fall on the finished-grade surface. Check sheet registration."))
        return
    s = np.array([p.s for p in pts])
    filled = np.interp(s, s[ok], e[ok])  # stations off the surface take the nearest known elevation
    for p, v in zip(pts, filled):
        p.elev = round(float(v), 2)
    covered = 100 * ok.mean()
    prj.alignment.profile_source = "Finished-grade surface along the centerline (calculated)" + (f"; {covered:.0f}% of the centerline is on the surface" if covered < 99 else "")


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
    surfaces = dbm.reports_for(db, pid, dbm.LANDXML_SURFACE)
    _ensure_false_origin(prj, overrides, surfaces, [(c["e"], c["n"]) for _, r in reports for c in r.control])
    result = assemble(reports, overrides.get("sheets", {}), prj.datum.elevation_offset_ft, prj.datum.false_origin)
    prj.sheets, prj.structures, prj.pipes = result["sheets"], result["structures"], result["pipes"]
    prj.existing, prj.proposed, prj.issues = result["existing"], result["proposed"], result["issues"]
    prj.alignment, prj.corridor = result["alignment"], result["corridor"]
    pdf_real_frame = result["real_frame"]
    if surfaces:
        _apply_landxml(db, prj, overrides, surfaces, pdf_real_frame, bool(prj.structures or prj.existing or prj.proposed))
    if pdf_real_frame:
        prj.datum.horizontal = f"Registered to control points or printed coordinates{f' (EPSG:{prj.datum.epsg})' if prj.datum.epsg else ''}"
    elif any(s.used for s in prj.imported_surfaces):
        prj.datum.horizontal = f"LandXML coordinates{f' (EPSG:{prj.datum.epsg})' if prj.datum.epsg else ''}"
    _centerline_profile(prj)
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
