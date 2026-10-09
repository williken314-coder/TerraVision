"""Turn per-sheet extraction reports into one project model.

Rules this module follows:

* Positions come only from plan sheets with a known scale, registered into
  one project coordinate frame through structures they share.
* Elevations and sizes come only from text printed on the sheets. When
  sheets disagree, the value is flagged and both readings are reported.
* A value is marked ``verified`` only when two independent sheets agree
  (for example plan and profile). Single readings stay ``ai`` with a
  confidence until a person confirms them.
* Nothing missing is filled in. Missing values are reported as issues with
  the information needed to resolve them.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..geometry.surface import TIN, Grid
from ..geometry.alignment import offset_label, station_label
from ..models import EDGE_CODES, EDGE_NAMES, Alignment, AlignmentPoint, CorridorTemplate, Fact, GridSurface, Issue, Pipe, RoadEdge, Sheet, Status, Structure
from .calibration import SheetCalibration, Similarity, fit_similarity, ft_per_pt
from .pdf import PLAN_CATEGORIES, PROFILE_CATEGORIES, ROAD_CATEGORIES, SheetReport
from .roadway import link_by_stationing, read_roadway_sheet

SYSTEM_BY_PREFIX = {"CB": "storm", "INL": "storm", "DI": "storm", "HW": "storm", "FES": "storm", "STMH": "storm",
                    "SMH": "san", "FH": "water", "WV": "water", "TX": "elec"}
KIND_BY_PREFIX = {"CB": "CB", "INL": "INL", "DI": "INL", "HW": "HW", "FES": "HW", "STMH": "MH", "SMH": "SMH",
                  "MH": "MH", "FH": "FH", "WV": "OTHER", "TX": "TX"}
PROPOSED_CODES = {"FG", "BC", "FL", "PAD", "HP", "LP", "BW"}
EXISTING_CODES = {"EG"}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def wall_ft(dia_in: float | None, material: str | None) -> float:
    if not dia_in:
        return 0.15
    m = (material or "").upper()
    if "RCP" in m:  # ASTM C76 wall B, approximately
        return round((dia_in / 12 + 1) / 12, 4)
    if "DIP" in m or m == "DI":
        return 0.03
    if "PVC" in m:
        return round(dia_in * 0.03 / 12, 4)
    return round(dia_in * 0.06 / 12, 4)


def sheet_key(doc_id: str, page: int) -> str:
    return f"{doc_id}:{page}"


def _system_for(prefix: str, category: str) -> str:
    if prefix == "MH":
        return "san" if "sanitary" in category else "storm"
    return SYSTEM_BY_PREFIX.get(prefix, "storm")


def _compass(dx: float, dz: float) -> str:
    ang = (math.degrees(math.atan2(dx, -dz)) + 360) % 360
    return COMPASS[int((ang + 22.5) // 45) % 8]


def assemble(reports: list[tuple[str, SheetReport]], overrides: dict | None = None, elevation_offset: float = 0.0,
             false_origin: list[float] | None = None) -> dict:
    """``false_origin`` [E0, N0] is subtracted from control-point coordinates (see ``Datum.false_origin``)."""
    overrides = overrides or {}
    e0, n0 = false_origin or (0.0, 0.0)
    issues: list[Issue] = []
    sheets: list[Sheet] = []
    cals: dict[str, SheetCalibration] = {}
    rep_by_key: dict[str, SheetReport] = {}
    doc_of: dict[str, str] = {}

    # 1. sheets and scale calibration -------------------------------------
    for doc_id, r in reports:
        key = sheet_key(doc_id, r.page)
        rep_by_key[key], doc_of[key] = r, doc_id
        sid = r.sheet_id or f"p{r.page + 1}"
        ov = overrides.get(key, {})
        scale = ov.get("scale_ft_per_in") or r.scale_ft_per_in
        method = "user scale" if ov.get("scale_ft_per_in") else ("scale text" if r.scale_ft_per_in else "none")
        cals[key] = SheetCalibration(key, ft_per_pt(scale) if scale else None, method)
        sheets.append(Sheet(id=sid, title=r.title or "(untitled)", category=r.category, page=r.page, document_id=doc_id,
                            scale_ft_per_in=scale, notes=f"{len(r.candidates)} labelled items, {r.line_segments} line segments"))
        if r.scale_source == "ocr" and not ov.get("scale_ft_per_in"):
            issues.append(Issue(severity="info", code="scale_by_ocr", sheet=sid, page=r.page,
                                message=f"{sid}: scale {r.scale_text}. It is drawn as line strokes, not text, so it was read from the drawing; check it in Plans → Sheets & calibration."))
        for n in r.notes:
            issues.append(Issue(severity="warning" if "scale" in n.lower() else "blocker", code="sheet_note", message=n, sheet=sid, page=r.page))
        if not r.sheet_id:
            issues.append(Issue(severity="info", code="no_sheet_number", message=f"Page {r.page + 1}: no sheet number found in the title block.", page=r.page))

    sid_of = {sheet_key(s.document_id, s.page): s.id for s in sheets}
    sheet_of = {sheet_key(s.document_id, s.page): s for s in sheets}

    # 1b. roadway centerlines, stationing and edges (sheet feet) -------------
    road: dict[str, dict] = {}
    for key, r in rep_by_key.items():
        if not r.linework:
            continue
        sid, cal = sid_of[key], cals[key]
        stations = [(c.x, c.y, c.values["station"], i) for i, c in enumerate(r.candidates) if c.kind == "station"]
        labels = [dict(x=c.x, y=c.y, kind=c.label, size=c.values.get("size", 6.0), text=c.text, bbox=c.bbox)
                  for c in r.candidates if c.kind == "road_label"]
        res = read_roadway_sheet(r.linework, stations, labels, cal.ft_per_pt, max(12.0, min(r.width_pt, r.height_pt) * 0.02))
        if not res:
            if len(stations) >= 2 and r.category in ROAD_CATEGORIES:
                issues.append(Issue(severity="info", code="no_centerline", sheet=sid, page=r.page,
                                    message=f"{sid} has station labels, but no centerline with road edges on both sides was found along them."))
            continue
        measured_in = res["measured_ft_per_pt"] * 72
        if cal.ft_per_pt is None:
            cal.ft_per_pt, cal.method = res["measured_ft_per_pt"], "stationing"
            sheet_of[key].scale_ft_per_in = round(measured_in, 2)
            issues[:] = [i for i in issues if not (i.code == "sheet_note" and i.page == r.page and i.sheet == sid and "scale" in i.message.lower())]
            issues.append(Issue(severity="info", code="scale_from_stationing", sheet=sid, page=r.page,
                                message=f"{sid} states no scale; using 1\" = {measured_in:.1f}' measured from its stationing. Set the scale in Plans if that's wrong."))
        elif abs(res["measured_ft_per_pt"] / cal.ft_per_pt - 1) > 0.02:
            issues.append(Issue(severity="warning", code="scale_vs_stationing", sheet=sid, page=r.page,
                                message=f"Stationing on {sid} measures 1\" = {measured_in:.1f}', but the sheet says 1\" = {cal.ft_per_pt * 72:.0f}'. "
                                        "A half-size print or a wrong scale note? Set the scale in Plans."))
        res["matchlines"] = [(c.values["station"], c.values.get("sheet_ref"), sid) for c in r.candidates if c.kind == "matchline"]
        road[key] = res

    # 2. structure positions on each calibrated plan sheet (sheet-local feet) ----
    local: dict[str, dict[str, tuple]] = {}
    for key, r in rep_by_key.items():
        cal = cals[key]
        if r.category not in PLAN_CATEGORIES or cal.ft_per_pt is None:
            continue
        pos = {}
        for c in r.candidates:
            if c.kind != "structure":
                continue
            xy = c.values.get("symbol_xy")
            conf = c.values.get("position_confidence", 0.4)
            if not xy:
                xy, conf = c.values.get("label_xy"), 0.35
            if not xy:
                continue  # read by the vision model without a location
            pos[c.label] = (xy[0] * cal.ft_per_pt, xy[1] * cal.ft_per_pt, conf, bool(c.values.get("symbol_xy")), c)
        if pos:
            local[key] = pos

    # 3. register sheets into one frame through shared structures, or matchline stationing ----
    registered: dict[str, Similarity] = {}
    anchors = [k for k in rep_by_key if k in local or k in road]  # report order: ties resolve the same way every time
    if not anchors:
        # no structures and no roadway: place the scaled plan sheet with the most to read (e.g. a grading-only set)
        scaled = [k for k, r in rep_by_key.items() if r.category in PLAN_CATEGORIES and cals[k].ft_per_pt and r.candidates]
        if scaled:
            anchors = [max(scaled, key=lambda k: len(rep_by_key[k].candidates))]
            for k in scaled:
                if k not in anchors:
                    issues.append(Issue(severity="warning", code="unregistered_sheet", sheet=sid_of[k], page=rep_by_key[k].page,
                                        message=f"{sid_of[k]} shares no labelled structures or stationing with {sid_of[anchors[0]]}, so its geometry is not placed. Add control points to register it."))
    if anchors:
        primary = max(anchors, key=lambda k: (len(local.get(k, {})), road[k]["s_range"][1] - road[k]["s_range"][0] if k in road else 0))
        ctrl = overrides.get(primary, {}).get("control_points")
        if ctrl and len(ctrl) >= 2:
            k = cals[primary].ft_per_pt
            registered[primary] = fit_similarity(np.array([[c[0] * k, c[1] * k] for c in ctrl]), np.array([[c[2] - e0, c[3] + n0] for c in ctrl]))
        else:
            registered[primary] = Similarity()
        changed = True
        while changed:
            changed = False
            for key in [k for k in road if k not in registered]:
                for rk in [k for k in road if k in registered]:
                    link = link_by_stationing(road[key], road[rk], registered[rk], road[key]["matchlines"] + road[rk]["matchlines"], sid_of[key], sid_of[rk])
                    if link:
                        registered[key] = link[0]
                        changed = True
                        issues.append(Issue(severity="info", code="placed_by_stationing", sheet=sid_of[key], page=rep_by_key[key].page,
                                            message=f"{sid_of[key]} placed next to {sid_of[rk]} by matching their centerline stationing at the matchline, STA {station_label(link[1], 2)}. "
                                                    "That fixes position well but direction less so; control points or two shared structures place it more precisely."))
                        break
            for key, pos in local.items():
                if key in registered:
                    continue
                ref_pts = {}
                for rk in registered:
                    for sid, p in local[rk].items():
                        x, z = registered[rk].apply(p[0], p[1])
                        ref_pts.setdefault(sid, (float(x), float(z)))
                common = [s for s in pos if s in ref_pts]
                if len(common) >= 2:
                    sim = fit_similarity(np.array([[pos[s][0], pos[s][1]] for s in common]), np.array([ref_pts[s] for s in common]))
                    registered[key] = sim
                    changed = True
                    if abs(sim.scale - 1) > 0.02:
                        issues.append(Issue(severity="warning", code="scale_mismatch", sheet=sid_of[key], page=rep_by_key[key].page,
                                            message=f"{sid_of[key]} disagrees with {sid_of[primary]} by {abs(sim.scale - 1) * 100:.1f}% in scale. Check the stated scale on both sheets."))
                    if sim.rms > 2.0:
                        issues.append(Issue(severity="warning", code="registration_residual", sheet=sid_of[key], page=rep_by_key[key].page,
                                            message=f"Shared structures on {sid_of[key]} line up with {sid_of[primary]} only to {sim.rms:.1f} ft RMS."))
        for key in local:
            if key not in registered:
                issues.append(Issue(severity="warning", code="unregistered_sheet", sheet=sid_of[key], page=rep_by_key[key].page,
                                    message=f"{sid_of[key]} shares fewer than two labelled structures with the other plan sheets, so its geometry is not placed. Add control points to register it."))
        for key in road:
            if key not in registered and key not in local:
                issues.append(Issue(severity="warning", code="unregistered_sheet", sheet=sid_of[key], page=rep_by_key[key].page,
                                    message=f"{sid_of[key]}'s roadway shares no labelled structures, and no matchline stationing, with the placed sheets, so it is not placed. Add control points to register it."))
    for key, sim in registered.items():
        cals[key].to_project = sim
        for s in sheets:
            if sheet_key(s.document_id, s.page) == key:
                s.registered = True

    # 4. merge structures ----------------------------------------------------
    by_id: dict[str, dict] = defaultdict(lambda: {"pos": [], "rims": [], "invs": [], "sheets": set(), "cats": set()})
    for key, r in rep_by_key.items():
        for c in r.candidates:
            if c.kind != "structure":
                continue
            e = by_id[c.label]
            e["sheets"].add(sid_of[key])
            e["cats"].add(r.category)
            if "rim" in c.values:
                e["rims"].append(dict(v=c.values["rim"] + elevation_offset, conf=c.values["rim_confidence"], sheet=sid_of[key], page=r.page,
                                      bbox=c.values.get("rim_bbox"), text=c.values.get("rim_text"), doc=doc_of[key]))
            for inv in c.values["inverts"]:
                e["invs"].append(dict(inv, v=inv["elev"] + elevation_offset, sheet=sid_of[key], page=r.page, doc=doc_of[key]))
            if key in registered and c.label in local.get(key, {}):
                p = local[key][c.label]
                x, z = registered[key].apply(p[0], p[1])
                e["pos"].append(dict(x=float(x), z=float(z), conf=p[2], sheet=sid_of[key], page=r.page, bbox=c.bbox, doc=doc_of[key], text=c.text))

    structures: list[Structure] = []
    invs_by_id: dict[str, list] = {}
    for sid, e in sorted(by_id.items()):
        prefix = sid.split("-")[0]
        cat = next((c for c in e["cats"] if c in PLAN_CATEGORIES), next(iter(e["cats"])))
        system = _system_for(prefix, cat)
        if not e["pos"]:
            issues.append(Issue(severity="warning", code="no_position", object_id=sid, message=f"{sid} appears on {', '.join(sorted(e['sheets']))} but not on a scaled, registered plan sheet, so it has no location."))
            continue
        xs = np.array([p["x"] for p in e["pos"]])
        zs = np.array([p["z"] for p in e["pos"]])
        spread = float(np.hypot(xs - xs.mean(), zs - zs.mean()).max())
        best_pos = max(e["pos"], key=lambda p: p["conf"])
        pos_conf = min(p["conf"] for p in e["pos"]) if spread < 2 else 0.4
        if spread >= 2:
            issues.append(Issue(severity="warning", code="position_disagreement", object_id=sid, message=f"{sid} is placed {spread:.1f} ft apart on different sheets."))
        position = Fact[list](value=[float(xs.mean()), float(zs.mean())], status=Status.verified if len(e["pos"]) >= 2 and spread < 1 and pos_conf >= 0.8 else Status.ai,
                              confidence=round(pos_conf, 2), source=best_pos["sheet"], page=best_pos["page"], bbox=best_pos["bbox"], text=best_pos["text"], document_id=best_pos["doc"])
        # headwalls and flared end sections have no rim; only report one if a sheet prints it
        no_rim = KIND_BY_PREFIX.get(prefix) == "HW" and not e["rims"]
        rim = _merge_values(e["rims"], f"{sid} rim", sid, [] if no_rim else issues)
        structures.append(Structure(id=sid, system=system, kind=KIND_BY_PREFIX.get(prefix, "OTHER"), x=position.value[0], z=position.value[1],
                                    position=position, rim=rim, type=Fact[str](value=None, status=Status.missing), sheet=best_pos["sheet"]))
        invs_by_id[sid] = e["invs"]

    # 5. pipes from plan line work ------------------------------------------
    st_by = {s.id: s for s in structures}
    pipes: list[Pipe] = []
    seen = set()
    counters = defaultdict(int)
    for key, r in rep_by_key.items():
        if key not in registered or not r.pipe_segments:
            continue
        cal = cals[key]
        sym_pts = {c.label: c.values["symbol_xy"] for c in r.candidates if c.kind == "structure" and c.values.get("symbol_xy")}
        snap = max(10.0, min(r.width_pt, r.height_pt) * 0.02)
        callouts = [c for c in r.candidates if c.kind == "pipe_callout"]

        def end_struct(x, y):
            best, bd = None, float("inf")
            for sid, (sx, sy) in sym_pts.items():
                d = math.hypot(sx - x, sy - y)
                if d < bd:
                    best, bd = sid, d
            return best if bd <= snap else None

        for x0, y0, x1, y1 in r.pipe_segments:
            a, b = end_struct(x0, y0), end_struct(x1, y1)
            if not a or not b or a == b or a not in st_by or b not in st_by:
                continue
            pair = frozenset((a, b))
            if pair in seen:
                continue
            seen.add(pair)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            seg_len_pt = math.hypot(x1 - x0, y1 - y0)
            co = min(callouts, key=lambda c: math.hypot(c.x - mx, c.y - my), default=None)
            if co and math.hypot(co.x - mx, co.y - my) > max(60.0, seg_len_pt * 0.6):
                co = None
            pipes.append(_make_pipe(st_by[a], st_by[b], invs_by_id, co, sid_of[key], r.page, doc_of[key], counters, issues, seg_len_pt * cal.ft_per_pt))

    # pipe sizes stated on profile sheets fill gaps left by plan callouts. A profile can show the
    # same structure in more than one run, so a callout must sit between both labels of one run.
    for key, r in rep_by_key.items():
        if r.category not in PROFILE_CATEGORIES:
            continue
        occ = defaultdict(list)
        for c in r.candidates:
            if c.kind == "structure":
                occ[c.label].append((c.x, c.y))
        for co in (c for c in r.candidates if c.kind == "pipe_callout"):
            best = None
            for p in pipes:
                for xa, ya in occ.get(p.from_id, []):
                    for xb, yb in occ.get(p.to_id, []):
                        if abs(ya - yb) > 40 or not (min(xa, xb) <= co.x <= max(xa, xb)) or abs(co.y - (ya + yb) / 2) > 260:
                            continue
                        span = abs(xa - xb)
                        if best is None or span < best[0]:
                            best = (span, p)
            if not best:
                continue
            p = best[1]
            for field_, val in (("dia_in", co.values["dia_in"]), ("material", co.values["material"])):
                f = getattr(p, field_)
                if f.value is None:
                    setattr(p, field_, type(f)(value=val, status=Status.ai, confidence=0.75, source=sid_of[key], page=r.page, bbox=co.bbox, text=co.text, document_id=doc_of[key]))
                elif f.value == val and f.source != sid_of[key]:
                    f.status, f.confidence = Status.verified, 0.95
                elif f.value != val:
                    f.status, f.confidence = Status.ai, 0.4
                    issues.append(Issue(severity="warning", code="callout_disagreement", object_id=p.id, sheet=sid_of[key], page=r.page,
                                        message=f"{p.id} {field_}: {f.source} says {f.value}, {sid_of[key]} says {val}."))
            p.wall_ft = wall_ft(p.dia_in.value, p.material.value)

    # 6. surfaces --------------------------------------------------------------
    pts = {"existing": [], "proposed": []}
    for key, r in rep_by_key.items():
        if key not in registered:
            continue
        cal = cals[key]
        for c in r.candidates:
            if c.kind == "spot":
                which = "proposed" if c.label in PROPOSED_CODES else "existing" if c.label in EXISTING_CODES else None
                if which:
                    x, z = cal.pt_to_project(c.x, c.y)
                    pts[which].append((float(x), float(z), c.values["elev"] + elevation_offset))
            elif c.kind == "contour":
                which = c.values.get("surface") or ("existing" if c.values["dashed"] else "proposed")
                arr = np.array(c.values["pts"])
                x, z = cal.pt_to_project(arr[:, 0], arr[:, 1])
                pts[which].extend((float(a), float(b), c.values["elev"] + elevation_offset) for a, b in zip(x, z))
    surfaces = {}
    for which, P in pts.items():
        if len(P) < 12:
            issues.append(Issue(severity="info", code=f"no_{which}_surface",
                                message=(f"Not enough {which} elevations to build a surface ({len(P)} found, 12+ needed). "
                                         + ("Upload the existing conditions / topographic survey sheet, with labelled contours or EG spot shots."
                                            if which == "existing" else "Upload the grading plan, with labelled proposed contours or FG/BC/FL spot elevations."))))
            continue
        arr = np.array(P)
        x0, z0 = arr[:, 0].min(), arr[:, 1].min()
        ext = max(arr[:, 0].max() - x0, arr[:, 1].max() - z0)
        cell = max(1.0, round(ext / 120))
        nx, nz = int(math.ceil((arr[:, 0].max() - x0) / cell)), int(math.ceil((arr[:, 1].max() - z0) / cell))
        try:
            g = TIN(arr[:, 0], arr[:, 1], arr[:, 2]).to_grid(x0, z0, cell, nx, nz)
        except Exception as ex:  # degenerate point sets
            issues.append(Issue(severity="warning", code=f"{which}_surface_failed", message=f"Could not triangulate the {which} surface: {ex}"))
            continue
        m = g.to_model("Existing ground" if which == "existing" else "Finished grade", "from uploaded plans")
        m.point_count = len(P)
        surfaces[which] = m

    # 7. roadway: one alignment and its edges, from the placed roadway sheets ---
    pieces = []
    for key, res in road.items():
        if key in registered:
            x, z = registered[key].apply(np.array(res["x"]), np.array(res["z"]))
            pieces.append(dict(key=key, sheet=sid_of[key], page=rep_by_key[key].page, doc=doc_of[key], res=res,
                               s=np.array(res["stations"]), x=np.asarray(x, float), z=np.asarray(z, float)))
    alignment, corridor = _build_road(pieces, issues) if pieces else (None, None)
    if corridor is not None:
        corridor.pavement_section = _pavement_section(rep_by_key, sid_of, doc_of, issues)

    return dict(sheets=sheets, structures=structures, pipes=pipes, existing=surfaces.get("existing"), proposed=surfaces.get("proposed"),
                alignment=alignment, corridor=corridor, issues=issues, calibrations={k: dict(ft_per_pt=c.ft_per_pt, method=c.method, registered=c.to_project is not None,
                                                     transform=c.to_project.as_dict() if c.to_project else None) for k, c in cals.items()})


def _fmt_vals(vals: list[dict]) -> str:
    return ", ".join("%.2f on %s" % (v["v"], v["sheet"]) for v in vals)


def _merge_values(vals: list[dict], what: str, oid: str, issues: list[Issue]) -> Fact:
    if not vals:
        issues.append(Issue(severity="warning", code="missing_value", object_id=oid, message=f"{what} not found on any sheet."))
        return Fact[float](value=None, status=Status.missing)
    distinct = sorted({round(v["v"], 2) for v in vals})
    best = max(vals, key=lambda v: v["conf"])
    if len(distinct) > 1:
        issues.append(Issue(severity="warning", code="value_disagreement", object_id=oid,
                            message=f"{what}: sheets disagree ({_fmt_vals(vals)})."))
        return Fact[float](value=round(best["v"], 2), status=Status.ai, confidence=0.4, source=best["sheet"], page=best["page"], bbox=best.get("bbox"), text=best.get("text"), document_id=best["doc"])
    agree_sheets = {v["sheet"] for v in vals}
    conf = min(0.99, best["conf"] + 0.1 * (len(agree_sheets) - 1))
    status = Status.verified if len(agree_sheets) >= 2 and conf >= 0.85 else Status.ai
    return Fact[float](value=round(best["v"], 2), status=status, confidence=round(conf, 2), source=best["sheet"], page=best["page"], bbox=best.get("bbox"), text=best.get("text"), document_id=best["doc"])


def _pick_invert(invs: list[dict], direction: str, toward: str | None) -> list[dict]:
    c = [i for i in invs if i.get("dir") == direction] or [i for i in invs if not i.get("dir")]
    if len(c) > 1 and toward:
        side = [i for i in c if i.get("side") and (i["side"] == toward or i["side"] in toward or toward in i["side"])]
        if side:
            c = side
    return c


def _make_pipe(a: Structure, b: Structure, invs_by_id: dict, co, sheet: str, page: int, doc: str, counters, issues: list[Issue], plan_len_ft: float) -> Pipe:
    ia, ib = invs_by_id.get(a.id, []), invs_by_id.get(b.id, [])
    best = None
    for up, dn in ((a, b), (b, a)):
        iu, idn = (ia, ib) if up is a else (ib, ia)
        outs = _pick_invert(iu, "OUT", _compass(dn.x - up.x, dn.z - up.z))
        ins = _pick_invert(idn, "IN", _compass(up.x - dn.x, up.z - dn.z))
        if outs and ins:
            o, i = outs[0], ins[0]
            if o["v"] >= i["v"]:
                amb = len({round(x["v"], 2) for x in outs}) > 1 or len({round(x["v"], 2) for x in ins}) > 1
                best = (up, dn, o, i, amb, iu, idn)
                break
    system = a.system
    prefix = {"storm": "ST", "san": "SA", "water": "WL", "elec": "EL"}[system]
    counters[prefix] += 1
    pid = f"{prefix}-{counters[prefix]}"
    dia = co.values["dia_in"] if co else None
    mat = co.values["material"] if co else None
    src = dict(source=sheet, page=page, document_id=doc)
    dia_f = Fact[float](value=dia, status=Status.ai if dia else Status.missing, confidence=0.85 if dia else None, bbox=co.bbox if co else None, text=co.text if co else None, **src)
    mat_f = Fact[str](value=mat, status=Status.ai if mat else Status.missing, confidence=0.85 if mat else None, bbox=co.bbox if co else None, text=co.text if co else None, **src)
    if not co:
        issues.append(Issue(severity="warning", code="no_callout", object_id=pid, sheet=sheet, page=page, message=f"{pid} ({a.id} to {b.id}): no size/material callout found near the pipe."))
    if best:
        up, dn, o, i, amb, iu_all, idn_all = best
        conf_o, conf_i = (o["confidence"] * (0.7 if amb else 1.0)), (i["confidence"] * (0.7 if amb else 1.0))
        inv_up = Fact[float](value=round(o["v"], 2), status=Status.ai, confidence=round(conf_o, 2), source=o["sheet"], page=o["page"], bbox=o.get("bbox"), text=o.get("text"), document_id=o["doc"])
        inv_dn = Fact[float](value=round(i["v"], 2), status=Status.ai, confidence=round(conf_i, 2), source=i["sheet"], page=i["page"], bbox=i.get("bbox"), text=i.get("text"), document_id=i["doc"])
        if co and co.values.get("slope_pct") is not None and plan_len_ft > 0:
            calc = (inv_up.value - inv_dn.value) / plan_len_ft * 100
            if abs(calc - co.values["slope_pct"]) > 0.15:
                issues.append(Issue(severity="warning", code="slope_mismatch", object_id=pid, sheet=sheet, page=page,
                                    message=f"{pid}: inverts give {calc:.2f}% over {plan_len_ft:.0f} ft, the callout says {co.values['slope_pct']:.2f}%."))
            else:
                inv_up.confidence = min(0.97, (inv_up.confidence or 0) + 0.1)
                inv_dn.confidence = min(0.97, (inv_dn.confidence or 0) + 0.1)
        if amb:
            issues.append(Issue(severity="warning", code="ambiguous_invert", object_id=pid, message=f"{pid}: more than one possible invert at {up.id} or {dn.id}. Confirm which one belongs to this pipe."))
        for fact, pool, chosen, dirn in ((inv_up, iu_all, o, "OUT"), (inv_dn, idn_all, i, "IN")):
            agree = {x["sheet"] for x in pool if abs(x["v"] - chosen["v"]) < 0.005 and (x.get("dir") in (dirn, None))}
            if len(agree) >= 2 and not amb:
                fact.confidence = round(min(0.98, (fact.confidence or 0) + 0.2 * (len(agree) - 1)), 2)
                if fact.confidence >= 0.85:
                    fact.status = Status.verified
    else:
        up, dn = a, b
        inv_up = Fact[float](value=None, status=Status.missing)
        inv_dn = Fact[float](value=None, status=Status.missing)
        issues.append(Issue(severity="warning", code="missing_inverts", object_id=pid, message=f"{pid} ({a.id} to {b.id}): matching INV OUT / INV IN not found. Check the profile sheet."))
    return Pipe(id=pid, system=system, from_id=up.id, to_id=dn.id, dia_in=dia_f, material=mat_f, wall_ft=wall_ft(dia, mat),
                inv_up=inv_up, inv_dn=inv_dn, sheet=sheet)


def _same_road(a: dict, b: dict) -> bool:
    """Two placed centerline pieces are one road if they agree where their stationing overlaps or meets."""
    lo, hi = max(a["s"][0], b["s"][0]), min(a["s"][-1], b["s"][-1])
    if hi - lo >= 5:
        ss = np.linspace(lo, hi, 5)
    elif lo - hi <= 10:
        ss = np.array([(lo + hi) / 2])
    else:
        return False
    d = np.hypot(np.interp(ss, a["s"], a["x"]) - np.interp(ss, b["s"], b["x"]), np.interp(ss, a["s"], a["z"]) - np.interp(ss, b["s"], b["z"]))
    return bool(d.max() <= 3.0 + max(0.0, lo - hi))


def _build_road(pieces: list[dict], issues: list[Issue]) -> tuple[Alignment, CorridorTemplate | None]:
    pieces = sorted(pieces, key=lambda p: -(p["s"][-1] - p["s"][0]))
    group, rest = [pieces[0]], pieces[1:]
    changed = True
    while changed:
        changed = False
        for p in list(rest):
            if any(_same_road(p, g) for g in group):
                group.append(p)
                rest.remove(p)
                changed = True
    for p in rest:
        issues.append(Issue(severity="info", code="other_alignment", sheet=p["sheet"], page=p["page"],
                            message=f"{p['sheet']} has another centerline (STA {station_label(p['s'][0])} to {station_label(p['s'][-1])}). "
                                    "Only one road per project is modeled so far."))
    # centerline points: each piece adds only the stations not already covered
    pts, covered = [], []
    for p in group:
        for s_, x_, z_ in zip(p["s"], p["x"], p["z"]):
            if not any(a - 0.5 <= s_ <= b + 0.5 for a, b in covered):
                pts.append((float(s_), float(x_), float(z_)))
        covered.append((float(p["s"][0]), float(p["s"][-1])))
    pts.sort()
    sheets = sorted({p["sheet"] for p in group})
    fits = [p["res"]["fit"] for p in group]
    conf = 0.9 if all(len(f["inliers"]) >= 3 and f["rms"] < 0.5 for f in fits) else 0.7
    alignment = Alignment(name=f"Centerline ({', '.join(sheets)})", source=", ".join(sheets), confidence=conf,
                          points=[AlignmentPoint(s=round(s_, 2), x=round(x_, 3), z=round(z_, 3)) for s_, x_, z_ in pts])

    # edges: the same kind and side from every piece, combined by how much road each covers
    by = defaultdict(list)
    for p in group:
        for e in p["res"]["edges"]:
            if e["kind"] in EDGE_CODES:
                by[(e["kind"], e["side"])].append((e, p))
    edges: list[RoadEdge] = []
    for (kind, side), lst in sorted(by.items()):
        offs = [e["offset"] for e, _ in lst]
        cov = [e["coverage"] for e, _ in lst]
        off = float(np.average(offs, weights=cov))
        main_e, main_p = max(lst, key=lambda t: t[0]["coverage"])
        eid = f"{EDGE_CODES[kind]}-{side}T"
        if max(offs) - min(offs) > 0.5:
            issues.append(Issue(severity="warning", code="edge_disagreement", object_id=eid,
                                message=f"{EDGE_NAMES[kind].capitalize()} {side}T is {', '.join(f'{o:.1f} ft on {p_['sheet']}' for (e_, p_), o in zip(lst, offs))}. "
                                        "The section may change along the road; only one width is modeled."))
        by_label = [t for t in lst if t[0]["named_by"] == "label"]
        lab = max(by_label, key=lambda t: t[0]["coverage"])[0]["label"] if by_label else None
        conf_e = 0.8 if by_label else 0.55
        if len({p_["sheet"] for _, p_ in lst}) >= 2 and max(offs) - min(offs) <= 0.2:
            conf_e += 0.1
        fact = Fact[float](value=round(off, 2), status=Status.ai, confidence=round(conf_e, 2), source=main_p["sheet"], page=main_p["page"],
                           document_id=main_p["doc"], bbox=lab["bbox"] if lab else None,
                           text=f"“{lab['text']}” label on a line {offset_label(-off if side == 'L' else off)}" if lab else f"line {offset_label(-off if side == 'L' else off)} of centerline, named from its order")
        edges.append(RoadEdge(id=eid, kind=kind, side=side, offset=fact, station_from=round(min(e["s0"] for e, _ in lst), 1),
                              station_to=round(max(e["s1"] for e, _ in lst), 1), named_by="label" if by_label else "order"))
    # symmetric sections: both sides read the same raises confidence
    for e in edges:
        o = next((x for x in edges if x.kind == e.kind and x.side != e.side), None)
        if o and abs(o.offset.value - e.offset.value) <= 0.3:
            e.offset.confidence = round(min(0.95, e.offset.confidence + 0.05), 2)
    ordered = [e.id for e in edges if e.named_by == "order"]
    if ordered:
        issues.append(Issue(severity="info", code="edges_by_order", message=f"No labels on {', '.join(ordered)}: named from their order outward from the centerline. Confirm them in Inspect."))
    if not edges:
        issues.append(Issue(severity="warning", code="no_road_edges", message=f"Centerline found on {alignment.source}, but no pavement, curb or sidewalk lines were identified along it."))
        return alignment, None
    pav = [e.offset.value for e in edges if e.kind == "pavement"]
    if len(pav) < 2:
        issues.append(Issue(severity="warning", code="no_pavement_edges", message=f"Edge of pavement wasn't identified on both sides of the centerline on {alignment.source}, so the road surface isn't drawn."))
    corridor = CorridorTemplate(pavement_half_width=round(float(np.mean(pav)), 2) if len(pav) == 2 else None, crown_slope_assumed=True,
                                curb_width=None, walk_inner=None, walk_outer=None, edges=edges,
                                pavement_section=Fact[str](value=None, status=Status.missing))
    issues.append(Issue(severity="info", code="crown_assumed", message="The road is drawn with a 2% crown; cross slope isn't read from the plans yet."))
    return alignment, corridor


def _pavement_section(rep_by_key: dict, sid_of: dict, doc_of: dict, issues: list[Issue]) -> Fact:
    """Pavement layers printed together (a typical section or a note), read top-down. Missing if none are found."""
    groups = []
    for key, r in rep_by_key.items():
        layers = sorted((c for c in r.candidates if c.kind == "pavement_layer"), key=lambda c: (c.y, c.x))
        for c in layers:
            g = next((g for g in groups if g["key"] == key and min(abs(c.y - o.y) + abs(c.x - o.x) * 0.25 for o in g["c"]) < 60), None)
            if g:
                g["c"].append(c)
            else:
                groups.append(dict(key=key, c=[c]))
    surface = ("ASPHALT", "HMA", "AC", "A.C.", "CONCRETE", "PCC", "P.C.C.")
    groups = [g for g in groups if len(g["c"]) >= 2 and any(any(w in c.values["material"] for w in surface) for c in g["c"])]
    if not groups:
        return Fact[str](value=None, status=Status.missing)
    rank = {"details": 0, "cross_section": 1, "roadway": 2}
    groups.sort(key=lambda g: (-len(g["c"]), rank.get(rep_by_key[g["key"]].category, 3)))
    g = groups[0]
    if len(groups) > 1:
        issues.append(Issue(severity="info", code="pavement_sections", message=f"{len(groups)} pavement sections found; using the one on {sid_of[g['key']]}. Others are on "
                            + ", ".join(sorted({sid_of[o['key']] for o in groups[1:]})) + "."))
    seen, parts = set(), []
    for c in g["c"]:
        t = f"{c.values['thickness_in']:g} in {c.values['material'].lower()}"
        if t not in seen:
            seen.add(t)
            parts.append(t)
    xs = [v for c in g["c"] for v in c.bbox]
    r = rep_by_key[g["key"]]
    return Fact[str](value=" / ".join(parts), status=Status.ai, confidence=0.7, source=sid_of[g["key"]], page=r.page, document_id=doc_of[g["key"]],
                     bbox=[min(xs[0::4]), min(xs[1::4]), max(xs[2::4]), max(xs[3::4])], text=" | ".join(dict.fromkeys(c.text for c in g["c"])))
