# TerraVision AI

Upload a civil plan set (PDF), and optionally the design surfaces (LandXML). TerraVision reads them, builds a 3D model of the jobsite from what is actually printed on the sheets, and flags everything it isn't sure about for a person to confirm.

```
apps/web/            The TerraVision interface (Three.js), served by the API at /
services/api/        Python backend: FastAPI, PDF reading, geometry engine, database
  gv/ingest/         Plan interpretation pipeline (PyMuPDF, OpenCV, optional vision model) and LandXML import
  gv/geometry/       Deterministic engineering math (alignments, TIN surfaces, pipes, crossings, earthwork, trenches, checks)
  gv/api.py          HTTP API
  samples/           A generated sample plan set and LandXML surfaces to test uploads with
  tests/             Test suite
docker-compose.yml   API + PostGIS
```

## Run it

### Local (SQLite, fastest)

```bash
cd services/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn gv.api:app --reload --port 8000
```

Open http://localhost:8000. The **Maple Ridge (demo)** project loads first. To try real processing:

1. Open the **Plans** tab, tap **New**, name a project.
2. Tap **Upload plan set (PDF)** and pick `services/api/samples/maple-ridge-sample-plans.pdf` (or your own plans).
3. When processing finishes, the model appears. Every value shows the sheet, page, and the exact text it was read from. Tap **Sheet** next to any value to see that spot on the drawing.
4. Optionally upload `services/api/samples/maple-ridge-surfaces.xml` (LandXML EG and FG surfaces) to the same project. See [LandXML surfaces](#landxml-surfaces).

### Docker (PostGIS)

```bash
docker compose up --build        # then open http://localhost:8000
```

### Optional AI features

Set `ANTHROPIC_API_KEY` on the server to turn on:

* **Vision reading of scanned sheets** (pages with no text layer). Model: `GV_VISION_MODEL` (default `claude-sonnet-5-5`).
* **Cross-checking text sheets with vision**: also set `GV_VISION_CROSSCHECK=1`. Values both readers agree on gain confidence; disagreements are flagged.
* **Open-ended assistant questions** beyond the built-in intents. Model: `GV_ASSISTANT_MODEL`.

Without a key, everything else works; scanned sheets are reported as unreadable instead of guessed.

### Tests

```bash
cd services/api && pytest
```

`test_geometry.py`, `test_extraction.py` and the parser tests in `test_landxml.py` need only numpy/scipy/pydantic. `test_pdf_and_api.py` runs the real PDF reader and the HTTP API end to end on the sample plan set (skipped if PyMuPDF or FastAPI isn't installed).

The sample set includes two roadway sheets drawn the way CAD exports them: C-6.0 with labels and structures, and C-6.1 rotated, with no structures or edge labels, which can only be placed through its matchline. Regenerate the samples with `python scripts/make_sample_plans.py` (needs reportlab and matplotlib) and `python scripts/make_sample_landxml.py`.

## How plan interpretation works

1. **Upload** (`POST /api/projects/{id}/documents`). The PDF is stored and processed in the background.
2. **Per-sheet extraction** (`gv/ingest/pdf.py`, PyMuPDF):
   * Sheet number (`C-4.0`, `G03`, `RW02` …) and title from the title block, whether it runs along the bottom or as a strip down the right edge with a rotated title. Text printed on most sheets of the set (client, project name, firm) is treated as boilerplate, so a project named "… Water System …" doesn't make every sheet a water sheet. Sheets are classified as grading, existing, storm, sanitary, water, roadway, profile, cross section, details, wall, etc.
   * Drawing scale from text like `1" = 40'` or `1:500`. When the scale is drawn as line strokes (AutoCAD SHX fonts plot every letter as tiny lines, so there is no text to read), the line under each view title is read by OCR (`gv/ingest/ocr.py`, optional `rapidocr-onnxruntime`), accepted only as a standard engineering scale, and flagged for confirmation.
   * Vector geometry: line segments with width and dash style, small closed symbols (structures), polylines (contours).
   * Labelled values: structure IDs (CB-4, SMH-2 …), RIM, INV IN/OUT with direction, pipe callouts (`31 LF 15" RCP @ 1.00%`), spot elevations (FG, BC, FL, EG …), contour labels, stations.
   * Labels are grouped with the nearest structure ID; structure IDs are snapped to the nearest symbol of the expected shape.
   * Roadway sheets (and any plan sheet with station labels): line work is chained back into whole lines (CAD plots dashed and center linetypes as hundreds of separate dashes, and arcs as curves), and station labels (`STA 10+00` or bare `10+00`), edge labels (EP, BC, SIDEWALK, R/W …), matchlines (`MATCHLINE STA 3+00 / SEE SHEET C-6.1`) and pavement layers (`1-1/2" HMA SURFACE COURSE`) are read. Profile grids aren't read as contours.
   * Scanned pages: OpenCV line/symbol detection plus the vision model (if enabled).
3. **Assembly** (`gv/ingest/assemble.py`):
   * Plan sheets with a known scale are converted to feet and **registered into one coordinate frame** through the structures they share (least-squares similarity fit), matchline stationing, or **reference marks**: two sheets at the same scale with at least two small marks at exactly the same spots, well apart (a plan set's "common control points"), are the same view. Marks repeated on most sheets (north arrows, legend symbols) don't count. Sheets that link to each other form groups; the model is built on the most useful group (survey control, then both surfaces, then structures), and every other sheet is listed with what it lines up with and a request for control points. Scale mismatches and poor fits are reported.
   * **Place by matching points**: when nothing ties a sheet to the model, **Plans → Sheets & calibration → Place** shows it beside a placed sheet; tap the same spot on both (a manhole, a road or pad corner), then a second spot far from the first (`PUT /api/projects/{id}/match`). The placement works in either direction (matching a placed sheet's neighbour places it too, and everything linked to it follows), is kept when plans are re-processed, and is reported with a warning if the points imply a different scale.
   * **Printed coordinates** (`gv/ingest/control.py`): N/E callouts such as an alignment control sheet's "STA. 10+00.00P2 … BEGINNING / N. 130437.12 / E. 1791742.12" are read with the point their leader marks (followed from the text to the arrow tip, since the text sits wherever it fits). Three or more that agree within 2 ft place the sheet in real coordinates on its own; points that don't fit (a leader followed to the wrong spot) are dropped and reported. They rank after control points a person sets, an earthwork model and structures.
   * Structures are merged across sheets. Elevations come only from printed text. When the plan and profile agree, the value is marked **verified**; a single reading stays **AI** with a confidence; disagreements are flagged with both readings.
   * Pipes come from thick plan line work connecting two structures. Inverts are matched by IN/OUT and compass side (e.g. `INV IN (W)`). Sizes/materials come from the nearest callout on the plan or between the two structures on the profile. Stated slope is checked against the inverts.
   * Existing and finished surfaces are triangulated (TIN) from labelled contours and spot elevations. A sheet titled existing or finished grade says what its contours are; otherwise dashed = existing, solid = proposed. Unlabelled contours between the labelled ones get elevations by counting, the way a person does (`gv/ingest/contours.py`): transects square to the line count the lines crossed out to the nearest labels on each side, or to two labels on one side at the top or bottom of a slope. A line that crosses a labelled contour, gives different answers along its length, or implies an interval that isn't a round number is left out. Counted contours are lower confidence and reported as such. Labelled contours whose value doesn't fit their neighbours (a "1070" clipped to "070") are left out and reported. Triangles longer than 250 ft are dropped, so a surface doesn't bridge gaps its plans never covered (finished grade drawn only inside the work limits).
   * **Roadway** (`gv/ingest/roadway.py`): the centerline is the line along which the station labels fit one consistent stationing, with road edges on both sides at irregular offsets (which rules out profile grids). The fit measures the scale, so a half-size print or a wrong scale note is reported, and a sheet with no scale gets one. Edges are lines at a constant offset from the centerline, named from labels, or from their order outward (pavement, curb, sidewalk) at lower confidence when there are no labels; dashed lines are never named by order. Roadway sheets that share no structures are placed through their matchline: same station, same point, same direction. Each edge (`EP-LT`, `BC-RT`, `SW-IN-LT` …) and the pavement section are values to confirm or correct like any other. Centerline elevations are sampled from the finished-grade surface (marked calculated); the 2% crown is assumed for drawing only.
   * Nothing missing is filled in. Missing values become issues that say what sheet or information is needed.
4. **Validation** (`gv/geometry/validate.py`): inverts above rims, adverse slopes, low cover, rims that don't match finished grade, elevation outliers (datum or misread digit), duplicate IDs.
5. **Analysis** (`gv/analysis.py`): slopes, cover, depths, utility crossings with vertical clearance, earthwork (cut/fill in CY with covered area), review list, status counts.

## LandXML surfaces

Upload a LandXML 1.x file (`.xml`, exported from Civil 3D, OpenRoads, Trimble Business Center, Carlson …) to a project, through the same **Upload** button as plan PDFs.

* **What is read:** TIN surfaces (`<Surface>/<Definition surfType="TIN">`), using the file's own triangles, so breaklines and boundaries are kept and nothing is interpolated outside the surface. Points are read in LandXML order (northing, easting, elevation). Linear units are converted to feet. `<CoordinateSystem epsgCode>` fills in the project EPSG if it is empty.
* **Which surface is which:** guessed from the name only (`EG`, `Existing`, `OG`, `Topo` → existing ground; `FG`, `Finished`, `Proposed`, `Design` → finished grade). Corridor layers such as `Datum`, `Subgrade` or volume surfaces, and anything unclear, are left unassigned and listed as issues. Pick or change the role in **Plans → Imported surfaces** (`PUT /api/projects/{id}/surfaces`).
* **Precedence:** an imported surface replaces the one built from plan contours for the same role. When the plan sheets are registered to the same coordinates (control points), the two are compared and the mean difference and RMS are reported. A consistent offset points to a vertical datum difference.
* **Coordinates:** state plane / UTM values are too large to draw precisely, so the first real-world coordinates a project receives (LandXML or control points) set a **false origin** (`datum.false_origin`, rounded to 1000 ft). Project coordinates are `x = E − E0`, `z = −(N − N0)`. Control points and LandXML use the same origin, so they line up. Coordinate readouts in the app add it back.
* If the plan sheets are still sheet-local (no control points), the surfaces won't line up with them, and an issue says so.
* The project's elevation offset (Datum & units) applies to LandXML elevations too.

Every extracted value is a `Fact` with `status` (verified, ai, user, missing), `confidence`, `source` sheet, `page`, `bbox` on the page, and the raw `text`.

## Engineering accuracy controls

| Control | Where |
|---|---|
| Drawing scale calibration | Read from each sheet; override per sheet in **Plans → Sheets & calibration** (`PUT /api/projects/{id}/calibration`) |
| Coordinate system | Sheets registered to each other automatically; register to real coordinates with control points (`control_points: [[x_pt, y_pt, X_ft, Z_ft], …]` on the calibration endpoint) and set `datum.epsg` |
| Elevation datum | **Plans → Datum & units**: vertical datum name and an offset applied to every plan elevation |
| Units | Imperial and metric scales (`1" = 40'`, `1:500`); everything stored in feet |
| Geometry validation | `gv/geometry/validate.py` |
| Confidence scores | On every extracted value; cross-sheet agreement raises confidence |
| Manual correction | Confirm / Correct on any value; corrections are stored and replayed when plans are re-processed |
| Source references | Sheet, page, location and raw text on every value, with a cropped image of the drawing |

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | Vision/OpenCV availability |
| GET/POST | `/api/projects` | List / create |
| GET/DELETE | `/api/projects/{id}` | Project model + analysis |
| POST | `/api/projects/{id}/documents` | Upload a PDF plan set or a LandXML file (multipart `file`) |
| GET | `/api/documents/{doc}` | Processing status and per-sheet summary |
| GET | `/api/documents/{doc}/pages/{page}.png?bbox=x0,y0,x1,y1` | Sheet image or crop around a value |
| PATCH | `/api/projects/{id}/fields` | `{object_id, field, action: confirm|set, value}` |
| PUT | `/api/projects/{id}/calibration` | Scale or control points for one sheet |
| PUT | `/api/projects/{id}/datum` | Vertical datum, offset, EPSG |
| PUT | `/api/projects/{id}/match` | Place a sheet from points matched with a placed sheet: `{document_id, page, ref_document_id, ref_page, pairs: [[x, y, ref_x, ref_y], …]}` (empty pairs clears) |
| PUT | `/api/projects/{id}/surfaces` | Role of an imported surface: `{document_id, index, role: existing|proposed|ignore}` |
| GET | `/api/projects/{id}/pipes/{pipe}/trench` | Trench excavation, bedding, backfill quantities |
| POST | `/api/projects/{id}/ask` | Assistant, `{q, selected}`; answers name their sheets |

Interactive docs: http://localhost:8000/docs

## Status against the Phase 2 spec

Working now:
* PDF upload, background processing, structured database (SQLite or PostgreSQL/PostGIS)
* Text and vector extraction, sheet identification, scale reading, scanned-sheet detection
* Structures, rims, inverts, pipe callouts, spot elevations, contours, stations
* Multi-sheet registration, plan-profile cross-checking, confidence, issues for missing data
* Existing and finished TIN surfaces, cut/fill engine, trench quantities
* 3D model from extracted data: surfaces, cut/fill heat map, structures, pipes, crossings
* Time Machine, bird's-eye, walk, operator, X-ray, finished, road-driving (when a roadway alignment exists)
* Manual correction with replay, calibration and datum controls, source-sheet crops
* LandXML TIN surface import, with roles, a false origin, and a cross-check against plan contours
* Roadway from plan sheets: centerline and stationing, pavement / curb / sidewalk / R/W edges per side, pavement section, sheets joined at matchlines
* Assistant grounded in the extracted data, with sheet references

Not yet extracted from uploaded plans (the demo project shows how they'll look):
* Driveways, parking lots, trails, intersections and curb returns; more than one road per project; widths that change along the road (one width per side is modeled, and differences are flagged)
* Road profiles from the profile half of plan & profile sheets (centerline elevations come from the finished-grade surface), and cross slope from typical sections
* Retaining walls, building footprints and foundations
* Water and sewer mains drawn as continuous lines without labelled structures
* Cross-section sheets, and linking profiles to plans by stationing (today they link by structure ID)
* DXF and IFC import; LandXML alignments, profiles and pipe networks (surfaces are imported)

Suggested next steps, in order: run real plan sets through and fix what they expose (the reader has so far been tested on generated samples, which can't show every firm's drafting habits), reading road profiles from plan & profile sheets by station, a calibration UI with click-to-place control points, then the React Three Fiber migration of `apps/web`.
