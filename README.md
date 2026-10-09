# GroundVision AI

Upload a civil plan set (PDF). GroundVision reads it, builds a 3D model of the jobsite from what is actually printed on the sheets, and flags everything it isn't sure about for a person to confirm.

```
apps/web/            The GroundVision interface (Three.js), served by the API at /
services/api/        Python backend: FastAPI, PDF reading, geometry engine, database
  gv/ingest/         Plan interpretation pipeline (PyMuPDF, OpenCV, optional vision model)
  gv/geometry/       Deterministic engineering math (alignments, TIN surfaces, pipes, crossings, earthwork, trenches, checks)
  gv/api.py          HTTP API
  samples/           A generated sample plan set to test uploads with
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

`test_geometry.py` and `test_extraction.py` need only numpy/scipy/pydantic. `test_pdf_and_api.py` runs the real PDF reader and the HTTP API end to end on the sample plan set (skipped if PyMuPDF or FastAPI isn't installed).

> Note: the first version of this code was written in an environment that couldn't install PyMuPDF, FastAPI, or SQLAlchemy. The geometry, assembly, and assistant tests passed there; `test_pdf_and_api.py` gets its first real run on your machine. If anything fails, run `pytest -x` and fix from the top.

## How plan interpretation works

1. **Upload** (`POST /api/projects/{id}/documents`). The PDF is stored and processed in the background.
2. **Per-sheet extraction** (`gv/ingest/pdf.py`, PyMuPDF):
   * Sheet number and title from the title block, classified as grading, storm, sanitary, water, roadway, profile, details, etc.
   * Drawing scale from text like `1" = 40'` or `1:500`.
   * Vector geometry: line segments with width and dash style, small closed symbols (structures), polylines (contours).
   * Labelled values: structure IDs (CB-4, SMH-2 …), RIM, INV IN/OUT with direction, pipe callouts (`31 LF 15" RCP @ 1.00%`), spot elevations (FG, BC, FL, EG …), contour labels, stations.
   * Labels are grouped with the nearest structure ID; structure IDs are snapped to the nearest symbol of the expected shape.
   * Scanned pages: OpenCV line/symbol detection plus the vision model (if enabled).
3. **Assembly** (`gv/ingest/assemble.py`):
   * Plan sheets with a known scale are converted to feet and **registered into one coordinate frame** through the structures they share (least-squares similarity fit). Scale mismatches and poor fits are reported.
   * Structures are merged across sheets. Elevations come only from printed text. When the plan and profile agree, the value is marked **verified**; a single reading stays **AI** with a confidence; disagreements are flagged with both readings.
   * Pipes come from thick plan line work connecting two structures. Inverts are matched by IN/OUT and compass side (e.g. `INV IN (W)`). Sizes/materials come from the nearest callout on the plan or between the two structures on the profile. Stated slope is checked against the inverts.
   * Existing and finished surfaces are triangulated (TIN) from labelled contours (dashed = existing, solid = proposed) and spot elevations.
   * Nothing missing is filled in. Missing values become issues that say what sheet or information is needed.
4. **Validation** (`gv/geometry/validate.py`): inverts above rims, adverse slopes, low cover, rims that don't match finished grade, elevation outliers (datum or misread digit), duplicate IDs.
5. **Analysis** (`gv/analysis.py`): slopes, cover, depths, utility crossings with vertical clearance, earthwork (cut/fill in CY with covered area), review list, status counts.

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
| POST | `/api/projects/{id}/documents` | Upload a PDF (multipart `file`) |
| GET | `/api/documents/{doc}` | Processing status and per-sheet summary |
| GET | `/api/documents/{doc}/pages/{page}.png?bbox=x0,y0,x1,y1` | Sheet image or crop around a value |
| PATCH | `/api/projects/{id}/fields` | `{object_id, field, action: confirm|set, value}` |
| PUT | `/api/projects/{id}/calibration` | Scale or control points for one sheet |
| PUT | `/api/projects/{id}/datum` | Vertical datum, offset, EPSG |
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
* Assistant grounded in the extracted data, with sheet references

Not yet extracted from uploaded plans (the demo project shows how they'll look):
* Roadway pavement, curb and gutter, sidewalks, driveways, parking lots, trails (need edge-of-pavement linework classification and the typical section)
* Retaining walls, building footprints and foundations
* Water and sewer mains drawn as continuous lines without labelled structures
* Cross-section sheets, and linking profiles to plans by stationing (today they link by structure ID)
* DXF, LandXML and IFC import (LandXML surfaces would be the quickest win)

Suggested next steps, in order: LandXML surface import, pavement edge extraction from roadway sheets, station-based plan/profile linking, a calibration UI with click-to-place control points, then the React Three Fiber migration of `apps/web`.
