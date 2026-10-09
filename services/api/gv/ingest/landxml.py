"""Read TIN surfaces from LandXML 1.x files (Civil 3D, OpenRoads, Trimble Business Center, Carlson ...).

What is read:

* ``<Surfaces>/<Surface>/<Definition surfType="TIN">``: points (``<P id>`` as
  "northing easting elevation", the LandXML order) and faces (``<F>`` as three
  point ids). Faces flagged invisible (``i="1"``) are outside the surface
  boundary and are dropped. The file's own triangles are kept as drawn, so
  breaklines and boundaries survive; nothing is re-triangulated unless the
  file has points but no faces.
* ``<Units>`` linear unit, converted to feet.
* ``<CoordinateSystem epsgCode>`` when present.

Whether a surface is existing ground or finished grade is guessed from its
name and description only. Anything unclear is left unassigned for a person
to decide; a guess is never used silently for an ambiguous name.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from ..geometry.surface import TIN, Grid, TriMesh
from ..models import GridSurface

# linearUnit -> feet. Survey and international feet are both taken as the project foot
# (2 ppm apart); metric is converted with the international foot.
UNIT_FT = {"foot": 1.0, "usfoot": 1.0, "ussurveyfoot": 1.0, "internationalfoot": 1.0, "ft": 1.0,
           "meter": 1 / 0.3048, "metre": 1 / 0.3048, "m": 1 / 0.3048, "millimeter": 1 / 304.8, "centimeter": 1 / 30.48,
           "kilometer": 1000 / 0.3048, "inch": 1 / 12, "mile": 5280.0}

PROPOSED_NAME = re.compile(r"(?<![A-Z])(FG|FIN(ISH(ED)?)?|PROP(OSED)?|DESIGN|GRADING|FINAL)(?![A-Z])", re.I)
EXISTING_NAME = re.compile(r"(?<![A-Z])(EG|EXG|OG|EXIST(ING)?|ORIG(INAL)?|TOPO(GRAPHY|GRAPHIC)?|SURVEY|NATURAL)(?![A-Z])", re.I)
# corridor/layer surfaces that are neither ground: never guessed into a role
OTHER_NAME = re.compile(r"(?<![A-Z])(DATUM|SUB\s*GRADE|SUBBASE|SUB-BASE|BASE|PAVE(MENT)?|VOL(UME)?|COMPARISON|ISOPACH|CUT|FILL)(?![A-Z])", re.I)


class LandXMLError(ValueError):
    pass


@dataclass
class LXSurface:
    index: int
    name: str
    desc: str
    e: np.ndarray  # easting, ft
    n: np.ndarray  # northing, ft
    z: np.ndarray  # elevation, ft
    faces: np.ndarray  # (m, 3) indices into the point arrays; empty if the file had none
    dropped_faces: int = 0  # invisible or referring to missing points
    role_guess: str | None = None  # existing | proposed | None

    def summary(self) -> dict:
        return dict(kind="landxml_surface", index=self.index, name=self.name, desc=self.desc, points=int(len(self.z)),
                    faces=int(len(self.faces)), dropped_faces=self.dropped_faces, role_guess=self.role_guess,
                    extent=[float(self.e.min()), float(self.n.min()), float(self.e.max()), float(self.n.max())] if len(self.z) else None,
                    elev_range=[float(self.z.min()), float(self.z.max())] if len(self.z) else None)


@dataclass
class LandXMLFile:
    units: str
    ft_per_unit: float
    epsg: int | None
    surfaces: list[LXSurface] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def is_landxml(data: bytes) -> bool:
    head = data[:4096].lstrip(b"\xef\xbb\xbf").lstrip()
    return head.startswith(b"<") and b"<LandXML" in head


def guess_role(name: str, desc: str = "") -> str | None:
    text = f"{name} {desc}"
    if OTHER_NAME.search(text):
        return None
    prop, exist = bool(PROPOSED_NAME.search(text)), bool(EXISTING_NAME.search(text))
    if prop == exist:  # neither, or both (e.g. "EG to FG comparison")
        return None
    return "proposed" if prop else "existing"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _child(el, name):
    return next((c for c in el if _local(c.tag) == name), None)


def parse_landxml(data: bytes) -> LandXMLFile:
    head = data[:65536]
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        raise LandXMLError("LandXML files with a DOCTYPE or entity declarations are not accepted.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as ex:
        raise LandXMLError(f"Not valid XML: {ex}") from None
    if _local(root.tag) != "LandXML":
        raise LandXMLError("Not a LandXML file (root element is not <LandXML>).")

    units, k = "foot", 1.0
    u = _child(root, "Units")
    if u is not None and len(u):
        sysel = u[0]
        units = sysel.get("linearUnit") or ("meter" if _local(sysel.tag) == "Metric" else "foot")
        key = units.replace(" ", "").lower()
        if key not in UNIT_FT:
            raise LandXMLError(f"Unknown linear unit '{units}' in <Units>.")
        k = UNIT_FT[key]
    epsg = None
    cs = _child(root, "CoordinateSystem")
    if cs is not None and (cs.get("epsgCode") or "").strip().isdigit():
        epsg = int(cs.get("epsgCode").strip())

    out = LandXMLFile(units=units, ft_per_unit=k, epsg=epsg)
    surfs = _child(root, "Surfaces")
    for el in _children(surfs, "Surface") if surfs is not None else []:
        name = (el.get("name") or f"Surface {len(out.surfaces) + 1}").strip()
        desc = (el.get("desc") or "").strip()
        d = _child(el, "Definition")
        if d is None:
            out.notes.append(f"Surface '{name}' has no <Definition>; skipped.")
            continue
        if (d.get("surfType") or "TIN").upper() != "TIN":
            out.notes.append(f"Surface '{name}' is a {d.get('surfType')} surface; only TIN surfaces are read.")
            continue
        pnts = _child(d, "Pnts")
        ids, coords = [], []
        for p in _children(pnts, "P") if pnts is not None else []:
            vals = (p.text or "").split()
            if len(vals) < 3:
                continue
            try:
                coords.append((float(vals[0]), float(vals[1]), float(vals[2])))
            except ValueError:
                continue
            ids.append(p.get("id") or str(len(ids) + 1))
        if len(coords) < 3:
            out.notes.append(f"Surface '{name}' has fewer than three points; skipped.")
            continue
        arr = np.asarray(coords) * k
        index_of = {pid: i for i, pid in enumerate(ids)}
        faces, dropped = [], 0
        fel = _child(d, "Faces")
        for f in _children(fel, "F") if fel is not None else []:
            if f.get("i") == "1":
                dropped += 1
                continue
            tri = (f.text or "").split()
            try:
                faces.append((index_of[tri[0]], index_of[tri[1]], index_of[tri[2]]))
            except (KeyError, IndexError):
                dropped += 1
        s = LXSurface(index=len(out.surfaces), name=name, desc=desc, n=arr[:, 0], e=arr[:, 1], z=arr[:, 2],
                      faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3), dropped_faces=dropped, role_guess=guess_role(name, desc))
        out.surfaces.append(s)
    if not out.surfaces:
        raise LandXMLError("No TIN surfaces found in this LandXML file." + (" " + " ".join(out.notes) if out.notes else ""))
    return out


def to_project_xz(e, n, false_origin: list[float] | None):
    """Real-world easting/northing (ft) to project x/z (see ``Datum.false_origin``)."""
    e0, n0 = false_origin or (0.0, 0.0)
    return np.asarray(e) - e0, -(np.asarray(n) - n0)


def grid_surface(s: LXSurface, false_origin: list[float] | None, elevation_offset: float, label: str, max_cells: int = 400) -> GridSurface:
    """Sample the surface's own triangles onto a regular grid (at most ``max_cells`` per side, whole-foot cells)."""
    x, z = to_project_xz(s.e, s.n, false_origin)
    x0, z0 = float(np.floor(x.min())), float(np.floor(z.min()))
    ext = max(x.max() - x0, z.max() - z0)
    cell = float(max(1, int(np.ceil(ext / max_cells))))
    nx, nz = max(1, int(np.ceil((x.max() - x0) / cell))), max(1, int(np.ceil((z.max() - z0) / cell)))
    e = s.z + elevation_offset
    g: Grid = TriMesh(x, z, e, s.faces).to_grid(x0, z0, cell, nx, nz) if len(s.faces) else TIN(x, z, e).to_grid(x0, z0, cell, nx, nz)
    m = g.to_model(label, f"LandXML: {s.name}")
    m.point_count = int(len(s.z))
    return m
