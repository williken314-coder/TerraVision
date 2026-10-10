"""Project data model.

Every value read from the plans is wrapped in a ``Fact`` that records exactly
where it came from (sheet, page, location on the page, raw text) and how much
to trust it. The geometry engine only consumes values; it never invents them.
Anything it derives is reported separately as "calculated" in
:mod:`gv.analysis`.
"""
from __future__ import annotations

from enum import Enum
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Status(str, Enum):
    verified = "verified"  # read directly from the plans with high certainty
    ai = "ai"  # interpreted by software/AI, needs a person to confirm
    user = "user"  # confirmed or corrected by a person
    missing = "missing"  # not found on any sheet


class Fact(BaseModel, Generic[T]):
    value: Optional[T] = None
    status: Status = Status.verified
    confidence: Optional[float] = None
    source: Optional[str] = None  # sheet number, e.g. "C-4.1"
    page: Optional[int] = None  # 0-based page in the uploaded PDF
    bbox: Optional[list[float]] = None  # [x0, y0, x1, y1] in PDF points
    text: Optional[str] = None  # raw text the value was read from
    document_id: Optional[str] = None

    def needs_review(self) -> bool:
        return self.status in (Status.ai, Status.missing)


System = Literal["storm", "san", "water", "elec"]


class Sheet(BaseModel):
    id: str
    title: str
    notes: str = ""
    category: str = "unknown"
    page: Optional[int] = None
    document_id: Optional[str] = None
    scale_ft_per_in: Optional[float] = None
    registered: bool = False  # transformed into the project coordinate system


class AlignmentPoint(BaseModel):
    s: float  # station, ft
    x: float
    z: float
    elev: Optional[float] = None  # finished-grade profile at centerline; None until a profile or surface gives it


class Alignment(BaseModel):
    name: str
    points: list[AlignmentPoint]
    source: Optional[str] = None  # sheet(s) the centerline and stationing were read from
    confidence: Optional[float] = None
    profile_source: Optional[str] = None  # where the elevations come from


EdgeKind = Literal["pavement", "curb_face", "curb_back", "walk_inner", "walk_outer", "row"]


EDGE_CODES = {"pavement": "EP", "curb_face": "FC", "curb_back": "BC", "walk_inner": "SW-IN", "walk_outer": "SW-OUT", "row": "RW"}
EDGE_NAMES = {"pavement": "edge of pavement", "curb_face": "face of curb", "curb_back": "back of curb", "walk_inner": "sidewalk, inner edge",
              "walk_outer": "sidewalk, outer edge", "row": "right-of-way"}


class RoadEdge(BaseModel):
    """A line running at a constant offset from the centerline, read from a roadway plan."""

    id: str  # e.g. "EP-LT", "BC-RT"
    kind: EdgeKind
    side: Literal["L", "R"]
    offset: Fact[float]  # ft from centerline, always positive
    station_from: float
    station_to: float
    named_by: Literal["label", "order"]  # a label on the plan, or its order outward from the centerline


class CorridorTemplate(BaseModel):
    """Typical section, offsets in feet from centerline.

    The symmetric fields describe the demo project. Corridors read from uploaded plans carry ``edges``
    (per side, with sources) and leave anything the plans don't show as None.
    """

    pavement_half_width: Optional[float] = 14.0
    crown_slope: float = 0.02
    crown_slope_assumed: bool = False  # not read from the plans; used only to draw the road
    curb_width: Optional[float] = 1.0
    curb_height: float = 0.5
    walk_inner: Optional[float] = 18.0
    walk_outer: Optional[float] = 23.0
    pavement_section: Fact[str] = Field(default_factory=lambda: Fact[str](value="1.5 in HMA surface / 2.5 in HMA base / 8 in aggregate base", source="C-8.0"))
    edges: list[RoadEdge] = []
    id: str = "SECTION"  # object id for confirming / correcting the pavement section

    def edge(self, kind: str, side: str) -> Optional[float]:
        e = next((e for e in self.edges if e.kind == kind and e.side == side), None)
        return e.offset.value if e else None


class GridSurface(BaseModel):
    """Elevations on a regular grid, row-major: index = j * (nx + 1) + i. NaN cells are serialized as null."""

    name: str
    x0: float
    z0: float
    cell: float
    nx: int
    nz: int
    elev: list[Optional[float]]
    source: Optional[str] = None
    point_count: Optional[int] = None  # number of source elevations behind the surface


class Structure(BaseModel):
    id: str
    system: System
    kind: Literal["CB", "SMH", "HW", "FH", "TX", "MH", "INL", "OTHER"]
    x: float
    z: float
    station: Optional[float] = None
    offset: Optional[float] = None
    position: Fact[list[float]] = Field(default_factory=lambda: Fact[list[float]]())
    rim: Fact[float]
    type: Fact[str]
    sheet: str


class Pipe(BaseModel):
    id: str
    system: System
    from_id: str
    to_id: str
    dia_in: Fact[float]
    material: Fact[str]
    wall_ft: float
    inv_up: Fact[float]
    inv_dn: Fact[float]
    sheet: str


class LinePoint(BaseModel):
    s: float
    x: float
    z: float
    ground: float
    center: float  # pipe centerline elevation


class LinearUtility(BaseModel):
    """A continuous main (water, conduit) run along an offset from the alignment."""

    id: str
    system: System
    name: str
    dia_in: Fact[float]
    material: Fact[str]
    od_ft: float
    cover: Fact[float]
    offset: float
    points: list[LinePoint]
    sheet: str


class Building(BaseModel):
    id: str
    lot: int
    station: float
    offset: float
    x: float
    z: float
    rotation: float
    width: float = 44.0
    depth: float = 34.0
    pad: float
    ffe: Fact[float]


class Tree(BaseModel):
    x: float
    z: float
    y: float
    h: float
    group: Literal["keep", "clear", "new"]


class Pond(BaseModel):
    x: float
    z: float
    rx: float
    rz: float
    pool: float


class Issue(BaseModel):
    """Something the system could not resolve on its own."""

    severity: Literal["blocker", "warning", "info"]
    code: str
    message: str
    object_id: Optional[str] = None
    sheet: Optional[str] = None
    page: Optional[int] = None


class Datum(BaseModel):
    vertical: str = "NAVD88"
    horizontal: str = "Sheet-local (unregistered)"
    epsg: Optional[int] = None  # e.g. 2227 for CA State Plane Zone III, US ft
    elevation_offset_ft: float = 0.0  # added to every plan elevation (datum conversion)
    units: Literal["ft", "m"] = "ft"
    # [E, N] subtracted from real-world coordinates (control points, LandXML) so project
    # coordinates stay small: x = E - E0, z = -(N - N0). Set once, the first time real coordinates arrive.
    false_origin: Optional[list[float]] = None


class Document(BaseModel):
    id: str
    filename: str
    pages: int
    status: Literal["queued", "processing", "done", "failed"] = "queued"
    error: Optional[str] = None
    uploaded_at: str
    vision_used: bool = False


class ImportedSurface(BaseModel):
    """A TIN surface from an imported LandXML file, and the role it plays in the model."""

    document_id: str
    index: int  # position among the file's surfaces
    name: str
    desc: str = ""
    points: int
    faces: int
    role: Optional[Literal["existing", "proposed", "ignore"]] = None
    role_source: Optional[Literal["name", "user"]] = None  # guessed from the surface name, or set by a person
    used: bool = False  # this surface is the project's existing / finished surface


class UnplacedStructure(BaseModel):
    """A structure read by name on sheets that aren't placed in the model, so it has no location yet."""

    id: str
    system: System
    sheets: list[str]
    rim: Optional[float] = None  # as read (a structure table, a callout); values, not a location
    inverts: list[dict] = []  # {elev, side, dir}
    type: Optional[str] = None
    source: Optional[str] = None  # sheet the rim was read on


class Project(BaseModel):
    id: str
    name: str
    demo: bool = False
    units: Literal["ft"] = "ft"
    datum: Datum = Field(default_factory=Datum)
    documents: list[Document] = []
    sheets: list[Sheet] = []
    alignment: Optional[Alignment] = None
    corridor: Optional[CorridorTemplate] = None
    existing: Optional[GridSurface] = None
    proposed: Optional[GridSurface] = None
    imported_surfaces: list[ImportedSurface] = []
    structures: list[Structure] = []
    unplaced: list[UnplacedStructure] = []
    pipes: list[Pipe] = []
    lines: list[LinearUtility] = []
    buildings: list[Building] = []
    trees: list[Tree] = []
    boundaries: list[list[list[float]]] = []  # polylines of [x, y, z]
    pond: Optional[Pond] = None
    issues: list[Issue] = []

    # convenience lookups -------------------------------------------------
    def structure(self, sid: str) -> Optional[Structure]:
        return next((s for s in self.structures if s.id == sid), None)

    def pipe(self, pid: str) -> Optional[Pipe]:
        return next((p for p in self.pipes if p.id == pid), None)

    def line(self, lid: str) -> Optional[LinearUtility]:
        return next((l for l in self.lines if l.id == lid), None)

    def road_edge(self, eid: str) -> Optional[RoadEdge]:
        return next((e for e in self.corridor.edges if e.id == eid), None) if self.corridor else None

    def find(self, oid: str):
        return self.structure(oid) or self.pipe(oid) or self.line(oid) or self.road_edge(oid)
