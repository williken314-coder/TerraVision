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
    elev: float  # finished-grade profile at centerline


class Alignment(BaseModel):
    name: str
    points: list[AlignmentPoint]


class CorridorTemplate(BaseModel):
    """Typical section, offsets in feet from centerline."""

    pavement_half_width: float = 14.0
    crown_slope: float = 0.02
    curb_width: float = 1.0
    curb_height: float = 0.5
    walk_inner: float = 18.0
    walk_outer: float = 23.0
    pavement_section: Fact[str] = Field(default_factory=lambda: Fact[str](value="1.5 in HMA surface / 2.5 in HMA base / 8 in aggregate base", source="C-8.0"))


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


class Document(BaseModel):
    id: str
    filename: str
    pages: int
    status: Literal["queued", "processing", "done", "failed"] = "queued"
    error: Optional[str] = None
    uploaded_at: str
    vision_used: bool = False


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
    structures: list[Structure] = []
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

    def find(self, oid: str):
        return self.structure(oid) or self.pipe(oid) or self.line(oid)
