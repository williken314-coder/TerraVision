"""Everything the geometry engine derives from a project, in one payload."""
from __future__ import annotations

from .geometry.alignment import Alignment
from .geometry.surface import Grid
from .geometry.utilities import earthwork, find_crossings, line_length, pipe_calc, structure_depth
from .geometry.validate import validate
from .models import Fact, Project, Status


def alignment_of(prj: Project) -> Alignment | None:
    if not prj.alignment or len(prj.alignment.points) < 2:
        return None
    pts = prj.alignment.points
    return Alignment([p.s for p in pts], [p.x for p in pts], [p.z for p in pts])


def station_fn(al: Alignment | None):
    def f(x, z):
        if al is None:
            return None, None
        s, d = al.to_so(x, z)
        return float(s[0]), float(d[0])
    return f


FIELDS = {"structure": ["rim"], "pipe": ["dia_in", "material", "inv_up", "inv_dn"], "line": ["dia_in", "material", "cover"],
          "edge": ["offset"], "section": ["pavement_section"]}


def _objects(prj: Project):
    yield from (("structure", s) for s in prj.structures)
    yield from (("pipe", p) for p in prj.pipes)
    yield from (("line", l) for l in prj.lines)
    if prj.corridor and prj.corridor.edges:  # read from uploaded plans (the demo's typical section is fixed)
        yield from (("edge", e) for e in prj.corridor.edges)
        yield ("section", prj.corridor)


def review_items(prj: Project) -> list[dict]:
    out = []
    for kind, obj in _objects(prj):
        for f in FIELDS[kind]:
            fact = getattr(obj, f)
            if kind == "structure" and obj.kind == "HW" and f == "rim" and fact.value is None:
                continue  # headwalls have no rim
            if isinstance(fact, Fact) and fact.needs_review():
                out.append(dict(object_id=obj.id, field=f, status=fact.status.value, confidence=fact.confidence, value=fact.value,
                                source=fact.source or getattr(obj, "sheet", None), page=fact.page, bbox=fact.bbox, text=fact.text,
                                document_id=fact.document_id))
    return out


def status_counts(prj: Project, calculated: int) -> dict:
    counts = {s.value: 0 for s in Status}
    for kind, obj in _objects(prj):
        for f in FIELDS[kind]:
            counts[getattr(obj, f).status.value] += 1
    counts["calculated"] = calculated
    return counts


def analyze(prj: Project) -> dict:
    fg = Grid.from_model(prj.proposed) if prj.proposed else None
    eg = Grid.from_model(prj.existing) if prj.existing else None
    al = alignment_of(prj)
    pipes = {p.id: pipe_calc(p, prj, fg).as_dict() for p in prj.pipes}
    depths = {s.id: structure_depth(s, prj) for s in prj.structures}
    lines = {l.id: dict(length=line_length(l)) for l in prj.lines}
    crossings = find_crossings(prj, station_fn(al))
    calculated = sum(1 for c in pipes.values() for k in ("length", "slope", "min_cover") if c[k] is not None) \
        + sum(1 for v in depths.values() if v is not None) + len(lines) + len(crossings)
    checks = validate(prj)
    return dict(
        pipes=pipes,
        structure_depths=depths,
        lines=lines,
        crossings=crossings,
        earthwork=earthwork(eg, fg),
        review=review_items(prj),
        status_counts=status_counts(prj, calculated),
        checks=[i.model_dump() for i in checks],
        issues=[i.model_dump() for i in prj.issues] + [i.model_dump() for i in checks],
    )
