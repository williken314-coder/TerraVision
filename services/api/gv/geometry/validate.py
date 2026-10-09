"""Engineering sanity checks on an assembled project. Findings become issues; nothing is changed."""
from __future__ import annotations

import numpy as np

from ..models import Issue, Project
from .surface import Grid
from .utilities import pipe_calc

MIN_COVER_FT = 3.0  # typical; actual requirement comes from the specs


def validate(prj: Project) -> list[Issue]:
    out: list[Issue] = []
    fg = Grid.from_model(prj.proposed) if prj.proposed else None
    ids = [s.id for s in prj.structures]
    for d in {i for i in ids if ids.count(i) > 1}:
        out.append(Issue(severity="warning", code="duplicate_id", object_id=d, message=f"{d} is used for more than one structure."))
    rims = [s.rim.value for s in prj.structures if s.rim.value is not None]
    med = float(np.median(rims)) if rims else None
    for s in prj.structures:
        if s.rim.value is not None and med is not None and abs(s.rim.value - med) > 150:
            out.append(Issue(severity="warning", code="elevation_outlier", object_id=s.id,
                             message=f"{s.id} rim {s.rim.value:.2f} is {abs(s.rim.value - med):.0f} ft from the site median. Check the datum or a misread digit."))
        if fg is not None and s.rim.value is not None:
            g = float(fg.sample(s.x, s.z))
            if np.isfinite(g) and abs(g - s.rim.value) > 2.0:
                out.append(Issue(severity="warning", code="rim_vs_grade", object_id=s.id,
                                 message=f"{s.id} rim {s.rim.value:.2f} differs from finished grade at its location ({g:.2f}) by {abs(g - s.rim.value):.1f} ft."))
    for p in prj.pipes:
        a, b = prj.structure(p.from_id), prj.structure(p.to_id)
        for sid, inv in ((a.id, p.inv_up.value), (b.id, p.inv_dn.value)):
            s = prj.structure(sid)
            if inv is not None and s.rim.value is not None and inv >= s.rim.value:
                out.append(Issue(severity="blocker", code="invert_above_rim", object_id=p.id, message=f"{p.id} invert {inv:.2f} at {sid} is at or above the rim {s.rim.value:.2f}."))
        c = pipe_calc(p, prj, fg)
        if c.slope is not None and p.system in ("storm", "san"):
            if c.slope <= 0:
                out.append(Issue(severity="blocker", code="adverse_slope", object_id=p.id, message=f"{p.id} runs flat or uphill ({c.slope * 100:.2f}%) from {p.from_id} to {p.to_id}."))
            elif c.slope < 0.003:
                out.append(Issue(severity="info", code="flat_slope", object_id=p.id, message=f"{p.id} slope {c.slope * 100:.2f}% is below 0.30%. Confirm it meets minimum velocity."))
        if c.min_cover is not None and c.min_cover < MIN_COVER_FT:
            out.append(Issue(severity="warning", code="low_cover", object_id=p.id,
                             message=f"{p.id} has {c.min_cover:.1f} ft of cover at its shallowest point (typical minimum {MIN_COVER_FT:.0f} ft; verify against the specs)."))
    return out
