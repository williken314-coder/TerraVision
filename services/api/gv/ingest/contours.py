"""Elevations for the contours a plan doesn't label.

Plans label every fifth contour or so; the ones between are left for the
reader to count. This module counts them, the way a person does:

* From an unlabelled line, look out to each side (a short transect, square to
  the line) and list the contour lines crossed in order.
* Between labels: 960 on one side, 965 on the other, and the count of lines
  crossed on each side fixes the interval and this line's place in it.
* At the top or bottom of a slope there are labels on one side only: the two
  nearest labels on that side give the interval and the direction.

Lines that aren't contours (road edges, property lines) are kept out by two
checks: a contour never crosses a labelled contour, and every transect along
the line must give the same elevation. A line that fails either is left
without an elevation, never guessed.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class _SegIndex:
    """Segments bucketed on a grid, for fast transect crossings."""

    def __init__(self, polys: list[np.ndarray], cell: float):
        self.cell = cell
        a, b, owner = [], [], []
        for i, P in enumerate(polys):
            a.append(P[:-1])
            b.append(P[1:])
            owner.append(np.full(len(P) - 1, i))
        self.a, self.b = np.vstack(a), np.vstack(b)
        self.owner = np.concatenate(owner)
        self.buckets: dict[tuple, list[int]] = defaultdict(list)
        lo = np.floor(np.minimum(self.a, self.b) / cell).astype(int)
        hi = np.floor(np.maximum(self.a, self.b) / cell).astype(int)
        for k in range(len(self.a)):
            for gx in range(lo[k, 0], hi[k, 0] + 1):
                for gy in range(lo[k, 1], hi[k, 1] + 1):
                    self.buckets[(gx, gy)].append(k)

    def crossings(self, p: np.ndarray, q: np.ndarray):
        """Polylines crossed by segment p->q: (owner, distance along p->q), nearest first, one per owner."""
        lo, hi = np.floor(np.minimum(p, q) / self.cell).astype(int), np.floor(np.maximum(p, q) / self.cell).astype(int)
        ks = {k for gx in range(lo[0], hi[0] + 1) for gy in range(lo[1], hi[1] + 1) for k in self.buckets.get((gx, gy), ())}
        if not ks:
            return []
        ks = np.fromiter(ks, int)
        a, b = self.a[ks], self.b[ks]
        r, s = q - p, b - a
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = np.abs(den) > 1e-12
        qa = a - p
        t = np.where(ok, (qa[:, 0] * s[:, 1] - qa[:, 1] * s[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (qa[:, 0] * r[1] - qa[:, 1] * r[0]) / np.where(ok, den, 1), -1)
        hit = ok & (t > 1e-6) & (t <= 1) & (u >= 0) & (u <= 1)
        best: dict[int, float] = {}
        for o, tt in zip(self.owner[ks[hit]], t[hit]):
            if o not in best or tt < best[o]:
                best[int(o)] = float(tt)
        return sorted(best.items(), key=lambda kv: kv[1])


def _interval_ok(step: float) -> bool:
    """Contour intervals are round numbers: 0.25, 0.5, 1, 2, 2.5, 5, 10 ..."""
    return any(abs(step - v) < 0.02 * v for v in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 4, 5, 10, 20, 25, 50))


def derive_elevations(polys: list[np.ndarray], labelled: dict[int, float], reach: float, samples: int = 5, passes: int = 6) -> dict[int, tuple[float, int]]:
    """Elevations for unlabelled polylines: {index: (elevation, transects that agree)}.

    ``polys``: contour-like polylines in page points. ``labelled``: {index: elevation} for those with a label.
    ``reach``: how far a transect looks out on each side, in points.
    Lines counted in one pass are references for the next, as a person counts on up a long slope; only lines
    that never cross a *labelled* contour qualify, whatever pass they're found in.
    """
    if len(labelled) < 2 or not polys:
        return {}
    idx = _SegIndex(polys, cell=max(reach / 2, 10.0))
    crosses_label = {i for i, P in enumerate(polys) if i not in labelled
                     and any(o in labelled for k in range(len(P) - 1) for o, _ in idx.crossings(P[k], P[k + 1]) if o != i)}
    known = dict(labelled)
    out: dict[int, tuple[float, int]] = {}
    for _ in range(passes):
        found = _pass(polys, known, idx, crosses_label, reach, samples)
        if not found:
            break
        out.update(found)
        known.update({i: e for i, (e, _) in found.items()})
    return out


def _pass(polys, known: dict[int, float], idx: _SegIndex, skip: set, reach: float, samples: int) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    for i, P in enumerate(polys):
        if i in known or i in skip or len(P) < 2:
            continue
        seg = np.hypot(*np.diff(P, axis=0).T)
        T = np.concatenate([[0.0], np.cumsum(seg)])
        if T[-1] <= 0:
            continue
        votes = []
        for f in (np.arange(samples) + 0.5) / samples:
            t = f * T[-1]
            k = min(len(seg) - 1, int(np.searchsorted(T, t, side="right") - 1))
            d = (P[k + 1] - P[k]) / (seg[k] or 1)
            x = P[k] + d * (t - T[k])
            n = np.array([-d[1], d[0]])
            sides = []
            for sgn in (1, -1):
                hits = [o for o, _ in idx.crossings(x + n * sgn * 0.01, x + n * sgn * reach) if o != i]
                sides.append(hits)
            e = _estimate(sides, known)
            if e is not None:
                votes.append(e)
        if len(votes) >= 2 and max(votes) - min(votes) < 1e-6:
            out[i] = (votes[0], len(votes))
    return out


def _estimate(sides: list[list[int]], labelled: dict[int, float]) -> float | None:
    """This line's elevation from the ordered crossings on its two sides, or None."""
    first = []  # (position, elevation) of the nearest labelled crossing on each side
    pairs = []  # two nearest labelled crossings on one side
    for hits in sides:
        labs = [(pos + 1, labelled[o]) for pos, o in enumerate(hits) if o in labelled]
        if labs:
            first.append(labs[0])
        if len(labs) >= 2:
            pairs.append(labs[:2])
    if len(first) == 2:  # between labels: positions count outward from this line on each side
        (pa, ea), (pb, eb) = first
        if ea == eb:
            return None
        step = (eb - ea) / (pa + pb)
        if not _interval_ok(abs(step)):
            return None
        return round(ea + pa * step, 3)
    for (p1, e1), (p2, e2) in pairs:  # one side only: the two nearest labels set interval and direction
        if p2 == p1 or e1 == e2:
            continue
        step = (e2 - e1) / (p2 - p1)
        if not _interval_ok(abs(step)):
            continue
        return round(e1 - p1 * step, 3)
    return None
