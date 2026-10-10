"""Placing one plan sheet against another from the contours both of them draw.

A labelled contour is the same line on the ground whichever sheet draws it: the
1010 contour on a 1" = 20' grading plan and on the 1" = 50' earthwork overview
are one curve, at two scales. With each sheet's scale known, the curves of
equal elevation on two sheets fix the rotation and offset between them.

The search is a correlation over rotations: each elevation's line work is
rasterised with its own complex phase, so lines only reinforce lines of the same
elevation; the best rotation and offset are then refined point to point (ICP,
same elevation only). A match counts only when most of the line work that
overlaps lies on the other sheet's line of the same elevation.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree

from .calibration import Similarity

STEP_FT = 2.0  # spacing of points sampled along each contour
NEAR_FT = 1.5  # a point "lies on" the other sheet's contour within this
MIN_MATCHED = 150  # points (300 ft of contour) that must lie on the other sheet's lines
MIN_FRACTION = 0.6  # of the overlapping line work
MAX_MEDIAN_FT = 0.25  # the same CAD line drawn twice agrees to hundredths; a near miss is a different place
MIN_ELEVATIONS = 3

_cache: dict[str, tuple | None] = {}


def sample_contours(lines) -> dict[tuple, np.ndarray]:
    """{(elev, role): points every STEP_FT} from [(elev, role, pts in feet)]."""
    out: dict[tuple, list] = {}
    for elev, role, pts in lines:
        p = np.asarray(pts, float)
        if len(p) < 2:
            continue
        seg = np.diff(p, axis=0)
        n = np.maximum(1, (np.hypot(*seg.T) / STEP_FT).astype(int))
        res = [p[:1]] + [a + v * (np.arange(1, k + 1)[:, None] / k) for a, v, k in zip(p[:-1], seg, n)]
        out.setdefault((elev, role), []).append(np.vstack(res))
    return {k: np.vstack(v) for k, v in out.items()}


def _rot(t):
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


def _raster(P, phase, origin, cell, n):
    img = np.zeros((n, n), complex)
    for e, p in P.items():
        ij = ((p - origin) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < n) & (ij[:, 1] >= 0) & (ij[:, 1] < n)
        np.add.at(img, (ij[ok, 1], ij[ok, 0]), phase[e])
    return img


def _search(A, B, common, degs, cell, centre_a):
    """Best (score, rotation, offset) of A onto B over the given rotations, by FFT correlation."""
    rng = np.random.default_rng(0)
    phase = {e: np.exp(2j * np.pi * rng.random()) for e in common}
    allb = np.vstack([B[e] for e in common])
    ra = max(np.hypot(*(A[e] - centre_a).T).max() for e in common)
    lo = allb.min(0) - ra - cell
    span = np.ptp(allb, 0).max() + 2 * ra + 2 * cell
    n = int(2 ** math.ceil(math.log2(max(16, span / cell))))  # B plus A's reach either side: no wrap-around
    Bf = np.fft.fft2(_raster({e: B[e] for e in common}, phase, lo, cell, n))
    best = None
    for deg in degs:
        R = _rot(math.radians(deg))
        Ar = {e: (A[e] - centre_a) @ R.T for e in common}
        # A is rasterised centred at the origin cell; the correlation peak is where that centre lands on B
        Af = np.fft.fft2(_raster(Ar, phase, np.array([-ra - cell, -ra - cell]), cell, n))
        c = np.fft.ifft2(Bf * np.conj(Af)).real
        k = int(np.argmax(c))
        if best is None or c.flat[k] > best[0]:
            iy, ix = np.unravel_index(k, c.shape)
            iy = iy - n if iy > n // 2 else iy
            ix = ix - n if ix > n // 2 else ix
            best = (c.flat[k], deg, lo + np.array([ix, iy]) * cell + ra + cell)
    return best


def _icp(A, trees, common, th, T, gates):
    """Rigid ICP between same-elevation points, one gate (ft) per iteration. Returns (theta, T) or None."""
    for g in gates:
        R = _rot(th)
        src, dst = [], []
        for e in common:
            d, i = trees[e].query(A[e] @ R.T + T, distance_upper_bound=g)
            m = d < g
            src.append(A[e][m])
            dst.append(trees[e].data[i[m]])
        src, dst = np.vstack(src), np.vstack(dst)
        if len(src) < 20:
            return None
        ms, md = src.mean(0), dst.mean(0)
        U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            return None
        th, T = math.atan2(Rm[1, 0], Rm[0, 0]), md - Rm @ ms
    return th, T


def match_contours(A: dict, B: dict) -> tuple[Similarity, dict] | None:
    """Rigid transform taking sheet A's contours (feet) onto B's, or None when they don't overlap convincingly.
    A, B: {(elev, role): (n, 2) points} as from sample_contours. Returns (Similarity, stats)."""
    common = sorted(set(A) & set(B))
    if len({e for e, _ in common}) < MIN_ELEVATIONS:
        return None
    key = hashlib.sha1(b"".join(repr(e).encode() + A[e].tobytes() + b"|" + B[e].tobytes() for e in common)).hexdigest()
    if key in _cache:
        return _cache[key]
    centre = np.vstack([A[e] for e in common]).mean(0)
    _, deg, pos = _search(A, B, common, np.arange(0, 360, 4.0), 10.0, centre)
    ra = max(np.hypot(*(A[e] - centre).T).max() for e in common)
    near = {e: B[e][np.hypot(*(B[e] - pos).T) < ra + 30] for e in common}  # the fine pass only needs B around the coarse spot
    near_common = [e for e in common if len(near[e])]
    if near_common:
        _, deg, pos = _search(A, near, near_common, np.arange(deg - 4, deg + 4.01, 1.0), 4.0, centre)
    th = math.radians(deg)
    T = pos - _rot(th) @ centre
    trees = {e: cKDTree(B[e]) for e in common}
    got = _icp(A, trees, common, th, T, [10.0] * 10 + [3.0] * 20 + [1.0] * 30)
    if got is None:
        _cache[key] = None
        return None
    # point-to-point ICP can settle a few feet along the contours; look around for a better seat
    sub = {e: A[e][::4] for e in common}
    def score(th, T):
        R = _rot(th)
        return sum(int((trees[e].query(sub[e] @ R.T + T, distance_upper_bound=0.5)[0] < 0.5).sum()) for e in common)
    th, T = got
    R = _rot(th)
    d = np.concatenate([trees[e].query(sub[e] @ R.T + T, distance_upper_bound=NEAR_FT)[0] for e in common])
    seated = bool((d < NEAR_FT).sum() >= 20 and np.median(d[d < NEAR_FT]) < 0.1)
    shifts = [] if seated else [np.array([dx, dy]) for dx in np.arange(-5, 5.01, 0.5) for dy in np.arange(-5, 5.01, 0.5)]
    best = max(shifts, key=lambda d: score(th, T + d)) if shifts else None
    if best is not None and best.any():
        alt = _icp(A, trees, common, th, T + best, [3.0] * 10 + [1.0] * 20)
        if alt is not None and score(*alt) > score(th, T):
            th, T = alt
    # how much of the line work that overlaps lies on the other sheet's line of the same elevation
    R = _rot(th)
    everything = cKDTree(np.vstack([B[e] for e in B]))
    matched = overlap = 0
    elevs, dev = set(), []
    for e in common:
        q = A[e] @ R.T + T
        d, _ = trees[e].query(q)
        near_any = everything.query(q, distance_upper_bound=6.0)[0] < np.inf
        on = d < NEAR_FT
        overlap += int(near_any.sum())
        matched += int(on.sum())
        dev += list(d[on])
        if on.sum() >= 10:
            elevs.add(e[0])
    stats = dict(matched=matched, overlap=overlap, fraction=matched / overlap if overlap else 0.0,
                 median_ft=float(np.median(dev)) if dev else None, elevations=len(elevs))
    ok = (matched >= MIN_MATCHED and stats["fraction"] >= MIN_FRACTION and len(elevs) >= MIN_ELEVATIONS
          and stats["median_ft"] is not None and stats["median_ft"] <= MAX_MEDIAN_FT)
    res = (Similarity(theta=th, tx=float(T[0]), tz=float(T[1]), rms=float(np.sqrt(np.mean(np.square(dev)))) if dev else 0.0, n=matched), stats) if ok else None
    if len(_cache) > 512:
        _cache.clear()
    _cache[key] = res
    return res
