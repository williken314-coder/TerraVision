"""Horizontal alignment: station/offset <-> plan coordinates.

Offsets are positive to the right of the direction of stationing (RT) and
negative to the left (LT), matching plan convention.
"""
from __future__ import annotations

import numpy as np


class Alignment:
    def __init__(self, stations, xs, zs):
        self.s = np.asarray(stations, dtype=float)
        self.x = np.asarray(xs, dtype=float)
        self.z = np.asarray(zs, dtype=float)
        tx, tz = np.gradient(self.x), np.gradient(self.z)
        ln = np.hypot(tx, tz)
        self.tx, self.tz = tx / ln, tz / ln
        self.nx, self.nz = -self.tz, self.tx

    @property
    def length(self) -> float:
        return float(self.s[-1] - self.s[0])

    def frame(self, s):
        s = np.clip(s, self.s[0], self.s[-1])
        f = lambda a: np.interp(s, self.s, a)
        return f(self.x), f(self.z), f(self.nx), f(self.nz), f(self.tx), f(self.tz)

    def to_xz(self, s, d):
        x, z, nx, nz, _, _ = self.frame(s)
        n = np.hypot(nx, nz)
        return x + nx / n * d, z + nz / n * d

    def to_so(self, x, z):
        """Station and offset of point(s). Extends past both ends of the alignment."""
        x = np.atleast_1d(np.asarray(x, dtype=float))[:, None]
        z = np.atleast_1d(np.asarray(z, dtype=float))[:, None]
        ax, az = self.x[:-1][None], self.z[:-1][None]
        vx, vz = np.diff(self.x)[None], np.diff(self.z)[None]
        l2 = vx * vx + vz * vz
        u = ((x - ax) * vx + (z - az) * vz) / l2
        uc = np.clip(u, 0.0, 1.0)
        uc[:, 0] = np.minimum(u[:, 0], 1.0)  # allow before the start
        uc[:, -1] = np.maximum(u[:, -1], 0.0)  # allow past the end
        px, pz = ax + vx * uc, az + vz * uc
        dx, dz = x - px, z - pz
        i = np.argmin(dx * dx + dz * dz, axis=1)
        r = np.arange(len(i))
        ln = np.sqrt(l2[0, i])
        ds = np.diff(self.s)[i]
        s = self.s[:-1][i] + ds * uc[r, i]
        d = dx[r, i] * (-vz[0, i] / ln) + dz[r, i] * (vx[0, i] / ln)
        return s, d


def station_label(s: float, decimals: int = 0) -> str:
    v = max(0.0, s)
    h = int(v // 100)
    r = round(v - h * 100, decimals)
    if r >= 100:
        h, r = h + 1, 0.0
    body = f"{r:.{decimals}f}"
    return f"{h}+{'0' if r < 10 else ''}{body}"


def offset_label(d: float) -> str:
    return f"{abs(d):.1f}' {'LT' if d < 0 else 'RT'}"
