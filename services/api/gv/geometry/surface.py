"""Terrain surfaces: TIN from points, regular grids for rendering and volumes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from ..models import GridSurface as GridSurfaceModel


@dataclass
class Grid:
    x0: float
    z0: float
    cell: float
    nx: int
    nz: int
    elev: np.ndarray  # shape (nz + 1, nx + 1)

    @classmethod
    def from_model(cls, m: GridSurfaceModel) -> "Grid":
        e = np.array([np.nan if v is None else v for v in m.elev], dtype=float)
        return cls(m.x0, m.z0, m.cell, m.nx, m.nz, e.reshape(m.nz + 1, m.nx + 1))

    def to_model(self, name: str, source: str | None = None) -> GridSurfaceModel:
        return GridSurfaceModel(name=name, x0=self.x0, z0=self.z0, cell=self.cell, nx=self.nx, nz=self.nz,
                                elev=[None if not np.isfinite(v) else round(float(v), 3) for v in self.elev.ravel()], source=source)

    def coords(self):
        xs = self.x0 + np.arange(self.nx + 1) * self.cell
        zs = self.z0 + np.arange(self.nz + 1) * self.cell
        return np.meshgrid(xs, zs)

    def sample(self, x, z):
        """Bilinear elevation at plan point(s). NaN outside the grid or where the surface has no data."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        ri, rj = (x - self.x0) / self.cell, (z - self.z0) / self.cell
        outside = (ri < -1e-6) | (rj < -1e-6) | (ri > self.nx + 1e-6) | (rj > self.nz + 1e-6)
        fi = np.clip(ri, 0, self.nx - 1e-9)
        fj = np.clip(rj, 0, self.nz - 1e-9)
        i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
        u, v = fi - i, fj - j
        e = self.elev
        out = (e[j, i] * (1 - u) * (1 - v) + e[j, i + 1] * u * (1 - v)
               + e[j + 1, i] * (1 - u) * v + e[j + 1, i + 1] * u * v)
        return np.where(outside, np.nan, out)

    @classmethod
    def from_function(cls, fn, x0, z0, cell, nx, nz) -> "Grid":
        g = cls(x0, z0, cell, nx, nz, np.zeros((nz + 1, nx + 1)))
        X, Z = g.coords()
        g.elev = np.asarray(fn(X, Z), dtype=float)
        return g


class TIN:
    """Triangulated irregular network from spot elevations / contour vertices."""

    def __init__(self, x, z, e):
        self.pts = np.column_stack([x, z]).astype(float)
        self.e = np.asarray(e, dtype=float)
        self.tri = Delaunay(self.pts)

    def sample(self, x, z):
        q = np.column_stack([np.ravel(x), np.ravel(z)])
        simplex = self.tri.find_simplex(q)
        out = np.full(len(q), np.nan)
        ok = simplex >= 0
        T = self.tri.transform[simplex[ok]]
        b = np.einsum("nij,nj->ni", T[:, :2, :], q[ok] - T[:, 2, :])
        bary = np.column_stack([b, 1 - b.sum(axis=1)])
        verts = self.tri.simplices[simplex[ok]]
        out[ok] = (self.e[verts] * bary).sum(axis=1)
        return out.reshape(np.shape(x))

    def to_grid(self, x0, z0, cell, nx, nz) -> Grid:
        g = Grid(x0, z0, cell, nx, nz, np.zeros((nz + 1, nx + 1)))
        X, Z = g.coords()
        g.elev = self.sample(X, Z)
        return g
