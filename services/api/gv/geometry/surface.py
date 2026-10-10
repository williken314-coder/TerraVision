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

    def __init__(self, x, z, e, max_edge: float | None = None):
        """``max_edge``: drop triangles with a longer side, so the surface doesn't bridge gaps the source never
        covered (finished-grade contours drawn only inside the work limits, say). Long triangles whose corners are
        all at one elevation are kept: they are flat ground inside one contour (a pad, pond floor, stockpile top)."""
        self.pts = np.column_stack([x, z]).astype(float)
        self.e = np.asarray(e, dtype=float)
        self.tri = Delaunay(self.pts)
        self.dropped = np.zeros(len(self.tri.simplices), bool)
        if max_edge:
            P = self.pts[self.tri.simplices]
            sides = np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2)
            E = self.e[self.tri.simplices]
            flat = np.ptp(E, axis=1) <= 0.01  # all three corners on one contour: a pad, pond floor or stockpile top
            self.dropped = (sides.max(axis=1) > max_edge) & ~flat
        a, b, c = (self.pts[self.tri.simplices[:, i]] for i in range(3))
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        self.dropped_area = float(area[self.dropped].sum())  # sq ft of the triangulation left empty

    def sample(self, x, z):
        q = np.column_stack([np.ravel(x), np.ravel(z)])
        simplex = self.tri.find_simplex(q)
        out = np.full(len(q), np.nan)
        ok = simplex >= 0
        ok[ok] = ~self.dropped[simplex[ok]]
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


class TriMesh:
    """A TIN with its triangles given (LandXML faces), so breaklines and boundaries are honoured.

    Grid nodes outside every triangle stay NaN: nothing is extrapolated past the surface boundary.
    """

    def __init__(self, x, z, e, faces):
        self.x = np.asarray(x, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.e = np.asarray(e, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    def to_grid(self, x0, z0, cell, nx, nz, chunk: int = 200_000) -> Grid:
        out = np.full((nz + 1) * (nx + 1), np.nan)
        F = self.faces
        if len(F) == 0:
            return Grid(x0, z0, cell, nx, nz, out.reshape(nz + 1, nx + 1))
        ax, az, ae = self.x[F[:, 0]], self.z[F[:, 0]], self.e[F[:, 0]]
        bx, bz, be = self.x[F[:, 1]], self.z[F[:, 1]], self.e[F[:, 1]]
        cx, cz, ce = self.x[F[:, 2]], self.z[F[:, 2]], self.e[F[:, 2]]
        det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
        keep = np.abs(det) > 1e-12  # drop degenerate (zero-area) faces
        # grid-index bounding box of each triangle
        i0 = np.clip(np.ceil((np.minimum(np.minimum(ax, bx), cx) - x0) / cell - 1e-9), 0, nx).astype(np.int64)
        i1 = np.clip(np.floor((np.maximum(np.maximum(ax, bx), cx) - x0) / cell + 1e-9), -1, nx).astype(np.int64)
        j0 = np.clip(np.ceil((np.minimum(np.minimum(az, bz), cz) - z0) / cell - 1e-9), 0, nz).astype(np.int64)
        j1 = np.clip(np.floor((np.maximum(np.maximum(az, bz), cz) - z0) / cell + 1e-9), -1, nz).astype(np.int64)
        ni, nj = np.maximum(0, i1 - i0 + 1), np.maximum(0, j1 - j0 + 1)
        cnt = np.where(keep, ni * nj, 0)
        tris = np.nonzero(cnt)[0]
        # process (triangle, grid node) candidate pairs in bounded chunks
        start = 0
        while start < len(tris):
            csum = np.cumsum(cnt[tris[start:]])
            stop = start + max(1, int(np.searchsorted(csum, chunk, side="right")))
            t = tris[start:stop]
            c = cnt[t]
            ft = np.repeat(t, c)
            local = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
            gi = i0[ft] + local % ni[ft]
            gj = j0[ft] + local // ni[ft]
            px, pz = x0 + gi * cell, z0 + gj * cell
            l1 = ((bz[ft] - cz[ft]) * (px - cx[ft]) + (cx[ft] - bx[ft]) * (pz - cz[ft])) / det[ft]
            l2 = ((cz[ft] - az[ft]) * (px - cx[ft]) + (ax[ft] - cx[ft]) * (pz - cz[ft])) / det[ft]
            l3 = 1 - l1 - l2
            inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
            val = l1 * ae[ft] + l2 * be[ft] + l3 * ce[ft]
            out[(gj * (nx + 1) + gi)[inside]] = val[inside]
            start = stop
        return Grid(x0, z0, cell, nx, nz, out.reshape(nz + 1, nx + 1))
