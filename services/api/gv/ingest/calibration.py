"""Drawing-scale calibration, sheet registration, and unit handling.

Sheet coordinates: PDF points, origin top-left, y down.
Project coordinates: feet, x east, z south (north is -z), so a plan sheet
with north up maps directly: x_ft = x_pt * k, z_ft = y_pt * k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

PT_PER_IN = 72.0
FT_PER_M = 3.280839895


def ft_per_pt(scale_ft_per_in: float) -> float:
    return scale_ft_per_in / PT_PER_IN


def to_ft(value: float, units: str) -> float:
    return value * FT_PER_M if units == "m" else value


@dataclass
class Similarity:
    """x' = s * R(theta) * x + t"""

    scale: float = 1.0
    theta: float = 0.0
    tx: float = 0.0
    tz: float = 0.0
    rms: float = 0.0
    n: int = 0

    def apply(self, x, z):
        c, s = math.cos(self.theta), math.sin(self.theta)
        x, z = np.asarray(x, float), np.asarray(z, float)
        return self.scale * (c * x - s * z) + self.tx, self.scale * (s * x + c * z) + self.tz

    def as_dict(self):
        return asdict(self)


def fit_similarity(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> Similarity:
    """Least-squares 2D similarity (Umeyama). src, dst: (n, 2)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 2:
        raise ValueError("Need at least two matching points")
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var = (a ** 2).sum() / n
    scale = float(np.trace(np.diag(S) @ D) / var) if allow_scale and var > 0 else 1.0
    t = md - scale * R @ ms
    sim = Similarity(scale=scale, theta=float(math.atan2(R[1, 0], R[0, 0])), tx=float(t[0]), tz=float(t[1]), n=n)
    px, pz = sim.apply(src[:, 0], src[:, 1])
    sim.rms = float(np.sqrt(((px - dst[:, 0]) ** 2 + (pz - dst[:, 1]) ** 2).mean()))
    return sim


@dataclass
class SheetCalibration:
    sheet_key: str  # "<document_id>:<page>"
    ft_per_pt: float | None
    method: str  # "scale text" | "user scale" | "control points" | "none"
    to_project: Similarity | None = None  # sheet feet -> project feet

    def pt_to_project(self, x_pt, y_pt):
        if self.ft_per_pt is None:
            return None
        x, z = np.asarray(x_pt, float) * self.ft_per_pt, np.asarray(y_pt, float) * self.ft_per_pt
        if self.to_project is not None:
            x, z = self.to_project.apply(x, z)
        return x, z
