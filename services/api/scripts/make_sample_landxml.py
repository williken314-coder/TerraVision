"""Generate a sample LandXML surface file from the Maple Ridge demo terrain.

Writes the existing-ground and finished-grade surfaces the sample plan set's
contours were drawn from, as TIN surfaces in LandXML 1.2, placed at
California State Plane Zone III (EPSG 2227, US survey feet) coordinates:

    E = SAMPLE_E0 + x,  N = SAMPLE_N0 - z   (x, z: demo project coordinates)

Use it with the sample PDF to exercise LandXML import, the false origin, and
the cross-check against surfaces read from plan contours.

    python scripts/make_sample_landxml.py  ->  samples/maple-ridge-surfaces.xml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gv.demo import build_demo  # noqa: E402
from gv.geometry.surface import Grid  # noqa: E402

SAMPLE_E0, SAMPLE_N0 = 6_052_400.0, 2_118_600.0
STEP = 10.0  # ft between TIN points


def surface_xml(name: str, desc: str, g: Grid) -> str:
    xs = np.arange(g.x0, g.x0 + g.nx * g.cell + 1e-6, STEP)
    zs = np.arange(g.z0, g.z0 + g.nz * g.cell + 1e-6, STEP)
    X, Z = np.meshgrid(xs, zs)
    E = g.sample(X, Z)
    ids = np.zeros(X.shape, dtype=int)
    pts, faces, n = [], [], 0
    for j in range(len(zs)):
        for i in range(len(xs)):
            if np.isfinite(E[j, i]):
                n += 1
                ids[j, i] = n
                pts.append(f'<P id="{n}">{SAMPLE_N0 - Z[j, i]:.3f} {SAMPLE_E0 + X[j, i]:.3f} {E[j, i]:.3f}</P>')
    for j in range(len(zs) - 1):
        for i in range(len(xs) - 1):
            a, b, c, d = ids[j, i], ids[j, i + 1], ids[j + 1, i], ids[j + 1, i + 1]
            if a and b and c and d:
                faces += [f"<F>{a} {b} {d}</F>", f"<F>{a} {d} {c}</F>"]
    return (f'  <Surface name="{name}" desc="{desc}">\n   <Definition surfType="TIN">\n    <Pnts>\n     '
            + "\n     ".join(pts) + "\n    </Pnts>\n    <Faces>\n     " + "\n     ".join(faces) + "\n    </Faces>\n   </Definition>\n  </Surface>\n")


def main(out: Path) -> None:
    prj = build_demo()
    eg, fg = Grid.from_model(prj.existing), Grid.from_model(prj.proposed)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2" date="2026-10-09" time="12:00:00">\n'
           ' <Units><Imperial areaUnit="squareFoot" linearUnit="USSurveyFoot" volumeUnit="cubicYard" temperatureUnit="fahrenheit" '
           'pressureUnit="inHG" angularUnit="decimal degrees" directionUnit="decimal degrees"/></Units>\n'
           ' <CoordinateSystem epsgCode="2227" name="NAD83 / California zone 3 (ftUS)"/>\n'
           ' <Project name="Maple Ridge Phase 1"/>\n'
           ' <Surfaces>\n'
           + surface_xml("EG", "Existing ground - topographic survey", eg)
           + surface_xml("FG", "Finished grade - grading plan", fg)
           + ' </Surfaces>\n</LandXML>\n')
    out.write_text(xml)
    print(f"wrote {out} ({len(xml) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "samples" / "maple-ridge-surfaces.xml")
