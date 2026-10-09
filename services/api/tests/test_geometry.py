import math

import numpy as np

from gv.analysis import analyze
from gv.demo import build_demo
from gv.edits import FieldEdit, apply_edit
from gv.geometry.alignment import Alignment, station_label
from gv.geometry.surface import TIN, Grid
from gv.geometry.utilities import earthwork
from gv.ingest.calibration import fit_similarity, Similarity


def test_alignment_roundtrip():
    t = np.arange(0, 501, 5.0)
    x, z = t, 20 * np.sin(t / 80)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])  # stations = arc length
    al = Alignment(s, x, z)
    for st, off in [(37.5, 12.0), (250, -9.0), (480, 30.0)]:
        x, z = al.to_xz(st, off)
        s2, d2 = al.to_so(x, z)
        assert abs(s2[0] - st) < 0.1 and abs(d2[0] - off) < 0.1, (st, off, s2[0], d2[0])


def test_station_label():
    assert station_label(375) == "3+75"
    assert station_label(5.5, 1) == "0+05.5"
    assert station_label(1299.96, 1) == "13+00.0"


def test_tin_reproduces_a_plane_exactly():
    rng = np.random.default_rng(0)
    x, z = rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)
    e = 1000 + 0.02 * x - 0.01 * z
    tin = TIN(x, z, e)
    q = tin.sample(np.array([40.0, 60.0]), np.array([50.0, 20.0]))
    assert np.allclose(q, 1000 + 0.02 * np.array([40, 60]) - 0.01 * np.array([50, 20]), atol=1e-9)


def test_earthwork_one_foot_cut_over_one_acre_section():
    eg = Grid.from_function(lambda X, Z: np.full_like(X, 101.0), 0, 0, 5, 40, 40)
    fg = Grid.from_function(lambda X, Z: np.full_like(X, 100.0), 0, 0, 5, 40, 40)
    ew = earthwork(eg, fg)
    assert ew["available"]
    assert math.isclose(ew["cut_cy"], 200 * 200 / 27, rel_tol=1e-9)
    assert ew["fill_cy"] == 0


def test_earthwork_refuses_without_both_surfaces():
    fg = Grid.from_function(lambda X, Z: np.full_like(X, 100.0), 0, 0, 5, 4, 4)
    ew = earthwork(None, fg)
    assert not ew["available"] and "existing ground" in ew["reason"]


def test_similarity_recovers_known_transform():
    truth = Similarity(scale=1.0, theta=0.3, tx=120.0, tz=-45.0)
    src = np.array([[0, 0], [100, 0], [30, 80], [60, -20.0]])
    dx, dz = truth.apply(src[:, 0], src[:, 1])
    fit = fit_similarity(src, np.column_stack([dx, dz]))
    assert abs(fit.theta - 0.3) < 1e-9 and abs(fit.tx - 120) < 1e-6 and fit.rms < 1e-6


def test_demo_crossing_and_recalc_after_correction():
    prj = build_demo()
    a = analyze(prj)
    x = a["crossings"][0]
    assert (x["upper"], x["lower"], x["severity"]) == ("ST-12", "W-1", "conflict")
    assert round(x["clearance_ft"] * 12) == 5
    apply_edit(prj, FieldEdit(object_id="ST-12", field="inv_up", action="set", value=1041.90))
    a2 = analyze(prj)
    assert a2["pipes"]["ST-12"]["slope"] > a["pipes"]["ST-12"]["slope"]
    assert prj.pipe("ST-12").inv_up.status.value == "user"
