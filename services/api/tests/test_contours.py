"""Counting unlabelled contours from the labelled ones."""
import numpy as np

from gv.ingest.contours import derive_elevations


def parallel(n, gap=10.0, length=400.0, wobble=0.0):
    """n contours running left to right, gap points apart, gently curved so transects aren't trivial."""
    xs = np.linspace(0, length, 40)
    return [np.column_stack([xs, 100 + i * gap + wobble * np.sin(xs / 60)]) for i in range(n)]


def test_between_labels():
    P = parallel(11, wobble=6)
    got = derive_elevations(P, {0: 100.0, 5: 105.0, 10: 110.0}, reach=60)
    assert {i: e for i, (e, _) in got.items()} == {1: 101, 2: 102, 3: 103, 4: 104, 6: 106, 7: 107, 8: 108, 9: 109}


def test_beyond_the_last_label_one_side_only():
    # the top of a slope: labels at 100 and 105 below, three unlabelled lines above them
    P = parallel(9)
    got = derive_elevations(P, {0: 100.0, 5: 105.0}, reach=60)
    assert {i: e for i, (e, _) in got.items() if i > 5} == {6: 106, 7: 107, 8: 108}


def test_a_line_crossing_contours_is_not_a_contour():
    P = parallel(11)
    road = np.array([[50.0, 80.0], [350.0, 220.0]])  # cuts across the slope
    got = derive_elevations(P + [road], {0: 100.0, 5: 105.0, 10: 110.0}, reach=60)
    assert 11 not in got and len(got) == 8


def test_odd_intervals_are_rejected():
    # 100 and 103 labelled with three lines between: a 0.75 ft interval is not a drafting interval
    P = parallel(5)
    assert derive_elevations(P, {0: 100.0, 4: 103.0}, reach=60) == {}


def test_needs_two_labels():
    assert derive_elevations(parallel(5), {0: 100.0}, reach=60) == {}
