"""Printed N/E coordinates with leaders, as control points."""
import math

import numpy as np
import pytest

from gv.ingest.assemble import _fit_printed_control
from gv.ingest.control import callout_blocks, read_control
from gv.ingest.pdf import Seg, TextSpan


def span(text, x0, y, size=11.2):
    w = len(text) * size * 0.55
    return TextSpan(text, x0 + w / 2, y, size, (x0, y - size / 2, x0 + w, y + size / 2))


def callout(x0, y0, n, e, sta="STA. 10+00.00A,"):
    return [span(sta, x0, y0), span("CONSTRUCTION ALIGNMENT", x0, y0 + 15), span("BEGINNING", x0, y0 + 30),
            span(f"N. {n:.2f}", x0, y0 + 45), span(f"E. {e:.2f}", x0, y0 + 60)]


def leader(x0, y, tx, ty):
    """A stub off the block's left edge, a knee, then a run to the target (and an arrowhead)."""
    kx = x0 - 10
    return [Seg(x0 - 3, y, kx, y, 0.5, False), Seg(kx, y, kx, y - 40, 0.5, False), Seg(kx, y - 40, tx, ty, 0.5, False),
            Seg(tx, ty, tx + 3, ty + 6, 0.5, False), Seg(tx, ty, tx + 6, ty + 2, 0.5, False)]


def test_block_and_leader():
    sp = callout(700, 900, 130437.12, 1791742.12) + [span("12+00A", 600, 1000)]
    blocks = callout_blocks(sp)
    assert len(blocks) == 1 and blocks[0]["n"] == 130437.12 and blocks[0]["e"] == 1791742.12
    assert blocks[0]["text"].startswith("STA. 10+00.00A,")
    got = read_control(sp, leader(700, 930, 640, 820))
    assert len(got) == 1 and (got[0]["x"], got[0]["y"]) == (640, 820)


def test_no_leader_no_point():
    assert read_control(callout(700, 900, 1.0e5, 1.7e6), []) == []
    # letters drawn as strokes inside the block aren't a leader
    strokes = [Seg(720, 945, 728, 940, 0.3, False), Seg(728, 940, 736, 950, 0.3, False)]
    assert read_control(callout(700, 900, 1.0e5, 1.7e6), strokes) == []


def test_one_line_callout():
    assert callout_blocks([span("CP-3  N 130420.55, E 1791802.10", 300, 300)])[0]["e"] == 1791802.10


def test_fit_drops_a_wrong_leader():
    k = 20 / 72
    rng = np.random.default_rng(2)
    pts = []
    for x, y in [(300, 300), (1500, 400), (900, 1200), (1800, 1100), (500, 1000)]:
        e, n = 1791000 + x * k, 131000 - y * k  # north up, 1" = 20'
        pts.append(dict(x=x + rng.normal(0, 1), y=y + rng.normal(0, 1), e=e, n=n))
    pts[1]["x"] += 250  # a leader followed to the wrong spot: ~70 ft off
    sim, used, dropped = _fit_printed_control(pts, k, 1791000, 131000)
    assert (used, dropped) == (4, 1) and sim.rms < 1.0 and abs(math.degrees(sim.theta)) < 0.5
    assert _fit_printed_control(pts[:2], k, 0, 0) is None  # two points can't show a wrong one


def test_bunched_points_dont_place_a_sheet():
    k = 20 / 72
    pts = [dict(x=500 + i, y=500 + i, e=1791000 + (500 + i) * k, n=131000 - (500 + i) * k) for i in range(0, 30, 10)]
    assert _fit_printed_control(pts, k, 1791000, 131000) is None
