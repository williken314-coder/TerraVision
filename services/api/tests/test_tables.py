"""Structure tables: rims, inverts, stations and types from a table, never locations."""
from gv.ingest.pdf import TextSpan, build_report
from gv.ingest.tables import structure_tables


def span(text, x0, y, size=11.2):
    w = len(text) * size * 0.55
    return TextSpan(text, x0 + w / 2, y, size, (x0, y - size / 2, x0 + w, y + size / 2))


# laid out like a real table: name, type, station, rim, then one invert per line
TABLE = [span("STORM STRUCTURE TABLE", 1930, 488),
         span("NAME", 1800, 516), span("TYPE", 1885, 516), span("STATION", 1985, 516), span("RIM", 2095, 516), span("PIPES IN", 2160, 516),
         span("I.E. 878.22, E", 2150, 540), span("23+92.7A, 5.9 L", 1975, 548),
         span("CB #5", 1800, 552), span("CB TYPE 2 48 IN. DIAM.", 1860, 552), span("ELEV. 882.58", 2075, 552), span("I.E. 878.22, W", 2150, 556),
         span("I.E. 876.83, W", 2150, 584),
         span("CB #7", 1800, 592), span("CB TYPE 2 48 IN. DIAM.", 1860, 592), span("24+21.2A, 11.5 R", 1975, 592), span("ELEV. 880.20", 2075, 592),
         span("I.E. 871.13, E", 2150, 600), span("WITH VANED GRATE", 1860, 604),
         span("GENERAL NOTES", 1990, 780)]


def test_rows():
    rows, used = structure_tables(TABLE)
    assert [r["id"] for r in rows] == ["CB-5", "CB-7"]
    cb5, cb7 = rows
    assert cb5["rim"] == 882.58 and [(i["elev"], i["side"]) for i in cb5["inverts"]] == [(878.22, "E"), (878.22, "W")]
    assert cb7["rim"] == 880.20 and [(i["elev"], i["side"]) for i in cb7["inverts"]] == [(876.83, "W"), (871.13, "E")]
    assert cb5["station"] == {"station": 2392.7, "alignment": "A", "offset": -5.9} and cb7["station"]["offset"] == 11.5
    assert cb5["type"] == "CB TYPE 2 48 IN. DIAM."
    assert all(i["dir"] is None for r in rows for i in r["inverts"])  # "PIPES IN" lists every pipe: matched by direction, not guessed
    assert not any(id(s) in used for s in TABLE if s.text == "GENERAL NOTES")


def test_a_rim_note_is_not_a_table():
    assert structure_tables([span("RIM", 500, 500), span("CB #3", 420, 530), span("ELEV. 880.00", 490, 530)])[0] == []


def test_table_rows_give_values_but_no_location():
    r = build_report(0, 2448, 1584, TABLE + [span("C09", 2278, 1537, 17.4), span("STORM PLAN", 2355, 740, 25.0)], [], [], [])
    rows = [c for c in r.candidates if c.kind == "structure"]
    assert {c.label for c in rows} == {"CB-5", "CB-7"}
    assert all(c.values["table"] and not c.values.get("label_xy") and not c.values.get("symbol_xy") for c in rows)
    assert {c.label: c.values["rim"] for c in rows} == {"CB-5": 882.58, "CB-7": 880.20}
