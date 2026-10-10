"""Construction assistant grounded in the project's extracted data.

Questions are answered from the project model and the geometry engine, and
every answer names the sheets it used. Common questions are handled by a
deterministic intent parser. Anything else goes to Claude (when
ANTHROPIC_API_KEY is set) together with the project facts, with instructions
to answer only from those facts.

Answers carry ``actions`` for the 3D view: select, isolate, mode, crossing,
trench, cutfill.
"""
from __future__ import annotations

import json
import os
import re

from .analysis import analyze
from .geometry.alignment import station_label
from .models import Project

SYS_WORDS = [("storm", r"storm|drain|catch|inlet|culvert"), ("san", r"sanit|sewer"), ("water", r"water|hydrant"), ("elec", r"electric|power|conduit|light")]
SYS_NAME = {"storm": "storm", "san": "sanitary sewer", "water": "water", "elec": "electrical"}
ID_RE = re.compile(r"\b(CB|SMH|STMH|MH|HW|FES|FH|INL|DI|WV|TX|ST|SA|WL|EL|W|E)\s?(?:#|NO\.?)?\s?-?\s?(\d{1,3}[A-Z]?)\b", re.I)


def _fmt(v, nd=2):
    return "not shown on the plans" if v is None else f"{v:.{nd}f}"


def _src(fact) -> str:
    if fact is None or fact.source is None:
        return ""
    page = f", p.{fact.page + 1}" if fact.page is not None else ""
    return f"{fact.source}{page}"


def _status(fact) -> str:
    if fact.status.value == "ai":
        return f" (AI-read, {round((fact.confidence or 0) * 100)}% confidence, needs confirmation)"
    if fact.status.value == "missing":
        return " (missing from the plans)"
    if fact.status.value == "user":
        return " (confirmed by your team)"
    return ""


def answer(prj: Project, q: str, selected: str | None = None) -> dict:
    a = analyze(prj)
    t = q.lower()
    sources: set[str] = set()

    def done(text, actions=(), extra_sources=()):
        sources.update(s for s in extra_sources if s)
        return dict(text=text, actions=list(actions), sources=sorted(sources), grounded=True)

    # "this pipe" refers to the selection in the 3D view
    m = ID_RE.search(q)
    oid = None
    if m:
        cand = f"{m.group(1).upper()}-{m.group(2).upper()}"
        oid = cand if prj.find(cand) else None
        u = next((u for u in prj.unplaced if u.id == cand), None)
        if not oid and u:
            if u.rim is None and not u.inverts:
                return done(f"**{cand}** is named on {', '.join(u.sheets)}, but no rim or inverts were read for it, and those sheets aren't placed in the model yet.", [], u.sheets)
            inv = "; ".join(f"{i['elev']:.2f}" + (f" ({i['side']})" if i.get("side") else "") for i in u.inverts)
            return done(f"**{cand}**{f' ({u.type})' if u.type else ''}: rim **{_fmt(u.rim)}**" + (f", inverts {inv}" if inv else "")
                        + f", from the structure table on {u.source or ', '.join(u.sheets)} (read by TerraVision; confirm against the plans). "
                        "It isn't shown in the 3D model yet because the sheets it's drawn on aren't placed: use Plans → Sheets & calibration → **Place**.",
                        [], [u.source] if u.source else u.sheets)
        if not oid:
            return done(f"I can't find **{cand}** in the extracted data. Known structures: {', '.join(s.id for s in prj.structures[:20]) or 'none yet'}.")
    elif selected and re.search(r"\bthis\b|\bit\b|selected", t) and prj.find(selected):
        oid = selected

    if oid:
        o = prj.find(oid)
        if re.search(r"trench|dig|excavat", t) and prj.pipe(oid):
            return done(f"Running the excavation simulator on **{oid}**.", [{"type": "trench", "id": oid}], [o.sheet])
        if prj.pipe(oid):
            p, c = o, a["pipes"][oid]
            sources.update(filter(None, [_src(p.inv_up), _src(p.inv_dn), _src(p.dia_in)]))
            deep = ""
            if re.search(r"deep|depth|below|cover", t):
                if c["min_cover"] is None:
                    deep = " Depth below finished grade can't be computed: " + ("the inverts are missing." if c["slope"] is None else "there is no finished-grade surface over this pipe yet (upload the grading plan).")
                else:
                    deep = f" Shallowest cover to finished grade: **{c['min_cover']:.1f} ft** (calculated from the grading surface)."
            slope = "" if c["slope"] is None else f", {c['length']:.1f} LF at {c['slope'] * 100:.2f}% (calculated)"
            desc = p.material.value if p.material.value and p.material.value[0].isdigit() else f"{_fmt(p.dia_in.value, 0)} in {p.material.value or '(material not called out)'}"
            return done(f"**{oid}**: {desc}, {p.from_id} to {p.to_id}. "
                        f"Invert {_fmt(p.inv_up.value)} ({_src(p.inv_up) or 'no source'}){_status(p.inv_up)} to {_fmt(p.inv_dn.value)} ({_src(p.inv_dn) or 'no source'}){_status(p.inv_dn)}{slope}.{deep}",
                        [{"type": "select", "id": oid}])
        if prj.structure(oid):
            s = o
            ins = [p for p in prj.pipes if p.to_id == oid]
            outs = [p for p in prj.pipes if p.from_id == oid]
            parts = [f"in from {p.from_id} {_fmt(p.inv_dn.value)} ({_src(p.inv_dn)}){_status(p.inv_dn)}" for p in ins] + \
                    [f"out to {p.to_id} {_fmt(p.inv_up.value)} ({_src(p.inv_up)}){_status(p.inv_up)}" for p in outs]
            sources.update(filter(None, [_src(s.rim)] + [_src(p.inv_dn) for p in ins] + [_src(p.inv_up) for p in outs]))
            depth = a["structure_depths"].get(oid)
            inv_txt = "Inverts: " + "; ".join(parts) + "." if parts else "No connected pipes with inverts were extracted."
            return done(f"**{oid}** rim {_fmt(s.rim.value)} ({_src(s.rim) or 'no source'}){_status(s.rim)}. {inv_txt}"
                        + (f" Depth rim to lowest invert: {depth:.2f} ft (calculated)." if depth is not None else ""), [{"type": "select", "id": oid}])
        line = prj.line(oid)
        return done(f"**{oid}**: {line.material.value}, design cover {_fmt(line.cover.value, 1)} ft ({_src(line.cover)}){_status(line.cover)}.", [{"type": "select", "id": oid}])

    if re.search(r"conflict|clash|cross|clearance", t):
        systems = [s for s, rx in SYS_WORDS if re.search(rx, t)]
        xs = a["crossings"]
        if len(systems) >= 2:
            xs = [x for x in xs if {_sys(prj, x["upper"]), _sys(prj, x["lower"])} >= set(systems[:2])]
        if not xs:
            what = " between " + " and ".join(SYS_NAME[s] for s in systems[:2]) if len(systems) >= 2 else ""
            return done(f"No utility crossings{what} were found in the extracted data. Crossings can only be checked where both utilities have plan locations and inverts.")
        lines = []
        for x in xs[:5]:
            up, lo = prj.find(x["upper"]), prj.find(x["lower"])
            sources.update(filter(None, [getattr(up, "sheet", None), getattr(lo, "sheet", None)]))
            where = f"near Sta {_sta(x['station'])}" if x["station"] is not None else f"at x {x['x']:.0f}, z {x['z']:.0f}"
            lines.append(f"**{x['upper']}** crosses over **{x['lower']}** {where}: estimated vertical clearance **{round(x['clearance_ft'] * 12)} in**"
                         + (" (potential conflict; required clearance must be verified against project specifications)" if x["severity"] != "clear" else ""))
        return done(" ".join(f"{l}." for l in lines), [{"type": "crossing", "id": xs[0]["id"]}])

    for sys, rx in SYS_WORDS:
        if re.search(rx, t) and re.search(r"show|every|all|list|where", t):
            ps = [p for p in prj.pipes if p.system == sys]
            ss = [s for s in prj.structures if s.system == sys]
            ls = [l for l in prj.lines if l.system == sys]
            if not (ps or ss or ls):
                found = [u for u in prj.unplaced if u.system == sys]
                if found:
                    on = sorted({x for u in found for x in u.sheets})
                    return done(f"I found {len(found)} {SYS_NAME[sys]} structures by name ({', '.join(u.id for u in found)}) on {', '.join(on)}, "
                                "but those sheets aren't placed in the model yet, so I can't show where they are. "
                                "Place one of them in Plans → Sheets & calibration → **Place**: tap the same two spots on it and on a placed sheet.",
                                [], on)
                return done(f"No {SYS_NAME[sys]} was extracted from the uploaded sheets.")
            lf = sum(a["pipes"][p.id]["length"] for p in ps) + sum(a["lines"][l.id]["length"] for l in ls)
            sizes = sorted({p.dia_in.value for p in ps if p.dia_in.value})
            sources.update(p.sheet for p in ps)
            sources.update(s.sheet for s in ss)
            review = sorted({r["object_id"] for r in a["review"] if _sys(prj, r["object_id"]) == sys})
            return done(f"Showing the {SYS_NAME[sys]} system: **{len(ps)} pipes, {len(ss)} structures, {lf:.0f} LF**"
                        + (f" ({', '.join(f'{d:g}' for d in sizes)} in)" if sizes else "") + "."
                        + (f" Needs review: {', '.join(review[:8])}." if review else ""), [{"type": "isolate", "system": sys}])

    if re.search(r"excavat|cut|fill|earth|volume|dirt|yard", t):
        ew = a["earthwork"]
        if not ew.get("available"):
            return done("I can't give an excavation quantity yet. " + ew["reason"])
        sources.update(filter(None, [prj.existing.source if prj.existing else None, prj.proposed.source if prj.proposed else None]))
        net = ew["net_cy"]
        return done(f"Mass earthwork, existing vs finished grade: **cut {ew['cut_cy']:,.0f} CY**, **fill {ew['fill_cy']:,.0f} CY**, net {abs(net):,.0f} CY {'export' if net > 0 else 'import'} "
                    f"over {ew['area_acres']:.2f} acres. Deepest cut {ew['max_cut_ft']:.1f} ft. Method: {ew['method']}. Trench excavation for each pipe is separate (tap a pipe, then Simulate trench).",
                    [{"type": "cutfill"}])

    if re.search(r"road|pav|finish|after|complete", t):
        if not prj.alignment:
            return done("Roadway surfaces (pavement, curb, sidewalk) haven't been extracted from these plans yet, so I can't show a finished roadway. I can show the finished-grade surface and utilities.", [{"type": "mode", "mode": "finished"}])
        cr = prj.corridor
        al_src = (prj.alignment.source or "").split(", ")
        if prj.demo or not cr or not cr.edges:
            body = "pavement, curb and gutter, and sidewalks built from the alignment and typical section." if cr else "the centerline only; road edges weren't identified."
            srcs = ["C-7.0", "C-8.0"] if prj.demo else al_src
            return done(f"Here's the finished roadway: {body}", [{"type": "mode", "mode": "finished"}], srcs)
        def side(kind):
            v = [f"{e.offset.value:.1f} ft {e.side}T" for e in cr.edges if e.kind == kind]
            return " / ".join(v) if v else "not found"
        parts = [f"edge of pavement {side('pavement')}", f"back of curb {side('curb_back')}"]
        if any(e.kind.startswith("walk") for e in cr.edges):
            parts.append(f"sidewalk {side('walk_inner')} to {side('walk_outer')}")
        sec = cr.pavement_section
        txt = (f"Here's the finished roadway along {prj.alignment.name}, STA {station_label(prj.alignment.points[0].s)} to {station_label(prj.alignment.points[-1].s)}: "
               + "; ".join(parts) + ". "
               + (f"Pavement section: {sec.value} ({sec.source})." if sec.value else "No pavement section was found in the plans.")
               + (f" Elevations: {prj.alignment.profile_source}." if prj.alignment.profile_source else ""))
        return done(txt, [{"type": "mode", "mode": "finished"}], al_src + ([sec.source] if sec.source else []))

    if re.search(r"review|confirm|verif|uncertain|missing|issue|problem", t):
        n = a["review"]
        blockers = [i for i in a["issues"] if i["severity"] == "blocker"]
        return done((f"{len(n)} extracted values need confirmation" if n else "Every extracted value is verified or confirmed")
                    + (f", and {len(blockers)} blocking issues: " + "; ".join(i["message"] for i in blockers[:4]) if blockers else "") + ". See the Review tab.",
                    [{"type": "tab", "tab": "review"}])

    llm = _llm_answer(prj, a, q, selected)
    if llm:
        return llm
    return done("I answer from the extracted plan data. Ask about a structure or pipe by ID (CB-4, ST-2), a system (storm, sewer, water), crossings, excavation, or the finished roadway.")


def _sys(prj: Project, oid: str):
    o = prj.find(oid)
    return getattr(o, "system", None)


def _sta(s: float) -> str:
    h = int(max(0, s) // 100)
    r = max(0, s) - h * 100
    return f"{h}+{r:05.2f}"


def _llm_answer(prj: Project, a: dict, q: str, selected: str | None) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    facts = dict(
        structures=[dict(id=s.id, system=s.system, rim=s.rim.value, rim_status=s.rim.status.value, rim_sheet=s.rim.source) for s in prj.structures],
        pipes=[dict(id=p.id, system=p.system, from_=p.from_id, to=p.to_id, dia_in=p.dia_in.value, material=p.material.value,
                    inv_up=p.inv_up.value, inv_dn=p.inv_dn.value, sheet=p.sheet, inv_status=p.inv_up.status.value, **a["pipes"][p.id]) for p in prj.pipes],
        crossings=a["crossings"], earthwork=a["earthwork"], sheets=[s.model_dump() for s in prj.sheets],
        issues=a["issues"][:40], selected=selected)
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("GV_ASSISTANT_MODEL", "claude-sonnet-5-5"), max_tokens=800,
        system=("You are a construction assistant for an excavation contractor. Answer ONLY from the project facts provided. "
                "Name the sheet for every number you use. If the facts don't contain the answer, say what's missing. Never estimate plan values. "
                'Reply with JSON: {"text": str, "actions": [{"type": "select", "id": str} | {"type": "isolate", "system": str}], "sources": [str]}'),
        messages=[{"role": "user", "content": f"Project facts:\n{json.dumps(facts, default=str)[:60000]}\n\nQuestion: {q}"}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    try:
        out = json.loads(text[text.find("{"): text.rfind("}") + 1])
        out["grounded"] = True
        return out
    except Exception:
        return dict(text=text, actions=[], sources=[], grounded=True)
