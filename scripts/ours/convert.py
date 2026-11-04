"""Convert our generator's State exports (data/ours/state/<home>/) to the common format.

  conda run -n fpe python scripts/ours/convert.py --kind apartment

Reads   data/ours/state/<home>/roomplan/openings.json   (rooms, doors, windows after the room-plan stage)
Writes  outputs/ours/<variant>/samples/<home>.json      common format, metres
        outputs/ours/<variant>/run_info.json

Two-storey houses: the State replay rejected rooms, openings and program entries because their net
areas exceed the brief's total area by more than 2% (a check added after this generator). The rejected
commands still carry the original geometry, so they are restored here (--restore-rejected, default on)
and counted in run_info.json. Houses are written as one sample per storey: <home>_s<k>.json.

State room polygons run along wall centre lines (the legacy net polygon moved outward by half the
wall behind each edge). Each edge is moved back by half its wall thickness, so rooms are net
polygons with wall gaps, as in the real datasets. The brief (total net area, room counts) and the
program graph (rooms and door/open/wall edges, produced by the generator's program stage) are kept
in `condition`.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpeval.format import canonical_room_type, make_sample, save_sample, validate  # noqa: E402

STATE = ROOT / "data" / "ours" / "state"
# Apartments: outdoor space is a balcony; houses: a terrace or garden ("outdoor"). Stairs stay "other".
EXTRA_ALIASES = {"apartment": {"circulation": "corridor", "outdoor": "balcony", "office": "study"},
                 "house": {"circulation": "corridor", "outdoor": "outdoor", "office": "study", "stairs": "other"}}


def restore_rejected(home_dir, s):
    """Add rooms, openings and program entries whose replay commands were rejected. Returns counts."""
    n = Counter()
    for line in open(home_dir / "replay.jsonl"):
        d = json.loads(line)
        if d.get("verdict") != "rejected" or d["stage"] not in ("program", "roomplan"):
            continue
        op, v = d["command"]["op"], d["command"]["value"]
        if op == "AddRoom":
            s["rooms"].append(v)
        elif op == "AddOpening":
            s["openings"].append(v)
        elif op == "AddNode":
            s["program"]["rooms"].append(v)
        elif op == "AddEdge":
            s["program"]["edges"].append(v)
        elif op == "SetArea":
            s["program_facts"]["rooms"].setdefault(v["room"], {})["area"] = v["area"]
        else:
            continue
        n[op] += 1
    return n


def net_polygon(room):
    """Room polygon with every edge moved inward by half its wall thickness."""
    pts = room["polygon"]
    poly = Polygon(pts)
    strips = []
    for k, e in enumerate(room["edges"]):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        strips.append(LineString([a, b]).buffer(e["thickness_m"] / 2, cap_style="square", join_style="mitre"))
    net = poly.difference(unary_union(strips))
    if net.geom_type == "MultiPolygon":
        net = max(net.geoms, key=lambda g: g.area)
    return net


def opening_polygon(rooms_by_id, o):
    """Rectangle of the opening in the wall: along edge `edge` of room_a, from `offset` for `width`."""
    room = rooms_by_id[o["room_a"]]
    pts = room["polygon"]
    a, b = pts[o["edge"]], pts[(o["edge"] + 1) % len(pts)]
    line = LineString([a, b])
    seg = shapely.ops.substring(line, o["offset"], o["offset"] + o["width"])
    t = room["edges"][o["edge"]]["thickness_m"]
    return seg.buffer(t / 2 + 0.02, cap_style="flat")


def ring(geom):
    return [[round(x, 4), round(y, 4)] for x, y in list(geom.exterior.coords)[:-1]]


def convert(home_dir, variant, kind, restore=True):
    """Return ([samples], restored counts); one sample per storey for houses."""
    s = json.load(open(home_dir / "roomplan" / "openings.json"))
    restored = restore_rejected(home_dir, s) if restore else Counter()
    storeys = sorted({r["storey"] for r in s["rooms"]})
    if kind == "apartment":
        return [_storey_sample(s, variant, kind, None, s["id"])], restored
    return [_storey_sample(s, variant, kind, k, f"{s['id']}_s{k}") for k in storeys], restored


def _storey_sample(s, variant, kind, storey, sample_id):
    by_id = {r["id"]: r for r in s["rooms"]}
    on = [r for r in s["rooms"] if storey is None or r["storey"] == storey]
    rooms, index = [], {}
    for r in on:
        net = net_polygon(r)
        index[r["id"]] = len(rooms)
        rooms.append({"type": canonical_room_type(r["type"], EXTRA_ALIASES[kind]), "type_native": r["type"],
                      "polygon": ring(net), "program_room": r.get("program_room")})
    doors, windows, edges = [], [], []
    for o in s["openings"]:
        if o["room_a"] not in index:
            continue
        poly = ring(opening_polygon(by_id, o))
        if o["kind"] == "window":
            windows.append({"polygon": poly})
            continue
        dtype = {"entrance": "front_door", "door": "interior_door", "open": "opening"}[o["kind"]]
        doors.append({"type": dtype, "polygon": poly, "width_m": o["width"]})
        if o.get("room_b") in index:
            edges.append([index[o["room_a"]], index[o["room_b"]], "door" if o["kind"] == "door" else "open"])
    boundary = unary_union([Polygon(r["polygon"]) for r in on])
    cond = {"type": "brief+program", "home": s["id"], "storey": storey, "n_storeys": len(s["storeys"]),
            "brief": s["brief"], "program": s["program"], "program_facts": s["program_facts"]}
    sample = make_sample("ours", variant, sample_id, rooms, units="m", scale_m_per_unit=1.0,
                         boundary=ring(boundary) if boundary.geom_type == "Polygon" else None,
                         doors=doors, windows=windows, graph={"edges": edges}, condition=cond)
    validate(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="apartment", choices=["apartment", "house"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--no-restore", action="store_true", help="keep the State as replayed (drops rejected rooms)")
    args = ap.parse_args()
    variant = args.variant or f"mason_old_{args.kind}_n100_seed0"
    out = ROOT / "outputs" / "ours" / variant
    (out / "samples").mkdir(parents=True, exist_ok=True)
    homes = sorted(p for p in STATE.glob(f"{args.kind}-*") if p.is_dir())
    done, failed, restored = 0, {}, {}
    for h in homes:
        try:
            samples, n = convert(h, variant, args.kind, not args.no_restore)
            for smp in samples:
                save_sample(smp, out / "samples" / f"{smp['id']}.json")
            if n:
                restored[h.name] = dict(n)
            done += 1
        except Exception as e:  # keep going; failures are recorded
            failed[h.name] = repr(e)
    info = {"source": "data/ours (mason_old, run 20261024-full-n100-seed0, state export)",
            "stage": "roomplan/openings.json", "n_homes": len(homes), "n_converted": done,
            "restored_from_rejected_commands": restored, "failures": failed, "date": datetime.now().isoformat(timespec="seconds")}
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    print(f"{done}/{len(homes)} converted -> {out}", failed or "")


if __name__ == "__main__":
    main()
