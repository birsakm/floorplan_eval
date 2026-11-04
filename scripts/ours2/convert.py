"""Convert Mason2 generated homes (data/ours2/<version>/homes/<seed>.json.gz, from fetch.py) to the common format.

  conda run -n fpe python scripts/ours2/convert.py --version next-living-bay-ad5110bb7b58

Writes  outputs/ours/<variant>/samples/<seed>.json   room layout, common format, metres (x, z of the home frame)
        outputs/ours/<variant>/objects/<seed>.json   placed objects: id, class, asset, room, world footprint, y range
        outputs/ours/<variant>/run_info.json

Geometry (Mason2 state v2.3):
- Room polygons are local to `room.placement` (p = [x, y, z], yaw). World 2D = (x, z).
- Interior walls are centred on the shared room edge; exterior walls lie outside the room edge. The API's
  `wall_envelopes` give each wall's solid (front a-b, back back_a-back_b), so net rooms = room minus all wall
  solids. Rooms then have wall gaps, as in the real datasets.
- Openings sit on walls (offset from wall.a along a->b, width; sill and height for windows). An opening's polygon
  is its slice of the wall solid. `door/entrance` = front door, `door/interior` = interior door, `passage` = open
  connection.
- The `landing` (stair landing outside the flat, outdoor) is not part of the dwelling: it is dropped, and the
  door to it is the front door, as in Swiss Dwellings. Balconies stay (native type `outdoor`, canonical balcony).

Condition: the generator draws its own brief (household, rooms, net areas) from Swiss priors; there is no external
brief. `brief` holds the program's room counts and summed indoor target net area, `program` its room-room edges
(door / open), so scripts/ours/evaluate_ours.py measures program adherence.
"""
import argparse
import gzip
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpeval.format import canonical_room_type, make_sample, save_sample, validate  # noqa: E402

# Mason2 node type -> native label of the old program vocabulary (scripts/ours/convert.py), so the baseline scripts
# (scripts/procthor, scripts/infinigen) read these programs unchanged. Outdoor nodes (balconies) are "outdoor".
NATIVE = {"hallway": "circulation", "living_room": "living", "dining_room": "dining"}
# native -> canonical
ALIASES = {"outdoor": "balcony", "small_bathroom": "bathroom", "circulation": "corridor", "office": "study"}
DROP = {"landing"}
# The program has no daylight flag; rooms that need an exterior wall (for the baselines' input) are these.
NEEDS_EXTERIOR = {"living", "bedroom", "kitchen", "dining", "outdoor"}


def world_polygon(room):
    p, yaw = room["placement"]["p"], room["placement"].get("yaw", 0.0) or 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    return [(p[0] + c * x - s * y, p[2] + s * x + c * y) for x, y in room["polygon"]]


def wall_solid(e):
    return Polygon([e["a"], e["b"], e["back_b"], e["back_a"]])


def opening_poly(wall, env, o, pad=0.02):
    a, b = np.array(wall["a"], float), np.array(wall["b"], float)
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    fa, ba = np.array(env["a"], float), np.array(env["back_a"], float)
    n = ba - fa
    nn = n / (np.linalg.norm(n) or 1)
    # front face of the solid, projected onto the wall line's parameter
    p0 = fa + u * (o["offset"] - np.dot(fa - a, u)) - nn * pad
    p1 = p0 + u * o["width"]
    q = n + 2 * pad * nn
    return [[round(float(x), 4), round(float(y), 4)] for x, y in (p0, p1, p1 + q, p0 + q)]


def ring(geom):
    return [[round(x, 4), round(y, 4)] for x, y in list(geom.exterior.coords)[:-1]]


def native(node):
    return "outdoor" if node.get("outdoor") else NATIVE.get(node["type"], node["type"])


def collect_objects(state):
    found = {}

    def walk(x, room=None):
        if isinstance(x, dict):
            if isinstance(x.get("id"), str) and x["id"].startswith("object:"):
                found[x["id"]] = (x, room)
            for k, v in x.items():
                walk(v, room)
        elif isinstance(x, list):
            for v in x:
                walk(v, room)
    for nd in state["program"]["nodes"]:
        walk(nd.get("room") or {}, nd["id"])
    return found


def objects_record(d):
    objs, out = collect_objects(d["state"]), []
    for oid, fr in d["frames"].items():
        info, room = objs.get(oid, ({}, None))
        h = (info.get("asset") or "").split(":")[-1]
        asset = d["state"]["assets"].get(info.get("asset") or "", {})
        h = (asset.get("hash") or h).split(":")[-1]
        bd = d["bounds"].get(h) or asset.get("bounds")
        rec = {"id": oid, "class": info.get("class") or asset.get("class"), "asset": h, "room": room,
               "decorated": info.get("decorated")}
        if bd:
            M = np.array(fr["model"], float).reshape(4, 4)
            lo, hi = np.array(bd["min"], float), np.array(bd["max"], float)
            corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            w = corners @ M.T
            foot = [(lo[0], lo[2]), (hi[0], lo[2]), (hi[0], hi[2]), (lo[0], hi[2])]
            rec["footprint"] = [[round(float((M @ [x, 0, z, 1])[0]), 4), round(float((M @ [x, 0, z, 1])[2]), 4)]
                                for x, z in foot]
            rec["y_range"] = [round(float(w[:, 1].min()), 4), round(float(w[:, 1].max()), 4)]
        out.append(rec)
    return out


def convert(d, variant):
    s = d["state"]
    nodes = {n["id"]: n for n in s["program"]["nodes"]}
    keep = [n for n in s["program"]["nodes"] if n["type"] not in DROP and n.get("room")]
    levels = {round(n["room"]["placement"]["p"][1], 2) for n in keep}
    if len(levels) > 1:  # rooms at several heights: storeys would be flattened onto each other
        raise NotImplementedError(f"multi-storey home (room heights {sorted(levels)}); converter handles one storey")
    solids = unary_union([wall_solid(e) for e in d["wall_envelopes"]])
    rooms, index = [], {}
    for n in keep:
        raw = Polygon(world_polygon(n["room"])).buffer(0)
        net = raw.difference(solids)
        if net.geom_type == "MultiPolygon":
            net = max(net.geoms, key=lambda g: g.area)
        if net.is_empty:
            continue
        index[n["id"]] = len(rooms)
        nt = native(n)
        rooms.append({"type": canonical_room_type(nt, ALIASES), "type_native": nt, "polygon": ring(net),
                      "program_room": n["id"], "target_area_m2": n.get("area_m2")})
    env = {e["id"]: e for e in d["wall_envelopes"]}
    doors, windows, edges = [], [], []
    for w in s["walls"]:
        ra, rb = (w["rooms"] + [None, None])[:2]
        for o in w.get("openings", []):
            poly = opening_poly(w, env[w["id"]], o)
            if o["kind"] == "window":
                r = ra if ra in index else rb
                windows.append({"polygon": poly, "room": index.get(r), "width_m": o["width"],
                                "sill_m": o.get("sill", 0.0), "head_m": o.get("sill", 0.0) + o.get("height", 0.0)})
                continue
            other = {ra, rb} - {None}
            if o.get("subtype") == "entrance" or any(nodes.get(r, {}).get("type") in DROP for r in other):
                doors.append({"type": "front_door", "polygon": poly, "width_m": o["width"]})
                continue
            kind = "open" if o["kind"] == "passage" else "door"
            doors.append({"type": "opening" if kind == "open" else "interior_door", "polygon": poly,
                          "width_m": o["width"]})
            if ra in index and rb in index:
                edges.append([index[ra], index[rb], kind])
    boundary = unary_union([Polygon(world_polygon(n["room"])).buffer(0) for n in keep])
    kept = {n["id"] for n in keep}
    prog_edges = [{"room_a": e["a"], "room_b": e["b"], "kind": "open" if e["kind"] == "passage" else e["kind"]}
                  for e in s["program"]["edges"] if e["a"] in kept and e["b"] in kept]
    entrance = next((x for e in s["program"]["edges"] for x, y in ((e["a"], e["b"]), (e["b"], e["a"]))
                     if x in kept and nodes.get(y, {}).get("type") in DROP), None)
    prog_rooms = [{"id": n["id"], "type": native(n), "target_net_area": n.get("area_m2"), "storey": 0,
                   "needs_exterior": native(n) in NEEDS_EXTERIOR} for n in keep]
    counts = Counter(native(n) for n in keep)
    net_target = sum(n.get("area_m2") or 0 for n in keep if not n.get("outdoor"))
    cond = {"type": "brief+program", "pipeline": "mason2", "home": str(d["seed"]), "storey": None, "n_storeys": 1,
            "version": d["version"], "key": d["key"],
            "brief": {"building_type": "apartment", "net_area_m2": net_target, "room_counts": dict(counts)},
            "program": {"rooms": prog_rooms, "edges": prog_edges, "entrance": entrance},
            "checks": {c["rule"]: c["status"] for c in d.get("checks", [])}}
    sample = make_sample("ours", variant, str(d["seed"]), rooms, units="m", scale_m_per_unit=1.0,
                         boundary=ring(boundary) if boundary.geom_type == "Polygon" else None,
                         doors=doors, windows=windows, graph={"edges": edges}, condition=cond)
    validate(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--reference", default="swiss_dwellings:all",
                    help="reference set the version is evaluated against (fpeval.eval_config.REFERENCES / STAT_DATASETS)")
    a = ap.parse_args()
    variant = a.variant or f"mason2_{a.version}"
    src = ROOT / "data" / "ours2" / a.version / "homes"
    out = ROOT / "outputs" / "ours" / variant
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.json.gz"), key=lambda p: int(p.name.split(".")[0]))
    done, failed, types = 0, {}, Counter()
    for f in files:
        try:
            d = json.load(gzip.open(f, "rt"))
            smp = convert(d, variant)
            save_sample(smp, out / "samples" / f"{smp['id']}.json")
            json.dump(objects_record(d), open(out / "objects" / f"{smp['id']}.json", "w"))
            types.update(r["type_native"] for r in smp["rooms"])
            done += 1
        except Exception as e:  # noqa: BLE001  keep going; failures are recorded
            failed[f.name] = repr(e)
    info = {"source": f"noface Mason2 API, generator version {a.version}", "reference": a.reference,
            "n_homes": len(files),
            "n_converted": done, "native_room_types": dict(types), "failures": failed,
            "date": datetime.now().isoformat(timespec="seconds")}
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    print(f"{done}/{len(files)} converted -> {out}", failed if failed else "")


if __name__ == "__main__":
    main()
