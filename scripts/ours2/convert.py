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

Benchmark runs (--benchmark rplan_g2p_test1000): the version was generated from the programs of
scripts/rplan/export_mason_programs.py (Table 1 block (a)). Each home is matched to its input by `input_id`
("rplan:<id>"), saved as samples/<rplan id>.json with the INPUT program as its condition (so Types / Graph / Area are
scored against what the method was given; the home's own program is kept in condition.mason), and the real layout
of that input goes to gt_samples/ (FID floor = GT of the same inputs).

Condition: the generator draws its own brief (household, rooms, net areas) from Swiss priors; there is no external
brief. `brief` holds the program's room counts and summed indoor target net area, `program` its room-room edges
(door / open), so scripts/ours/evaluate_ours.py measures program adherence.
"""
import argparse
import gzip
import json
import math
import re
import shutil
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


def storey_of(s, keep):
    """{node id: storey index}: from the node's `storey` id (state.storeys), else from its placement height."""
    idx = {st["id"]: st.get("index", i) for i, st in enumerate(s.get("storeys") or [])}
    if all(n.get("storey") in idx for n in keep):
        return {n["id"]: idx[n["storey"]] for n in keep}
    levels = sorted({round(n["room"]["placement"]["p"][1], 2) for n in keep})
    return {n["id"]: levels.index(round(n["room"]["placement"]["p"][1], 2)) for n in keep}


def convert(d, variant):
    """[samples] of one home: one sample (id = seed) for a single storey, else one per storey (<seed>_s<k>), as
    scripts/ours/convert.py writes houses. Walls are separated by storey (wall_envelopes[].floor = the storey's
    elevation). Stairs connect rooms of different storeys: program edges of kind "stair" (the program's stair
    connections; scripts/ours/evaluate_ours.py adds them to the door graph), and condition.stairs lists the stairs the
    layout actually built (state.stairs[].connects) so a missing stair is visible."""
    s = d["state"]
    nodes = {n["id"]: n for n in s["program"]["nodes"]}
    keep = [n for n in s["program"]["nodes"] if n["type"] not in DROP and n.get("room")]
    st_of = storey_of(s, keep)
    storey_idx = {st["id"]: st.get("index", i) for i, st in enumerate(s.get("storeys") or [])}
    n_storeys = len(set(st_of.values()))
    elev = {st.get("index", i): st.get("elevation") for i, st in enumerate(s.get("storeys") or [])}
    env = {e["id"]: e for e in d["wall_envelopes"]}

    def wall_storey(w):
        on = {st_of[r] for r in w["rooms"] if r in st_of}
        if len(on) == 1:
            return on.pop()
        f = env[w["id"]].get("floor")
        return next((k for k, z in elev.items() if z is not None and f is not None and abs(z - f) < 1e-3), 0)

    kept = {n["id"] for n in keep}
    prog_edges = [{"room_a": e["a"], "room_b": e["b"], "kind": "open" if e["kind"] == "passage" else e["kind"]}
                  for e in s["program"]["edges"] if e["a"] in kept and e["b"] in kept]
    entrance = next((x for e in s["program"]["edges"] for x, y in ((e["a"], e["b"]), (e["b"], e["a"]))
                     if x in kept and nodes.get(y, {}).get("type") in DROP), None)
    prog_rooms = [{"id": n["id"], "type": native(n), "target_net_area": n.get("area_m2"), "storey": st_of[n["id"]],
                   "needs_exterior": native(n) in NEEDS_EXTERIOR} for n in keep]
    counts = Counter(native(n) for n in keep)
    net_target = sum(n.get("area_m2") or 0 for n in keep if not n.get("outdoor"))
    built_stairs = [[c["room"] for c in x.get("connects", [])] for x in s.get("stairs") or []]
    out = []
    for k in sorted(set(st_of.values())):
        on = [n for n in keep if st_of[n["id"]] == k]
        walls = [w for w in s["walls"] if wall_storey(w) == k]
        solids = unary_union([wall_solid(env[w["id"]]) for w in walls])
        # stairwell voids in this storey's floor are not net floor area (the program's area_m2 excludes them)
        voids = [Polygon(ring_) for x in s.get("stairs") or [] for o in x.get("openings", [])
                 if storey_idx.get(o.get("storey")) == k for ring_ in o.get("footprint", [])]
        solids = unary_union([solids] + voids) if voids else solids
        rooms, index = [], {}
        for n in on:
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
        doors, windows, edges = [], [], []
        for w in walls:
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
        boundary = unary_union([Polygon(world_polygon(n["room"])).buffer(0) for n in on])
        multi = n_storeys > 1
        cond = {"type": "brief+program", "pipeline": "mason2", "home": str(d["seed"]), "storey": k if multi else None,
                "n_storeys": n_storeys, "version": d["version"], "key": d["key"],
                "brief": {"building_type": "house" if multi else "apartment", "net_area_m2": net_target,
                          "room_counts": dict(counts)},
                "program": {"rooms": prog_rooms, "edges": prog_edges, "entrance": entrance},
                "checks": {c["rule"]: c["status"] for c in d.get("checks", [])}}
        if multi:
            cond["stairs"] = built_stairs
        sid = f"{d['seed']}_s{k}" if multi else str(d["seed"])
        sample = make_sample("ours", variant, sid, rooms, units="m", scale_m_per_unit=1.0,
                             boundary=ring(boundary) if boundary.geom_type == "Polygon" else None,
                             doors=doors, windows=windows, graph={"edges": edges}, condition=cond)
        validate(sample)
        out.append(sample)
    return out


BENCHMARKS = {"rplan_g2p_test1000": ROOT / "outputs" / "rplan" / "g2p_test1000_realdoors_programs"}
INPUT_ID = re.compile(r'"(rplan:[0-9]+)"')


def input_id(d):
    """The benchmark input a generated home was made from ("rplan:<id>"): an explicit field if present, else the one
    "rplan:<id>" string in the record."""
    for holder in (d, d.get("state") or {}, d.get("params") or {}, d.get("profile") if isinstance(d.get("profile"), dict) else {},
                   (d.get("state") or {}).get("program") or {}):
        if isinstance(holder.get("input_id"), str):
            return holder["input_id"]
    found = set(INPUT_ID.findall(json.dumps(d)))
    if len(found) != 1:
        raise ValueError(f"no unique input_id in the home record (found {sorted(found)[:3]})")
    return found.pop()


def as_benchmark(smp, d, bench):
    """Re-key a converted home by its benchmark input and give it the input's program as condition."""
    rid = input_id(d).split(":", 1)[1]
    src = BENCHMARKS[bench] / "samples" / f"{rid}.json"
    inp = json.load(open(src))["condition"]
    inp = dict(inp, mason={"seed": smp["id"], "key": d.get("key"), "version": d.get("version"),
                           "program": smp["condition"]["program"]})
    smp.update(id=rid, condition=inp)
    return smp, BENCHMARKS[bench] / "gt_samples" / f"{rid}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--reference", default="swiss_dwellings:all",
                    help="reference set the version is evaluated against (fpeval.eval_config.REFERENCES / STAT_DATASETS)")
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), default=None,
                    help="the version was generated from these benchmark programs (Table 1 block (a))")
    a = ap.parse_args()
    variant = a.variant or f"mason2_{a.version}"
    src = ROOT / "data" / "ours2" / a.version / "homes"
    out = ROOT / "outputs" / "ours" / variant
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "objects").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.json.gz"), key=lambda p: int(p.name.split(".")[0]))
    done, failed, types, storeys = 0, {}, Counter(), Counter()
    for f in files:
        try:
            d = json.load(gzip.open(f, "rt"))
            smps = convert(d, variant)
            if a.benchmark and len(smps) > 1:
                raise NotImplementedError("multi-storey home for a single-storey benchmark input")
            smp = smps[0]
            if a.benchmark:
                smp, gt = as_benchmark(smp, d, a.benchmark)
                if (out / "samples" / f"{smp['id']}.json").exists():
                    raise ValueError(f"second home for input {smp['id']}")
                (out / "gt_samples").mkdir(exist_ok=True)
                shutil.copy(gt, out / "gt_samples" / gt.name)
            for smp in smps:
                save_sample(smp, out / "samples" / f"{smp['id']}.json")
                types.update(r["type_native"] for r in smp["rooms"])
            home_id = smps[0]["condition"]["home"] if len(smps) > 1 else smps[0]["id"]
            json.dump(objects_record(d), open(out / "objects" / f"{home_id}.json", "w"))
            storeys[len(smps)] += 1
            done += 1
        except Exception as e:  # noqa: BLE001  keep going; failures are recorded
            failed[f.name] = repr(e)
    info = {"source": f"noface Mason2 API, generator version {a.version}", "reference": a.reference,
            "benchmark": a.benchmark,
            "n_homes": len(files),
            "n_converted": done, "n_storeys": {str(k): v for k, v in sorted(storeys.items())},
            "kind": "house" if sum(v for k, v in storeys.items() if k > 1) > done / 2 else "apartment", "native_room_types": dict(types), "failures": failed,
            "date": datetime.now().isoformat(timespec="seconds")}
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    print(f"{done}/{len(files)} converted (storeys {dict(storeys)}) -> {out}", failed if failed else "")


if __name__ == "__main__":
    main()
