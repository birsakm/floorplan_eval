"""Convert Infinigen Indoors floor plans (raw/<home>.json from scripts/infinigen/generate.py) to the common format.

One sample per storey, with the same ids and the same condition (brief + program) as our samples of that home,
so the brief and program metrics of scripts/ours/evaluate_ours.py apply unchanged.
  rooms    Infinigen's room polygons are wall centre lines that share edges; each is shrunk by half the wall
           thickness (mitred) to the net room polygon, as in our samples. Types and program links come from the
           program room the Infinigen node was made from (generate.py).
  doors    Infinigen's door, open and entrance cutters, clipped to the walls: interior_door, opening, front_door.
  windows  window cutters clipped to the walls, with width_m, sill_m and head_m (from the cutter's z extent).
  graph    one edge per door or opening between two rooms.

    PYTHONPATH=. conda run -n fpe python scripts/infinigen/convert.py --kind apartment
"""
import argparse
import json
import sys
from pathlib import Path

import shapely
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpeval.format import canonical_room_type, make_sample, save_sample  # noqa: E402

# Same type mapping as our samples (scripts/ours/convert.py).
EXTRA_ALIASES = {"apartment": {"circulation": "corridor", "outdoor": "balcony", "office": "study", "small_bathroom": "bathroom"},
                 "house": {"circulation": "corridor", "outdoor": "outdoor", "office": "study", "stairs": "other"}}


def ring(p):
    return [[round(float(x), 4), round(float(y), 4)] for x, y in p.exterior.coords[:-1]]


def ours_samples(kind, source=None):
    """{(home, storey): (sample id, condition)} of our samples."""
    out = {}
    for p in sorted((ROOT / "outputs" / (source or f"ours/mason_old_{kind}_n100_seed0") / "samples").glob("*.json")):
        c = json.load(open(p))["condition"]
        out[(c["home"], c["storey"] or 0)] = (p.stem, c)
    return out


def merge_stair_rooms(storey):
    """Merge the synthetic stair room "stairs:auto@<room>" (scripts/infinigen/generate.py with_stair_rooms) into
    <room>, as Mason2 has its stair inside that room. If the two do not touch, the stair stays a room of its own."""
    out = [dict(r) for r in storey if not str(r.get("program_room")).startswith("stairs:auto@")]
    by = {r["program_room"]: r for r in out}
    for s in storey:
        pr = str(s.get("program_room"))
        if not pr.startswith("stairs:auto@"):
            continue
        host = by.get(pr.split("@", 1)[1])
        u = unary_union([Polygon(host["polygon"]).buffer(0), Polygon(s["polygon"]).buffer(0)]) if host else None
        if u is not None and u.geom_type == "Polygon":
            host["polygon"] = [list(c) for c in u.exterior.coords[:-1]]
            host["merged"] = s["name"]
        else:
            out.append(dict(s, program_room=None, stair_room=True))
    return out


def convert_storey(raw, k, kind, variant, sample_id, cond):
    wt = raw["wall_thickness"]
    prog = {r["id"]: r for r in cond["program"]["rooms"]}
    rooms, index, centre = [], {}, []
    raw_rooms = merge_stair_rooms(raw["storeys"][k])
    for r in raw_rooms:
        c = Polygon(r["polygon"]).buffer(0)
        net = c.buffer(-wt / 2, join_style="mitre")
        if net.geom_type != "Polygon" or net.is_empty:
            net = max(getattr(net, "geoms", [c]), key=lambda g: g.area)
        native = "stairs" if r.get("stair_room") else prog[r["program_room"]]["type"]
        index[r["name"]] = len(rooms)
        if r.get("merged"):  # openings of the merged stair room belong to its host
            index[r["merged"]] = len(rooms)
        centre.append(c)
        rooms.append({"type": canonical_room_type(native, EXTRA_ALIASES[kind]), "type_native": native,
                      "polygon": ring(net), "program_room": r["program_room"], "infinigen_type": r["infinigen_type"]})
    walls = unary_union([c.boundary for c in centre]).buffer(wt / 2, cap_style="flat", join_style="mitre")
    doors, windows, edges = [], [], []
    for o in raw["openings"][k]:
        g = wkt.loads(o["footprint"]).intersection(walls)
        if g.is_empty or g.area < 1e-4:
            continue
        rect = shapely.minimum_rotated_rectangle(g)
        xy = list(rect.exterior.coords)
        width = max(shapely.distance(shapely.Point(xy[0]), shapely.Point(xy[1])),
                    shapely.distance(shapely.Point(xy[1]), shapely.Point(xy[2])))
        if o["kind"] == "window":
            # Infinigen's floor top is at wt / 2 (BlueprintSolidifier.make_window_cutter: z_loc = ... + wt / 2).
            windows.append({"polygon": ring(rect), "width_m": round(width, 3),
                            "sill_m": round(o["z"][0] - wt / 2, 3), "head_m": round(o["z"][1] - wt / 2, 3),
                            "room": index.get(o["rooms"][0])})
            continue
        dtype = {"door": "interior_door", "opening": "opening", "front_door": "front_door"}[o["kind"]]
        doors.append({"type": dtype, "polygon": ring(rect), "width_m": round(width, 3)})
        if len(o["rooms"]) == 2 and all(n in index for n in o["rooms"]) and index[o["rooms"][0]] != index[o["rooms"][1]]:
            edges.append([index[o["rooms"][0]], index[o["rooms"][1]], "door" if o["kind"] == "door" else "open"])
    boundary = unary_union(centre)
    cond = dict(cond, type="brief+program")
    return make_sample("infinigen", variant, sample_id, rooms, units="m", scale_m_per_unit=1.0,
                       boundary=ring(boundary) if boundary.geom_type == "Polygon" else None,
                       doors=doors, windows=windows, graph={"edges": edges}, condition=cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="apartment", choices=["apartment", "house"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contour", default="fixed", choices=["fixed", "free"])
    ap.add_argument("--source", default=None, help="our variant whose programs are used (default ours/mason_old_<kind>_n100_seed0)")
    ap.add_argument("--name", default=None, help="output variant name (default <kind>_program[...]_n100_seed<seed>)")
    a = ap.parse_args()
    variant = a.name or f"{a.kind}_program{'' if a.contour == 'fixed' else '_freecontour'}_n100_seed{a.seed}"
    vdir = ROOT / "outputs" / "infinigen" / variant
    ours = ours_samples(a.kind, a.source)
    n, missing = 0, []
    for p in sorted((vdir / "raw").glob("*.json")):
        raw = json.load(open(p))
        for k in range(raw["n_storeys"]):
            sid, cond = ours[(raw["home"], k)]
            save_sample(convert_storey(raw, k, a.kind, variant, sid, cond), vdir / "samples" / f"{sid}.json")
            n += 1
    homes = {h for h, _ in ours}
    missing = sorted(homes - {p.stem for p in (vdir / "raw").glob("*.json")})
    print(f"{variant}: {n} samples; {len(missing)} homes without a plan: {missing}")
    json.dump({"source": "Infinigen Indoors floor plan solver (layout stage only), our programs as input",
               "infinigen": "1.12.3", "n_samples": n, "missing_homes": missing},
              open(vdir / "run_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
