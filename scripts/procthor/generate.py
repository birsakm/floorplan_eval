"""ProcTHOR floor plans for the same room programs as our homes (layout stage only), written in the common format.

ProcTHOR (procthor/generation, submodule external/methods/procthor) normally samples a hand-written RoomSpec from
PROCTHOR10K_ROOM_SPEC_SAMPLER. Here the RoomSpec is built from our program (outputs/ours/<variant>/samples/*.json,
condition.program), one per storey, so ProcTHOR and ours solve the same problem:
  spec    one LeafRoom per program room, ratio = target net area (m², rounded). ProcTHOR only places doors between
          siblings of the RoomSpec tree, so the door graph is expressed as a tree: a room together with the rooms that
          hang only off it (program degree 1, e.g. an en-suite bathroom) becomes a MetaRoom; everything else is at
          the top level. Those hanging rooms get avoid_doors_from_metarooms (as ProcTHOR's bathrooms).
          LeafRoom.room_type is ProcTHOR's (Kitchen, LivingRoom, Bedroom, Bathroom) or None for other types; our
          type is kept on our side. ProcTHOR's layout uses room types only for the outdoor door and open
          connections, and both come from our program here.
  size    grid of cells of interior_boundary_scale metres, as in ProcTHOR (scale ~ U(1.6, 2.2)); the grid is sized so
          that the boundary (after ProcTHOR's random corner cuts) has about the program's area, and the scale is then
          set so that it matches exactly (the analogue of Infinigen's fixed contour).
  doors   ProcTHOR's select_openings / select_door_walls on the RoomSpec tree. A selected pair that is an "open" edge
          in our program becomes an open connection over the whole shared wall (ProcTHOR's empty wall); everything
          else a door, its width a ProcTHOR Doorway asset width (asset-database.json, train split) that fits the
          wall. The front door goes to our entrance room on the ground storey when it has an exterior wall, else
          ProcTHOR's select_outdoor_openings picks it. Upper storeys get no outdoor door.
Everything else is ProcTHOR's: interior boundary cuts, rectangle and L-shape growth, candidate scoring
(generate_floorplan, 100 candidates), wall extraction. Walls have no thickness in ProcTHOR (room polygons share
edges). Windows, furniture and the AI2-THOR stages are not run (they need the simulator and asset pipeline).

    VENV=/datawaha/cggroup/birsakm/venvs/fpe-procthor; PYTHONPATH=. $VENV/bin/python scripts/procthor/generate.py --kind apartment
"""
import argparse
import json
import random
import sys
import types
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "methods" / "procthor"))
# The layout modules import ai2thor / moviepy only for the simulator stages; stub them.
for name, attr in [("ai2thor", None), ("ai2thor.controller", "Controller"), ("moviepy", None),
                   ("moviepy.editor", "ImageSequenceClip")]:
    mod = sys.modules.setdefault(name, types.ModuleType(name))
    if attr:
        setattr(mod, attr, type(attr, (), {}))

from fpeval.format import canonical_room_type, make_sample, save_sample  # noqa: E402
from procthor.constants import OUTDOOR_ROOM_ID  # noqa: E402
from procthor.generation.doors import (get_room_spec_neighbors, select_door_walls, select_openings,  # noqa: E402
                                       select_outdoor_openings)
from procthor.generation.floorplan_generation import generate_floorplan  # noqa: E402
from procthor.generation.generation import (consolidate_walls, find_walls, get_xz_poly_map,  # noqa: E402
                                            scale_boundary_groups)
from procthor.generation.interior_boundaries import sample_interior_boundary  # noqa: E402
from procthor.generation.room_specs import RoomSpec  # noqa: E402
from procthor.utils.types import InvalidFloorplan, LeafRoom, MetaRoom  # noqa: E402

PROCTHOR_TYPES = {"living": "LivingRoom", "kitchen": "Kitchen", "bedroom": "Bedroom", "bathroom": "Bathroom",
                  "small_bathroom": "Bathroom"}
# Same type mapping as our samples (scripts/ours/convert.py).
EXTRA_ALIASES = {"apartment": {"circulation": "corridor", "outdoor": "balcony", "office": "study", "small_bathroom": "bathroom"},
                 "house": {"circulation": "corridor", "outdoor": "outdoor", "office": "study", "stairs": "other"}}
SCALE_RANGE = (1.6, 2.2)  # procthor/utils/types.py: interior_boundary_scale
DOOR_T = 0.1  # drawn door / opening depth across the (zero-thickness) wall
N_BOUNDARY_TRIES = 500
DOORWAY_WIDTHS = sorted({round(d["boundingBox"]["x"], 3) for d in json.load(open(
    ROOT / "external/methods/procthor/procthor/databases/asset-database.json"))["Doorway"] if d["split"] == "train"})


def load_programs(kind, source=None):
    """[(home, storey, sample id, condition)] of our samples, and {home: index}."""
    out = []
    for p in sorted((ROOT / "outputs" / (source or f"ours/mason_old_{kind}_n100_seed0") / "samples").glob("*.json")):
        c = json.load(open(p))["condition"]
        out.append((c["home"], c["storey"] or 0, p.stem, c))
    homes = {h: i for i, h in enumerate(sorted({o[0] for o in out}))}
    return out, homes


def room_spec(program, storey):
    """RoomSpec for one storey and {leaf id: program room}."""
    rooms = [r for r in program["rooms"] if r["storey"] == storey]
    ids = {r["id"]: i + 2 for i, r in enumerate(rooms)}  # 0 and 1 are reserved (empty, outdoor)
    nb = defaultdict(set)
    for e in program["edges"]:
        if e["kind"] in ("door", "open") and e["room_a"] in ids and e["room_b"] in ids:
            nb[e["room_a"]].add(e["room_b"])
            nb[e["room_b"]].add(e["room_a"])
    ent = program.get("entrance")
    ent = ent.get("room") if isinstance(ent, dict) else ent
    root = ent if ent in ids else next((r["id"] for r in rooms if r["type"] == "stairs"),
                                       max(ids, key=lambda i: len(nb[i])))

    def leaf(r, hanging):
        return LeafRoom(room_id=ids[r["id"]], ratio=max(1, round(r["target_net_area"])),
                        room_type=PROCTHOR_TYPES.get(r["type"]), avoid_doors_from_metarooms=hanging)

    by_id = {r["id"]: r for r in rooms}
    hanging = {i for i in ids if len(nb[i]) == 1 and i != root}
    grouped, spec = set(), []
    for r in rooms:
        kids = [n for n in sorted(nb[r["id"]]) if n in hanging and r["id"] not in hanging] if r["id"] != root else []
        if kids:
            children = [leaf(r, False)] + [leaf(by_id[k], True) for k in kids]
            spec.append(MetaRoom(ratio=sum(c.ratio for c in children), children=children))
            grouped |= {r["id"], *kids}
    spec = [leaf(r, False) for r in rooms if r["id"] not in grouped] + spec
    return RoomSpec(room_spec_id="program", sampling_weight=1, spec=spec), {v: by_id[k] for k, v in ids.items()}


def layout(spec, area):
    """ProcTHOR floor plan for spec with a total area of `area` m²: (boundary groups in metres, scale)."""
    n = len(spec.room_type_map)
    for _ in range(N_BOUNDARY_TRIES):
        scale = random.uniform(*SCALE_RANGE)
        cells = area / scale ** 2
        aspect = random.uniform(1.0, 2.0)
        x = max(1, round(np.sqrt(cells * aspect)))
        z = max(1, round(cells / x))
        if x * z < n:
            continue
        boundary = sample_interior_boundary(num_rooms=n, dims=(x, z))
        try:
            floorplan = generate_floorplan(room_spec=spec, interior_boundary=boundary)
        except InvalidFloorplan:
            continue
        scale = float(np.sqrt(area / (boundary != OUTDOOR_ROOM_ID).sum()))
        floorplan = np.pad(floorplan, pad_width=1, mode="constant", constant_values=OUTDOOR_ROOM_ID)
        return scale_boundary_groups(consolidate_walls(find_walls(floorplan)), scale), scale
    raise InvalidFloorplan(f"no floor plan after {N_BOUNDARY_TRIES} interior boundaries")


def ring(p):
    return [[round(float(a), 4), round(float(b), 4)] for a, b in p.exterior.coords[:-1]]


def door_on(seg, width):
    """Door rectangle of `width` at a uniformly random position along wall segment seg."""
    (x0, z0), (x1, z1) = seg
    line = LineString([(x0, z0), (x1, z1)])
    t = random.uniform(width / 2, line.length - width / 2)
    a, b = line.interpolate(t - width / 2), line.interpolate(t + width / 2)
    return LineString([a, b]).buffer(DOOR_T / 2, cap_style="flat")


def solve(args):
    kind, home, storey, sid, cond, seed, variant = args
    random.seed(seed)
    np.random.seed(seed)
    program = cond["program"]
    spec, leaf_room = room_spec(program, storey)
    area = sum(r["target_net_area"] for r in leaf_room.values())
    groups, scale = layout(spec, area)
    polys = {k: Polygon([p0 for p0, _ in loop]) for k, loop in get_xz_poly_map(groups, set(leaf_room)).items()}
    order = sorted(polys)
    index = {k: i for i, k in enumerate(order)}
    rooms = [{"type": canonical_room_type(leaf_room[k]["type"], EXTRA_ALIASES[kind]),
              "type_native": leaf_room[k]["type"], "polygon": ring(polys[k]), "program_room": leaf_room[k]["id"]}
             for k in order]
    kinds = {frozenset((e["room_a"], e["room_b"])): e["kind"] for e in program["edges"]}
    openings = select_openings(neighboring_rooms=set(groups), room_spec_neighbors=get_room_spec_neighbors(spec.spec),
                               room_spec=spec)
    doors, edges = [], []
    for pair, seg in select_door_walls(openings, groups).items():
        a, b = pair
        if kinds.get(frozenset((leaf_room[a]["id"], leaf_room[b]["id"]))) == "open":
            walls = [LineString(w) for w in groups[pair]]
            g = unary_union([w.buffer(DOOR_T / 2, cap_style="flat") for w in walls])
            doors.append({"type": "opening", "polygon": ring(g.envelope), "width_m": round(sum(w.length for w in walls), 3)})
            edges.append([index[a], index[b], "open"])
        else:
            length = LineString(seg).length
            width = random.choice([w for w in DOORWAY_WIDTHS if w <= length - 0.2] or [length])
            doors.append({"type": "interior_door", "polygon": ring(door_on(seg, width)), "width_m": round(width, 3)})
            edges.append([index[a], index[b], "door"])
    if storey == 0:
        ent = program.get("entrance")
        ent = ent.get("room") if isinstance(ent, dict) else ent
        ent_leaf = next((k for k, r in leaf_room.items() if r["id"] == ent), None)
        outdoor = ([(OUTDOOR_ROOM_ID, ent_leaf)] if (OUTDOOR_ROOM_ID, ent_leaf) in groups else
                   select_outdoor_openings(boundary_groups=groups, room_type_map=spec.room_type_map))
        for pair, seg in select_door_walls(outdoor, groups).items():
            length = LineString(seg).length
            width = random.choice([w for w in DOORWAY_WIDTHS if w <= length - 0.2] or [length])
            doors.append({"type": "front_door", "polygon": ring(door_on(seg, width)), "width_m": round(width, 3)})
    outline = unary_union(list(polys.values()))
    sample = make_sample("procthor", variant, sid, rooms, units="m", scale_m_per_unit=1.0,
                         boundary=ring(outline) if outline.geom_type == "Polygon" else None,
                         doors=doors, windows=[], graph={"edges": edges},
                         condition=dict(cond, type="brief+program"))
    return sid, sample, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="apartment", choices=["apartment", "house"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--source", default=None, help="our variant whose programs are used (default ours/mason_old_<kind>_n100_seed0)")
    ap.add_argument("--name", default=None, help="output variant name (default <kind>_program[...]_n100_seed<seed>)")
    a = ap.parse_args()
    variant = a.name or f"{a.kind}_program_n100_seed{a.seed}"
    out = ROOT / "outputs" / "procthor" / variant
    plans, homes = load_programs(a.kind, a.source)
    jobs = [(a.kind, h, k, sid, c, a.seed * 100000 + homes[h] * 10 + k, variant) for h, k, sid, c in plans]
    failed, scales = [], []
    with Pool(a.workers) as pool:
        for job, res in zip(jobs, pool.imap(_safe_solve, jobs)):
            if isinstance(res, str):
                failed.append(job[3])
                print(job[3], "failed:", res, flush=True)
                continue
            sid, sample, scale = res
            save_sample(sample, out / "samples" / f"{sid}.json")
            scales.append(scale)
    print(f"{variant}: {len(jobs) - len(failed)} samples, {len(failed)} failed; scale {np.mean(scales):.2f} m/cell "
          f"(range {min(scales):.2f}-{max(scales):.2f})")
    json.dump({"source": "ProcTHOR layout stage (procthor/generation, 53d5bd4) with RoomSpecs built from our programs",
               "n_samples": len(jobs) - len(failed), "failed": failed,
               "scale_m_per_cell": [round(min(scales), 3), round(float(np.mean(scales)), 3), round(max(scales), 3)]},
              open(out / "run_info.json", "w"), indent=1)


def _safe_solve(job):
    try:
        return solve(job)
    except Exception as e:  # noqa: BLE001 - reported per plan
        return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
