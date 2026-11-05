"""Infinigen Indoors floor plans for the same room programs as our homes (layout stage only).

Infinigen's FloorPlanSolver (infinigen/core/constraints/example_solver/room/floor_plan.py) normally samples its own
room graph from home_room_constraints() and sizes a rectangular outline from typical room areas. Here both inputs
come from our program (outputs/ours/<variant>/samples/*.json, condition.program), so Infinigen and ours solve the
same problem:
  graph   one RoomGraph per storey: our rooms with Infinigen types, door/open edges between rooms of that storey,
          the Exterior node linked to rooms that need an exterior wall and to the entrance room
  outline area = sum of the program's target net areas on that storey (Infinigen's slackness factor is kept);
          aspect ratio and the rest are Infinigen's own. With --contour fixed (default; Infinigen's fixed_contour
          mode, which home_room_constraints() switches off) the outline keeps that size, so the net area follows
          the brief; with --contour free the annealing may grow it (homes then come out ~2.3x the brief's area).
Everything else is Infinigen's: outline segmentation, room assignment, simulated annealing, outline decoration.
Blender solidification (walls, doors, windows) and furniture are skipped.

Writes outputs/infinigen/<kind>_program[_freecontour]_n100_seed0/raw/<home>.json: per storey, room names, types and polygons.

    /datawaha/cggroup/birsakm/miniconda3/envs/infinigen/bin/python scripts/infinigen/generate.py --kind apartment
"""
import argparse
import json
import sys
import traceback
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TYPES = {"living": "LivingRoom", "kitchen": "Kitchen", "bedroom": "Bedroom", "bathroom": "Bathroom", "small_bathroom": "Bathroom",
         "circulation": "Hallway", "outdoor": "Balcony", "storage": "Closet", "laundry": "Utility",
         "office": "Office", "dining": "DiningRoom", "stairs": "StaircaseRoom"}


def load_programs(kind, source=None):
    """{home: (program, n_storeys, seed)} from our converted samples."""
    out = {}
    for p in sorted((ROOT / "outputs" / (source or f"ours/mason_old_{kind}_n100_seed0") / "samples").glob("*.json")):
        c = json.load(open(p))["condition"]
        out.setdefault(c["home"], (c["program"], c["n_storeys"]))
    return {h: (prog, n, i) for i, (h, (prog, n)) in enumerate(sorted(out.items()))}


STAIR_M2 = 3.5  # synthetic staircase room (Mason2 stairs: 1 m wide straight flight, ~3.2 m^2 stairwell void)


def with_stair_rooms(program, n_storeys):
    """Infinigen's multi-storey solver needs a StaircaseRoom on every storey. Old Mason programs have stair rooms;
    Mason2 programs instead put the stair inside a room (program edge kind "stair", e.g. hallway:0 -- hallway:1).
    Then each such room gets an adjacent synthetic stair room "stairs:auto@<room>", open to it; scripts/infinigen/
    convert.py merges it back into that room, so the layouts have the program's rooms, like ours."""
    if n_storeys < 2 or any(r["type"] == "stairs" for r in program["rooms"]):
        return program
    rooms = {r["id"]: r for r in program["rooms"]}
    hosts = {}
    for e in program["edges"]:
        if e["kind"] == "stair":
            for x in (e["room_a"], e["room_b"]):
                if x in rooms:
                    hosts.setdefault(rooms[x]["storey"], x)
    for lvl in range(n_storeys):  # a storey without a stair edge: its largest circulation room, else its largest room
        if lvl not in hosts:
            on = [r for r in program["rooms"] if r["storey"] == lvl]
            hall = [r for r in on if r["type"] == "circulation"] or on
            hosts[lvl] = max(hall, key=lambda r: r["target_net_area"] or 0)["id"]
    extra = [{"id": f"stairs:auto@{h}", "type": "stairs", "target_net_area": STAIR_M2, "storey": lvl,
              "needs_exterior": False} for lvl, h in sorted(hosts.items())]
    edges = [{"room_a": x["id"], "room_b": x["id"].split("@", 1)[1], "kind": "open"} for x in extra]
    return dict(program, rooms=program["rooms"] + extra, edges=program["edges"] + edges)


def room_graphs(program, n_storeys):
    """One Infinigen RoomGraph per storey, nodes in BFS order from the entrance (exterior node last)."""
    import networkx as nx
    from infinigen.core.constraints.example_solver.room.base import RoomGraph, room_name
    from infinigen.core.tags import Semantics

    rooms = {r["id"]: r for r in program["rooms"]}
    G = nx.Graph()
    G.add_nodes_from(rooms)
    G.add_edges_from((e["room_a"], e["room_b"]) for e in program["edges"]
                     if e["kind"] in ("door", "open") and e["room_a"] in rooms and e["room_b"] in rooms
                     and rooms[e["room_a"]]["storey"] == rooms[e["room_b"]]["storey"])
    ent = program.get("entrance") or {}
    ent_room = ent.get("room") if isinstance(ent, dict) else ent
    graphs, names_of = [], {}
    for lvl in range(n_storeys):
        ids = [i for i, r in rooms.items() if r["storey"] == lvl]
        H = G.subgraph(ids)
        start = ent_room if ent_room in ids else next(
            (i for i in ids if rooms[i]["type"] == "stairs"), max(ids, key=H.degree))
        order = []
        for comp_start in [start] + ids:  # BFS, then any disconnected rest
            if comp_start not in order:
                order += [n for n in nx.bfs_tree(H, comp_start) if n not in order]
        count = defaultdict(int)
        names = []
        for i in order:
            t = Semantics(Semantics[TYPES[rooms[i]["type"]]].value)
            names.append(room_name(t, lvl, count[t]))
            count[t] += 1
            names_of[names[-1]] = i
        names.append(room_name(Semantics.Exterior, lvl))
        ext = len(names) - 1
        children = [[order.index(j) for j in H.neighbors(i)] for i in order]
        outside = [k for k, i in enumerate(order) if rooms[i].get("needs_exterior") or i == ent_room]
        children.append(outside)
        graphs.append(RoomGraph(children, names, order.index(ent_room) if ent_room in order else None))
    return graphs, names_of


def cutters(solidifier, state):
    """Doors, openings, entrance and windows of one storey, placed by Infinigen's own BlueprintSolidifier rules.

    Runs the cutter part of BlueprintSolidifier.solidify (room meshes are not built) and reads each Blender cutter
    back: footprint (world xy convex hull), bottom and top (z) and kind (Door, Entrance, Open, Window).
    """
    import bpy
    import numpy as np
    import shapely
    from infinigen.core.constraints.example_solver.room import solidifier as sol
    from infinigen.core.constraints.example_solver.room.base import room_type, valid_rooms
    from infinigen.core.tags import Semantics

    wt = solidifier.constants.wall_thickness
    segments = {k: o.polygon for k, o in valid_rooms(state)}
    shared = {k: {r.target_name: sol.canonicalize_mls(r.value) for r in o.relations} for k, o in valid_rooms(state)}
    ext = next(k for k in state.objs if room_type(k) == Semantics.Exterior)
    ext_edges = {r.target_name: sol.mls_ccw(sol.canonicalize_mls(r.value), state, r.target_name)
                 for r in state[ext].relations}
    ext_buffer = shapely.simplify(state[ext].polygon.buffer(-wt / 2 - sol._eps, join_style="mitre"), 1e-3)
    open_c, door_c, inner_c = solidifier.make_interior_cutters(
        solidifier.graph.valid_neighbours, shared, segments, ext_buffer)
    window_c, entrance_c = solidifier.make_exterior_cutters(ext_edges, state[ext].polygon)
    bpy.context.view_layer.update()
    out = []
    for kind, group in [("door", door_c), ("opening", open_c), ("window", inner_c), ("window", window_c),
                        ("front_door", entrance_c)]:
        seen = set()
        for rooms, c in solidifier.unroll(group):
            if c.name in seen:  # interior cutters are stored under both rooms
                continue
            seen.add(c.name)
            v = np.array([c.matrix_world @ p.co for p in c.data.vertices])
            out.append({"kind": kind, "rooms": list(rooms), "footprint": shapely.MultiPoint(v[:, :2]).convex_hull.wkt,
                        "z": [float(v[:, 2].min()), float(v[:, 2].max())]})
    return out


def solve(args):
    home, (program, n_storeys, idx), out_dir, seed, fixed = args
    out = out_dir / f"{home}.json"
    if out.exists():
        return home, "exists"
    try:
        import numpy as np
        from infinigen.core.constraints.example_solver.room import floor_plan, graph as graph_mod
        from infinigen.core.constraints.example_solver.room.base import room_level, room_type
        from infinigen.core.tags import Semantics
        from infinigen.core.util.math import FixedSeed
        from infinigen_examples.constraints import home as home_constraints

        program = with_stair_rooms(program, n_storeys)
        graphs, names_of = room_graphs(program, n_storeys)
        storey_area = defaultdict(float)
        for r in program["rooms"]:
            storey_area[r["storey"]] += r["target_net_area"]

        def make_graph(self, i):
            return graphs[self.level]

        def suggest_dimensions(self, graph, width=None, height=None):
            area = storey_area[self.level] * self.slackness
            if width is None and height is None:
                aspect = np.random.uniform(*self.constants.aspect_ratio_range)
            else:
                aspect = width / height
            return (self.constants.unit_cast(np.sqrt(area * aspect).item()),
                    self.constants.unit_cast(np.sqrt(area / aspect).item()))

        graph_mod.GraphMaker.make_graph = make_graph
        graph_mod.GraphMaker.__call__ = make_graph
        graph_mod.GraphMaker.suggest_dimensions = suggest_dimensions

        # With shapely 2.1, an annealing move that collapses a room raises ValueError (fewer than 4 ring
        # coordinates) where Infinigen expects NotImplementedError ("Polygon empty"), its signal to reject the move.
        from infinigen.core.constraints.constraint_language.constants import RoomConstants
        canonicalize = RoomConstants.canonicalize

        def canonicalize_or_reject(self, p):
            try:
                return canonicalize(self, p)
            except ValueError as e:
                raise NotImplementedError(str(e))

        RoomConstants.canonicalize = canonicalize_or_reject

        s = seed * 1000 + idx
        with FixedSeed(s):
            consgraph = home_constraints.home_room_constraints()
            consgraph.constants.n_stories = n_storeys
            consgraph.constants.fixed_contour = fixed
            fps = floor_plan.FloorPlanSolver(s, consgraph)
            # FloorPlanSolver.solve() up to (excluding) Blender solidification.
            state = floor_plan.State(graphs=fps.graphs)
            states = []
            while len(states) < fps.n_stories:
                pholder = fps.contour_factory.add_staircase(fps.contours[-1])
                state.objs, states = {}, []
                for j in range(fps.n_stories):
                    for _ in range(fps.n_divide_trials * (j + 1) ** 2):
                        st = fps.segment_makers[j].build_segments(pholder)
                        if st is not None:
                            states.append(st)
                            state.objs.update(st.objs)
                            break
                    else:
                        break
            state = fps.simulated_anneal(state)
            fps.contour_factory.decorate(state)
            openings = [cutters(fps.solidifiers[j], floor_plan.State(
                {k: v for k, v in state.objs.items() if room_level(k) == j})) for j in range(n_storeys)]
        storeys = [[] for _ in range(n_storeys)]
        for name, ob in state.objs.items():
            if room_type(name) in (Semantics.Exterior, Semantics.Staircase) or ob.polygon is None:
                continue
            storeys[room_level(name)].append({"name": name, "infinigen_type": room_type(name).value,
                                              "program_room": names_of.get(name),
                                              "polygon": [list(map(float, c)) for c in ob.polygon.exterior.coords[:-1]]})
        c = consgraph.constants
        json.dump({"home": home, "seed": s, "n_storeys": n_storeys, "wall_thickness": float(c.wall_thickness),
                   "wall_height": float(c.wall_height), "graphs": [str(g) for g in graphs], "storeys": storeys,
                   "openings": openings}, open(out, "w"))
        return home, "ok"
    except Exception:
        (out_dir / f"{home}.err").write_text(traceback.format_exc())
        return home, "failed"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="apartment", choices=["apartment", "house"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--homes", nargs="*", help="only these homes")
    ap.add_argument("--contour", default="fixed", choices=["fixed", "free"])
    ap.add_argument("--source", default=None, help="our variant whose programs are used (default ours/mason_old_<kind>_n100_seed0)")
    ap.add_argument("--name", default=None, help="output variant name (default <kind>_program[...]_n100_seed<seed>)")
    a = ap.parse_args()
    progs = load_programs(a.kind, a.source)
    if a.homes:
        progs = {h: progs[h] for h in a.homes}
    tag = "" if a.contour == "fixed" else "_freecontour"
    out_dir = ROOT / "outputs/infinigen" / (a.name or f"{a.kind}_program{tag}_n100_seed{a.seed}") / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(h, p, out_dir, a.seed, a.contour == "fixed") for h, p in progs.items()]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for home, status in pool.imap_unordered(solve, jobs):
            print(home, status, flush=True)


if __name__ == "__main__":
    sys.exit(main())
