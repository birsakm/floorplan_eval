"""Real door graphs of the 1,000 Table 1(a) inputs (from the original RPLAN masks), compared with the synthesised
door graphs the bubble-diagram methods received.

  PYTHONPATH=. python scripts/rplan/real_door_graphs.py   # -> results/rplan/door_graphs.json, .md

Rooms come from the masks themselves (Graph2Plan's room boxes do not follow the walls): free space splits into
wall-enclosed regions, each identified by the RPLAN room centroid(s) in it and matched to the input's rooms by
category and position. A real door (data/rplan_kaggle: one box per door run) connects the regions on its two sides.
Open connection: several rooms in one region (no wall, no door between them); bubble diagrams have no such edge.
"""
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from fpeval.datasets.rplan_kaggle import door_boxes, load_pickle, pickle_index  # noqa: E402

PROGRAMS = ROOT / "outputs" / "rplan" / "g2p_test1000_programs"  # the synthesised-door inputs it compares with
SCALE = 18 / 256
PK = None


CAT = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom", "StudyRoom", "SecondRoom",
       "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in"]


def sides(box):
    """Pixels just beyond the two long sides of a door box."""
    (x0, y0), (x1, y1) = box[0], box[2]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 - x0 >= y1 - y0:
        xs = range(x0 + 1, max(x0 + 2, x1 - 1))
        return [(min(y1 + 1, 255), x) for x in xs], [(max(y0 - 2, 0), x) for x in xs]
    ys = range(y0 + 1, max(y0 + 2, y1 - 1))
    return [(y, min(x1 + 1, 255)) for y in ys], [(y, max(x0 - 2, 0)) for y in ys]


def match_rooms(prooms, g2p):
    """RPLAN room (category, centroid) -> input room index: same category, nearest input polygon centroid
    (one-to-one per category)."""
    from scipy.optimize import linear_sum_assignment
    cats = [CAT.index(r["type_native"]) for r in g2p["rooms"]]
    cen = [np.mean(np.asarray(r["polygon"], float), 0) for r in g2p["rooms"]]  # (x, y) px
    out = {}
    for c in set(r["category"] for r in prooms):
        P = [k for k, r in enumerate(prooms) if r["category"] == c]
        G = [k for k, t in enumerate(cats) if t == c]
        D = np.array([[np.hypot(prooms[a]["centroid"][1] - cen[b][0], prooms[a]["centroid"][0] - cen[b][1]) for b in G]
                      for a in P])
        for a, b in zip(*linear_sum_assignment(D)):
            out[P[a]] = G[b]
    return out


def one(gid):
    """Rooms from the masks: free space (inside, not wall, not door) splits into regions; a region holds the RPLAN
    room(s) whose centroid lies in it (a region without centroid is a hall: it joins the living room, as RPLAN
    labels such space; a region with several is an open connection between them). A door connects the regions on
    its two sides."""
    g2p = json.load(open(ROOT / "data" / "rplan_graph2plan" / "converted" / "samples" / f"test_{gid}.json"))
    prog = json.load(open(PROGRAMS / "samples" / f"{gid}.json"))["condition"]["program"]
    gt = json.load(open(PROGRAMS / "gt_samples" / f"{gid}.json"))
    node_of = [r["node"] for r in gt["rooms"]]
    pid = {f"r{n}": n for n in node_of}  # input room id -> node index = Graph2Plan room index
    p = load_pickle(PK[gid])
    wall = (p[2] > 0) | (p[1] == 127)
    door = p[3] > 0
    free = (p[0] > 0) & ~wall & ~door
    comp, nc = ndimage.label(free)
    m = match_rooms(p[4], g2p)
    living = next(k for k, r in enumerate(g2p["rooms"]) if r["type_native"] == "LivingRoom")
    rooms_in = {c: [] for c in range(1, nc + 1)}
    for k, r in enumerate(p[4]):
        c = comp[r["centroid"][0], r["centroid"][1]]
        if c > 0 and k in m:
            rooms_in[c].append((k, m[k]))
    halls = sum(1 for v in rooms_in.values() if not v)

    def room_at(pts):
        cs = Counter(comp[q] for q in pts if comp[q] > 0)
        if not cs:
            return None
        c = cs.most_common(1)[0][0]
        if not rooms_in[c]:
            return living
        y, x = np.mean(pts, 0)  # several rooms in one region: the one with the nearest centroid
        k, g = min(rooms_in[c], key=lambda kg: np.hypot(p[4][kg[0]]["centroid"][0] - y, p[4][kg[0]]["centroid"][1] - x))
        return g
    real = set()
    for b in door_boxes(door, wall):
        a, c = (room_at(s) for s in sides(b))
        if a is not None and c is not None and a != c:
            real.add(tuple(sorted((a, c))))
    opened = set()
    for v in rooms_in.values():
        gs = sorted({g for _, g in v})
        opened |= {(a, b) for i, a in enumerate(gs) for b in gs[i + 1:]}
    syn = {tuple(sorted((pid[e["room_a"]], pid[e["room_b"]]))) for e in prog["edges"]}
    types = [r["type"] for r in g2p["rooms"]]
    return {"id": gid, "n_rooms": len(types), "types": types, "real_doors": sorted(real), "open": sorted(opened - real),
            "synthetic": sorted(syn), "halls": halls, "unmatched_rplan_rooms": len(p[4]) - len(m)}


def typed(types, edges):
    G = nx.Graph()
    G.add_nodes_from((i, {"t": t}) for i, t in enumerate(types))
    G.add_edges_from(edges)
    return G


def main():
    global PK
    PK = pickle_index()
    ids = sorted(f.stem for f in (PROGRAMS / "gt_samples").glob("*.json"))
    with Pool(16) as pool:
        R = pool.map(one, ids, chunksize=8)
    S = Counter()
    pair_types = Counter()
    for r in R:
        real, syn, opn = set(map(tuple, r["real_doors"])), set(map(tuple, r["synthetic"])), set(map(tuple, r["open"]))
        t = r["types"]
        Gr, Gs = typed(t, real), typed(t, syn)
        Gro = typed(t, real | opn)
        S["n"] += 1
        S["equal"] += real == syn
        S["isomorphic"] += nx.is_isomorphic(Gr, Gs, node_match=lambda a, b: a["t"] == b["t"])
        S["jaccard"] += len(real & syn) / max(1, len(real | syn))
        S["recall_syn_in_real"] += len(real & syn) / max(1, len(syn))
        S["real_edges"] += len(real)
        S["syn_edges"] += len(syn)
        S["open_edges"] += len(opn)
        S["real_connected"] += nx.is_connected(Gr)
        S["real_open_connected"] += nx.is_connected(Gro)
        S["real_tree"] += nx.is_tree(Gr)
        S["has_open"] += bool(opn)
        S["non_living_door"] += any("living_room" not in (t[a], t[b]) for a, b in real)
        for a, b in real:
            pair_types[" - ".join(sorted((t[a], t[b])))] += 1
    n = S["n"]
    out = {"n": n,
           "real_equals_synthetic": S["equal"] / n,
           "real_isomorphic_to_synthetic (typed)": S["isomorphic"] / n,
           "edge_jaccard_mean": S["jaccard"] / n,
           "synthetic_edges_that_are_real": S["recall_syn_in_real"] / n,
           "door_edges_per_plan_real": S["real_edges"] / n, "door_edges_per_plan_synthetic": S["syn_edges"] / n,
           "open_connections_per_plan": S["open_edges"] / n, "plans_with_open_connections": S["has_open"] / n,
           "real_door_graph_connected": S["real_connected"] / n,
           "real_doors_plus_open_connected": S["real_open_connected"] / n,
           "real_door_graph_is_tree": S["real_tree"] / n,
           "plans_with_a_door_between_two_non_living_rooms": S["non_living_door"] / n,
           "real_door_pairs_by_type": dict(pair_types.most_common(15))}
    res = ROOT / "results" / "rplan"
    json.dump({"summary": out, "plans": R}, open(res / "door_graphs.json", "w"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
