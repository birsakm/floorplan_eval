"""RPLAN's real door graph of one plan, on the Graph2Plan rooms (shared by real_door_graphs.py and the inputs of
Table 1(a): scripts/houseganpp/g2p_to_hgjson.py --real-doors).

Rooms come from the original RPLAN masks (data/rplan_kaggle): free space (inside, not wall, not door) splits into
wall-enclosed regions; each region holds the RPLAN room centroid(s) in it, matched to the Graph2Plan rooms by
category and position (one-to-one per category). A region without a centroid is a hall and counts as the living
room, as RPLAN labels such space. Each real door (one box per door run, split where a wall meets it from the side)
connects the rooms of the regions on its two sides.
Rooms the real doors leave unconnected (7.9% of the 1,000 test plans, mostly balconies without a drawn door) get an
INFERRED door to the room of the connected part with which they share the longest wall.
"""
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from fpeval.datasets.rplan_kaggle import door_boxes, load_pickle, pickle_index  # noqa: E402

G2P = ROOT / "data" / "rplan_graph2plan" / "converted" / "samples"
CAT = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom", "StudyRoom", "SecondRoom",
       "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in"]
_PK = {}


def _pickles():
    if not _PK:
        _PK.update(pickle_index())
    return _PK


def sides(box):
    """Pixels (row, col) just beyond the two long sides of a door box."""
    (x0, y0), (x1, y1) = box[0], box[2]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 - x0 >= y1 - y0:
        xs = range(x0 + 1, max(x0 + 2, x1 - 1))
        return [(min(y1 + 1, 255), x) for x in xs], [(max(y0 - 2, 0), x) for x in xs]
    ys = range(y0 + 1, max(y0 + 2, y1 - 1))
    return [(y, min(x1 + 1, 255)) for y in ys], [(y, max(x0 - 2, 0)) for y in ys]


def match_rooms(prooms, g2p_rooms):
    from scipy.optimize import linear_sum_assignment
    cats = [CAT.index(r["type_native"]) for r in g2p_rooms]
    cen = [np.mean(np.asarray(r["polygon"], float), 0) for r in g2p_rooms]  # (x, y) px
    out = {}
    for c in set(r["category"] for r in prooms):
        P = [k for k, r in enumerate(prooms) if r["category"] == c]
        G = [k for k, t in enumerate(cats) if t == c]
        D = np.array([[np.hypot(prooms[a]["centroid"][1] - cen[b][0], prooms[a]["centroid"][0] - cen[b][1]) for b in G]
                      for a in P])
        for a, b in zip(*linear_sum_assignment(D)):
            out[P[a]] = G[b]
    return out


def door_graph(rplan_id):
    """{'doors': [(a, b, box)], 'inferred': [(a, b)], 'open': [(a, b)], 'front': room, 'front_box': box,
    'halls': n, 'n_rooms': n}; rooms are Graph2Plan room indices, boxes in the 256 px frame (x = col, y = row)."""
    g2p = json.load(open(G2P / f"test_{rplan_id}.json"))
    rooms = g2p["rooms"]
    p = load_pickle(_pickles()[str(rplan_id)])
    wall = (p[2] > 0) | (p[1] == 127)
    door = p[3] > 0
    free = (p[0] > 0) & ~wall & ~door
    comp, nc = ndimage.label(free)
    m = match_rooms(p[4], rooms)
    living = next(k for k, r in enumerate(rooms) if r["type_native"] == "LivingRoom")
    rooms_in = {c: [] for c in range(1, nc + 1)}
    for k, r in enumerate(p[4]):
        c = comp[r["centroid"][0], r["centroid"][1]]
        if c > 0 and k in m:
            rooms_in[c].append((k, m[k]))

    def region_room(c, at):
        if not rooms_in[c]:
            return living
        y, x = at
        return min(rooms_in[c], key=lambda kg: np.hypot(p[4][kg[0]]["centroid"][0] - y,
                                                         p[4][kg[0]]["centroid"][1] - x))[1]

    def room_at(pts):
        cs = Counter(comp[q] for q in pts if comp[q] > 0)
        return region_room(cs.most_common(1)[0][0], np.mean(pts, 0)) if cs else None

    doors = []
    for b in door_boxes(door, wall):
        a, c = (room_at(s) for s in sides(b))
        if a is not None and c is not None and a != c:
            doors.append((min(a, c), max(a, c), b))
    opened = set()
    for v in rooms_in.values():
        gs = sorted({g for _, g in v})
        opened |= {(a, b) for i, a in enumerate(gs) for b in gs[i + 1:]}
    # front door: the room just inside
    fb = door_boxes(p[1] == 255)
    front = None
    if fb:
        cands = [room_at(s) for s in sides(fb[0])]
        front = next((r for r in cands if r is not None), living)
    # inferred doors for rooms the real doors (and open connections) leave unconnected
    G = nx.Graph()
    G.add_nodes_from(range(len(rooms)))
    G.add_edges_from((a, b) for a, b, _ in doors)
    G.add_edges_from(opened)
    inferred = []
    region_of = {}
    for c, v in rooms_in.items():
        for _, g in v:
            region_of.setdefault(g, np.zeros_like(free))
            region_of[g] |= comp == c
    while not nx.is_connected(G):
        main = nx.node_connected_component(G, front if front is not None else living)
        best = None
        for c in nx.connected_components(G):
            if c == main:
                continue
            for a in c:
                if a not in region_of:
                    continue
                near = ndimage.binary_dilation(region_of[a], iterations=6)  # across a wall (<= 5 px)
                for b in main:
                    if b in region_of:
                        s = int((near & region_of[b]).sum())
                        if s and (best is None or s > best[0]):
                            best = (s, a, b)
        if best is None:  # no wall contact at all: attach to the living room
            c = next(c for c in nx.connected_components(G) if c != main)
            best = (0, min(c), living)
        _, a, b = best
        inferred.append((min(a, b), max(a, b)))
        G.add_edge(a, b)
    halls = sum(1 for v in rooms_in.values() if not v)
    return {"doors": doors, "inferred": inferred, "open": sorted(opened), "front": front,
            "front_box": fb[0] if fb else None, "halls": halls, "n_rooms": len(rooms)}
