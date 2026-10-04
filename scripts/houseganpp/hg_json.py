"""Helpers for RPLAN in House-GAN++ JSON format (shared by houseganpp and house_diffusion converters).

HG++ JSON (from Housegan-data-reader): room_type (HG++ ids), boxes, edges [x0,y0,x1,y1,t,t'], ed_rm.
Both methods centre the plan in the 256 px canvas before use (shift = bbox centre -> 128); the
GT layouts exported here use the same shift so they share the frame of the generated samples.
"""
import json

import numpy as np

from fpeval.format import canonical_room_type, make_sample

HG_CLASS = {1: "living_room", 2: "kitchen", 3: "bedroom", 4: "bathroom", 5: "balcony",
            6: "entrance", 7: "dining room", 8: "study room", 10: "storage", 15: "front door",
            16: "unknown", 17: "interior_door"}
DOOR_TYPES = {15: "front_door", 17: "interior_door"}


def hg_room(t):
    native = HG_CLASS.get(int(t), str(t))
    return {"type": canonical_room_type(native), "type_native": native}


def chain(segs):
    """Order a closed loop of segments [(x0,y0,x1,y1)] into a vertex list."""
    segs = [((s[0], s[1]), (s[2], s[3])) for s in segs]
    poly = [segs[0][0], segs[0][1]]
    used = {0}
    while len(used) < len(segs):
        cur = poly[-1]
        for k, (a, b) in enumerate(segs):
            if k in used:
                continue
            if a == cur or b == cur:
                poly.append(b if a == cur else a)
                used.add(k)
                break
        else:
            break  # not a single loop; keep what we have
    if poly[-1] == poly[0]:
        poly = poly[:-1]
    # drop collinear points
    out = []
    n = len(poly)
    for i in range(n):
        p0, p1, p2 = np.array(poly[i - 1]), np.array(poly[i]), np.array(poly[(i + 1) % n])
        if abs(np.cross(p1 - p0, p2 - p1)) > 1e-9:
            out.append([float(p1[0]), float(p1[1])])
    return out if len(out) >= 3 else [[float(x), float(y)] for x, y in poly]


def load(path):
    d = json.load(open(path))
    types = [int(t) for t in d["room_type"]]
    boxes = np.array(d["boxes"], float)
    tl, br = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    shift = (tl + br) / 2.0 - 128.0
    polys = []
    for k in range(len(types)):
        segs = [np.array(e[:4]) - np.tile(shift, 2) for e, r in zip(d["edges"], d["ed_rm"]) if r[0] == k]
        polys.append(chain([tuple(s) for s in segs]))
    # node adjacency exactly as HG++/HD build_graph: any shared edge in ed_rm
    adj = set()
    for r in d["ed_rm"]:
        if len(r) > 1:
            a, b = sorted(r[:2])
            if a != b:
                adj.add((a, b))
    return {"types": types, "polys": polys, "node_edges": sorted(adj), "shift": shift.tolist()}


def room_graph(types, node_edges):
    """Room-room edges (node indices) from the node graph: 'door' if a shared interior door node
    touches both rooms, else 'adjacent' (shared wall). Front-door-adjacent rooms returned separately."""
    rooms = [i for i, t in enumerate(types) if t not in DOOR_TYPES]
    nb = {i: set() for i in range(len(types))}
    for a, b in node_edges:
        nb[a].add(b)
        nb[b].add(a)
    edges = {}
    for a, b in node_edges:
        if a in rooms and b in rooms:
            edges[(a, b)] = "adjacent"
    for d, t in enumerate(types):
        if t == 17:
            rs = sorted(r for r in nb[d] if r in rooms)
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    edges[(rs[i], rs[j])] = "door"
    front = sorted({r for d, t in enumerate(types) if t == 15 for r in nb[d] if r in rooms})
    return edges, front


def condition(ref_id, types, node_edges):
    return {"type": "bubble_diagram", "ref_id": f"rplan_{ref_id}", "format": "housegan++ json",
            "nodes": [HG_CLASS.get(t, str(t)) for t in types], "node_type_ids": types,
            "node_edges": [list(e) for e in node_edges]}


def assemble(source, variant, sid, node_polys, types, node_edges, ref_id, extra_cond=None):
    """node_polys: per input node, a list of polygons (possibly empty). Rooms -> rooms list (one entry
    per component), door nodes -> doors. Graph edges map to the first (largest) component per node."""
    rooms, doors, first = [], [], {}
    for k, (t, ps) in enumerate(zip(types, node_polys)):
        for p in ps:
            if len(p) < 3:
                continue
            if t in DOOR_TYPES:
                doors.append({"type": DOOR_TYPES[t], "polygon": p})
            else:
                first.setdefault(k, len(rooms))
                rooms.append({**hg_room(t), "polygon": p, "node": k})
    redges, front = room_graph(types, node_edges)
    g = [[first[a], first[b], kind] for (a, b), kind in sorted(redges.items()) if a in first and b in first]
    cond = condition(ref_id, types, node_edges)
    cond["front_door_rooms"] = front
    cond["missing_nodes"] = [k for k, t in enumerate(types) if t not in DOOR_TYPES and k not in first]
    if extra_cond:
        cond.update(extra_cond)
    return make_sample(source, variant, sid, rooms, units="px", doors=doors, graph={"edges": g},
                       condition=cond)


def gt_sample(source, variant, path, ref_id):
    h = load(path)
    return assemble(source, variant, ref_id, [[p] for p in h["polys"]], h["types"], h["node_edges"], ref_id,
                    extra_cond={"gt_of": ref_id, "frame_shift_px": h["shift"]})
