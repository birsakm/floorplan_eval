"""Build House-GAN++-format JSON bubble-diagram inputs from the Graph2Plan-preprocessed RPLAN split
(/datawaha/cggroup/datasets/RPLAN/Network/data/data_{test,train}.mat), used when the original
HG++ JSON (from raw RPLAN PNGs) are not available.

APPROXIMATIONS (Graph2Plan data has no interior doors and no wall gaps):
  * rooms: Graph2Plan rBoundary polygons (wall-centre lines, 256 px RPLAN frame, x=col, y=row);
    RPLAN label -> HG++ id exactly as Housegan-data-reader/read_dd.py.
  * wall adjacency (ed_rm pairs): two rooms share a boundary piece of length >= MIN_SHARED px
    (polygon edges are split at the other rooms' vertices so each piece has one neighbour).
  * front door (type 15): from the Graph2Plan boundary (first two points = front-door segment),
    attached to the room(s) whose boundary overlaps it.
  * interior doors (type 17): SYNTHESISED. Every non-living room gets one door: to the living room
    if they share a wall, otherwise to the neighbour with the longest shared wall (preference
    living > dining > entrance > any). Gives #doors = #rooms-1, as in the bundled RPLAN samples.
    The door is a 2 x min(12, shared-2) px rectangle centred on the shared wall.
Output: <out_dir>/<rplan_name>.json (+ list.txt). Run in the `fpe` env.
"""
import argparse
import json
import os

import numpy as np
import scipy.io as sio
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge

G2P_DIR = "/datawaha/cggroup/datasets/RPLAN/Network/data"
# RPLAN label -> HG++ id (read_dd.py)
RPLAN_TO_HG = {0: 1, 1: 3, 2: 2, 3: 4, 4: 7, 5: 3, 6: 8, 7: 3, 8: 3, 9: 5, 10: 6, 11: 10}
MIN_SHARED = 3.0
DOOR_PREF = {1: 0, 7: 1, 6: 2}


def plan_rooms(p):
    types = np.atleast_1d(p.rType).astype(int).tolist()
    rb = p.rBoundary if len(types) > 1 else [p.rBoundary]
    polys = [np.asarray(x, dtype=float).reshape(-1, 2) for x in rb]
    return types, polys


def valid(types, polys):
    return all(q.size >= 6 and np.all(np.isfinite(q)) and q.min() >= 0 and q.max() <= 256 for q in polys) \
        and any(t == 0 for t in types)


def split_segments(poly, cut_pts):
    """Polygon edges split at any cut point lying on them."""
    segs = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if np.allclose(a, b):
            continue
        d = b - a
        L = np.linalg.norm(d)
        ts = [0.0, 1.0]
        for c in cut_pts:
            t = np.dot(c - a, d) / (L * L)
            if 1e-6 < t < 1 - 1e-6 and np.linalg.norm(a + t * d - c) < 1e-6:
                ts.append(t)
        ts = sorted(set(ts))
        for t0, t1 in zip(ts[:-1], ts[1:]):
            segs.append((a + t0 * d, a + t1 * d))
    return segs


def shared_line(pa, pb):
    inter = Polygon(pa).boundary.intersection(Polygon(pb).boundary)
    lines = [g for g in getattr(inter, "geoms", [inter]) if g.geom_type in ("LineString", "MultiLineString")]
    if not lines:
        return None
    m = linemerge(lines) if len(lines) > 1 else lines[0]
    parts = list(getattr(m, "geoms", [m]))
    return max(parts, key=lambda g: g.length)


def door_rect(line, length=12.0, half=1.0):
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    L = min(length, max(line.length - 2, 2))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if abs(x1 - x0) >= abs(y1 - y0):  # horizontal wall
        return [cx - L / 2, cy - half, cx + L / 2, cy + half]
    return [cx - half, cy - L / 2, cx + half, cy + L / 2]


def rect_edges(b):
    x0, y0, x1, y1 = b
    return [(np.array([x0, y0]), np.array([x0, y1])), (np.array([x0, y1]), np.array([x1, y1])),
            (np.array([x1, y1]), np.array([x1, y0])), (np.array([x1, y0]), np.array([x0, y0]))]


def convert(p):
    rtypes, polys = plan_rooms(p)
    types = [RPLAN_TO_HG.get(t, 16) for t in rtypes]
    n = len(types)
    shared = {}
    for i in range(n):
        for j in range(i + 1, n):
            ln = shared_line(polys[i], polys[j])
            if ln is not None and ln.length >= MIN_SHARED:
                shared[(i, j)] = ln
    # synthesized interior doors
    living = [i for i, t in enumerate(types) if t == 1]
    doors = []  # (rect, [room_a, room_b], type)
    for r in range(n):
        if types[r] == 1:
            continue
        nbs = [(j, shared[tuple(sorted((r, j)))]) for j in range(n) if j != r and tuple(sorted((r, j))) in shared]
        if not nbs:
            continue
        nbs.sort(key=lambda x: (DOOR_PREF.get(types[x[0]], 9), -x[1].length))
        j, ln = nbs[0]
        doors.append((door_rect(ln), [r, j], 17))
    # front door from boundary
    b = np.asarray(p.boundary)
    fd = LineString([b[0, :2], b[1, :2]])
    fd_rooms = [i for i in range(n) if Polygon(polys[i]).boundary.intersection(fd).length > 1]
    if not fd_rooms and living:
        fd_rooms = [min(living, key=lambda i: Polygon(polys[i]).distance(fd))]
    if fd_rooms:
        doors.append((door_rect(fd, length=fd.length + 0, half=1.0), fd_rooms[:1], 15))

    all_types = types + [d[2] for d in doors]
    boxes, edges, ed_rm = [], [], []
    verts = [q for q in polys]
    for k in range(n):
        cut = np.concatenate([verts[j] for j in range(n) if j != k], 0)
        for a, c in split_segments(polys[k], cut):
            mid = LineString([a, c])
            nb = [j for j in range(n) if j != k and (tuple(sorted((k, j))) in shared)
                  and Polygon(polys[j]).boundary.intersection(mid).length > 0.5 * mid.length]
            edges.append([float(a[0]), float(a[1]), float(c[0]), float(c[1]), types[k], types[nb[0]] if nb else 0])
            ed_rm.append([k, nb[0]] if nb else [k])
        q = polys[k]
        boxes.append([float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())])
    for di, (rect, rooms, t) in enumerate(doors):
        d = n + di
        es = rect_edges(rect)
        # attach the two long edges to the rooms on either side (or the single room for front door)
        long_idx = [0, 2] if (rect[3] - rect[1]) > (rect[2] - rect[0]) else [1, 3]
        for ei, (a, c) in enumerate(es):
            rm = None
            if ei in long_idx:
                side = long_idx.index(ei)
                rm = rooms[side] if side < len(rooms) else None
                if t == 15:
                    rm = rooms[0]
            edges.append([float(a[0]), float(a[1]), float(c[0]), float(c[1]), t, types[rm] if rm is not None else 0])
            ed_rm.append([d, rm] if rm is not None else [d])
        boxes.append([float(x) for x in rect])
    return {"room_type": all_types, "boxes": boxes, "edges": edges, "ed_rm": ed_rm,
            "source": "graph2plan_rplan (synthesized interior doors)", "rplan_name": str(p.name)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None, help="file of RPLAN names to keep (default: all valid)")
    ap.add_argument("--n_random", type=int, default=None, help="random subset size (seed 0) if no --ids")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = sio.loadmat(os.path.join(G2P_DIR, f"data_{a.split}.mat"), squeeze_me=True, struct_as_record=False)["data"]
    by_name = {str(p.name): p for p in data}
    if a.ids:
        names = [l.strip() for l in open(a.ids) if l.strip()]
    else:
        names = list(by_name)
        if a.n_random:
            names = sorted(np.random.RandomState(0).choice(names, a.n_random, replace=False).tolist(), key=int)
    os.makedirs(a.out, exist_ok=True)
    ok, skipped = [], []
    for nm in names:
        p = by_name[nm]
        if not valid(*plan_rooms(p)):
            skipped.append(nm)
            continue
        try:
            js = convert(p)
        except Exception as e:  # noqa: BLE001
            skipped.append(nm)
            print("skip", nm, e)
            continue
        path = os.path.join(a.out, f"{nm}.json")
        json.dump(js, open(path, "w"))
        ok.append(os.path.abspath(path))
    open(os.path.join(a.out, "list.txt"), "w").write("\n".join(ok) + "\n")
    json.dump({"skipped": skipped}, open(os.path.join(a.out, "skipped.json"), "w"))
    print(f"wrote {len(ok)} HG++ JSONs, skipped {len(skipped)} -> {a.out}")


if __name__ == "__main__":
    main()
