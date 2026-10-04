"""Build DS2D test inputs (run in the `fpe` env).

ProcTHOR: HF ludolara/DStruct2Design `validation.json` (the official DS2D test split; their
run_generation_procthor.py notes "test set was used for validation, just naming difference").
Order = np.random.seed(12345) permutation as in run_generation_procthor.py.

RPLAN: DS2D's RPLAN conversion needs the raw RPLAN PNGs (not available here). We rebuild the
DS2D RPLAN record format (rooms=[y0,x0,y1,x1,c1,c2,area,h,w], polygons, edges) from the
Graph2Plan RPLAN test split (data_test.mat: rType + rBoundary room polygons, 256px grid).
DS2D RPLAN variants are room-count hold-outs (model NR never saw N-room plans), so the
test input for model NR is the set of N-room plans.

Output: data/method_inputs/ds2d/{procthor,rplan}/test_inputs*.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DA = ROOT / "data" / "method_inputs" / "ds2d"
G2P_TEST = Path("/datawaha/cggroup/datasets/RPLAN/Network/data/data_test.mat")

ROOM_LABEL = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom", 4: "DiningRoom",
              5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom", 8: "GuestRoom", 9: "Balcony",
              10: "Entrance", 11: "Storage", 12: "Wall-in", 13: "External", 14: "ExteriorWall",
              15: "FrontDoor", 16: "InteriorWall", 17: "InteriorDoor"}
PIXEL2LEN = 18 / 256
PIXEL2AREA = PIXEL2LEN ** 2


# --- copied from external/methods/ds2d/rplan_dataset_convert.py (edge extraction) ---
def collide2d(b1, b2, th=0):
    return not ((b1[0] - th > b2[2]) or (b1[2] + th < b2[0]) or (b1[1] - th > b2[3]) or (b1[3] + th < b2[1]))


def point_box_relation(u, vbox):
    uy, ux = u
    vy0, vx0, vy1, vx1 = vbox
    if (ux < vx0 and uy <= vy0) or (ux == vx0 and uy == vy0): return 0
    elif (vx0 <= ux < vx1 and uy <= vy0): return 3
    elif (vx1 <= ux and uy < vy0) or (ux == vx1 and uy == vy0): return 8
    elif (vx1 <= ux and vy0 <= uy < vy1): return 7
    elif (vx1 < ux and vy1 <= uy) or (ux == vx1 and uy == vy1): return 9
    elif (vx0 < ux <= vx1 and vy1 <= uy): return 6
    elif (ux <= vx0 and vy1 < uy) or (ux == vx0 and uy == vy1): return 1
    elif (ux <= vx0 and vy0 < uy <= vy1): return 2
    elif (vx0 < ux < vx1 and vy0 < uy < vy1): return 4
    return 4


def get_edges(rooms, th=9):
    edges = []
    for u in range(len(rooms)):
        for v in range(u + 1, len(rooms)):
            if not collide2d(rooms[u][:4], rooms[v][:4], th=th):
                continue
            uy0, ux0, uy1, ux1 = rooms[u][:4]
            vy0, vx0, vy1, vx1 = rooms[v][:4]
            uc = (uy0 + uy1) / 2, (ux0 + ux1) / 2
            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                rel = 5
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                rel = 4
            else:
                rel = point_box_relation(uc, rooms[v][:4])
            edges.append([u, v, rel])
    return edges


def shoelace(poly):
    a = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        a += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(a) / 2.0


def rplan_record(plan):
    """Graph2Plan plan -> DS2D raw record + DS2D 'new'-format JSON (data2json)."""
    rtypes = np.atleast_1d(plan.rType).astype(int).tolist()
    rb = plan.rBoundary
    if len(rtypes) == 1:
        rb = [rb]
    rooms, polygons = [], []
    for k, (c1, poly) in enumerate(zip(rtypes, rb)):
        poly = np.asarray(poly, dtype=int).reshape(-1, 2)  # Graph2Plan: (x=col, y=row)
        verts = [(int(r), int(c)) for c, r in poly]        # DS2D: vertex = (row, col)
        rows = [v[0] for v in verts]; cols = [v[1] for v in verts]
        area = shoelace(verts)
        width = max(rows) - min(rows)   # DS2D naming: min_x/max_x over vertex[0] -> "width"
        height = max(cols) - min(cols)
        rooms.append([min(rows), min(cols), max(rows), max(cols), c1, k + 1, area, height, width])
        polygons.append(verts)
    edges = get_edges(rooms)
    # data2json ('new' format)
    total = sum(r[6] for r in rooms)
    js = {"room_count": len(rooms), "total_area": float(f"{total * PIXEL2AREA:.2f}"),
          "room_types": [ROOM_LABEL[r[4]] for r in rooms], "rooms": []}
    for i, r in enumerate(rooms):
        js["rooms"].append({"area": float(f"{r[6] * PIXEL2AREA:.2f}"), "room_type": ROOM_LABEL[r[4]],
                            "floor_polygon": [{"x": x, "z": z} for x, z in polygons[i]],
                            "height": float(f"{r[7] * PIXEL2LEN:.2f}"), "width": float(f"{r[8] * PIXEL2LEN:.2f}"),
                            "id": f"room|{i}"})
    return {"rooms_raw": rooms, "edges": edges, "gt": js}


def prep_rplan(n_per, out):
    import scipy.io as sio
    d = sio.loadmat(str(G2P_TEST), squeeze_me=True, struct_as_record=False)["data"]
    by_n = {5: [], 6: [], 7: [], 8: []}
    for i, p in enumerate(d):
        n = len(np.atleast_1d(p.rType))
        rb = p.rBoundary if n > 1 else [p.rBoundary]
        # skip corrupted plans: 34 have an empty room polygon, a few have non-finite coordinates
        if any(np.asarray(x).size < 6 or not np.all(np.isfinite(np.asarray(x, dtype=float)))
               or np.asarray(x, dtype=float).min() < 0 or np.asarray(x, dtype=float).max() > 256 for x in rb):
            continue
        if n in by_n:
            by_n[n].append(i)
    for n, idxs in by_n.items():
        np.random.seed(12345)
        perm = np.random.permutation(len(idxs))[:n_per]
        path = out / f"test_inputs_{n}R.jsonl"
        with open(path, "w") as f:
            for j in perm:
                i = idxs[j]
                rec = rplan_record(d[i])
                rec.update({"id": f"{i:05d}", "g2p_index": int(i), "rplan_name": str(d[i].name)})
                f.write(json.dumps(rec) + "\n")
        print(f"rplan {n}R: {len(idxs)} plans with {n} rooms in G2P test split, wrote {len(perm)} -> {path}")


def procthor_full_spec(s):
    """Assumed 'prompt' field of DS2D's (unreleased) ProcTHOR datasets: all room attributes
    except floor_polygon, plus room_count/total_area/room_types."""
    return {"room_count": s["room_count"], "total_area": s["total_area"], "room_types": s["room_types"],
            "rooms": [{k: v for k, v in r.items() if k != "floor_polygon"} for r in s["rooms"]]}


def prep_procthor(n, out):
    data = [json.loads(l) for l in open(out / "validation.json") if l.strip()]
    np.random.seed(12345)
    perm = np.random.permutation(len(data))[:n]
    path = out / "test_inputs.jsonl"
    with open(path, "w") as f:
        for i in perm:
            s = dict(data[i]); s["prompt"] = procthor_full_spec(s)
            f.write(json.dumps({"id": f"{int(i):04d}", "split_index": int(i), "sample": s}) + "\n")
    print(f"procthor: {len(data)} in validation.json (=DS2D test), wrote {len(perm)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rplan_n", type=int, default=500)
    ap.add_argument("--procthor_n", type=int, default=1000)
    a = ap.parse_args()
    (DA / "rplan").mkdir(parents=True, exist_ok=True)
    prep_rplan(a.rplan_n, DA / "rplan")
    prep_procthor(a.procthor_n, DA / "procthor")
