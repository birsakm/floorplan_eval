"""Build GSDiff conditioning inputs (bubble diagrams, boundary images) + GT samples from public RPLAN-derived data.

GSDiff's own inputs come from raw RPLAN PNGs (Google-Form gated) via datasets/rplan-process*.py. Instead we
use DiffPlanner's public release (dataset.zip -> dataset_json/data_test.json), which is the Graph2Plan
RPLAN test split (12002 plans) with room polygons that tile the interior, 6 room classes and adjacencies,
all in the RPLAN 256 px frame (same frame as GSDiff's normalized coords * 128 + 128).

Per plan we write <out>/<name>.npz with
  bb_semantics  (n,) int   GSDiff 7-class ids of the rooms (bubble-diagram nodes)
  bb_adjacency  (n,n) uint8 rooms sharing a wall segment (GSDiff rplan-process8 definition, recomputed
                            from the polygons; shared length > 1 px)
  boundary      (k,2) int   boundary corners (Graph2Plan boundary minus the 2 front-door points)
  boundary_img  (256,256,3) uint8 boundary image drawn exactly as rplang_edge_semantics_simplified_78_10_prerunCNN
and GT samples in the common format to <gt_dir>/<name>.json.

Run in the fpe env with repo root on PYTHONPATH.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from fpeval.format import make_sample, save_sample

# DiffPlanner category -> GSDiff simplified class
DP_TO_GS = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 2}
DP_NAMES = {0: ("living_room", "LivingRoom/DiningRoom/Entrance/Wall-in"), 1: ("bedroom", "Master/Child/Study/Second/GuestRoom"),
            2: ("kitchen", "Kitchen"), 3: ("bathroom", "Bathroom"), 4: ("balcony", "Balcony"), 5: ("storage", "Storage")}


def boundary_image(corners):
    img = Image.new("RGB", (256, 256), "white")
    dr = ImageDraw.Draw(img)
    n = len(corners)
    for w in (7, 5, 3, 1):
        for i in range(n):
            dr.line([tuple(corners[i]), tuple(corners[(i + 1) % n])], fill="black", width=w)
    for p in corners:
        dr.rectangle([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill="black")
    return np.array(img)


def adjacency(polys, min_len=1.0):
    n = len(polys)
    a = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            inter = polys[i].boundary.intersection(polys[j].boundary)
            if inter.length > min_len:
                a[i, j] = a[j, i] = 1
    return a


def door_rect(b):
    (x0, y0), (x1, y1) = b[0][:2], b[1][:2]
    w = 2
    if y0 == y1:
        return [[min(x0, x1), y0 - w], [max(x0, x1), y0 - w], [max(x0, x1), y0 + w], [min(x0, x1), y0 + w]]
    return [[x0 - w, min(y0, y1)], [x0 + w, min(y0, y1)], [x0 + w, max(y0, y1)], [x0 - w, max(y0, y1)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="DiffPlanner dataset_json/data_test.json")
    ap.add_argument("--ids", required=True, help="text file with RPLAN names to use")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--gt_variant", default="rplan_gt")
    args = ap.parse_args()
    data = {d["name"]: d for d in json.load(open(args.json))}
    ids = [l.strip() for l in open(args.ids) if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"n": 0, "rooms_hist": {}, "adj_agree_with_rEdge": 0, "adj_pairs": 0}
    for name in ids:
        d = data[name]
        rooms = d["rooms"]
        polys = [Polygon(r["r_boundary"]).buffer(0) for r in rooms]
        adj = adjacency(polys)
        sem = np.array([DP_TO_GS[r["category"]] for r in rooms], dtype=np.int64)
        b = d["boundary"]
        corners = [[int(p[0]), int(p[1])] for p in b if int(p[3]) == 0]
        np.savez_compressed(out / f"{name}.npz", bb_semantics=sem, bb_adjacency=adj,
                            boundary=np.array(corners, dtype=np.int32), boundary_img=boundary_image(corners))
        # sanity vs. Graph2Plan rEdge adjacency
        ref = {tuple(sorted(e)) for e in d.get("adjacencies", [])}
        mine = {(i, j) for i in range(len(rooms)) for j in range(i + 1, len(rooms)) if adj[i, j]}
        stats["adj_pairs"] += len(ref | mine)
        stats["adj_agree_with_rEdge"] += len(ref & mine)
        stats["n"] += 1
        k = str(len(rooms))
        stats["rooms_hist"][k] = stats["rooms_hist"].get(k, 0) + 1
        if args.gt_dir:
            gt_rooms = [{"type": DP_NAMES[r["category"]][0], "type_native": DP_NAMES[r["category"]][1],
                         "polygon": [[int(x), int(y)] for x, y in r["r_boundary"]]} for r in rooms]
            edges = [[int(i), int(j), "adjacent"] for i, j in sorted(mine)]
            s = make_sample("rplan_diffplanner_json", args.gt_variant, name, gt_rooms, units="px",
                            boundary=[[int(p[0]), int(p[1])] for p in b],
                            doors=[{"type": "front_door", "polygon": door_rect(b)}], graph={"edges": edges},
                            condition={"type": "ground_truth", "ref_id": f"rplan_{name}"})
            save_sample(s, Path(args.gt_dir) / f"{name}.json")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
