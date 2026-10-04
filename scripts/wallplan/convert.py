"""Convert WallPlan raw outputs (raw/<name>.pkl from scripts/wallplan/infer.py) to the common format.

WallPlan outputs a wall graph (nodes at 120 px, (row, col)) plus "room circles" (closed node
cycles with a room category), a front door, interior doors and windows. Room polygons are the
room circles (wall centre lines); coordinates are mapped to the 256 px RPLAN frame used by
WallPlan's own pre-processing (p256 = 2 * p120 + 8, i.e. 1/2 of the 512 px render) and
swapped to (x, y). Door/window rectangles use the sizes of WallPlan's floorplan_render().

Variants:
  rplan_boundary_test1000 (--source g2p): Graph2Plan RPLAN test boundaries; boundary = Graph2Plan
      boundary; GT for the same ids -> gt_samples/.
  bundled_boundary_test500 (--source bundled): the 500 boundaries shipped in the repo; boundary =
      outer ring of the input boundary wall graph; no GT (the pkls carry no room labels and their
      file names do NOT correspond to the RPLAN plans with the same id).

    PYTHONPATH=. conda run -n fpe python scripts/wallplan/convert.py outputs/wallplan/<variant> \
        [--g2p_mat data/method_inputs/diffplanner/graph2plan_data/Network/data/data_test.mat]
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from rplan_g2p import (load_g2p, g2p_boundary_xy, g2p_front_door, g2p_condition, gt_sample,  # noqa: E402
                       make_sample, save_sample)
from fpeval.format import canonical_room_type  # noqa: E402

WP_LABELS = ["LivingRoom", "Bedroom", "Kitchen", "Bathroom", "Balcony", "Storage"]


def to256(p):  # (row, col) at 120 px -> [x, y] at 256 px
    return [float(2 * p[1] + 8), float(2 * p[0] + 8)]


def rect(center_rc120, ori, half_along, half_across):
    """Rectangle (256 px, [x, y]) around a 120 px (row, col) centre; ori 0 = horizontal."""
    x, y = to256(center_rc120)
    if ori == 0:
        return [[x - half_along, y - half_across], [x + half_along, y - half_across],
                [x + half_along, y + half_across], [x - half_along, y + half_across]]
    return [[x - half_across, y - half_along], [x + half_across, y - half_along],
            [x + half_across, y + half_along], [x - half_across, y + half_along]]


def bundled_boundary(wall_graph):
    """Outer ring of the input wall graph (256 px frame of the pkl, (row, col) -> [x, y])."""
    segs = []
    for n in wall_graph:
        if n:
            for c in n['connect']:
                if c:
                    a, b = n['pos'], wall_graph[c]['pos']
                    segs.append(LineString([(a[1], a[0]), (b[1], b[0])]))
    polys = list(polygonize(unary_union(segs)))
    outer = unary_union(polys)
    if outer.geom_type != "Polygon":
        outer = max(outer.geoms, key=lambda g: g.area)
    return [[float(x), float(y)] for x, y in list(outer.exterior.coords)[:-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--g2p_mat", default=None, help="set for g2p-input variants (boundary + GT)")
    args = ap.parse_args()
    root = Path(args.variant_dir)
    variant = root.name
    g2p = load_g2p(args.g2p_mat) if args.g2p_mat else None
    n = 0
    for f in sorted((root / "raw").glob("*.pkl"), key=lambda p: int(p.stem)):
        name = f.stem
        res = pickle.load(open(f, "rb"))
        g = res["graph"]
        rooms = []
        for rc in res["room_circles"]:
            if rc is None:
                continue
            cyc = list(rc["circle"])
            if len(cyc) > 1 and cyc[0] == cyc[-1]:
                cyc = cyc[:-1]
            if len(cyc) < 3:
                continue
            native = WP_LABELS[int(rc["category"])]
            rooms.append({"type": canonical_room_type(native), "type_native": native,
                          "polygon": [to256(g[i]['pos']) for i in cyc]})
        doors = []
        for d in res["frontdoor"]:
            doors.append({"type": "front_door", "polygon": rect(d['pos'], d['ori'], 5.0, 1.5)})
        for d in res["doors"]:
            if d['category'] != 4:
                doors.append({"type": "interior_door", "polygon": rect(d['pos'], d['ori'], 5.0, 1.75)})
            else:  # balcony door: 2/3 of the wall segment between the two vertices
                v1, v2 = (np.array(g[d['vertex'][0]]['pos']), np.array(g[d['vertex'][1]]['pos']))
                mid = (v1 + v2) / 2.0
                k = 1 if d['ori'] == 0 else 0
                half = abs(v1[k] - v2[k]) * 2.0 / 3.0  # in 256 px: |d120| * 2 * (2/3) / 2
                doors.append({"type": "interior_door", "polygon": rect(mid, d['ori'], half, 1.75)})
        windows = [{"polygon": rect(w['pos'], w['ori'], 8.5, 1.25)} for w in res["wins"]]
        windows += [{"polygon": rect(w['pos'], w['ori'], 13.5, 1.25)} for w in res["livwins"]]
        if g2p is not None:
            d = g2p[name]
            boundary = g2p_boundary_xy(d.boundary)
            cond = g2p_condition(name)
        else:
            boundary = bundled_boundary(res["input"]["wall_graph"])
            cond = {"type": "boundary", "ref_id": f"wallplan_bundled_{name}",
                    "dataset": "WallPlan repo test/input/*.pkl"}
        s = make_sample("wallplan", variant, name, rooms, units="px", boundary=boundary, doors=doors,
                        windows=windows, condition=cond)
        save_sample(s, root / "samples" / f"{name}.json")
        if g2p is not None:
            save_sample(gt_sample(g2p[name], "rplan_gt", variant), root / "gt_samples" / f"{name}.json")
        n += 1
    print(f"converted {n} samples -> {root / 'samples'}")


if __name__ == "__main__":
    main()
