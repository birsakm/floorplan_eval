"""Convert DiffPlanner outputs (raw/b_post.json = sampled + post-processed plans) to the common format.

Rooms use the aligned room polygons ("r_boundary_aligned") produced by DiffPlanner's
post_processing.py; rooms whose polygon came out empty are dropped (counted in
raw/convert_stats.json). Doors/windows from decorate.py are segments
[room, x, y, dx, dy, dir]; they are stored as thin rectangles (half width 1.5 px, matching the
official 512 px visualization). Coordinates are DiffPlanner's 256 px RPLAN frame, i.e. the
same frame as the Graph2Plan boundary. Ground truth for the same ids goes to gt_samples/.

    PYTHONPATH=. conda run -n fpe python scripts/diffplanner/convert.py outputs/diffplanner/<variant> \
        --g2p_mat data/method_inputs/diffplanner/graph2plan_data/Network/data/data_test.mat
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from rplan_g2p import (load_g2p, g2p_boundary_xy, g2p_front_door, g2p_condition, gt_sample,  # noqa: E402
                       seg_rect, make_sample, save_sample)
from fpeval.format import canonical_room_type  # noqa: E402

DP_LABELS = ["LivingRoom", "Bedroom", "Kitchen", "Bathroom", "Balcony", "Storage"]


def seg_poly(seg, hw=1.5):
    _, x, y, dx, dy = seg[:5]
    return seg_rect([x, y], [x + dx, y + dy], hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--g2p_mat", required=True)
    ap.add_argument("--raw", default="raw/b_post.json")
    ap.add_argument("--no_gt", action="store_true")
    args = ap.parse_args()
    root = Path(args.variant_dir)
    variant = root.name
    g2p = load_g2p(args.g2p_mat)
    data = json.load(open(root / args.raw, encoding="utf-8"))
    stats = {"plans": 0, "rooms": 0, "rooms_dropped_empty_polygon": 0, "plans_with_dropped_rooms": []}
    for d in data:
        name = d["name"]
        rooms, idx_map = [], {}
        for i, r in enumerate(d["rooms"]):
            poly = r.get("r_boundary_aligned") or []
            if len(poly) and poly[0] == poly[-1]:
                poly = poly[:-1]
            if len(poly) < 3:
                stats["rooms_dropped_empty_polygon"] += 1
                if name not in stats["plans_with_dropped_rooms"]:
                    stats["plans_with_dropped_rooms"].append(name)
                continue
            native = DP_LABELS[int(r["category"])]
            idx_map[i] = len(rooms)
            rooms.append({"type": canonical_room_type(native), "type_native": native,
                          "polygon": [[float(x), float(y)] for x, y in poly]})
        edges = [[idx_map[a], idx_map[b], "adjacent"] for a, b in d.get("adjacencies_aligned", [])
                 if a in idx_map and b in idx_map]
        doors = [g2p_front_door(d["boundary"])]
        doors += [{"type": "interior_door", "polygon": seg_poly(s)} for s in d.get("doors", [])]
        windows = [{"polygon": seg_poly(s)} for s in d.get("windows", [])]
        s = make_sample("diffplanner", variant, name, rooms, units="px",
                        boundary=[[int(p[0]), int(p[1])] for p in d["boundary"]], doors=doors, windows=windows,
                        graph={"edges": edges}, condition=g2p_condition(name))
        save_sample(s, root / "samples" / f"{name}.json")
        if not args.no_gt:
            save_sample(gt_sample(g2p[name], "rplan_gt", variant), root / "gt_samples" / f"{name}.json")
        stats["plans"] += 1
        stats["rooms"] += len(rooms)
    json.dump(stats, open(root / "raw" / "convert_stats.json", "w"), indent=1)
    print({k: v if not isinstance(v, list) else len(v) for k, v in stats.items()})


if __name__ == "__main__":
    main()
