"""Convert House-GAN raw outputs (scripts/housegan/infer.py .npz files) to the common format.

Generated rooms: each 32x32 generator mask is binarized (>0), upsampled to 256x256 exactly as in the
repo's draw_masks (cv2.INTER_AREA, threshold 127) and vectorized with cv2.findContours (external
contours, one polygon per connected component, components < 64 px^2 = one mask cell dropped).
Coordinates: 256 px frame, x = mask column, y = mask row (the convention of the repo's mask_to_bb /
bb_to_im_fid visualizations used in the paper figures).

GT rooms: the input floorplan's room boxes (House-GAN data has only boxes), same centering as the
dataset loader; dataset boxes index masks as mask[x0:x1, y0:y1], so in image convention the polygon is
x in [y0, y1], y in [x0, x1] (scaled by 256).

Rooms are ordered by decreasing area (as bb_to_im_fid draws them) so small rooms render on top; graph
edges (bubble-diagram adjacency, the conditioning input) are remapped accordingly.

Usage (fpe env, repo root on PYTHONPATH):
  python scripts/housegan/convert.py outputs/housegan/lifull_bubble_testD [--gt]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from fpeval.format import canonical_room_type, make_sample, save_sample

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6,
              "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
ID_TO_NAME = {v: k for k, v in ROOM_CLASS.items()}
EXTRA = {"missing": "other", "laundry_room": "laundry"}
MIN_AREA = 64.0


def mask_polygons(mask):
    m = (mask.astype(np.float32) > 0).astype(np.uint8) * 255
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(m, 127, 255, 0)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        pts = c[:, 0, :].astype(float)  # (col, row) -> (x, y)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def build(rooms_per_node, nodes, edges, sid, variant, source, cond):
    """rooms_per_node: list (per input node) of list of polygons (np arrays)."""
    items = []  # (area, node_idx, poly)
    for k, polys in enumerate(rooms_per_node):
        for p in polys:
            items.append((poly_area(p), k, p))
    items.sort(key=lambda t: -t[0])
    rooms, node_to_rooms = [], {}
    for _, k, p in items:
        native = ID_TO_NAME[int(nodes[k])]
        node_to_rooms.setdefault(k, []).append(len(rooms))
        rooms.append({"type": canonical_room_type(native, EXTRA), "type_native": native,
                      "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in p]})
    gedges = []
    for k, w, l in edges:
        if w > 0 and k in node_to_rooms and l in node_to_rooms:
            gedges.append([node_to_rooms[k][0], node_to_rooms[l][0], "adjacent"])
    return make_sample(source, variant, sid, rooms, units="px", graph={"edges": gedges}, condition=cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--gt", action="store_true", help="also write gt_samples/ from the input boxes")
    args = ap.parse_args()
    root = Path(args.variant_dir)
    variant = args.variant or root.name
    n = n_missing_rooms = 0
    gt_done = set()
    for f in sorted((root / "raw").glob("*.npz")):
        d = np.load(f)
        gid, v = f.stem.split("_v")  # raw files are <graph>_v<k>
        sid = gid if v == "0" else f.stem  # first variation keeps the GT id
        nodes, edges = d["nodes"], d["edges"]
        cond = {"type": "bubble_diagram", "ref_id": f"housegan_lifull_testD_{gid}"}
        per_node = [mask_polygons(m) for m in d["masks"]]
        n_missing_rooms += sum(1 for p in per_node if not p)
        save_sample(build(per_node, nodes, edges, sid, variant, "housegan", cond), root / "samples" / f"{sid}.json")
        if args.gt and gid not in gt_done:
            per_node_gt = []
            for x0, y0, x1, y1 in d["gt_bbs"] * 256.0:
                per_node_gt.append([np.array([[y0, x0], [y1, x0], [y1, x1], [y0, x1]], dtype=float)])
            save_sample(build(per_node_gt, nodes, edges, gid, variant, "housegan_lifull_gt", cond),
                        root / "gt_samples" / f"{gid}.json")
            gt_done.add(gid)
        n += 1
    print(f"converted {n} samples; {n_missing_rooms} input rooms produced an empty mask; {len(gt_done)} GT samples")


if __name__ == "__main__":
    main()
