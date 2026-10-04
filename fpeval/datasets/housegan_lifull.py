"""Convert the House-GAN vectorized LIFULL data (housegan_clean_data.npy) to the common format.

Raw data: data/housegan_lifull/raw/ (scripts/datasets/download_housegan_lifull.sh).
Each entry: [room_types, room_bboxes, edges[x0,y0,x1,y1,*,*], edge->rooms, door edge ids, rgb_name].
Room polygons are rebuilt by polygonizing the edges assigned to each room (fallback: bbox).
Coordinates are in the 256x256 pixel frame of the raster-to-vector output (no metric scale).

Splits: the clean data has no official split (House-GAN evaluates by holding out plans
by room count). dataset_paper/{train,valid}_data.npy hold boxes from a different
processing run and do not match the clean data, so every plan gets split "all".

    python -m fpeval.datasets.housegan_lifull [--limit N] [--workers 16]
"""
import argparse
from multiprocessing import Pool

import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import polygonize, unary_union

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, polygons_of, seg_rect, write_splits

RAW = DATA / "housegan_lifull" / "raw"
ROOM_CLASS = {1: "living_room", 2: "kitchen", 3: "bedroom", 4: "bathroom", 5: "missing",
              6: "closet", 7: "balcony", 8: "corridor", 9: "dining_room", 10: "laundry_room"}
EXTRA = {"missing": "other", "laundry_room": "laundry"}


def convert_one(args):
    idx, entry, split, root = args
    types, boxes, edges, e2r, doors, rgb = entry
    rooms, kept = [], []
    for r, (t, bb) in enumerate(zip(types, boxes)):
        segs = [LineString([(e[0], e[1]), (e[2], e[3])]) for e, rr in zip(edges, e2r)
                if r in rr and (e[0], e[1]) != (e[2], e[3])]
        geom = None
        if segs:
            polys = list(polygonize(unary_union(segs)))
            if polys:
                geom = unary_union(polys)
        bbox = box(*[float(v) for v in bb])
        # sanity: polygon should roughly fill its bbox
        if geom is None or geom.area < 0.5 * bbox.area:
            geom = bbox
        native = ROOM_CLASS.get(int(t), f"class_{int(t)}")
        for ring in polygons_of(geom, min_area=1.0):
            rooms.append({"type": canonical_room_type(native, EXTRA), "type_native": native,
                          "polygon": ring})
            kept.append(r)
    if not rooms:
        return idx, None
    first = {}
    for k, r in enumerate(kept):
        first.setdefault(r, k)
    door_set = set(int(d) for d in doors)
    gedges = set()
    door_polys = []
    for ei, (e, rr) in enumerate(zip(edges, e2r)):
        if ei in door_set:
            door_polys += [{"type": "interior_door" if len(rr) == 2 else "front_door",
                            "polygon": p} for p in seg_rect((e[0], e[1]), (e[2], e[3]), 1.0)]
        if len(rr) == 2 and rr[0] in first and rr[1] in first and rr[0] != rr[1]:
            a, b = sorted((first[rr[0]], first[rr[1]]))
            gedges.add((a, b, "door" if ei in door_set else "adjacent"))
    # if a pair has a door, drop its "adjacent" duplicate
    dp = {(a, b) for a, b, k in gedges if k == "door"}
    gedges = sorted(e for e in gedges if e[2] == "door" or (e[0], e[1]) not in dp)
    footprint = unary_union([box(*[float(v) for v in bb]) for bb in boxes])
    outer = [LineString([(e[0], e[1]), (e[2], e[3])]) for e, rr in zip(edges, e2r) if len(rr) == 1]
    fp = list(polygonize(unary_union(outer))) if outer else []
    bnd = polygons_of(unary_union(fp)) if fp else polygons_of(footprint.convex_hull)
    sid = f"{idx:06d}"
    sample = make_sample(
        "housegan_lifull", "clean", sid, rooms, units="px",
        boundary=max(bnd, key=lambda p: len(p)) if len(bnd) == 1 else None,
        doors=door_polys, graph={"edges": [list(e) for e in gedges]},
        condition={"split": split, "rgb_name": str(rgb), "frame": "256x256 px"})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return idx, split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    data = np.load(f"{args.raw}/housegan_clean_data.npy", allow_pickle=True)
    root = out_dirs("housegan_lifull", args.out)
    n = len(data) if args.limit is None else min(args.limit, len(data))
    jobs = ((i, data[i], "all", root) for i in range(n))
    splits, failed = {}, 0
    with Pool(args.workers) as pool:
        for idx, sp in pool.imap_unordered(convert_one, jobs, chunksize=256):
            if sp is None:
                failed += 1
            else:
                splits[f"{idx:06d}"] = sp
    counts = write_splits(root, splits, {"note": "no official split for housegan_clean_data.npy (House-GAN holds out by room count)"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
