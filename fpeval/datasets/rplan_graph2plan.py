"""Convert RPLAN in Graph2Plan format (data_{train,valid,test}.mat) to the common format.

Source: local copy /datawaha/cggroup/datasets/RPLAN/Network/data/ (read-only; RPLAN itself
is obtained via the RPLAN Google Form, Graph2Plan's processed .mat files via its README).
data/rplan_graph2plan/raw is a symlink to that directory.

Per plan: name (RPLAN id), boundary [x, y, dir, isNew] (first two points = front door),
rType (RPLAN category), rBoundary (room polygons, x/y in the 256x256 RPLAN frame), rEdge
[u, v, relation] (Graph2Plan's spatial-relation graph; relation 0-9 = direction class).
Room polygons are RPLAN's wall-centred room regions (rooms share edges, no wall gaps).

Units: px of the 256 frame; RPLAN's frame spans 18 m (as used by DS2D), so
scale_m_per_unit = 18/256. Graph edges are labelled "adjacent" (rEdge has no door info);
the Graph2Plan relation id is kept in condition.relations. Plans with empty / non-finite /
out-of-frame room polygons (a few dozen) are skipped.
Splits: Graph2Plan train / valid / test (written as train / val / test).

    python -m fpeval.datasets.rplan_graph2plan [--limit N]
"""
import argparse
from pathlib import Path

import numpy as np

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, polygons_of, seg_rect, write_splits

RAW = DATA / "rplan_graph2plan" / "raw"
ROOM_LABEL = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom",
              "StudyRoom", "SecondRoom", "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in"]
EXTRA = {"Wall-in": "closet"}
SPLITS = {"train": "data_train.mat", "val": "data_valid.mat", "test": "data_test.mat"}


def convert(p, split, root):
    from shapely.geometry import Polygon
    types = np.atleast_1d(p.rType).astype(int).tolist()
    rb = p.rBoundary if len(types) > 1 else [p.rBoundary]
    rooms = []
    for t, poly in zip(types, rb):
        a = np.asarray(poly, dtype=float).reshape(-1, 2) if np.asarray(poly).size else np.zeros((0, 2))
        if len(a) < 3 or not np.all(np.isfinite(a)) or a.min() < 0 or a.max() > 256:
            return None
        rings = polygons_of(Polygon(a).buffer(0), min_area=1.0)
        if not rings:
            return None
        native = ROOM_LABEL[t] if 0 <= t < len(ROOM_LABEL) else f"class_{t}"
        rooms.append({"type": canonical_room_type(native, EXTRA), "type_native": native,
                      "polygon": rings[0]})
    b = np.asarray(p.boundary)
    boundary = b[b[:, 3] == 0, :2].astype(float).tolist()
    door = seg_rect(tuple(b[0, :2]), tuple(b[1, :2]), 1.5)
    edges = np.atleast_2d(np.asarray(p.rEdge)) if np.asarray(p.rEdge).size else np.zeros((0, 3), int)
    gedges = sorted({(int(min(u, v)), int(max(u, v)), "adjacent") for u, v, _ in edges if u != v})
    sid = f"{split}_{str(p.name)}"
    sample = make_sample(
        "rplan_graph2plan", "graph2plan_mat", sid, rooms, units="px", scale_m_per_unit=18 / 256,
        boundary=boundary, doors=[{"type": "front_door", "polygon": r} for r in door],
        graph={"edges": [list(e) for e in gedges]},
        condition={"split": split, "rplan_name": str(p.name),
                   "relations": [[int(u), int(v), int(r)] for u, v, r in edges]})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return sid


def main():
    import scipy.io as sio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = out_dirs("rplan_graph2plan", args.out)
    splits, skipped = {}, {}
    for split, fn in SPLITS.items():
        d = sio.loadmat(str(Path(args.raw) / fn), squeeze_me=True, struct_as_record=False)["data"]
        d = d[: args.limit] if args.limit else d
        for p in d:
            sid = convert(p, split, root)
            if sid is None:
                skipped[split] = skipped.get(split, 0) + 1
            else:
                splits[sid] = split
    counts = write_splits(root, splits, {"note": "Graph2Plan data_train/valid/test.mat splits", "skipped_corrupt": skipped})
    print("converted", len(splits), "skipped", skipped, counts)


if __name__ == "__main__":
    main()
