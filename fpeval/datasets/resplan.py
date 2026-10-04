"""Convert ResPlan (17,000 residential plans, CC BY 4.0) to the common format.

Raw data: data/resplan/raw/{ResPlan.pkl,split.json} (scripts/datasets/download_resplan.sh).
Official loader: external/datasets/resplan/resplan_utils.py (used for the room graph).

Geometry in the pickle is in a normalized frame (plan width 256 units). The metric
scale per plan is sqrt(area_m2 / inner.area) -- with it, 99.3% of wall depths land
in 10-40 cm, matching the ResPlan README. We write coordinates in metres (units "m").

Rooms: living (all parts unioned, as in resplan_utils.plan_to_graph), kitchen,
bedroom, bathroom, balcony, storage, stair (one room per polygon part), plus garden,
parking and pool (outside spaces; not graph nodes).
Graph: resplan_utils.plan_to_graph + add_adjacency_edges (the paper/Kaggle definition);
edge labels door / adjacent / window. Front-door links are dropped (not a room).
Splits: split.json (train / val / test / augmented).

    python -m fpeval.datasets.resplan [--limit N] [--workers 16] [--no-graph]
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, REPO, out_dirs, polygons_of, write_splits

RAW = DATA / "resplan" / "raw"
sys.path.insert(0, str(REPO / "external" / "datasets" / "resplan"))

ROOM_KEYS = ["living", "kitchen", "bedroom", "bathroom", "balcony", "storage", "stair",
             "garden", "parking", "pool"]
EXTRA = {"stair": "other", "garden": "outdoor", "parking": "garage", "pool": "outdoor"}
EDGE_LABEL = {"via_door": "door", "adjacency": "adjacent", "via_window": "window"}
_STATE = {}


def _parts(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


def convert_one(plan):
    import resplan_utils as ru
    root, split_of, with_graph = _STATE["root"], _STATE["split_of"], _STATE["graph"]
    plan = ru.normalize_keys(dict(plan))
    pid = int(plan["id"])
    inner = plan.get("inner")
    s = (float(plan["area"]) / inner.area) ** 0.5 if inner is not None and inner.area > 0 else None
    if not s:
        return pid, None
    tf = lambda g: affinity.scale(g, xfact=s, yfact=s, origin=(0, 0))
    rooms, node_of = [], {}
    for key in ROOM_KEYS:
        parts = _parts(plan.get(key))
        if key == "living" and parts:
            parts = [unary_union(parts)]
        for i, g in enumerate(parts):
            node = f"{key}_{i}"
            for ring in polygons_of(tf(g), min_area=1e-3):
                node_of.setdefault(node, len(rooms))
                rooms.append({"type": canonical_room_type(key, EXTRA), "type_native": key,
                              "polygon": ring})
    if not rooms:
        return pid, None
    doors = [{"type": "interior_door", "polygon": r} for g in _parts(plan.get("door"))
             for r in polygons_of(tf(g))]
    doors += [{"type": "front_door", "polygon": r} for g in _parts(plan.get("front_door"))
              for r in polygons_of(tf(g))]
    windows = [{"polygon": r} for g in _parts(plan.get("window")) for r in polygons_of(tf(g))]
    walls = [{"polygon": r} for g in _parts(plan.get("wall")) for r in polygons_of(tf(g))]
    bnd = polygons_of(tf(inner)) if inner is not None else []
    graph = None
    if with_graph:
        G = ru.add_adjacency_edges(ru.plan_to_graph(plan))
        edges = set()
        for u, v, d in G.edges(data=True):
            lab = EDGE_LABEL.get(d.get("type"))
            if lab and u in node_of and v in node_of and node_of[u] != node_of[v]:
                a, b = sorted((node_of[u], node_of[v]))
                edges.add((a, b, lab))
        graph = {"edges": [list(e) for e in sorted(edges)]}
    split = split_of.get(pid, "unlisted")
    sid = f"{pid:05d}"
    sample = make_sample(
        "resplan", "github_release", sid, rooms, units="m", scale_m_per_unit=1.0,
        boundary=max(bnd, key=lambda r: Polygon(r).area) if bnd else None,
        doors=doors, windows=windows, walls=walls, graph=graph,
        condition={"split": split, "area_m2": float(plan["area"]),
                   "native_m_per_unit": s})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return pid, split


def _init(root, split_of, graph):
    _STATE.update(root=root, split_of=split_of, graph=graph)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    with open(f"{args.raw}/ResPlan.pkl", "rb") as f:
        data = pickle.load(f)
    split_of = {}
    for sp, ids in json.load(open(f"{args.raw}/split.json")).items():
        for i in ids:
            split_of[int(i)] = sp
    data = data[: args.limit] if args.limit else data
    root = out_dirs("resplan", args.out)
    splits, failed = {}, 0
    with Pool(args.workers, initializer=_init, initargs=(root, split_of, not args.no_graph)) as pool:
        for pid, sp in pool.imap_unordered(convert_one, data, chunksize=32):
            if sp is None:
                failed += 1
            else:
                splits[f"{pid:05d}"] = sp
    counts = write_splits(root, splits, {"note": "from ResPlan split.json; 'augmented' = geometric augmentations of other plans"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
