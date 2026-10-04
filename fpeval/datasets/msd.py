"""Convert Modified Swiss Dwellings (MSD v1, 4TU.ResearchData, CC BY 4.0) to the common format.

Raw data: data/msd/raw/{graph_out,full_out,graph_in,struct_in}/ (scripts/datasets/download_msd.sh).
MSD samples are whole building floors (often several apartments), derived from Swiss Dwellings.

Only the train release (ids 0..4166) has ground-truth outputs (graph_out/*.pickle with room
polygons in metres + access graph); the test release (ids 4167..5556) only has inputs
(struct_in, graph_in) because it was a hidden challenge set, so it is not converted.
Coordinates are kept as stored (they match the y-down layout of full_out/*.npy).

The pickles are networkx graphs whose `centroid` attributes are torch tensors; we unpickle
with torch stubbed out (centroids are not needed), so torch is not required.
Room classes (msd constants.ROOM_NAMES): 0 Bedroom, 1 Livingroom, 2 Kitchen, 3 Dining,
4 Corridor, 5 Stairs, 6 Storeroom, 7 Bathroom, 8 Balcony, 9 Structure.
Structure nodes (if any) go to `walls`. Edge labels: door / entrance / passage.

    python -m fpeval.datasets.msd [--limit N] [--workers 16]
"""
import argparse
import ast
import pickle
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import Polygon

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, polygons_of, write_splits

RAW = DATA / "msd" / "raw"
ROOM_NAMES = ['Bedroom', 'Livingroom', 'Kitchen', 'Dining', 'Corridor', 'Stairs', 'Storeroom',
              'Bathroom', 'Balcony', 'Structure', 'Door', 'Entrance Door', 'Window']
EXTRA = {"Livingroom": "living_room", "Dining": "dining_room", "Stairs": "other",
         "Storeroom": "storage"}
EDGE_LABEL = {"door": "door", "entrance": "entrance", "passage": "passage"}


class _NoTorch(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch"):
            return lambda *a, **k: None
        return super().find_class(module, name)


def load_graph(path):
    with open(path, "rb") as f:
        return _NoTorch(f).load()


def _poly(geom):
    if isinstance(geom, str):
        geom = ast.literal_eval(geom)
    if hasattr(geom, "exterior"):
        return geom
    return Polygon(geom)


def convert_one(args):
    path, root = args
    G = load_graph(path)
    rooms, walls, node_of = [], [], {}
    for n, d in G.nodes(data=True):
        t = int(d.get("room_type", d.get("roomtype", -1)))
        native = ROOM_NAMES[t] if 0 <= t < len(ROOM_NAMES) else f"class_{t}"
        rings = polygons_of(_poly(d["geometry"]), min_area=1e-3)
        if native == "Structure":
            walls += [{"polygon": r, "type": "structure"} for r in rings]
            continue
        for r in rings:
            node_of.setdefault(n, len(rooms))
            rooms.append({"type": canonical_room_type(native, EXTRA), "type_native": native,
                          "polygon": r})
    if not rooms:
        return None
    edges = set()
    for u, v, d in G.edges(data=True):
        lab = EDGE_LABEL.get(d.get("connectivity"), str(d.get("connectivity")))
        if u in node_of and v in node_of and node_of[u] != node_of[v]:
            a, b = sorted((node_of[u], node_of[v]))
            edges.add((a, b, lab))
    sid = Path(path).stem
    sample = make_sample(
        "msd", "v1_4tu", sid, rooms, units="m", scale_m_per_unit=1.0, walls=walls,
        graph={"edges": [list(e) for e in sorted(edges)]},
        condition={"split": "train", "level": "building_floor",
                   "inputs": {"struct_in": f"struct_in/{sid}.npy", "graph_in": f"graph_in/{sid}.pickle"}})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return sid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    files = sorted(Path(args.raw, "graph_out").glob("*.pickle"), key=lambda p: int(p.stem))
    files = files[: args.limit] if args.limit else files
    root = out_dirs("msd", args.out)
    splits, failed = {}, 0
    with Pool(args.workers) as pool:
        for sid in pool.imap_unordered(convert_one, ((f, root) for f in files), chunksize=16):
            if sid is None:
                failed += 1
            else:
                splits[sid] = "train"
    counts = write_splits(root, splits, {"note": "MSD train release only; the test release (ids 4167-5556) has no public ground truth"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
