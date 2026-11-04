"""RPLAN with its real doors: the Graph2Plan room polygons plus the interior doors and the front door of the original
RPLAN images (Kaggle upload "RPLAN pickle files", scripts/datasets/download_rplan_kaggle.py).

Graph2Plan's processed RPLAN (fpeval/datasets/rplan_graph2plan.py) dropped the interior doors. The Kaggle pickles are
the original 80,788 RPLAN plans as masks, in the same 256 px frame and with the same ids (Graph2Plan 'name'):
  [0] inside mask, [1] boundary (127 exterior wall, 255 front door), [2] interior walls, [3] interior doors,
  [4] rooms [{category (RPLAN 0-12), centroid (row, col)}]
Graph2Plan's room polygons are the net room interiors; the door masks fill the wall gaps between them (5 px deep), so
each door's rectangle touches the two rooms it connects. Per Graph2Plan sample (same id, e.g. test_50962) we keep the
rooms and the boundary, replace the front door by the mask's and add one 'interior_door' per door-mask component
(axis-aligned bounding box; pixel (r, c) spans [c, c+1] x [r, r+1]). graph.edges gets 'door' edges for room pairs
connected by a door (fpeval.metrics.door_edges); the 'adjacent' edges of Graph2Plan stay.
A plan is skipped when the pickle's room categories differ from Graph2Plan's (counted in splits.json).

    python -m fpeval.datasets.rplan_kaggle [--workers 16] [--ids <file of RPLAN ids>]
"""
import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..format import load_sample, save_sample
from ._common import DATA, out_dirs, write_splits

RAW = DATA / "rplan_kaggle" / "raw"
G2P = DATA / "rplan_graph2plan" / "converted"
_NUMPY = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
          ("numpy", "ndarray"), ("numpy", "dtype"),
          ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer")}


class _NumpyOnly(pickle.Unpickler):
    """The pickles come from a third-party upload: allow numpy arrays and nothing else."""

    def find_class(self, module, name):
        if (module, name) not in _NUMPY:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        import importlib
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                return getattr(importlib.import_module(module.replace("numpy.core", "numpy._core")), name)


def load_pickle(path):
    with open(path, "rb") as f:
        return _NumpyOnly(f).load()


def pickle_index(raw=RAW):
    return {p.stem: p for p in (Path(raw) / "pickle").glob("*/*.pkl")}


def door_boxes(mask, min_len=8):
    """One box per door: doors are straight runs of the door mask; two doors meeting at a corner form one connected
    component, so horizontal and vertical runs are separated first (opening with a 1 x min_len / min_len x 1 line)."""
    from scipy import ndimage
    h = ndimage.binary_opening(mask, structure=np.ones((1, min_len), bool))
    v = ndimage.binary_opening(mask, structure=np.ones((min_len, 1), bool))
    both = h & v  # a square door block (rare): keep it once, as horizontal
    return boxes(h) + boxes(v & ~both)


def boxes(mask):
    """Axis-aligned boxes of the connected components of a mask, as polygons in (x, y) pixel-edge coordinates."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for sl in ndimage.find_objects(lab):
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append([[float(c0), float(r0)], [float(c1), float(r0)], [float(c1), float(r1)], [float(c0), float(r1)]])
    return out


def convert(args):
    from ..geometry import rooms_as_shapes
    from ..metrics import door_edges
    from .. import rules
    sid, pkl, root = args
    s = load_sample(G2P / "samples" / f"{sid}.json")
    p = load_pickle(pkl)
    cats = sorted(int(r["category"]) for r in p[4])
    g2p_cats = sorted(s["condition"].get("rplan_types") or [])
    native = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom", "StudyRoom", "SecondRoom",
              "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in"]
    if not g2p_cats:
        g2p_cats = sorted(native.index(r["type_native"]) for r in s["rooms"] if r["type_native"] in native)
    if cats != g2p_cats:
        return sid, "categories differ"
    front = boxes(p[1] == 255)
    if not front:
        return sid, "no front door"
    s["doors"] = [{"type": "front_door", "polygon": front[0]}] + [{"type": "interior_door", "polygon": b}
                                                                  for b in door_boxes(p[3] > 0)]
    shapes = rooms_as_shapes(s)
    de = sorted({tuple(e) for e in door_edges(s, shapes, rules.WALL_HALF_THICKNESS_M / s["scale_m_per_unit"])})
    adj = [e for e in s["graph"]["edges"] if e[2] != "door"]
    s["graph"]["edges"] = adj + [[i, j, "door"] for i, j in de]
    s["source"], s["variant"] = "rplan_kaggle", "graph2plan_rooms+rplan_doors"
    s["condition"]["rplan_pickle"] = str(Path(pkl).relative_to(DATA))
    s["condition"]["mask_reach"] = mask_reach(p)
    save_sample(s, Path(root) / "samples" / f"{sid}.json")
    return sid, None


def mask_reach(p):
    """Reachability on the original RPLAN masks: walk from the front door through free space (inside, not wall) and
    the interior doors; a room is reached when its centroid is. Openings without a door count as passable, as they
    are in the real building. Graph2Plan's room polygons are simplified boxes that overlap and do not follow the
    walls, so a door graph between them misses connections that exist in the plan; the masks do not.
    Returns {"all": bool, "unreached": [RPLAN category of each unreached room]}."""
    from scipy import ndimage
    wall = (p[2] > 0) | (p[1] == 127)
    passable = ((p[0] > 0) & ~wall) | (p[3] > 0) | (p[1] == 255)
    lab, _ = ndimage.label(passable)
    start = set(np.unique(lab[p[1] == 255]).tolist()) - {0}
    unreached = [int(r["category"]) for r in p[4] if lab[r["centroid"][0], r["centroid"][1]] not in start]
    return {"all": not unreached, "unreached": unreached}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ids", default=None, help="file with RPLAN ids (one per line) to convert; default: all")
    a = ap.parse_args()
    root = out_dirs("rplan_kaggle", a.out)
    g2p = json.load(open(G2P / "splits.json"))["splits"]
    split_of = {sid: sp for sp, ids in g2p.items() for sid in ids}
    pk = pickle_index(a.raw)
    want = {line.strip() for line in open(a.ids)} if a.ids else None
    jobs = [(sid, pk[sid.split("_", 1)[1]], root) for sid in split_of
            if sid.split("_", 1)[1] in pk and (want is None or sid.split("_", 1)[1] in want)]
    with Pool(a.workers) as pool:
        res = pool.map(convert, jobs, chunksize=16)
    skipped = {sid: why for sid, why in res if why}
    ok = {sid: split_of[sid] for sid, why in res if not why}
    missing = len(split_of) - len(jobs) if want is None else len(want) - len(jobs)
    counts = write_splits(root, ok, {"note": "Graph2Plan rooms and splits + RPLAN doors (Kaggle pickles)",
                                     "skipped": skipped, "no_pickle": missing})
    print("converted", counts, "skipped", len(skipped), "without pickle", missing)


if __name__ == "__main__":
    main()
