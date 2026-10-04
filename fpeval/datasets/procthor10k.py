"""Convert ProcTHOR-10K houses (allenai/procthor-10k, Apache-2.0) to the common format.

NOTE: these houses are procedurally generated (synthetic), not real floorplans. They are
included as a reference distribution (DS2D trains on them), not as real-world GT.

Raw data: data/procthor10k/raw/{train,val,test}.jsonl.gz (scripts/datasets/download_procthor10k.sh).
Rooms from `floorPolygon` (x, z) in metres (z is used as the down-pointing y axis).
Doors/windows: the hole's horizontal extent along its wall (wall ids encode the wall's
endpoints: wall|<room>|x0|z0|x1|z1), drawn as a thin rectangle. Graph edges come from
doors between two different rooms ("door").

    python -m fpeval.datasets.procthor10k [--limit N]
"""
import argparse
import gzip
import json

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, seg_rect, write_splits

RAW = DATA / "procthor10k" / "raw"
EXTRA = {"LivingRoom": "living_room"}


def _hole_segment(wall_id, hole):
    parts = wall_id.split("|")
    x0, z0, x1, z1 = map(float, parts[-4:])
    L = ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5 or 1.0
    ux, uz = (x1 - x0) / L, (z1 - z0) / L
    a, b = hole[0]["x"], hole[1]["x"]
    return (x0 + ux * a, z0 + uz * a), (x0 + ux * b, z0 + uz * b)


def convert(house, split, idx, root):
    rooms, index = [], {}
    for r in house["rooms"]:
        poly = [[round(p["x"], 4), round(p["z"], 4)] for p in r["floorPolygon"]]
        if len(poly) < 3:
            continue
        index[r["id"]] = len(rooms)
        rooms.append({"type": canonical_room_type(r["roomType"], EXTRA),
                      "type_native": r["roomType"], "polygon": poly})
    if not rooms:
        return None
    doors, windows, edges = [], [], set()
    for d in house.get("doors", []):
        try:
            p, q = _hole_segment(d["wall0"], d["holePolygon"])
        except (KeyError, ValueError, IndexError):
            continue
        a, b = index.get(d.get("room0")), index.get(d.get("room1"))
        ext = "exterior" in d.get("wall1", "") or a == b
        doors += [{"type": "front_door" if ext else "interior_door", "polygon": r}
                  for r in seg_rect(p, q, 0.05)]
        if a is not None and b is not None and a != b:
            edges.add((min(a, b), max(a, b), "door"))
    for w in house.get("windows", []):
        try:
            p, q = _hole_segment(w["wall0"], w["holePolygon"])
        except (KeyError, ValueError, IndexError):
            continue
        windows += [{"polygon": r} for r in seg_rect(p, q, 0.05)]
    sid = f"{split}_{idx:05d}"
    sample = make_sample("procthor10k", "main", sid, rooms, units="m", scale_m_per_unit=1.0,
                         doors=doors, windows=windows,
                         graph={"edges": [list(e) for e in sorted(edges)]},
                         condition={"split": split, "synthetic": True})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return sid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = out_dirs("procthor10k", args.out)
    splits, failed = {}, 0
    for split in ("train", "val", "test"):
        with gzip.open(f"{args.raw}/{split}.jsonl.gz", "rt") as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    break
                sid = convert(json.loads(line), split, i, root)
                if sid is None:
                    failed += 1
                else:
                    splits[sid] = split
    counts = write_splits(root, splits, {"note": "official ProcTHOR-10K train/val/test"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
