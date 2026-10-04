"""Convert the MagicPlan floor dataset released with PuzzleFusion (NeurIPS 2023) to the common format.

Source: Google Drive folder linked from github.com/sepidsh/PuzzleFussion (README "MagicPlan";
"dataset follows same license as code": non-commercial research, GPLv3 terms).
Raw data: data/magicplan/raw/ (scripts/datasets/download_magicplan.sh). We convert the
reader-ready floors in magicplan_mini_set/raw_data_magicplan_withdata_reader/:
  SFU_train_samples.pickle  (list of floors; split "train")
  SFU_samples_floor/sfu_test.pickle (100 floors; split "test")
Each floor is a list of rooms {room_type, points (m, room-local), delta_x/delta_y (room
offset), windoor [{category door|window, points: 2 endpoints}], furnitures}.
Rooms are captured with the MagicPlan AR app by consumers, so they are real but come
with small gaps/overlaps between rooms. Floors include some non-residential spaces.

Coordinates are metres; y is negated so that the layout matches the reader's matplotlib
plot (y-up) when drawn y-down. Doors/windows -> thin (10 cm) rectangles. Graph: rooms
whose door segments lie within 30 cm of each other are linked with "door".

    python -m fpeval.datasets.magicplan [--limit N] [--workers 16]

Rooms / doors / windows with non-finite coordinates are dropped (a few raw floors contain NaN).
"""
import argparse
import math
import pickle
from multiprocessing import Pool
from pathlib import Path

from shapely.geometry import LineString, Polygon

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, polygons_of, seg_rect, write_splits

RAW = DATA / "magicplan" / "raw" / "magicplan_mini_set" / "raw_data_magicplan_withdata_reader"
EXTRA = {
    "Master Bedroom": "bedroom", "Vestibule": "entrance", "Halldentree": "entrance",
    "Hall": "corridor", "Halldenuit": "corridor", "Halldecave": "corridor",
    "Furnace Room": "storage", "Cellar": "storage", "Cave": "storage",
    "Unfinished Basement": "storage", "Attic": "storage", "Grenier": "storage",
    "Archives": "storage", "Equipment Room": "storage", "Maintenance Room": "storage",
    "Laundry Room": "laundry", "Buanderie": "laundry", "Renholdssone": "laundry",
    "Kitchenette": "kitchen", "Sejour": "living_room", "Den": "living_room",
    "Playroom": "living_room", "Music Room": "living_room",
    "Master Bathroom": "bathroom", "Half Bathroom": "bathroom", "Toilet": "bathroom",
    "Restrooms": "bathroom", "Rest Room": "bathroom",
    "Private Office": "study", "Shared Office": "study",
    "Deck": "balcony", "Porch": "outdoor", "Patio": "outdoor", "Outbuilding": "outdoor",
    "Stairway": "other", "Elevators": "other", "Piece": "other", "Open Space": "other",
}


def convert(floor, split, idx, root):
    rooms, doors, windows, door_lines = [], [], [], []
    for r in floor:
        dx, dy = float(r.get("delta_x", 0)), float(r.get("delta_y", 0))
        tf = lambda pts: [(float(x) + dx, -(float(y) + dy)) for x, y in pts]
        pts = tf(r["points"])
        if len(pts) < 3 or not all(map(math.isfinite, (v for q in pts for v in q))):
            continue
        rings = polygons_of(Polygon(pts).buffer(0), min_area=1e-3)
        if not rings:
            continue
        k = len(rooms)
        rooms.append({"type": canonical_room_type(r["room_type"], EXTRA),
                      "type_native": r["room_type"], "polygon": rings[0]})
        for w in r.get("windoor", []):
            seg = tf(w["points"])
            if len(seg) != 2 or seg[0] == seg[1] or not all(map(math.isfinite, (v for q in seg for v in q))):
                continue
            rect = seg_rect(seg[0], seg[1], 0.05)
            if w.get("category") == "door":
                doors += [{"type": "door", "polygon": p} for p in rect]
                door_lines.append((k, LineString(seg)))
            elif w.get("category") == "window":
                windows += [{"polygon": p} for p in rect]
    if not rooms:
        return None
    edges = set()
    for i, (a, la) in enumerate(door_lines):
        for b, lb in door_lines[i + 1:]:
            if a != b and la.distance(lb) < 0.3:
                edges.add((min(a, b), max(a, b), "door"))
    sid = f"{split}_{idx:06d}"
    sample = make_sample("magicplan", "puzzlefusion_sfu", sid, rooms, units="m",
                         scale_m_per_unit=1.0, doors=doors, windows=windows,
                         graph={"edges": [list(e) for e in sorted(edges)]},
                         condition={"split": split})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return sid


def _job(a):
    return a[1], convert(*a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = out_dirs("magicplan", args.out)
    splits, failed = {}, 0
    for split, rel in (("train", "SFU_train_samples.pickle"), ("test", "SFU_samples_floor/sfu_test.pickle")):
        with open(Path(args.raw) / rel, "rb") as f:
            floors = pickle.load(f)
        floors = floors[: args.limit] if args.limit else floors
        with Pool(args.workers) as pool:
            for sp, sid in pool.imap_unordered(_job, ((fl, split, i, root) for i, fl in enumerate(floors)), chunksize=256):
                if sid is None:
                    failed += 1
                else:
                    splits[sid] = sp
    counts = write_splits(root, splits, {"note": "train = SFU_train_samples.pickle, test = sfu_test.pickle (PuzzleFusion release)"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
