"""Convert Swiss Dwellings v3.0.0 (Zenodo 7788422, CC BY 4.0) to the common format.

Raw data: data/swiss_dwellings/raw/swiss-dwellings-v3.0.0/geometries.csv
(scripts/datasets/download_swiss_dwellings.sh). Geometries are WKT polygons in metres.

One sample per residential apartment and floor (apartment_id, floor_id); maisonettes
therefore yield one sample per floor. Rooms = `area` entities of the apartment (SHAFT,
VOID, LIGHTWELL, OUTDOOR_VOID dropped), doors/windows = `opening` entities, walls =
`separator` entities. The y axis is flipped (source is y-up) so that y points down.
There is no official split: every sample gets split "all"; site/building/plan ids are kept
in `condition` so a site-disjoint split can be made later (MSD uses this data, see msd.py).

    python -m fpeval.datasets.swiss_dwellings [--limit N] [--workers 16]
"""
import argparse
from multiprocessing import Pool

from shapely import affinity, wkt

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, polygons_of, write_splits

RAW = DATA / "swiss_dwellings" / "raw" / "swiss-dwellings-v3.0.0"
SKIP_AREAS = {"SHAFT", "VOID", "LIGHTWELL", "OUTDOOR_VOID", "AIR"}
EXTRA = {
    "ROOM": "bedroom",  # generic "Zimmer"; MSD maps it to Bedroom too
    "LIVING_DINING": "living_room", "KITCHEN_DINING": "kitchen", "DINING": "dining_room",
    "CORRIDORS_AND_HALLS": "corridor", "LOBBY": "entrance", "FOYER": "entrance",
    "STOREROOM": "storage", "BASEMENT_COMPARTMENT": "storage", "BASEMENT": "storage",
    "BIKE_STORAGE": "storage", "PRAM_AND_BIKE_STORAGE_ROOM": "storage", "PRAM": "storage",
    "WAREHOUSE": "storage", "ARCHIVE": "storage", "TECHNICAL_AREA": "storage",
    "HEATING": "storage", "OIL_TANK": "storage", "HOUSE_TECHNICS_FACILITIES": "storage",
    "ELECTRICAL_SUPPLY": "storage", "WATER_SUPPLY": "storage", "GAS": "storage",
    "WASH_AND_DRY_ROOM": "laundry", "SANITARY_ROOMS": "bathroom",
    "LOGGIA": "balcony", "WINTERGARTEN": "balcony", "TERRACE": "balcony",
    "GARDEN": "outdoor", "PATIO": "outdoor", "CARPARK": "garage", "GARAGE": "garage",
    "OFFICE": "study", "OFFICE_SPACE": "study", "STUDIO": "study", "CLOAKROOM": "closet",
    "STAIRCASE": "other", "ELEVATOR": "other", "NOT_DEFINED": "other",
}


def _geom(s):
    return affinity.scale(wkt.loads(s), xfact=1.0, yfact=-1.0, origin=(0, 0))


def convert_one(args):
    (apt, floor), rows, root = args
    rooms, doors, windows, walls = [], [], [], []
    meta = rows[0]
    for r in rows:
        et, st = r["entity_type"], r["entity_subtype"]
        if et == "area":
            if st in SKIP_AREAS:
                continue
            for ring in polygons_of(_geom(r["geometry"]), min_area=1e-3):
                rooms.append({"type": canonical_room_type(st, EXTRA), "type_native": st,
                              "polygon": ring})
        elif et == "opening":
            for ring in polygons_of(_geom(r["geometry"])):
                if st == "WINDOW":
                    windows.append({"polygon": ring})
                else:
                    doors.append({"type": "front_door" if st == "ENTRANCE_DOOR" else "interior_door",
                                  "polygon": ring})
        elif et == "separator":
            for ring in polygons_of(_geom(r["geometry"])):
                walls.append({"polygon": ring, "type": st.lower()})
    if not rooms:
        return None
    sid = f"{apt}_{int(floor)}"
    sample = make_sample(
        "swiss_dwellings", "v3.0.0", sid, rooms, units="m", scale_m_per_unit=1.0,
        doors=doors, windows=windows, walls=walls,
        condition={"split": "all", "apartment_id": apt, "floor_id": int(floor),
                   "site_id": int(meta["site_id"]), "building_id": int(meta["building_id"]),
                   "plan_id": int(meta["plan_id"]), "unit_id": int(meta["unit_id"]),
                   "elevation": float(meta["elevation"])})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return sid


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    df = pd.read_csv(f"{args.raw}/geometries.csv")
    df = df[(df.unit_usage == "RESIDENTIAL") & df.apartment_id.notna()]
    df = df[df.entity_type != "feature"]
    root = out_dirs("swiss_dwellings", args.out)
    groups = df.groupby(["apartment_id", "floor_id"], sort=True)
    keys = list(groups.groups)[: args.limit] if args.limit else list(groups.groups)
    print("apartment-floors:", len(keys))

    def jobs():
        for k in keys:
            yield k, groups.get_group(k).to_dict("records"), root

    splits, failed = {}, 0
    with Pool(args.workers) as pool:
        for sid in pool.imap_unordered(convert_one, jobs(), chunksize=64):
            if sid is None:
                failed += 1
            else:
                splits[sid] = "all"
    counts = write_splits(root, splits, {"note": "Swiss Dwellings has no official split; use condition.site_id for site-disjoint splits"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
