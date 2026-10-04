"""Convert CubiCasa5k (Zenodo 2613548, CC BY-NC-SA 4.0) to the common format.

Raw data: data/cubicasa5k/raw/cubicasa5k/{high_quality,high_quality_architectural,colorful}/<n>/model.svg
plus {train,val,test}.txt (scripts/datasets/download_cubicasa5k.sh). Official parser:
external/datasets/cubicasa5k/floortrans/loaders/house.py (rasterizes; we read the same SVG
elements but keep them as vectors).

One sample per `Floorplan Floor-k` group in model.svg (multi-storey houses give several
samples: <subset>_<n>_f<k>). Rooms = `Space <Type> [<Subtype>]` groups (first polygon);
type_native is the full class (e.g. "Outdoor Balcony"), room["label"] the visible (mostly
Finnish) name label. "UserDefined" and "Room" spaces are typed from their Finnish label (FINNISH; e.g.
"Room"/"H" (huone) -> bedroom, "TUPA" -> living_room, unknown -> other).
The ~11k "Undefined" spaces (label "UNDEFINED") stay "other". Overlay/void spaces (Below150cm, OpenToBelow) are dropped.
Walls/railings = `Wall` / `Railing` groups, doors/windows = `Door` / `Window` groups (first
polygon, as in house.py). SVG coordinates are y-down pixels.

Metric scale: each Space carries a hidden dimension annotation (marks at the room's
bounding-box extent and an imperial label like 9'8" x 13'4"). The median of
label_m / extent_px is 0.0100 +- 0.0001 on every plan we checked, i.e. SVG units are cm.
We therefore write metres (units "m") using a fixed 0.01 m/px for all plans and store the
per-floor estimate in condition.m_per_px_est (None if the floor has no labels).

Splits: official train.txt / val.txt / test.txt (3 of the 5000 folders are in none: "unlisted").

    python -m fpeval.datasets.cubicasa5k [--limit N] [--workers 16]
"""
import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from lxml import etree

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, polygons_of, write_splits

RAW = DATA / "cubicasa5k" / "raw" / "cubicasa5k"
SVG = "{http://www.w3.org/2000/svg}"
SKIP_SPACES = {"Below150cm", "OpenToBelow", "Skylights"}
EXTRA = {
    "Bath": "bathroom", "Sauna": "bathroom", "HotTub": "bathroom",
    "LivingRoom": "living_room", "Lounge": "living_room", "RecreationRoom": "living_room",
    "MediaRoom": "living_room", "SunRoom": "living_room", "Conservatory": "living_room",
    "Kitchen": "kitchen", "Dining": "dining_room", "EatingArea": "dining_room",
    "Entry": "entrance", "DraughtLobby": "entrance", "Hall": "entrance", "HallWay": "corridor",
    "Closet": "closet", "DressingRoom": "closet",
    "Storage": "storage", "Pantry": "storage", "TechnicalRoom": "storage", "Attic": "storage",
    "Basement": "storage", "Garbage": "storage",
    "Utility": "laundry",  # Finnish KHH (kodinhoitohuone) = utility/laundry room
    "Office": "study", "Library": "study",
    "Den": "living_room",  # mostly "TH" (takkahuone, fireplace room)
    "Garage": "garage", "CarPort": "garage", "Outdoor": "outdoor",
    "Outdoor Balcony": "balcony", "Outdoor Balcony Glazed": "balcony",
    "Outdoor Balcony Covered": "balcony", "Outdoor Terrace": "balcony",
    "Outdoor Terrace Covered": "balcony", "Outdoor Terrace Covered Open": "balcony",
    "Outdoor Terrace Roof": "balcony", "Outdoor Veranda": "balcony",
    "Outdoor Veranda Glazed": "balcony", "Elevated": "other", "Bedroom Guest": "bedroom", "Room Cold": "storage",
    "StairWell": "other", "Elevator": "other", "Alcove": "other", "Room": "other",
    "Undefined": "other", "UserDefined": "other",
}
M_PER_PX = 0.01  # SVG model units are centimetres (verified via dimension labels)
# Finnish name-label abbreviations, used only for "UserDefined"/"Room" spaces (first token of the
# label before "/", "+" or space decides).
FINNISH = {
    "PH": "bathroom", "KH": "bathroom", "KPH": "bathroom", "WC": "bathroom", "SH": "bathroom",
    "PSH": "bathroom", "INV.WC": "bathroom", "S": "bathroom", "SAUNA": "bathroom",
    "KHH": "laundry", "PKH": "laundry", "K": "kitchen", "KK": "kitchen", "KT": "kitchen",
    "MH": "bedroom", "H": "bedroom", "OH": "living_room", "TUPA": "living_room",
    "OLESKELU": "living_room", "TH": "living_room",
    "RUOK": "dining_room", "R": "dining_room", "TSTO": "study", "TYÖH": "study",
    "TYÖTILA": "study", "KIRJASTO": "study", "VAR": "storage", "VARASTO": "storage",
    "KELL": "storage", "TEKN": "storage", "LÄMM": "storage", "KYLMÄ": "storage",
    "KOM": "closet", "KOMERO": "closet", "KYLMIÖ": "storage", "VH": "closet", "PUKUH": "closet", "ET": "entrance", "TK": "entrance",
    "AULA": "entrance", "KÄYT": "corridor", "KÄYTÄVÄ": "corridor", "HALLI": "entrance",
    "PARVEKE": "balcony", "LASITETTU": "balcony", "AVOKUISTI": "balcony", "TERASSI": "balcony",
    "AUTOTALLI": "garage", "AT": "garage", "AK": "garage",
}
_FTIN = re.compile(r"(\d+)'\s*(\d+)\"")


def _points(g):
    """First direct <polygon> child's points (house.py get_points), as float [[x, y], ...]."""
    for c in g:
        if c.tag == SVG + "polygon":
            vals = [float(v) for v in re.split(r"[ ,]+", c.get("points", "").strip()) if v]
            pts = list(zip(vals[0::2], vals[1::2]))
            # drop duplicated closing points
            while len(pts) > 1 and pts[-1] == pts[0]:
                pts.pop()
            out = []
            for p in pts:
                if not out or out[-1] != p:
                    out.append(p)
            return out if len(out) >= 3 else None
    return None


def _clean(pts):
    from shapely.geometry import Polygon
    return polygons_of(Polygon(pts).buffer(0), min_area=1.0) if pts else []


def _translate(el):
    m = re.search(r"matrix\(([^)]*)\)", el.get("transform", "") or "")
    if not m:
        return 0.0, 0.0
    v = [float(x) for x in re.split(r"[ ,]+", m.group(1).strip())]
    return v[4], v[5]


def _name_label(space):
    lab = next((e for e in space.iter(SVG + "g") if "NameLabel" in (e.get("class") or "")), None)
    return "".join(lab.itertext()).strip().upper() if lab is not None else ""


def _ft(s):
    m = _FTIN.search(s)
    return (int(m.group(1)) + int(m.group(2)) / 12.0) * 0.3048 if m else None


def _space_scale(space):
    """m/px estimate from a Space's dimension marks + imperial label, or None."""
    dim = next((e for e in space.iter(SVG + "g") if (e.get("class") or "") == "Dimension"), None)
    if dim is None:
        return []
    marks = {e.get("id"): _translate(e) for e in dim.iter(SVG + "g") if (e.get("class") or "") == "DimensionMark"}
    lab = next((e for e in dim.iter(SVG + "g") if "DimensionMeasureLabel" in (e.get("class") or "")), None)
    if lab is None or "mark2" not in marks or "mark3" not in marks:
        return []
    txt = "".join(lab.itertext())
    parts = txt.split("x")
    if len(parts) != 2:
        return []
    wm, hm = _ft(parts[0]), _ft(parts[1])
    wpx, hpx = marks["mark2"][0], marks["mark3"][1]
    out = []
    if wm and wpx > 20:
        out.append(wm / wpx)
    if hm and hpx > 20:
        out.append(hm / hpx)
    return out


def convert_svg(args):
    rel, split, raw, root = args
    path = Path(raw) / rel.strip("/") / "model.svg"
    tree = etree.parse(str(path), etree.XMLParser(huge_tree=True, recover=True))
    floors = [g for g in tree.iter(SVG + "g") if (g.get("class") or "").startswith("Floorplan Floor-")]
    if not floors:
        floors = [tree.getroot()]
    out = []
    base = rel.strip("/").replace("/", "_")
    for k, fl in enumerate(floors, 1):
        rooms, doors, windows, walls, scales = [], [], [], [], []
        for g in fl.iter(SVG + "g"):
            cls = (g.get("class") or "").split()
            gid = g.get("id") or ""
            if cls and cls[0] == "Space" and len(cls) > 1:
                native = " ".join(cls[1:])
                scales += _space_scale(g)
                if cls[1] in SKIP_SPACES:
                    continue
                label = _name_label(g)
                t = EXTRA.get(native) or canonical_room_type(cls[1], EXTRA)
                if cls[1] in ("UserDefined", "Room") and label:
                    t = FINNISH.get(re.split(r"[/+ ]", label)[0], "other")
                for ring in _clean(_points(g)):
                    rooms.append({"type": t, "type_native": native, "label": label,
                                  "polygon": ring})
            elif gid in ("Wall", "Railing"):
                for ring in _clean(_points(g)):
                    walls.append({"polygon": ring, "type": gid.lower()})
            elif gid == "Door":
                for ring in _clean(_points(g)):
                    doors.append({"type": "door", "polygon": ring})
            elif gid == "Window":
                for ring in _clean(_points(g)):
                    windows.append({"polygon": ring})
        if not rooms:
            continue
        s_est = float(np.median(scales)) if scales else None
        s = M_PER_PX
        f = lambda items: [dict(it, polygon=[[round(x * s, 4), round(y * s, 4)] for x, y in it["polygon"]]) for it in items]
        rooms, doors, windows, walls = f(rooms), f(doors), f(windows), f(walls)
        sid = f"{base}_f{k}"
        sample = make_sample(
            "cubicasa5k", "v1", sid, rooms, units="m", scale_m_per_unit=1.0,
            doors=doors, windows=windows, walls=walls,
            condition={"split": split, "folder": rel, "floor": k, "n_floors": len(floors),
                       "m_per_px": s, "m_per_px_est": s_est, "n_scale_labels": len(scales)})
        save_sample(sample, Path(root) / "samples" / f"{sid}.json")
        out.append((sid, split))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    raw = Path(args.raw)
    split_of = {}
    for sp in ("train", "val", "test"):
        for line in (raw / f"{sp}.txt").read_text().split():
            split_of["/" + line.strip("/") + "/"] = sp
    folders = sorted("/" + str(p.parent.relative_to(raw)) + "/" for p in raw.glob("*/*/model.svg"))
    folders = folders[: args.limit] if args.limit else folders
    root = out_dirs("cubicasa5k", args.out)
    jobs = [(f, split_of.get(f, "unlisted"), str(raw), str(root)) for f in folders]
    splits, failed = {}, []
    with Pool(args.workers) as pool:
        for f, res in zip(folders, pool.imap(convert_svg, jobs, chunksize=8)):
            if not res:
                failed.append(f)
            for sid, sp in res:
                splits[sid] = sp
    counts = write_splits(root, splits, {"note": "official CubiCasa5k train/val/test.txt (by folder); one sample per floor", "failed_folders": failed})
    print("folders", len(folders), "samples", len(splits), "failed folders", len(failed), counts)


if __name__ == "__main__":
    main()
