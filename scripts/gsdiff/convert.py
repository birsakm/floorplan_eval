"""Convert GSDiff native outputs (vr4stat_<i>.npy structural-graph results) to the common format.

GSDiff outputs a wall graph (corners + edges, 512x512 px frame = 2x RPLAN's 256 px frame; we write
coordinates in the RPLAN 256 px frame by default, matching the GT/conditioning data) with per-corner semantics. Room
polygons are the planar faces of that graph, extracted by GSDiff's own
get_cycle_basis_and_semantic_3_semansimplified (run at inference time and stored in vr4stat as
simple_cycles_test / simple_cycles_semantics_test); each face gets the semantic voted by its corners.
Faces that are not simple (pinched at a repeated vertex) are split with shapely.make_valid into
polygonal parts that keep the face's label.

GSDiff simplified 7-class RPLAN semantics:
  0 LivingRoom+DiningRoom+Entrance, 1 Master/Child/Study/Second/GuestRoom, 2 Storage+Wall-in,
  3 Kitchen, 4 Bathroom, 5 Balcony, 6 External (never assigned to a face)

Usage (fpe env, repo root on PYTHONPATH):
  python scripts/gsdiff/convert.py outputs/gsdiff/rplan_uncond [--variant rplan_uncond]
"""
import argparse
import re
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from fpeval.format import make_sample, save_sample

GSDIFF_CLASSES = {
    0: ("living_room", "LivingRoom/DiningRoom/Entrance"),
    1: ("bedroom", "MasterRoom/ChildRoom/StudyRoom/SecondRoom/GuestRoom"),
    2: ("storage", "Storage/Wall-in"),
    3: ("kitchen", "Kitchen"),
    4: ("bathroom", "Bathroom"),
    5: ("balcony", "Balcony"),
    6: ("outdoor", "External"),
}
MIN_AREA = 1.0  # px^2 (output frame); drop degenerate slivers


def polygon_parts(coords):
    if len(coords) >= 2 and tuple(coords[0]) == tuple(coords[-1]):
        coords = coords[:-1]
    if len(coords) < 3:
        return []
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    geoms = getattr(poly, "geoms", [poly])
    out = []
    for g in geoms:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and p.area >= MIN_AREA:
                p = p.simplify(0)  # drop collinear duplicates
                out.append(p)
    return out


def convert_cycles(cycles, semantics, scale=0.5):
    rooms, shapes = [], []
    for cyc, sem in zip(cycles, semantics):
        canon, native = GSDIFF_CLASSES.get(int(sem), ("other", str(sem)))
        coords = [(float(p[0]) * scale, float(p[1]) * scale) for p in cyc]
        for p in polygon_parts(coords):
            ext = [[round(x, 2), round(y, 2)] for x, y in list(p.exterior.coords)[:-1]]
            if len(ext) < 3:
                continue
            rooms.append({"type": canon, "type_native": native, "polygon": ext})
            shapes.append(p)
    boundary = None
    if shapes:
        u = unary_union([s.buffer(0.5, join_style=2) for s in shapes]).buffer(-0.5, join_style=2)
        if u.geom_type == "Polygon":
            boundary = [[round(x, 2), round(y, 2)] for x, y in list(u.exterior.coords)[:-1]]
    return rooms, boundary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--raw_subdir", default="raw")
    ap.add_argument("--out_subdir", default="samples")
    ap.add_argument("--condition_type", default=None)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="GSDiff works in a 512 frame = 2x the RPLAN 256 px frame; 0.5 maps back to RPLAN pixels")
    args = ap.parse_args()
    root = Path(args.variant_dir)
    variant = args.variant or root.name
    files = sorted((root / args.raw_subdir).glob("vr4stat_*.npy"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    n_ok = n_empty = 0
    for f in files:
        sid = re.findall(r"vr4stat_(.+)", f.stem)[0]
        vr = np.load(f, allow_pickle=True).item()
        rooms, boundary = convert_cycles(vr.get("simple_cycles_test", []), vr.get("simple_cycles_semantics_test", []), args.scale)
        if not rooms:
            n_empty += 1
        cond = None
        if args.condition_type:
            cond = {"type": args.condition_type, "ref_id": vr.get("ref_id", sid)}
        sample = make_sample("gsdiff", variant, sid, rooms, units="px", boundary=boundary, condition=cond)
        save_sample(sample, root / args.out_subdir / f"{sid}.json")
        n_ok += 1
    print(f"converted {n_ok} samples ({n_empty} with no rooms) -> {root / args.out_subdir}")


if __name__ == "__main__":
    main()
