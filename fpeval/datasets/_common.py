"""Shared helpers for dataset converters."""
import json
from pathlib import Path

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"


def out_dirs(dataset, out=None):
    root = Path(out) if out else DATA / dataset / "converted"
    (root / "samples").mkdir(parents=True, exist_ok=True)
    return root


def write_splits(root, splits, extra=None):
    """splits: {sample_id: split_name}. Writes splits.json as {split: [ids]}."""
    by = {}
    for sid, sp in sorted(splits.items()):
        by.setdefault(sp, []).append(sid)
    info = {"counts": {k: len(v) for k, v in by.items()}, "splits": by}
    if extra:
        info.update(extra)
    with open(Path(root) / "splits.json", "w") as f:
        json.dump(info, f)
    return info["counts"]


def polygons_of(geom, min_area=0.0):
    """Shapely geometry -> list of exterior rings ([[x, y], ...], not closed).

    Holes are dropped (the common format has simple polygons only).
    """
    if geom is None or geom.is_empty:
        return []
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, Polygon):
        parts = [geom]
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        parts = [g for g in geom.geoms if isinstance(g, Polygon)]
        parts += [p for g in geom.geoms if isinstance(g, MultiPolygon) for p in g.geoms]
    else:
        return []
    out = []
    for p in parts:
        if p.area <= min_area:
            continue
        ring = [[round(float(x), 4), round(float(y), 4)] for x, y in p.exterior.coords[:-1]]
        if len(ring) >= 3:
            out.append(ring)
    return out


def seg_rect(p, q, half_width):
    """Thin rectangle around segment p-q (for doors/windows given as lines)."""
    from shapely.geometry import LineString
    return polygons_of(LineString([p, q]).buffer(half_width, cap_style=2, join_style=2))
