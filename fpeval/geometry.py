"""Shapely helpers for common-format floorplans."""
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def to_polygon(coords):
    """Polygon from a coordinate list, repaired to a valid (Multi)Polygon.

    Returns (geometry, was_valid). Degenerate input gives an empty Polygon.
    """
    if len(coords) < 3:
        return Polygon(), False
    poly = Polygon(coords)
    if poly.is_valid:
        return poly, True
    fixed = shapely.make_valid(poly)
    parts = [g for g in getattr(fixed, "geoms", [fixed]) if isinstance(g, (Polygon, MultiPolygon))]
    return (unary_union(parts) if parts else Polygon()), False


def rooms_as_shapes(sample):
    """[(type, geometry, was_valid)] for every room with non-zero area."""
    out = []
    for room in sample["rooms"]:
        geom, valid = to_polygon(room["polygon"])
        if not geom.is_empty and geom.area > 0:
            out.append((room["type"], geom, valid))
    return out


def boundary_shape(sample):
    if not sample.get("boundary"):
        return None
    geom, _ = to_polygon(sample["boundary"])
    return None if geom.is_empty else geom


def closing(geom, radius):
    """Morphological closing: fills gaps narrower than 2 * radius (e.g. walls)."""
    return geom.buffer(radius, join_style="mitre").buffer(-radius, join_style="mitre")


def fill_holes(geom):
    parts = getattr(geom, "geoms", [geom])
    return unary_union([Polygon(p.exterior) for p in parts if isinstance(p, Polygon)])


def min_rect_sides(geom):
    """(short, long) side lengths of the minimum rotated bounding rectangle."""
    rect = geom.minimum_rotated_rectangle
    if not isinstance(rect, Polygon):
        return 0.0, 0.0
    xy = np.asarray(rect.exterior.coords)[:4]
    a = np.linalg.norm(xy[1] - xy[0])
    b = np.linalg.norm(xy[2] - xy[1])
    return min(a, b), max(a, b)


def _edges(geom):
    """(dx, dy) of every ring edge of a (Multi)Polygon."""
    out = []
    for part in getattr(geom, "geoms", [geom]):
        for ring in [part.exterior, *part.interiors]:
            out.append(np.diff(np.asarray(ring.coords), axis=0))
    return np.concatenate(out) if out else np.zeros((0, 2))


def dominant_orientation(geoms):
    """Length-weighted dominant edge orientation (radians, modulo 90 deg) of a plan."""
    d = np.concatenate([_edges(g) for g in geoms]) if geoms else np.zeros((0, 2))
    length = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    return 0.25 * np.arctan2((length * np.sin(4 * ang)).sum(), (length * np.cos(4 * ang)).sum())


def rectilinear_fraction(geom, theta=0.0, angle_tol_deg=5.0, min_edge=0.0):
    """Fraction of perimeter aligned (within angle_tol) with the axes rotated by theta.

    Edges shorter than min_edge are ignored, so staircase corner cuts from raster
    vectorization don't count against rectilinearity.
    """
    d = _edges(geom)
    length = np.hypot(d[:, 0], d[:, 1])
    keep = length >= min_edge
    if not keep.any():
        return 1.0
    d, length = d[keep], length[keep]
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]) - theta, np.pi / 2)
    tol = np.deg2rad(angle_tol_deg)
    ok = (ang < tol) | (ang > np.pi / 2 - tol)
    return length[ok].sum() / length.sum()


def n_corners(geom, simplify_tol):
    """Number of polygon corners after light simplification (exterior rings)."""
    simple = geom.simplify(simplify_tol, preserve_topology=True)
    return sum(len(p.exterior.coords) - 1 for p in getattr(simple, "geoms", [simple]))
