"""Per-sample and paired floorplan metrics.

All lengths are converted to meters with `scale` (m per unit). When a source
has no known scale, pass a nominal one: tolerances stay reasonable, but
metric-unit rules must then be ignored (the evaluator marks them as NaN).
"""
from collections import Counter

import networkx as nx
import numpy as np
import shapely
from shapely.ops import unary_union

from . import rules
from .format import ROOM_TYPES
from .geometry import (boundary_shape, closing, dominant_orientation, fill_holes,
                       min_rect_sides, n_corners, rectilinear_fraction, rooms_as_shapes)

NAN = float("nan")


def adjacency_graph(shapes, tol, min_shared):
    """Room adjacency from geometry: rooms whose tol-buffers overlap along >= min_shared."""
    g = nx.Graph()
    g.add_nodes_from(range(len(shapes)))
    buffered = [geom.buffer(tol) for _, geom, _ in shapes]
    tree = shapely.STRtree(buffered)
    for i, j in zip(*tree.query(buffered, predicate="intersects")):
        if i >= j:
            continue
        # Overlap of two tol-buffers along a shared wall is ~ length * 2 * tol.
        shared = buffered[i].intersection(buffered[j]).area / (2 * tol)
        if shared >= min_shared:
            g.add_edge(int(i), int(j))
    return g


def room_records(shapes, scale):
    """Per-room geometric attributes (meters)."""
    total = sum(geom.area for _, geom, _ in shapes) or 1.0
    theta = dominant_orientation([geom for _, geom, _ in shapes])
    min_edge = rules.RECTILINEAR_MIN_EDGE_M / scale
    recs = []
    for rtype, geom, valid in shapes:
        short, long_ = min_rect_sides(geom)
        width = 2 * shapely.maximum_inscribed_circle(geom).length if geom.area > 0 else 0.0
        hull = geom.convex_hull.area
        recs.append({
            "type": rtype,
            "area_m2": geom.area * scale ** 2,
            "area_frac": geom.area / total,
            "width_m": width * scale,
            "aspect": long_ / short if short > 0 else np.inf,
            "convexity": geom.area / hull if hull > 0 else 0.0,
            "rectilinear": rectilinear_fraction(geom, theta, min_edge=min_edge),
            "corners": n_corners(geom, simplify_tol=0.05 / scale),
            "valid": valid,
        })
    return recs


def sample_metrics(sample, scale, metric_scale_known=True):
    """Unpaired metrics for one sample.

    Returns (summary dict, per-room records, adjacency graph, room shapes).
    """
    shapes = rooms_as_shapes(sample)
    tol = rules.WALL_HALF_THICKNESS_M / scale
    out = {"n_rooms": len(shapes), "n_rooms_declared": len(sample["rooms"])}
    for t in ROOM_TYPES:
        out[f"n_{t}"] = sum(1 for s in shapes if s[0] == t)
    if not shapes:
        return out, [], nx.Graph(), shapes

    geoms = [g for _, g, _ in shapes]
    union = unary_union(geoms)
    area_sum = sum(g.area for g in geoms)
    out["frac_invalid_polygons"] = np.mean([not v for _, _, v in shapes])
    out["overlap_ratio"] = (area_sum - union.area) / union.area

    # Holes: enclosed space wider than a wall that belongs to no room.
    closed = closing(union, tol)
    filled = fill_holes(closed)
    out["hole_ratio"] = max(0.0, filled.area - closed.area) / filled.area if filled.area > 0 else NAN
    out["n_components"] = len(getattr(closed, "geoms", [closed]))

    bnd = boundary_shape(sample)
    if bnd is not None:
        out["boundary_coverage"] = closed.intersection(bnd).area / bnd.area
        out["outside_boundary_ratio"] = union.difference(bnd.buffer(tol)).area / union.area
    else:
        out["boundary_coverage"] = out["outside_boundary_ratio"] = NAN

    g = adjacency_graph(shapes, tol, rules.MIN_SHARED_WALL_M / scale)
    out["adj_components"] = nx.number_connected_components(g)
    out["frac_isolated_rooms"] = sum(1 for n in g if g.degree(n) == 0) / len(shapes)

    recs = room_records(shapes, scale)
    out["mean_rectilinear"] = np.mean([r["rectilinear"] for r in recs])
    out["frac_nonrectilinear_rooms"] = np.mean([r["rectilinear"] < 0.98 for r in recs])
    out["mean_convexity"] = np.mean([r["convexity"] for r in recs])
    out["mean_corners"] = np.mean([r["corners"] for r in recs])
    out["frac_aspect_violation"] = np.mean([
        rules.MAX_ASPECT[r["type"]] is not None and r["aspect"] > rules.MAX_ASPECT[r["type"]]
        for r in recs])
    if metric_scale_known:
        out["total_area_m2"] = union.area * scale ** 2
        out["frac_area_violation"] = np.mean([r["area_m2"] < rules.MIN_AREA_M2[r["type"]] for r in recs])
        out["frac_width_violation"] = np.mean([r["width_m"] < rules.MIN_WIDTH_M[r["type"]] for r in recs])
    else:
        out["total_area_m2"] = out["frac_area_violation"] = out["frac_width_violation"] = NAN
    out["any_rule_violation"] = float(
        out["overlap_ratio"] > 0.02 or out["hole_ratio"] > 0.02 or out["adj_components"] > 1
        or out["frac_aspect_violation"] > 0
        or (metric_scale_known and (out["frac_area_violation"] > 0 or out["frac_width_violation"] > 0)))
    return out, recs, g, shapes


def _type_pair_edges(g, shapes):
    return Counter(tuple(sorted((shapes[i][0], shapes[j][0]))) for i, j in g.edges)


def _multiset_f1(pred, ref):
    tp = sum((pred & ref).values())
    n_pred, n_ref = sum(pred.values()), sum(ref.values())
    if n_pred == 0 and n_ref == 0:
        return 1.0
    if n_pred == 0 or n_ref == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_ref
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def rasterize_types(shapes, bounds, res):
    """Label map (res x res) of room type indices over bounds; -1 = empty."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = bounds
    s = res / max(x1 - x0, y1 - y0, 1e-9)
    img = Image.new("I", (res, res), -1)
    draw = ImageDraw.Draw(img)
    for rtype, geom, _ in shapes:
        for part in getattr(geom, "geoms", [geom]):
            pts = [((x - x0) * s, (y - y0) * s) for x, y in part.exterior.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=ROOM_TYPES.index(rtype))
    return np.asarray(img)


def paired_metrics(gen, gt, frame_aligned, res=256):
    """Metrics comparing a generated sample with its GT (both from sample_metrics)."""
    (gs, grecs, gg, gshapes), (ts, trecs, tg, tshapes) = gen, gt
    out = {}
    n, n_gt = gs["n_rooms"], ts["n_rooms"]
    out["room_count_abs_err"] = abs(n - n_gt)
    out["room_count_match"] = float(n == n_gt)
    gtypes = Counter(s[0] for s in gshapes)
    ttypes = Counter(s[0] for s in tshapes)
    out["type_hist_l1"] = sum(((gtypes - ttypes) + (ttypes - gtypes)).values())
    out["type_multiset_match"] = float(gtypes == ttypes)
    out["adjacency_f1"] = _multiset_f1(_type_pair_edges(gg, gshapes), _type_pair_edges(tg, tshapes))

    if frame_aligned and gshapes and tshapes:
        bounds = unary_union([g for _, g, _ in tshapes]).bounds
        a = rasterize_types(gshapes, bounds, res)
        b = rasterize_types(tshapes, bounds, res)
        ious = []
        for k in set(np.unique(a)) | set(np.unique(b)):
            if k < 0:
                continue
            inter = np.sum((a == k) & (b == k))
            uni = np.sum((a == k) | (b == k))
            ious.append(inter / uni)
        out["type_miou"] = float(np.mean(ious)) if ious else NAN
        inside = b >= 0
        out["pixel_type_acc"] = float(np.mean(a[inside] == b[inside])) if inside.any() else NAN
    else:
        out["type_miou"] = out["pixel_type_acc"] = NAN
    return out
