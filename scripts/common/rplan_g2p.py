"""Shared helpers for the boundary-conditioned RPLAN methods (iPLAN, DiffPlanner, WallPlan).

All three are run on boundaries from the Graph2Plan-preprocessed RPLAN test split
(Data.zip -> Network/data/data_test.mat; same 256 px frame as RPLAN, (x, y) with y down),
and all outputs are converted into that frame so samples are directly comparable.

Run in the `fpe` env with the repo root on PYTHONPATH.
"""
import numpy as np
import cv2
import scipy.io as sio

from fpeval.format import canonical_room_type, make_sample, save_sample

# RPLAN room label ids (also used by Graph2Plan rType and iPLAN rTypes)
RPLAN_LABELS = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom", "StudyRoom",
                "SecondRoom", "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in"]
RPLAN_EXTRA = {"Wall-in": "closet"}


def rplan_type(label_id):
    native = RPLAN_LABELS[int(label_id)]
    return canonical_room_type(native, RPLAN_EXTRA), native


def load_g2p(mat_path):
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["data"]
    return {str(d.name): d for d in data}


def g2p_boundary_xy(boundary):
    """Graph2Plan boundary rows (x, y, dir, isNew) -> [[x, y], ...] (all points, incl. door points)."""
    b = np.asarray(boundary)
    return [[int(x), int(y)] for x, y in b[:, :2]]


def seg_rect(p0, p1, half_width):
    """Axis-aligned rectangle around segment p0-p1 (points [x, y]) with given half thickness."""
    (x0, y0), (x1, y1) = p0, p1
    if abs(y1 - y0) <= abs(x1 - x0):  # horizontal
        xa, xb = sorted([x0, x1])
        return [[xa, y0 - half_width], [xb, y0 - half_width], [xb, y0 + half_width], [xa, y0 + half_width]]
    ya, yb = sorted([y0, y1])
    return [[x0 - half_width, ya], [x0 + half_width, ya], [x0 + half_width, yb], [x0 - half_width, yb]]


def g2p_front_door(boundary, half_width=2):
    b = np.asarray(boundary)
    return {"type": "front_door", "polygon": seg_rect(b[0, :2].tolist(), b[1, :2].tolist(), half_width)}


def g2p_condition(name):
    return {"type": "boundary", "ref_id": f"rplan_{name}",
            "dataset": "RPLAN (Graph2Plan-preprocessed test split)"}


def gt_sample(d, source, variant):
    """Ground-truth layout of one Graph2Plan record in the common format (256 px frame)."""
    rooms = []
    rb = d.rBoundary if isinstance(d.rBoundary, np.ndarray) and d.rBoundary.dtype == object else [d.rBoundary]
    for t, poly in zip(np.atleast_1d(d.rType), rb):
        poly = np.asarray(poly, dtype=float).reshape(-1, 2)
        if len(poly) < 3:
            continue
        ctype, native = rplan_type(t)
        rooms.append({"type": ctype, "type_native": native, "polygon": poly.tolist()})
    edges = []
    rEdge = np.asarray(d.rEdge).reshape(-1, 3) if np.size(d.rEdge) else np.zeros((0, 3), int)
    for a, b_, _ in rEdge:
        if a < len(rooms) and b_ < len(rooms):
            edges.append([int(a), int(b_), "adjacent"])
    return make_sample(source, variant, str(d.name), rooms, units="px",
                       boundary=g2p_boundary_xy(d.boundary), doors=[g2p_front_door(d.boundary)],
                       graph={"edges": edges}, condition=g2p_condition(str(d.name)))


def mask_polygons(mask, scale=1.0, offset=(0.0, 0.0), min_pixels=4, up=4):
    """Binary raster mask -> list of polygons ([[x, y], ...]) tracing pixel edges, one per
    connected component (holes dropped). Pixel (r, c) covers [c, c+1) x [r, r+1); output
    coordinates are x = c * scale + offset[0], y = r * scale + offset[1]."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=4)
    polys = []
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        if comp.sum() < min_pixels:
            continue
        big = cv2.resize(comp, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea).reshape(-1, 2)
        # contour runs through boundary pixels of the upsampled mask: left/top pixels sit at
        # up*c, right/bottom ones at up*c+up-1, so round(u/up) gives the exact pixel edges.
        c = np.round(c / up).astype(np.int32)
        keep = [i for i in range(len(c)) if not np.array_equal(c[i], c[i - 1])]
        c = c[keep]
        if len(c) >= 3:
            c = cv2.approxPolyDP(c.reshape(-1, 1, 2), 0.1, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        polys.append([[float(x * scale + offset[0]), float(y * scale + offset[1])] for x, y in c])
    return polys


__all__ = ["RPLAN_LABELS", "rplan_type", "load_g2p", "g2p_boundary_xy", "g2p_front_door", "g2p_condition",
           "gt_sample", "mask_polygons", "seg_rect", "save_sample", "make_sample"]
