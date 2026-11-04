"""Render common-format floorplans to PNG with a fixed per-type palette."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .format import ROOM_TYPES, load_sample

PALETTE = {
    "living_room": (238, 77, 77), "kitchen": (192, 192, 224), "bedroom": (255, 165, 0),
    "bathroom": (65, 105, 225), "balcony": (35, 139, 34), "entrance": (220, 220, 50),
    "dining_room": (200, 120, 200), "study": (139, 90, 43), "storage": (150, 150, 150),
    "corridor": (240, 220, 180), "garage": (90, 90, 120), "laundry": (100, 200, 200),
    "closet": (180, 140, 100), "outdoor": (150, 220, 150), "other": (210, 210, 210),
}
assert set(PALETTE) == set(ROOM_TYPES)

# Types drawn in one colour by render_layout: the datasets split these differently.
LAYOUT_MERGE = {"entrance": "corridor", "closet": "storage", "outdoor": "balcony"}


def render(sample, size=512, margin=16, rooms_only=False):
    """Render a sample. rooms_only=True skips boundary/doors/windows."""
    pts = [p for r in sample["rooms"] for p in r["polygon"]]
    if sample.get("boundary") and not rooms_only:
        pts += sample["boundary"]
    img = Image.new("RGB", (size, size), (255, 255, 255))
    if not pts:
        return img
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = min(xs), min(ys)
    extent = max(max(xs) - x0, max(ys) - y0, 1e-9)
    s = (size - 2 * margin) / extent

    def tf(poly):
        return [((x - x0) * s + margin, (y - y0) * s + margin) for x, y in poly]

    draw = ImageDraw.Draw(img)
    if sample.get("boundary") and not rooms_only:
        draw.polygon(tf(sample["boundary"]), fill=(245, 245, 245), outline=(0, 0, 0))
    for room in sample["rooms"]:
        if len(room["polygon"]) >= 3:
            draw.polygon(tf(room["polygon"]), fill=PALETTE[room["type"]], outline=(0, 0, 0))
    if rooms_only:
        return img
    for door in sample.get("doors") or []:
        draw.polygon(tf(door["polygon"]), fill=(255, 255, 255), outline=(60, 60, 60))
    for win in sample.get("windows") or []:
        draw.polygon(tf(win["polygon"]), fill=(150, 220, 255), outline=(0, 0, 255))
    return img


def render_layout(sample, size=256, margin=8, wall_px=4, close=0.3, merge=LAYOUT_MERGE, align=True):
    """Uniform raster of the room layout for FID/KID, the same for every method and dataset.

    House-GAN / GRE-Diff style: white background, rooms filled with their type colour in decreasing area order,
    black walls of wall_px pixels on every room boundary. Rooms keep their true shape; wall gaps between rooms of
    up to 2 * close (sample units) are filled, each gap pixel going to the nearest room, so the boundary lies on the
    wall centreline whatever wall thickness a source draws. merge maps room types to the type whose colour they get.
    align rotates the plan so its dominant wall direction (length-weighted, modulo 90°) is axis-parallel, as in
    RPLAN-style data.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt
    from shapely import affinity
    from shapely.geometry import Polygon

    merge = merge or {}
    polys = [(Polygon(r["polygon"]).buffer(0), r["type"]) for r in sample["rooms"] if len(r["polygon"]) >= 3]
    edges = [np.diff(np.asarray(p.exterior.coords), axis=0) for p, _ in polys
             if p.geom_type == "Polygon" and not p.is_empty]
    if align and edges:
        # Wall direction (modulo 90°) with the most edge length; plans with walls in several directions keep
        # the main one axis-parallel instead of averaging between them.
        e = np.concatenate(edges)
        w, th = np.hypot(*e.T), np.degrees(np.arctan2(e[:, 1], e[:, 0])) % 90
        hist = np.bincount(np.round(th).astype(int) % 90, weights=w, minlength=90)
        peak = np.argmax(hist + np.roll(hist, 1) + np.roll(hist, -1))
        d = (th - peak + 45) % 90 - 45  # offset from the peak in (-45, 45]
        near = np.abs(d) <= 2
        rot = peak + np.average(d[near], weights=w[near]) if near.any() else 0.0
        polys = [(affinity.rotate(p, -rot, origin=(0, 0)), t) for p, t in polys]
    rooms = [(p, merge.get(t, t)) for p0, t in polys for p in getattr(p0, "geoms", [p0]) if p.area > 0]
    labels = Image.new("I", (size, size), 0)
    s = 1.0
    if rooms:
        x0, y0, x1, y1 = np.array([p.bounds for p, _ in rooms]).T
        x0, y0 = x0.min(), y0.min()
        s = (size - 2 * margin) / max(x1.max() - x0, y1.max() - y0, 1e-9)
        draw = ImageDraw.Draw(labels)
        rooms.sort(key=lambda r: -r[0].area)
        for k, (p, _) in enumerate(rooms):
            draw.polygon([((x - x0) * s + margin, (y - y0) * s + margin) for x, y in p.exterior.coords], fill=k + 1)
    lab = np.array(labels)
    gap_px = 2 * close * s
    if gap_px > 0 and rooms:
        # A background pixel is wall gap when it lies between two rooms: their nearest points are less than
        # gap_px apart in total and in roughly opposite directions (so outer corners are not filled). It takes
        # the nearest room.
        pix = np.indices(lab.shape)
        dist, vec = [], []  # per room: distance to it and vector from its nearest point
        for k in range(1, len(rooms) + 1):
            d, i = distance_transform_edt(lab != k, return_indices=True)
            dist.append(d)
            vec.append(pix - i)
        dist, vec = np.stack(dist), np.stack(vec)
        best = np.argmin(dist, axis=0)
        d1 = np.take_along_axis(dist, best[None], 0)[0]
        v1 = np.take_along_axis(vec, best[None, None], 0)[0]
        cos = (vec * v1).sum(1) / np.maximum(dist * d1, 1e-9)
        between = (np.arange(len(rooms))[:, None, None] != best) & (dist + d1 <= gap_px) & (cos <= -0.5)
        gap = (lab == 0) & between.any(0)
        lab[gap] = best[gap] + 1
    # Pixels on both sides of every label change (rooms and background), then widened to wall_px.
    dv, dh = lab[1:] != lab[:-1], lab[:, 1:] != lab[:, :-1]
    edge = np.zeros(lab.shape, bool)
    edge[1:] |= dv
    edge[:-1] |= dv
    edge[:, 1:] |= dh
    edge[:, :-1] |= dh
    if wall_px > 2:
        edge = binary_dilation(edge, np.ones((wall_px - 1, wall_px - 1), bool))
    colours = np.array([(255, 255, 255)] + [PALETTE[t] for _, t in rooms], np.uint8)
    rgb = colours[lab]
    rgb[edge] = 0
    return Image.fromarray(rgb)


def main():
    ap = argparse.ArgumentParser(description="Render samples/*.json to renders/*.png")
    ap.add_argument("sample_dir", help="directory containing samples/ (e.g. outputs/<m>/<v>)")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.sample_dir)
    out = root / "renders"
    out.mkdir(exist_ok=True)
    for p in sorted((root / "samples").glob("*.json")):
        render(load_sample(p), args.size).save(out / f"{p.stem}.png")


if __name__ == "__main__":
    main()
