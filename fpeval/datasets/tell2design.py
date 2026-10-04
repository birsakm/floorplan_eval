"""Convert the Tell2Design floorplan images (RPLAN-format PNGs) to the common format.

Tell2Design (ACL 2023, CC BY-NC 4.0) ships 80,788 floorplan images
(General Data/floorplan_image/<id>.png) derived from RPLAN (the count equals RPLAN's),
rendered as 256x256 RGB with a fixed per-type colour palette (see COLORS), white interior
walls, black exterior walls and a red front door. Interior doors are NOT drawn.
Raw data: data/tell2design/raw/ (scripts/datasets/download_tell2design.sh).

T2D merges RPLAN's child/second/guest rooms into "common room" (-> bedroom); RPLAN study,
entrance and wall-in have no separate colour, except for a rare yellow class we could not
identify (-> other, type_native "unknown_yellow").

Vectorization: one polygon per 4-connected component of each room colour, traced with
cv2.findContours + approxPolyDP (eps 1 px); rooms are the wall-free room interiors.
Boundary: contour of the non-white area. Front door: minAreaRect of the red component.
Graph: rooms within ~4 px (one wall) of each other -> "adjacent" (no door info available).
Units are pixels of the 256x256 frame (no metric scale is published).

Splits: Tell2Design's eval set (Separated Data/eval_data, human instructions) -> "test";
its 2nd-finetune set (human instructions) -> "train_human"; all others -> "train".

    python -m fpeval.datasets.tell2design [--limit N] [--workers 16]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2

cv2.setNumThreads(1)
import numpy as np

from ..format import canonical_room_type, make_sample, save_sample
from ._common import DATA, out_dirs, write_splits

RAW = DATA / "tell2design" / "raw" / "Tell2Design Data"
# Colour -> Tell2Design room label, recovered by sampling the T5 annotation box centres of
# the 2,308 eval plans (each colour matched one label in >97% of cases).
COLORS = {
    (238, 232, 170): "living room", (255, 165, 0): "master room", (255, 215, 0): "common room",
    (240, 128, 128): "kitchen", (173, 216, 230): "bathroom", (107, 142, 35): "balcony",
    (218, 112, 214): "dining room", (221, 160, 221): "storage",
    (255, 255, 0): "unknown_yellow",  # rare, room-sized, never in the annotated sets
}
FRONT_DOOR = (255, 0, 0)
EXTRA = {"master room": "bedroom", "common room": "bedroom", "unknown_yellow": "other"}


def _contours(mask, eps=1.0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 4:
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            # pixel-corner coordinates: contour runs through pixel centres; shift by 0.5
            out.append([[float(x) + 0.5, float(y) + 0.5] for x, y in a])
    return out


def _rects(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    out = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 2:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)))
        out.append(([[float(x) + 0.5, float(y) + 0.5] for x, y in box], lab == k))
    return out


def _key(c):
    return (c[0] << 16) | (c[1] << 8) | c[2]


def convert_one(args):
    path, split, root = args
    from PIL import Image
    im = np.array(Image.open(path).convert("RGB")).astype(np.int32)
    code = (im[..., 0] << 16) | (im[..., 1] << 8) | im[..., 2]
    rooms, masks = [], []
    for col, native in COLORS.items():
        m = code == _key(col)
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
        for j in range(1, n):
            comp = lab == j
            if comp.sum() < 9:
                continue
            for poly in _contours(comp):
                rooms.append({"type": canonical_room_type(native, EXTRA), "type_native": native,
                              "polygon": poly})
                masks.append(comp)
    if not rooms:
        return None
    k5 = np.ones((5, 5), np.uint8)
    near = [cv2.dilate(m.astype(np.uint8), k5) > 0 for m in masks]
    boxes = []
    for m in near:
        ys, xs = np.nonzero(m)
        boxes.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    edges = set()
    for a in range(len(masks)):
        ya0, ya1, xa0, xa1 = boxes[a]
        for b in range(a + 1, len(masks)):
            yb0, yb1, xb0, xb1 = boxes[b]
            y0, y1, x0, x1 = max(ya0, yb0), min(ya1, yb1), max(xa0, xb0), min(xa1, xb1)
            if y0 >= y1 or x0 >= x1:
                continue
            if (near[a][y0:y1, x0:x1] & near[b][y0:y1, x0:x1]).sum() >= 3:
                edges.add((a, b, "adjacent"))
    doors, front = [], None
    for poly, dm in _rects(code == _key(FRONT_DOOR)):
        doors.append({"type": "front_door", "polygon": poly})
        front = [i for i, d in enumerate(near) if (d & dm).any()]
    inside = (code != _key((255, 255, 255))).astype(np.uint8)
    inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, k5) > 0
    b = _contours(inside)
    sid = Path(path).stem
    sample = make_sample(
        "tell2design", "rplan_images", sid, rooms, units="px",
        boundary=max(b, key=lambda r: cv2.contourArea(np.array(r, np.float32))) if b else None,
        doors=doors, graph={"edges": [list(e) for e in sorted(edges)]},
        condition={"split": split, "front_door_rooms": front, "frame": "256x256 px (RPLAN-derived)"})
    save_sample(sample, root / "samples" / f"{sid}.json")
    return sid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    raw = Path(args.raw)
    sep = raw / "Separated Data"
    test_ids = {p.stem for p in (sep / "eval_data" / "annotated_imgs").glob("*.png")}
    human_ids = {p.stem for p in (sep / "2nd_finetune_data" / "annotated_imgs").glob("*.png")}
    files = sorted((raw / "General Data" / "floorplan_image").glob("*.png"), key=lambda p: int(p.stem))
    files = files[: args.limit] if args.limit else files
    root = out_dirs("tell2design", args.out)

    def split_of(stem):
        return "test" if stem in test_ids else "train_human" if stem in human_ids else "train"

    splits, failed = {}, 0
    with Pool(args.workers) as pool:
        for sid in pool.imap_unordered(convert_one, ((f, split_of(f.stem), root) for f in files), chunksize=64):
            if sid is None:
                failed += 1
            else:
                splits[sid] = split_of(sid)
    counts = write_splits(root, splits, {"note": "test = Tell2Design eval_data/annotated_imgs ids; train_human = 2nd_finetune_data/annotated_imgs ids; rest train"})
    print("converted", len(splits), "failed", failed, counts)


if __name__ == "__main__":
    main()
