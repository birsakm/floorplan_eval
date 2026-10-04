"""Convert ChatHouseDiffusion raw label maps (RPLAN labels 0..17, 64x64) to the common format.

  conda run -n fpe python scripts/chathousediffusion/convert.py outputs/chathousediffusion/<variant> \
      --data data/method_inputs/chathousediffusion [--gt]

Generated: <variant>/raw/label/<id>.png -> <variant>/samples/<id>.json
GT (--gt):  <data>/image_test/<id>.png     -> <variant>/gt_samples/<id>.json (+ gt_renders/)
Boundary = interior of the condition mask (<data>/mask_test/<id>.png != 255).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription on the shared machine
import argparse
from functools import partial
from multiprocessing import Pool
import json
from pathlib import Path

import cv2

cv2.setNumThreads(0)
import numpy as np

from fpeval.format import canonical_room_type, make_sample, save_sample
from fpeval.render import render

SOURCE = "chathousediffusion"
RPLAN_LABELS = ["LivingRoom", "MasterRoom", "Kitchen", "Bathroom", "DiningRoom", "ChildRoom",
                "StudyRoom", "SecondRoom", "GuestRoom", "Balcony", "Entrance", "Storage", "Wall-in",
                "External", "ExteriorWall", "FrontDoor", "InteriorWall", "InteriorDoor"]
EXTRA = {"Wall-in": "closet", "ChildRoom": "bedroom", "StudyRoom": "study", "GuestRoom": "bedroom"}
UP = 8  # contour on an 8x nearest-upsampled mask -> polygon follows pixel edges (within 1/16 px)


def mask_polygons(mask, min_px=2, eps=0.5):
    """One simplified polygon per 4-connected component of a binary mask (pixel units)."""
    polys = []
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    for k in range(1, n):
        comp = cc == k
        if comp.sum() < min_px:
            continue
        big = cv2.resize(comp.astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, eps * UP, True)[:, 0, :].astype(float)
        if len(c) < 3:
            continue
        # contour runs through pixel centres of the upsampled grid; shift to pixel edges
        c = (c + 0.5) / UP
        polys.append([[round(x, 3), round(y, 3)] for x, y in c])
    return polys


def label_to_sample(lab, variant, sid, boundary_mask, condition):
    rooms = []
    for li in range(13):
        for poly in mask_polygons(lab == li):
            native = RPLAN_LABELS[li]
            rooms.append({"type": canonical_room_type(native, EXTRA), "type_native": native, "polygon": poly})
    doors = [{"type": "front_door", "polygon": p} for p in mask_polygons(lab == 15, min_px=1)]
    doors += [{"type": "interior_door", "polygon": p} for p in mask_polygons(lab == 17, min_px=1)]
    bpolys = mask_polygons(boundary_mask, min_px=1)
    boundary = max(bpolys, key=lambda p: cv2.contourArea(np.array(p, np.float32))) if bpolys else None
    return make_sample(SOURCE, variant, sid, rooms, boundary=boundary, doors=doors, condition=condition)


def _job(p, sdir, rdir, vdir, variant, data):
    sid = p.stem
    lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    cond = cv2.imread(str(data / "mask_test" / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
    tpath = vdir / "raw/text" / f"{sid}.json"
    graph_json = json.loads(tpath.read_text()) if tpath.exists() else None
    condition = {"type": "text_graph+boundary", "ref_id": f"tell2design_{sid}",
                 "dataset": "Tell2Design test (RPLAN-based); text pre-parsed by moonshot-v1-8k (upstream)",
                 "graph_json": graph_json}
    s = label_to_sample(lab, variant if sdir == "samples" else variant + "_gt", sid, cond != 255, condition)
    save_sample(s, vdir / sdir / f"{sid}.json")
    if sdir == "gt_samples":  # samples/ are rendered with `python -m fpeval.render`
        render(s).save(vdir / rdir / f"{sid}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--data", default="data/method_inputs/chathousediffusion")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    vdir = Path(args.variant_dir)
    variant = vdir.name
    data = Path(args.data)
    jobs = [("samples", "renders", sorted((vdir / "raw/label").glob("*.png")))]
    if args.gt:
        ids = {p.stem for p in jobs[0][2]}
        jobs.append(("gt_samples", "gt_renders", sorted(p for p in (data / "image_test").glob("*.png") if p.stem in ids)))
    for sdir, rdir, files in jobs:
        if sdir == "gt_samples":
            (vdir / rdir).mkdir(parents=True, exist_ok=True)
        with Pool(args.workers) as pool:
            pool.map(partial(_job, sdir=sdir, rdir=rdir, vdir=vdir, variant=variant, data=data), files, chunksize=32)
        print(f"{sdir}: {len(files)}")


if __name__ == "__main__":
    main()
