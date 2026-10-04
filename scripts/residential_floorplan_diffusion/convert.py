"""Convert Residential Floorplan Diffusion stage-2 outputs (64x64 RGB) to the common format.

  conda run -n fpe python scripts/residential_floorplan_diffusion/convert.py outputs/residential_floorplan_diffusion/<variant>

raw/stage2/<id>.png -> samples/<id>.json. Pixels are assigned to the nearest palette colour
(RPLAN-toolbox colours used by the bundled condition images; walls are salmon in stage 2, see
test/stage2_psnr.png); each room class is vectorized per 4-connected component. Boundary = outer
contour of all non-background pixels. NOTE: written against the reference image in the repo only;
the model weights were not accessible, so this has not been run on real model outputs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # avoid thread oversubscription on the shared machine
import argparse
import json
from pathlib import Path

import cv2

cv2.setNumThreads(0)
import numpy as np

from fpeval.format import canonical_room_type, make_sample, save_sample

SOURCE = "residential_floorplan_diffusion"
PALETTE = {  # name -> RGB
    "background": (255, 255, 255),
    "living_room": (244, 242, 229),
    "bedroom": (253, 244, 171),
    "kitchen": (234, 216, 214),
    "toilet": (205, 233, 252),
    "balcony": (208, 216, 135),
    "wall": (250, 190, 175),
}
ROOMS = ["living_room", "bedroom", "kitchen", "toilet", "balcony"]
UP = 8


def mask_polygons(mask, min_px=2, eps=0.5):
    polys = []
    n, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    for k in range(1, n):
        comp = cc == k
        if comp.sum() < min_px:
            continue
        big = cv2.resize(comp.astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps * UP, True)[:, 0, :].astype(float)
        if len(c) >= 3:
            polys.append([[round(x, 3), round(y, 3)] for x, y in (c + 0.5) / UP])
    return polys


def convert(png, variant, cond):
    rgb = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB).astype(float)
    names = list(PALETTE)
    pal = np.array([PALETTE[n] for n in names], float)
    cls = np.argmin(((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1), -1)
    rooms = []
    for n in ROOMS:
        # 2x2 opening removes 1-px slivers (anti-aliased salmon wall pixels close to kitchen colour)
        m = cv2.morphologyEx((cls == names.index(n)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        for poly in mask_polygons(m, min_px=4):
            rooms.append({"type": canonical_room_type(n), "type_native": n, "polygon": poly})
    fg = mask_polygons(cls != names.index("background"), min_px=1)
    boundary = max(fg, key=lambda p: cv2.contourArea(np.array(p, np.float32))) if fg else None
    return make_sample(SOURCE, variant, png.stem, rooms, boundary=boundary, condition=cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    args = ap.parse_args()
    vdir = Path(args.variant_dir)
    meta = json.loads((vdir / "raw/meta.json").read_text()) if (vdir / "raw/meta.json").exists() else {}
    files = sorted((vdir / "raw/stage2").glob("*.png"))
    for p in files:
        c = str(int(p.stem.split("_")[0][1:]))
        m = meta.get(c, {})
        cond = {"type": "per_type_room_masks (living, bedroom, kitchen, toilet, balcony)",
                "ref_id": f"resdiff_test_{c}", "inputs": m.get("inputs"),
                "target_counts_living_bed_toilet_kitchen_balcony": m.get("target_counts_living_bed_toilet_kitchen_balcony")}
        save_sample(convert(p, vdir.name, cond), vdir / "samples" / f"{p.stem}.json")
    print("samples:", len(files))


if __name__ == "__main__":
    main()
