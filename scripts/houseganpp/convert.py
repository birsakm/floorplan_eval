"""Convert House-GAN++ raw outputs (raw/<id>.npz with 64x64 masks per input node) to the common format.

Vectorization mirrors HG++'s own draw_masks(): each node mask (logits in [-1,1]) is thresholded at 0,
upsampled to 256 px (cv2 INTER_AREA, >=0.5), and rooms are painted in node order (later nodes
overwrite earlier ones). Each room's visible region is split into connected components and traced
with cv2.findContours (+ approxPolyDP, eps=1 px). Door nodes (15 front door, 17 interior door) are
traced from their own masks (largest component) into `doors`.
Also writes gt_samples/ from the input JSONs (same centred 256 px frame).

Usage (fpe env, repo root on PYTHONPATH):
  python scripts/houseganpp/convert.py outputs/houseganpp/<variant>
"""
import argparse
from multiprocessing import Pool
import os
import sys
from pathlib import Path

import cv2

cv2.setNumThreads(1)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hg_json  # noqa: E402
from fpeval.format import save_sample  # noqa: E402

SRC = "houseganpp"


def trace(binary, min_area=4.0, largest_only=False):
    cs, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_area]
    cs.sort(key=cv2.contourArea, reverse=True)
    if largest_only:
        cs = cs[:1]
    out = []
    for c in cs:
        c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :]
        if len(c) >= 3:
            # pixel-index contour -> pixel-edge coordinates (+0.5 centre offset is negligible; keep px)
            out.append([[float(x), float(y)] for x, y in c])
    return out


def masks_to_polys(masks, types, size=256):
    up = []
    for m in masks.astype(np.float32):
        b = (m > 0).astype(np.float32)
        up.append(cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA) >= 0.5)
    label = np.full((size, size), -1, int)
    for k, t in enumerate(types):
        if t not in hg_json.DOOR_TYPES:
            label[up[k]] = k
    polys = []
    for k, t in enumerate(types):
        if t in hg_json.DOOR_TYPES:
            polys.append(trace(up[k], largest_only=True))
        else:
            polys.append(trace(label == k))
    return polys


def _one(job):
    p, root, variant = job
    r = np.load(p, allow_pickle=True)
    types = [int(t) for t in r["room_type"]]
    in_json = str(r["input_json"])
    ref_id = os.path.splitext(os.path.basename(in_json))[0]
    h = hg_json.load(in_json)
    assert h["types"] == types, p
    polys = masks_to_polys(r["masks"], types)
    s = hg_json.assemble(SRC, variant, p.stem, polys, types, h["node_edges"], ref_id,
                         extra_cond={"frame_shift_px": h["shift"]})
    save_sample(s, root / "samples" / f"{p.stem}.json")
    return ref_id, in_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("CONVERT_WORKERS", "8")),
                    help="process pool size (<=16); each worker single-threaded (OMP_NUM_THREADS=1)")
    a = ap.parse_args()
    root = Path(a.variant_dir)
    variant = root.name
    jobs = [(p, root, variant) for p in sorted((root / "raw").glob("*.npz"))]
    with Pool(min(a.workers, 16)) as pool:
        res = pool.map(_one, jobs, chunksize=16)
    gts = dict(res)
    for ref_id, in_json in gts.items():
        save_sample(hg_json.gt_sample("rplan", variant, in_json, ref_id), root / "gt_samples" / f"{ref_id}.json")
    print(f"converted {len(res)} samples, {len(gts)} gt layouts -> {root}")


if __name__ == "__main__":
    main()
