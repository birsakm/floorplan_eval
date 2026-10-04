"""Convert HouseDiffusion raw outputs (raw/<id>.npz, see infer.py) to the common format.

HouseDiffusion directly outputs one vector polygon per input node in the centred 256 px RPLAN frame
(x right, y down). Room nodes (HD types 1-10) -> rooms, door nodes (11 front door, 12 interior
door) -> doors. Consecutive duplicate corners are merged; polygons with <3 distinct corners are
dropped (recorded in condition.missing_nodes). Polygons are kept as generated (no snapping /
validity repair). Also writes gt_samples/ from the input JSONs (same frame).

Usage (fpe env, repo root on PYTHONPATH):
  python scripts/house_diffusion/convert.py outputs/house_diffusion/<variant>
"""
import argparse
from multiprocessing import Pool
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "houseganpp"))
import hg_json  # noqa: E402
from fpeval.format import save_sample  # noqa: E402

SRC = "house_diffusion"
HD_TO_HG = {11: 15, 12: 17, 13: 16}


def clean(poly):
    out = []
    for x, y in poly:
        p = [round(float(x), 3), round(float(y), 3)]
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out if len({tuple(p) for p in out}) >= 3 else []


def _one(job):
    p, root, variant = job
    r = np.load(p, allow_pickle=True)
    types = [HD_TO_HG.get(int(t), int(t)) for t in r["types"]]
    in_json = str(r["input_json"])
    ref_id = os.path.splitext(os.path.basename(in_json))[0]
    h = hg_json.load(in_json)
    assert h["types"] == types, (p, h["types"], types)
    polys = [[q] if q else [] for q in (clean(pp) for pp in r["polys"])]
    s = hg_json.assemble(SRC, variant, p.stem, polys, types, h["node_edges"], ref_id,
                         extra_cond={"num_corners": [len(pp) for pp in r["polys"]],
                                     "frame_shift_px": h["shift"]})
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
