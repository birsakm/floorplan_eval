"""Write the images FID/KID were computed on to disk, for looking at them side by side.

Every plan in a FID feature cache (fid_features*.npz under outputs/*/*/eval and data/*/converted/eval) is rendered
again with fpeval.distribution._render_file, i.e. exactly as for the features, to
    <dir>/samples/<id>.json    -> <dir>/fid_renders/<id>.png
    <dir>/gt_samples/<id>.json -> <dir>/gt_fid_renders/<id>.png
results/fid_renders.csv lists which images went into which feature set (cache), so a set can be looked at as a whole.
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fpeval.distribution import _render_file
from fpeval.eval_config import REFERENCES, VARIANTS

ROOT = Path(__file__).resolve().parents[2]


def scale_of(src_dir):
    """m per unit of the plans in a samples/ or gt_samples/ directory."""
    d = src_dir.parent.relative_to(ROOT).as_posix()
    if d.startswith("outputs/"):
        return VARIANTS[d[len("outputs/"):]]["scale"]
    return next(r[2] for r in REFERENCES.values() if r[0] == d)


def out_path(p):
    p = Path(p)
    return p.parent.parent / {"samples": "fid_renders", "gt_samples": "gt_fid_renders"}[p.parent.name] / f"{p.stem}.png"


def _job(args):
    src, scale, dst = args
    dst.parent.mkdir(exist_ok=True)
    _render_file((src, scale)).save(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    caches = sorted([*ROOT.glob("outputs/*/*/eval/fid_features*.npz"), *ROOT.glob("data/*/converted/eval/*/fid_features*.npz")])
    rows, files = [], set()
    for c in caches:
        for k in np.load(c).keys():
            rows.append((c.relative_to(ROOT).as_posix(), Path(k).relative_to(ROOT).as_posix(),
                         out_path(k).relative_to(ROOT).as_posix()))
            files.add(k)
    jobs = [(f, scale_of(Path(f).parent), out_path(f)) for f in sorted(files)]
    jobs = [j for j in jobs if a.overwrite or not j[2].exists()]
    print(f"{len(caches)} feature sets, {len(files)} plans, {len(jobs)} to render")
    with Pool(a.workers) as pool:
        pool.map(_job, jobs, chunksize=16)
    with open(ROOT / "results" / "fid_renders.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "sample", "render"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
