"""Convert iPLAN raw outputs (raw/<name>.npz from scripts/iplan/infer.py) to the common format.

iPLAN works on a 128x128 raster; each room instance (living room = remainder region, other
rooms = predicted boxes clipped to the interior, painted in order) is vectorized per connected
component and scaled x2 into the 256 px RPLAN / Graph2Plan frame. The input boundary is the
Graph2Plan boundary (256 px). Ground truth for the same ids goes to gt_samples/.

    PYTHONPATH=. conda run -n fpe python scripts/iplan/convert.py outputs/iplan/<variant> \
        --g2p_mat data/method_inputs/diffplanner/graph2plan_data/Network/data/data_test.mat
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from rplan_g2p import (load_g2p, rplan_type, g2p_boundary_xy, g2p_front_door, g2p_condition,  # noqa: E402
                       gt_sample, mask_polygons, make_sample, save_sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--g2p_mat", required=True)
    ap.add_argument("--no_gt", action="store_true")
    args = ap.parse_args()
    root = Path(args.variant_dir)
    variant = root.name
    g2p = load_g2p(args.g2p_mat)
    n = 0
    for f in sorted((root / "raw").glob("*.npz"), key=lambda p: int(p.stem)):
        r = np.load(f)
        name = str(r["name"])
        inst, types = r["inst"], r["inst_types"]
        rooms = []
        for k, t in enumerate(types):
            ctype, native = rplan_type(t)
            for poly in mask_polygons(inst == k, scale=2.0, min_pixels=2):
                rooms.append({"type": ctype, "type_native": native, "polygon": poly})
        d = g2p[name]
        s = make_sample("iplan", variant, name, rooms, units="px", boundary=g2p_boundary_xy(d.boundary),
                        doors=[g2p_front_door(d.boundary)], condition=g2p_condition(name))
        save_sample(s, root / "samples" / f"{name}.json")
        if not args.no_gt:
            save_sample(gt_sample(d, "rplan_gt", variant), root / "gt_samples" / f"{name}.json")
        n += 1
    print(f"converted {n} samples -> {root / 'samples'}")


if __name__ == "__main__":
    main()
