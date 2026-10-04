"""House-GAN batched inference on the held-out LIFULL test group (default: target set D, 10-12 rooms).

Mirrors variation_bbs_with_target_graph_segments_suppl.py (split='eval' of FloorplanGraphDataset on
train_data.npy, first 5000 graphs of the held-out group) but batches graphs and saves the raw
32x32 generator masks instead of montage images. Submodule code is imported unmodified.

Raw output: one .npz per test graph with
  masks     (N,32,32) float16  generator output in [-1,1] (room present where > 0)
  nodes     (N,)      int      House-GAN room class (1..10, see ROOM_CLASS)
  edges     (E,3)     int      [k, +1/-1, l] input bubble-diagram triples (+1 = adjacent)
  gt_bbs    (N,4)     float    GT boxes (normalized, dataset convention: mask[x0:x1, y0:y1])
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

HG = os.path.join(os.path.dirname(__file__), "..", "..", "external", "methods", "housegan")
sys.path.insert(0, os.path.abspath(HG))
# utils.py does `from pygraphviz import *` (only used for drawing bubble diagrams); stub it to avoid a
# graphviz system dependency.
import types  # noqa: E402
sys.modules.setdefault("pygraphviz", types.ModuleType("pygraphviz"))
import torchvision.transforms as transforms  # noqa: E402
from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn  # noqa: E402
from models import Generator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing train_data.npy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_set", default="D")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_variations", type=int, default=1)
    ap.add_argument("--max_graphs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    opt = ap.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    os.makedirs(opt.out_dir, exist_ok=True)

    gen = Generator().cuda().eval()
    gen.load_state_dict(torch.load(opt.checkpoint, map_location="cuda"))

    ds = FloorplanGraphDataset(opt.data_dir, transforms.Normalize(mean=[0.5], std=[0.5]),
                               target_set=opt.target_set, split="eval")
    n = min(len(ds), opt.max_graphs)
    t0 = time.time()
    for start in range(0, n, opt.batch_size):
        idx = list(range(start, min(n, start + opt.batch_size)))
        items = [ds[i] for i in idx]
        # GT boxes (same preprocessing as __getitem__: /256 and centred)
        gts = []
        for i in idx:
            bbs = np.stack(ds.subgraphs[i][1]).astype(np.float64) / 256.0
            tl, br = bbs[:, :2].min(0), bbs[:, 2:].max(0)
            shift = (tl + br) / 2.0 - 0.5
            bbs[:, :2] -= shift
            bbs[:, 2:] -= shift
            gts.append(bbs)
        mks, nds, eds, nd_to_sample, _ = floorplan_collate_fn(items)
        nds_c, eds_c = nds.cuda(), eds.cuda()
        nd_to_sample = nd_to_sample.numpy()
        for v in range(opt.num_variations):
            z = torch.randn(nds.shape[0], 128, device="cuda")
            with torch.no_grad():
                gen_mks = gen(z, nds_c, eds_c).cpu().numpy()
            for b, i in enumerate(idx):
                sel = nd_to_sample == b
                nodes = np.where(nds[sel].numpy() == 1)[-1] + 1
                np.savez_compressed(
                    os.path.join(opt.out_dir, f"{i:05d}_v{v}.npz"),
                    masks=gen_mks[sel].astype(np.float16), nodes=nodes,
                    edges=items[b][2].numpy(), gt_bbs=gts[b])
        print(f"{min(n, start + opt.batch_size)}/{n} graphs, {time.time() - t0:.1f}s", flush=True)
    print(f"done: {n} graphs x {opt.num_variations} variations in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
