"""House-GAN++ inference driver (mirrors external/methods/houseganpp/test.py).

Differences from test.py: saves the raw 64x64 generator masks (+ input graph) per sample as
.npz instead of only a PNG, supports several samples per input graph (--n_per_graph) and a
fixed seed. The generation procedure itself (initial pass + iterative refinement fixing room
types in sorted order, 10 rounds) is identical to test.py.

Run from inside external/methods/houseganpp (imports its modules), e.g.
  python scripts/houseganpp/infer.py --list data_list.txt --out outputs/houseganpp/<v>/raw
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms

HG_DIR = os.environ.get("HG_DIR", os.getcwd())
sys.path.insert(0, HG_DIR)
from dataset.floorplan_dataset_maps_functional_high_res import (  # noqa: E402
    FloorplanGraphDataset, floorplan_collate_fn)
from misc.utils import _init_input, draw_masks  # noqa: E402
from models.models import Generator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=os.path.join(HG_DIR, "checkpoints/pretrained.pth"))
ap.add_argument("--list", required=True, help="text file with one HG++ json path per line")
ap.add_argument("--out", required=True)
ap.add_argument("--n_per_graph", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save_png", action="store_true", help="also save the native HG++ rendering")
opt = ap.parse_args()
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))  # avoid CPU oversubscription
os.makedirs(opt.out, exist_ok=True)
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

model = Generator()
model.load_state_dict(torch.load(opt.checkpoint, map_location="cpu"), strict=True)
model = model.eval().cuda()

paths = [l.strip() for l in open(opt.list) if l.strip()]
ds = FloorplanGraphDataset(opt.list, transforms.Normalize(mean=[0.5], std=[0.5]), split="test")
assert len(ds) == len(paths)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=floorplan_collate_fn)


def _infer(graph, prev_state):
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    with torch.no_grad():
        masks = model(z.cuda(), given_masks_in.cuda(), given_nds.cuda(), given_eds.cuda())
    return masks.detach().cpu().numpy()


t0 = time.time()
n_done = 0
for i, sample in enumerate(loader):
    mks, nds, eds, _, _ = sample
    real_nodes = np.where(nds.detach().cpu() == 1)[-1]  # type index = HG++ room_type - 1
    graph = [nds, eds]
    in_id = os.path.splitext(os.path.basename(paths[i]))[0]
    _types = sorted(list(set(real_nodes)))
    selected_types = [_types[:k + 1] for k in range(10)]
    for s in range(opt.n_per_graph):
        state = {"masks": None, "fixed_nodes": []}
        masks = _infer(graph, state)
        for _types_k in selected_types:
            _fixed = np.concatenate([np.where(real_nodes == t)[0] for t in _types_k]) \
                if len(_types_k) > 0 else np.array([])
            state = {"masks": masks, "fixed_nodes": _fixed}
            masks = _infer(graph, state)
        sid = f"{in_id}_{s}"
        np.savez_compressed(os.path.join(opt.out, f"{sid}.npz"), masks=masks.astype(np.float16),
                            room_type=real_nodes + 1, edges=eds.numpy(), input_json=paths[i])
        if opt.save_png:
            draw_masks(masks.copy(), real_nodes).save(os.path.join(opt.out, f"{sid}.png"))
        n_done += 1
    if i % 50 == 0:
        print(f"[{i+1}/{len(paths)}] {n_done} samples, {time.time()-t0:.1f}s", flush=True)

info = {"n_inputs": len(paths), "n_samples": n_done, "seconds": time.time() - t0}
json.dump(info, open(os.path.join(opt.out, "_infer_info.json"), "w"), indent=1)
print(info)
