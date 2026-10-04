"""HouseDiffusion inference driver.

Re-implements the 'eval' path of house_diffusion/rplanhg_datasets.py + scripts/image_sample.py
for an arbitrary list of House-GAN++ JSON files, without touching the submodule:
  * graph / GT polygon extraction: identical to RPlanhgDataset (uses its reader/build_graph);
  * "syn" conditions (what the model is sampled with): number of corners per room drawn from a
    per-room-type corner-count distribution (cndist), exactly like the eval set construction;
  * sampling: diffusion.p_sample_loop (1000 steps, discrete denoising for last 32 steps), last
    step taken, coordinates mapped to [0,256) like save_samples().
Raw output per sample: npz with polygons (list of Nx2 in 256px space), types (HD ids), graph.

Run with PYTHONPATH=external/methods/house_diffusion.
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import cv2 as cv
import numpy as np
import torch as th

from house_diffusion import rplanhg_datasets as R
from house_diffusion.script_util import (
    add_dict_to_argparser, args_to_dict, create_model_and_diffusion,
    model_and_diffusion_defaults, update_arg_parser)

MAX_PTS = 100
get_one_hot = lambda x, z: np.eye(z)[x]  # noqa: E731


def load_house(path):
    """Return (org_house [[contour(Nx2, 256px), type]], graph triples) as RPlanhgDataset does."""
    ds = object.__new__(R.RPlanhgDataset)
    ds.set_name = "eval"
    rms_type, fp_eds, rms_bbs, eds_to_rms = R.reader(path)
    fp_eds = np.array(fp_eds)
    rms_bbs = np.array(rms_bbs)
    tl = np.min(rms_bbs[:, :2], 0)
    br = np.max(rms_bbs[:, 2:], 0)
    shift = (tl + br) / 2.0 - 0.5
    fp_eds[:, :2] -= shift
    fp_eds[:, 2:] -= shift
    nodes, triples, masks = ds.build_graph(rms_type, fp_eds, eds_to_rms)
    house = []
    for m, t in zip(masks, nodes):
        m = cv.resize(m.astype(np.uint8), (256, 256), interpolation=cv.INTER_AREA)
        contours, _ = cv.findContours(m, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:  # node vanishes in the 64 px rasterisation (original code would crash)
            raise ValueError("empty room mask")
        t = int(t)
        if t > 10:
            t = {15: 11, 17: 12, 16: 13}[t]
        house.append([contours[0][:, 0, :], t])
    return house, np.array(triples), [int(x) for x in rms_type]


def make_cond(house, graph, cndist, rng):
    """Build the syn_* model kwargs for one house (mirrors the eval branch of RPlanhgDataset)."""
    while True:
        ncs = [cndist[t][rng.randint(0, len(cndist[t]) - 1)] for _, t in house]
        if sum(ncs) < MAX_PTS:
            break
    rows, bounds, n = [], [], 0
    for i, (_, t) in enumerate(house):
        k = ncs[i]
        rtype = np.repeat(get_one_hot(t, 25)[None], k, 0)
        ridx = np.repeat(get_one_hot(i + 1, 32)[None], k, 0)
        cidx = np.array([get_one_hot(x, 32) for x in range(k)])
        pad = np.ones((k, 1))
        conn = np.array([[j, (j + 1) % k] for j in range(k)]) + n
        bounds.append([n, n + k])
        n += k
        rows.append(np.concatenate((np.zeros([k, 2]), rtype, cidx, ridx, pad, conn), 1))
    hl = np.concatenate(rows, 0)
    hl = np.concatenate((hl, np.zeros((MAX_PTS - len(hl), 94))), 0)
    gen_mask = np.ones((MAX_PTS, MAX_PTS))
    gen_mask[:n, :n] = 0
    door_mask = np.ones((MAX_PTS, MAX_PTS))
    self_mask = np.ones((MAX_PTS, MAX_PTS))
    living = next(i for i, (_, t) in enumerate(house) if t == 1)
    for i in range(len(bounds)):
        connected = False
        for j in range(len(bounds)):
            if i == j:
                self_mask[bounds[i][0]:bounds[i][1], bounds[j][0]:bounds[j][1]] = 0
            elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                door_mask[bounds[i][0]:bounds[i][1], bounds[j][0]:bounds[j][1]] = 0
                connected = True
        if not connected:
            door_mask[bounds[i][0]:bounds[i][1], bounds[living][0]:bounds[living][1]] = 0
    c = 2
    return {
        "syn_door_mask": door_mask, "syn_self_mask": self_mask, "syn_gen_mask": gen_mask,
        "syn_room_types": hl[:, c:c + 25], "syn_corner_indices": hl[:, c + 25:c + 57],
        "syn_room_indices": hl[:, c + 57:c + 89], "syn_src_key_padding_mask": 1 - hl[:, c + 89],
        "syn_connections": hl[:, c + 90:c + 92],
    }


def decode(sample, cond):
    """sample: [S,2] in [-1,1]; returns polygons (256px) + HD types, like save_samples()."""
    pad = cond["syn_src_key_padding_mask"]
    ridx = cond["syn_room_indices"].argmax(1)
    rtyp = cond["syn_room_types"].argmax(1)
    polys, types = [], []
    for j in range(len(pad)):
        if pad[j] == 1:
            continue
        p = (sample[j] / 2 + 0.5) * 256
        if not polys or ridx[j] != cur:
            polys.append([])
            types.append(int(rtyp[j]))
            cur = ridx[j]
        polys[-1].append([float(p[0]), float(p[1])])
    return polys, types


def main():
    defaults = dict(dataset="rplan", model_path="", list="", out="", n_per_graph=1,
                    batch_size=64, seed=0, cndist="", clip_denoised=True)
    defaults.update(model_and_diffusion_defaults())
    ap = argparse.ArgumentParser()
    add_dict_to_argparser(ap, defaults)
    args = ap.parse_args()
    th.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))  # avoid CPU oversubscription
    update_arg_parser(args)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.cuda().eval()

    paths = [l.strip() for l in open(args.list) if l.strip()]
    houses, skipped = {}, {}
    for p in paths:
        try:
            houses[p] = load_house(p)
            if sum(len(h[0]) for h in houses[p][0]) > MAX_PTS:  # original drops houses with >100 corners
                raise ValueError("more than 100 GT corners")
        except Exception as e:  # noqa: BLE001
            skipped[p] = str(e)
            houses.pop(p, None)
    paths = [p for p in paths if p in houses]
    print(f"{len(paths)} inputs, skipped {len(skipped)}", flush=True)
    cndist = {int(k): v for k, v in json.load(open(args.cndist)).items()}

    jobs = [(p, s) for p in paths for s in range(args.n_per_graph)]
    t0 = time.time()
    for b in range(0, len(jobs), args.batch_size):
        batch = jobs[b:b + args.batch_size]
        conds = [make_cond(houses[p][0], houses[p][1], cndist, rng) for p, _ in batch]
        kw = {k: th.tensor(np.stack([c[k] for c in conds])).cuda() for k in conds[0]}
        shape = (len(batch), 2, MAX_PTS)
        sample = diffusion.p_sample_loop(model, shape, clip_denoised=args.clip_denoised,
                                         model_kwargs=kw, analog_bit=args.analog_bit)
        sample = sample[-1].permute([0, 2, 1]).cpu().numpy()  # [B, S, 2]
        for (p, s), smp, c in zip(batch, sample, conds):
            polys, types = decode(smp, c)
            house, graph, rms_type = houses[p]
            in_id = os.path.splitext(os.path.basename(p))[0]
            np.savez_compressed(
                os.path.join(args.out, f"{in_id}_{s}.npz"),
                polys=np.array([np.array(q) for q in polys], dtype=object), types=np.array(types),
                graph=graph, hg_room_type=np.array(rms_type),
                gt_polys=np.array([h[0] for h in house], dtype=object),
                gt_types=np.array([h[1] for h in house]), input_json=p)
        print(f"batch {b // args.batch_size + 1}/{(len(jobs) + args.batch_size - 1) // args.batch_size} "
              f"{time.time() - t0:.1f}s", flush=True)
    json.dump({"n_inputs": len(paths), "n_samples": len(jobs), "seconds": time.time() - t0, "skipped": skipped},
              open(os.path.join(args.out, "_infer_info.json"), "w"), indent=1)


def compute_cndist(list_file, out):
    """Per-HD-type corner-count lists, computed exactly as the train cndist (contours of 64px masks)."""
    d = defaultdict(list)
    for p in [l.strip() for l in open(list_file) if l.strip()]:
        try:
            house = load_house(p)[0]
        except ValueError:
            continue
        for poly, t in house:
            d[t].append(int(len(poly)))
    json.dump({str(k): v for k, v in d.items()}, open(out, "w"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cndist":
        compute_cndist(sys.argv[2], sys.argv[3])
    else:
        main()
