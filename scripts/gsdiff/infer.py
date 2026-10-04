"""GSDiff inference for the three generation settings (RPLAN-trained models).

Standalone re-implementation of scripts/test_main.py (uncond), test_topo.py (topo / bubble diagram) and
test_boun.py (boundary). Sampling code (cosine schedule, 1000 DDPM steps, x0 thresholding, stage-2 edge
transformer, planar-cycle extraction with get_cycle_basis_and_semantic_3_semansimplified,
merge_points=False, align_points=False, resolution 512) is copied from those scripts; only data loading
differs:
  uncond: the original only takes tensor shapes from the RPLAN test loader -> zeros of shape (bs,53,10).
  topo:   bubble diagram (<=8 rooms, 7 classes, adjacency) from prepare_inputs.py npz files.
  boun:   boundary image (drawn as in rplang_edge_semantics_simplified_78_10_prerunCNN) -> frozen boundary
          CNN (structure-78-12) -> 16x16 feature map, as in prerunningCNN.py, computed on the fly.
Submodule code is imported unmodified.

Raw output per sample: vr4stat_<id>.npy (same dict as the test scripts, plus 'ref_id') and graph_<id>.npz.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "methods", "gsdiff"))
sys.path.insert(0, GS)
from gsdiff.utils import (edges_remove_padding, edges_to_coordinates,  # noqa: E402
                          get_cycle_basis_and_semantic_3_semansimplified,
                          inverse_normalize_and_remove_padding_100_4testing)

CKPTS = {
    "uncond": {"node": "outputs/structure-1/model1000000.pt", "edge": "outputs/structure-2/model_stage2_best_061000.pt"},
    "topo": {"node": "topo-params/structure-80-106-2/model1000000.pt",
             "edge": "topo-params/structure-56-35-interval1000/model_stage2_best_076000.pt",
             "enc": "topo-params/structure-57-16/model_stage0_best_006000.pt"},
    "boun": {"node": "outputs/structure-81-106-3/model1000000.pt",
             "edge": "outputs/structure-56-36-interval1000/model_stage2_best_065000.pt",
             "enc": "structure-78-12/model_stage0_best_006700.pt"},
}


def schedule(T=1000):
    alpha_bar = lambda t: math.cos(t / 1.0 * math.pi / 2) ** 2
    betas = np.array([min(1 - alpha_bar((i + 1) / T) / alpha_bar(i / T), 0.999) for i in range(T)], dtype=np.float64)
    alphas = 1.0 - betas
    ac = np.cumprod(alphas)
    ac_prev = np.append(1.0, ac[:-1])
    return dict(posterior_variance=betas * (1.0 - ac_prev) / (1.0 - ac), sqrt_recip=np.sqrt(1.0 / ac),
                sqrt_recipm1=np.sqrt(1.0 / ac - 1), coef1=betas * np.sqrt(ac_prev) / (1.0 - ac),
                coef2=(1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - ac))


def load(model, path, device):
    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    for p in model.parameters():
        p.requires_grad = False
    return model


def boundary_feat16(enc, imgs, device):
    """prerunningCNN.py feature extraction. imgs: (bs,256,256,3) uint8."""
    x = torch.tensor((imgs.astype(np.int32) - 128) / 128, device=device).permute(0, 3, 1, 2).float()
    e1 = enc.Conv1(x)
    e2 = enc.Conv2(enc.Maxpool1(e1)) + enc.shortcut2(enc.Maxpool1(e1))
    e3 = enc.Maxpool2(e2)
    e3 = enc.Conv3(e3) + enc.shortcut3(e3)
    e4 = enc.Maxpool3(e3)
    e4 = enc.Conv4(e4) + enc.shortcut4(e4)
    e5 = enc.Maxpool4(e4)
    return enc.Conv5(e5) + enc.shortcut5(e5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uncond", "topo", "boun"], required=True)
    ap.add_argument("--ckpt_root", required=True, help="checkpoints/gsdiff")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--inputs", default=None, help="dir of prepare_inputs.py npz files (topo/boun)")
    ap.add_argument("--ids", default=None, help="id list (topo/boun)")
    ap.add_argument("--num_samples", type=int, default=3000, help="uncond only")
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    opt = ap.parse_args()
    device, res, T = "cuda:0", 512, 1000
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    os.makedirs(opt.out_dir, exist_ok=True)
    s = {k: torch.tensor(v, device=device) for k, v in schedule(T).items()}
    ck = {k: os.path.join(opt.ckpt_root, v) for k, v in CKPTS[opt.mode].items()}

    if opt.mode == "uncond":
        from gsdiff.house_nn1 import HeterHouseModel
        from gsdiff.house_nn2 import EdgeModel
        node_model, edge_model = load(HeterHouseModel(), ck["node"], device), load(EdgeModel(), ck["edge"], device)
        ids = [str(i) for i in range(opt.num_samples)]
    else:
        ids = [l.strip() for l in open(opt.ids) if l.strip()]
        inp = [np.load(Path(opt.inputs) / f"{i}.npz") for i in ids]
        if opt.mode == "topo":
            from gsdiff.heterhouse_80_106_2 import TopoHeterHouseModel
            from gsdiff.bubble_diagram_57_9 import TopoGraphModel
            from gsdiff.heterhouse_56_31 import TopoEdgeModel
            node_model, edge_model = load(TopoHeterHouseModel(), ck["node"], device), load(TopoEdgeModel(), ck["edge"], device)
            enc = load(TopoGraphModel(), ck["enc"], device)
        else:
            from gsdiff.heterhouse_81_106_3 import BoundHeterHouseModel
            from gsdiff.boundary_78_10 import BoundaryModel
            from gsdiff.heterhouse_56_32 import BoundEdgeModel
            node_model, edge_model = load(BoundHeterHouseModel(), ck["node"], device), load(BoundEdgeModel(), ck["edge"], device)
            enc = load(BoundaryModel(), ck["enc"], device)
    n_total = len(ids)

    def cond_batch(idx):
        """Returns per-batch conditioning tensors (as in the dataset classes)."""
        if opt.mode == "topo":
            bs = len(idx)
            sem = np.zeros((bs, 8, 7)); adj = np.zeros((bs, 8, 8), dtype=np.uint8); pad = np.zeros((bs, 8, 1), dtype=np.uint8)
            for b, i in enumerate(idx):
                ss = inp[i]["bb_semantics"]; n = len(ss)
                assert n <= 8, f"{ids[i]}: {n} rooms > 8 (GSDiff topo supports 4-8)"
                sem[b, :n] = np.eye(7)[ss]; adj[b, :n, :n] = inp[i]["bb_adjacency"]; pad[b, :n] = 1
            sem_t = torch.tensor(sem, device=device).float()
            adj_t = torch.tensor(adj, device=device); pad_t = torch.tensor(pad, device=device)
            emb = enc.semantics_embedding(sem_t)
            for layer in enc.transformer_layers:
                emb = layer(emb, adj_t)
            emb = emb * pad_t
            return {"emb": emb, "pad": pad_t}
        if opt.mode == "boun":
            imgs = np.stack([inp[i]["boundary_img"] for i in idx])
            return {"feat": boundary_feat16(enc, imgs, device)}
        return {}

    t0 = time.time()
    res_c, res_s, res_n, conds = [], [], [], []
    with torch.no_grad():
        for start in range(0, n_total, opt.batch_size):
            idx = list(range(start, min(n_total, start + opt.batch_size)))
            bs = len(idx)
            cb = cond_batch(idx)
            attn = torch.ones((bs, 53, 53), dtype=torch.uint8, device=device)
            x = torch.randn(bs, 53, 10, device=device, dtype=torch.float64)
            for step in range(T - 1, -1, -1):
                t = torch.tensor([step], device=device)
                if opt.mode == "uncond":
                    o1, o2 = node_model(x, attn, t)
                elif opt.mode == "topo":
                    o1, o2 = node_model(x, attn, t, cb["emb"], cb["pad"])
                else:
                    o1, o2 = node_model(x, attn, t, cb["feat"])
                eps = torch.cat((o1, o2), dim=2)
                x0 = s["sqrt_recip"][t][:, None, None] * x - s["sqrt_recipm1"][t][:, None, None] * eps
                x0[:, :, 0:2] = torch.clamp(x0[:, :, 0:2], -1, 1)
                x0[:, :, 2:9] = x0[:, :, 2:9] >= 0.5
                x0[:, :, 9:10] = x0[:, :, 9:10] >= 0.75
                mean = s["coef1"][t][:, None, None] * x0 + s["coef2"][t][:, None, None] * x
                x = mean + torch.sqrt(s["posterior_variance"][t][:, None, None]) * torch.randn_like(x)
            for b in range(bs):
                res_c.append(x[b, :, :2][None]); res_s.append(x[b, :, 2:9][None]); res_n.append(x[b, :, 9:10][None].view(-1))
                conds.append({k: v[b:b + 1] for k, v in cb.items()})
            print(f"stage1 {start + bs}/{n_total} {time.time() - t0:.0f}s", flush=True)
        t_stage1 = time.time() - t0

        corners_all, seman_all = inverse_normalize_and_remove_padding_100_4testing(res_c, res_s, res_n, resolution=res)
        edges_out, ncorn = [], []
        for i in range(n_total):
            c = torch.zeros((1, 53, 2), dtype=torch.float64, device=device)
            ct = (torch.tensor(corners_all[i], dtype=torch.float64, device=device) - res // 2) / (res // 2)
            c[:, :ct.shape[1]] = ct
            sm = torch.zeros((1, 53, 7), dtype=torch.float64, device=device)
            sm[:, :ct.shape[1]] = torch.tensor(seman_all[i], dtype=torch.float64, device=device)
            a = torch.zeros((1, 53, 53), dtype=torch.bool, device=device)
            a[:, :ct.shape[1], :ct.shape[1]] = True
            m = torch.zeros((1, 53, 1), dtype=torch.uint8, device=device)
            m[:, :ct.shape[1]] = 1
            if opt.mode == "uncond":
                oe, _, _ = edge_model(c, a, m, sm)
            elif opt.mode == "topo":
                oe, _, _ = edge_model(c, a, m, sm, conds[i]["emb"], conds[i]["pad"])
            else:
                oe, _, _ = edge_model(c, a, m, sm, conds[i]["feat"])
            oe = F.one_hot(torch.argmax(F.softmax(oe, dim=2), dim=2), num_classes=2)
            edges_out.append(oe); ncorn.append(int(m.sum().item()))
        edges_all = edges_remove_padding(edges_out, ncorn)
    print(f"stage2 done {time.time() - t0:.0f}s", flush=True)

    n_fail = 0
    for i in range(n_total):
        corners, edges, seman = corners_all[i], edges_all[i], seman_all[i]
        st = np.where(seman == 1, np.indices(seman.shape)[-1], 99999)
        pts = [tuple(p) for p in np.concatenate((corners, st), axis=-1).tolist()[0]]
        n = len(pts)
        e_coords = edges_to_coordinates(np.triu(edges[0, :, 1].reshape(n, n)).reshape(-1), pts)
        vr = {"ref_id": ids[i], "output_points_test": pts, "output_edges_test": list(e_coords)}
        try:
            d_rev, cycles, cyc_sem = get_cycle_basis_and_semantic_3_semansimplified(pts, list(e_coords))
        except Exception as ex:
            n_fail += 1
            d_rev, cycles, cyc_sem = {}, [], []
            vr["error"] = repr(ex)
        vr.update(d_rev_test=d_rev, simple_cycles_test=cycles, simple_cycles_semantics_test=cyc_sem)
        np.save(os.path.join(opt.out_dir, f"vr4stat_{ids[i]}.npy"), vr)
        np.savez_compressed(os.path.join(opt.out_dir, f"graph_{ids[i]}.npz"), corners=corners[0], semantics=seman[0],
                            adjacency=edges[0, :, 1].reshape(n, n))
    total = time.time() - t0
    log = {"mode": opt.mode, "n": n_total, "cycle_failures": n_fail, "seconds": total, "seconds_stage1": t_stage1,
           "checkpoints": CKPTS[opt.mode], "seed": opt.seed, "batch_size": opt.batch_size}
    json.dump(log, open(os.path.join(opt.out_dir, "_log.json"), "w"), indent=1)
    print(json.dumps(log))


if __name__ == "__main__":
    main()
