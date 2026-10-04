"""Headless batch driver for iPLAN (boundary -> room types -> room locations -> partition).

Re-implements the three synth/test_*.py scripts of the iPLAN repo as one per-sample
pipeline, without modifying the submodule:
  * the per-stage FloorPlan classes (synth/floorplan/*.py) and models are used as-is;
  * shims: test_roompartition.py calls `models.FloorPlanRNNTest`, which does not exist (the
    class is FloorPlanRNN in room_partition/models/floorplan_rnn_test.py), and loss_layer imports a non-existent `room_partition.models.utils.box_utils`
    -> we import floorplan_rnn_test directly and alias room_partition/utils/box_utils.py;
  * inputs are built from the Graph2Plan-preprocessed RPLAN boundary (256 px, (x, y, dir,
    isNew)) converted to iPLAN's 128 px (row, col, dir, isNew) format: floor(p[:, [1, 0]] / 2).

Run in the fpe-iplan env with CUDA_VISIBLE_DEVICES set. Writes raw/<name>.npz per sample.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import scipy.io as sio
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IPLAN = os.path.join(REPO, "external", "methods", "iplan")
sys.path.insert(0, IPLAN)
sys.path.insert(1, os.path.join(IPLAN, "room_partition"))  # floorplan_rnn.py does `import utils`

# --- shim for the broken box_utils import in room_partition/models/loss_layer.py
_spec = importlib.util.spec_from_file_location(
    "room_partition.models.utils.box_utils", os.path.join(IPLAN, "room_partition", "utils", "box_utils.py"))
_box_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_box_utils)
_pkg = types.ModuleType("room_partition.models.utils")
_pkg.box_utils = _box_utils
sys.modules["room_partition.models.utils"] = _pkg
sys.modules["room_partition.models.utils.box_utils"] = _box_utils


def to_iplan_boundary(b256):
    b = np.asarray(b256, dtype=int)
    out = np.zeros_like(b)
    out[:, 0] = b[:, 1] // 2
    out[:, 1] = b[:, 0] // 2
    out[:, 2:] = b[:, 2:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p_mat", required=True, help="Graph2Plan Network/data/data_test.mat")
    ap.add_argument("--ids", required=True, help="text file with RPLAN names, one per line")
    ap.add_argument("--ckpt", required=True, help="checkpoints/iplan/iPLAN")
    ap.add_argument("--out", required=True, help="raw output dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for k in ("g2p_mat", "ids", "ckpt", "out"):
        setattr(args, k, os.path.abspath(getattr(args, k)))

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_work")
    os.makedirs(os.path.join(work, "weights"), exist_ok=True)
    rpk = os.path.join(work, "weights", "renderer.pkl")
    if not os.path.lexists(rpk):
        os.symlink(os.path.join(args.ckpt, "room_partition", "renderer.pkl"), rpk)
    os.chdir(work)  # RendererNet loads ./weights/renderer.pkl

    from synth.floorplan.roomtype_fp import FloorPlan as FPType
    from synth.floorplan.roomlocation_fp import FloorPlan as FPLoc
    from synth.floorplan.roompartition_fp import FloorPlan as FPPart
    from room_type import models as rt_models
    from room_location.Living import models as living
    from room_location.Location import models as location
    from room_partition.models.floorplan_rnn_test import FloorPlanRNN as FloorPlanRNNTest
    from room_partition.models.loss_layer import LossFun
    from synth import test_roomlocation as T2
    from synth import test_roompartition as T3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = sio.loadmat(args.g2p_mat, squeeze_me=True, struct_as_record=False)["data"]
    by_name = {str(d.name): d for d in data}
    names = [l.strip() for l in open(args.ids) if l.strip()]
    if args.limit:
        names = names[:args.limit]

    # ---- models (same hyper-parameters as synth/test_*.py)
    max_room_per_type = [1, 2, 1, 2, 1, 1, 1, 3, 1, 3, 1, 1, 1]
    cvae = rt_models.cvae(modul_name="roomtype", model_name="cvae", input_dim=int(np.sum(max_room_per_type)),
                          hidden_dim1=128, hidden_dim2=64, z_dim=32)
    cvae.load_model(os.path.join(args.ckpt, "room_type", "roomtype_cvae_150.pth"))
    cvae.cuda().eval()

    loc = os.path.join(args.ckpt, "room_location")
    living_model = living.model(module_name="living", model_name="resnet18_fc1", input_channel=3,
                                output_channel=2, pretrained=False)
    living_connect = living.connect(module_name="living", model_name="resnet18_fc1", input_channel=512,
                                    output_channel=2, reshape=True)
    living_model.load_model(f"{loc}/living_resnet18_300.pth")
    living_connect.load_model(f"{loc}/living_fc1_300.pth")
    location_model = location.model(module_name="location", model_name="resnet18_up1", input_channel=17,
                                    output_channel=16, pretrained=True,
                                    pretrained_path=f"{loc}/resnet18-5c106cde.pth")
    location_connect = location.connect(module_name="location", model_name="resnet18_up1", input_channel=512,
                                        output_channel=16, reshape=False)
    location_embedding = location.embedding(module_name="location", model_name="resnet18_up1", input_channel=13,
                                            output_channel=256, reshape=False)
    location_model.load_model(f"{loc}/location_resnet18_100.pth")
    location_connect.load_model(f"{loc}/location_up1_100.pth")
    location_embedding.load_model(f"{loc}/location_embed_100.pth")
    for m in (living_model, living_connect, location_model, location_connect, location_embedding):
        m.cuda().eval()

    popt = argparse.Namespace(load_netG_path=os.path.join(args.ckpt, "room_partition", "G_net_210.pth"),
                              max_iter=args.max_iter, gpu_ids=[0], lr=10, coverage=1, inside=0.5, mutex=0)
    fp_rnn = FloorPlanRNNTest(popt)
    fp_rnn.load_networks(popt)
    criterion = LossFun(device=fp_rnn.device)

    log = {"ok": [], "failed": {}}
    t0 = time.time()
    for k, name in enumerate(names):
        out_path = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(out_path):
            log["ok"].append(name)
            continue
        d = by_name[name]
        mat_path = os.path.join(work, f"{name}.mat")
        b128 = to_iplan_boundary(d.boundary)
        gtb = np.asarray(d.gtBoxNew, dtype=int)
        sio.savemat(mat_path, {"data": {
            "name": name, "gt_rTypes": np.asarray(d.rType, dtype=int),
            "gt_rBoxes": np.stack([gtb[:, 1], gtb[:, 0], gtb[:, 3], gtb[:, 2]], 1) // 2,
            "Boundary": b128, "rTypes": np.array([]), "rBoxes": np.array([]), "rCenters": np.array([])}})
        try:
            # ---- stage 1: room types (synth/test_roomtype.py)
            fp = FPType(mat_path)
            img = torch.FloatTensor(fp.init_input_img(fp.exterior_boundary)).unsqueeze(0).unsqueeze(0).cuda()
            img = fp.normalize(img)
            with torch.no_grad():
                emb = cvae.embed(img)
                z = torch.randn(img.size(0), 32).cuda()
                sample = cvae.decoder(torch.cat([z, emb], 1)).view(-1, 19)
            fp.update_rTypes(sample.squeeze().cpu().numpy(), max_room_per_type)
            fp.rBoxes = np.array([])
            fp.rCenters = np.array([])
            sio.savemat(mat_path, {"data": fp.to_dict()})

            # ---- stage 2: room locations (synth/test_roomlocation.py)
            fp = FPLoc(mat_path)
            inp = fp.get_composite_living().unsqueeze(0).cuda()
            with torch.no_grad():
                lc = torch.round(living_connect(living_model(inp))).cpu().squeeze().numpy()
            node = {"category": 0, "centroid": (int(lc[0]), int(lc[1]))}
            fp.add_room(node)
            fp.living_node = node
            ok = False
            for _ in range(50):
                pc, pt, flag = T2.get_rcenters(fp, location_model, location_connect, location_embedding)
                if flag:
                    fp.update_rCenters(pc, pt)
                    sio.savemat(mat_path, {"data": fp.to_dict()})
                    ok = True
                    break
            if not ok:
                raise RuntimeError("room location failed after 50 attempts")

            # ---- stage 3: partition (synth/test_roompartition.py)
            fp = FPPart(mat_path)
            img, rCenters, rTypes, inside, boundary = fp.get_input()
            fp_rnn.evaluate(img, rCenters, rTypes, inside)
            boundary = boundary[:, [1, 0, 2, 3]]
            pred = fp_rnn.pred_rBoxes[:, [1, 0, 3, 2]]
            layout = T3.get_image(img, pred, rTypes, inside)
            _, layout, living_mask = T3.obtain_living(layout, boundary)
            criterion.get_initial_mutex(pred, boundary)
            it = 0
            while T3.get_ratio(layout) > 0.0005 and it < args.max_iter:
                it += 1
                pred = T3.fine_turning(pred, boundary, living_mask, criterion, popt)
                layout = T3.get_image(img, pred, rTypes, inside, living_mask)
            # instance map with the same painting order as get_image (idx 0 = living room)
            boxes = (pred.detach().cpu() * 127).long().numpy()
            ins = inside.numpy()
            inst = np.full((128, 128), -1, dtype=np.int16)
            inst[(living_mask.numpy() == 1) & (ins > 0)] = 0
            for r in range(len(rTypes)):
                x0, y0, x1, y1 = boxes[r]
                m = np.zeros((128, 128), bool)
                m[y0:y1 + 1, x0:x1 + 1] = True  # same (unclamped) slicing as get_image
                inst[m & (ins > 0)] = r + 1
            np.savez_compressed(
                out_path, name=name, layout=layout.numpy().astype(np.int16), inst=inst,
                inst_types=np.concatenate([[0], rTypes.numpy()]).astype(int),
                boxes_xyxy_128=boxes, rTypes=fp.rTypes, rCenters=fp.rCenters, boundary128=b128,
                boundary256=np.asarray(d.boundary, dtype=int), finetune_iters=it)
            log["ok"].append(name)
        except Exception as e:  # noqa
            log["failed"][name] = f"{type(e).__name__}: {e}"
        finally:
            if os.path.exists(mat_path):
                os.remove(mat_path)
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{len(names)} ok={len(log['ok'])} fail={len(log['failed'])} "
                  f"{(time.time() - t0) / (k + 1):.2f}s/sample", flush=True)
    log["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("done", len(log["ok"]), "ok", len(log["failed"]), "failed", f"{log['seconds']:.0f}s")


if __name__ == "__main__":
    main()
