"""Batch driver for Residential Floorplan Diffusion (two-stage DDPM), no submodule edits.

Reuses Diffusion_stage1/2 and the helpers from upstream predict.py, but:
  * uses matched condition sets  test/stage1_input/<k>_{living_room,bedroom,kitchen,toilet,balcony}.png
    (upstream's demo mixes 1_living_room with 0_* inputs),
  * samples stage 1 in batches per condition until `--per_cond` samples pass the upstream
    room-count check (Room_Judgment) or `--max_rounds` batches were drawn,
  * runs stage 2 on every accepted stage-1 image and applies upstream remove_noise(),
  * skips the upstream PSNR filter (it compares against one fixed reference image with
    data_range=255 on [0,1] images, so it never removes anything).
Must run with cwd = submodule root (weights are loaded from ./logs/model_stage{1,2}.pth unless
--ckpt is given).

Usage (fpe-resdiff env):
  python infer.py --ckpt checkpoints/residential_floorplan_diffusion --out OUT_RAW --per_cond 48
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
# upstream nets/unet.py overwrites os.environ['CUDA_VISIBLE_DEVICES'] with a garbage string at import
# time; initialise CUDA first so the caller's CUDA_VISIBLE_DEVICES is the one that counts.
torch.cuda.init()
import predict as P  # noqa: E402  (upstream module; its __main__ block is not executed)
from ddpm import Diffusion_stage1, Diffusion_stage2  # noqa: E402

TYPES = ["living_room", "bedroom", "kitchen", "toilet", "balcony"]


def room_counts(img_rgb):
    """Upstream Room_Judgment counting on a stage-1 output (RGB uint8)."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # upstream applies BGR2RGB to an RGB image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return [P.num_room(hsv, P.lower_gray, P.upper_gray), P.num_room(hsv, P.lower_yellow, P.upper_yellow),
            P.num_room(hsv, P.lower_blue, P.upper_blue), P.num_room(hsv, P.lower_pink, P.upper_pink),
            P.num_room(hsv, P.lower_green, P.upper_green)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./logs")
    ap.add_argument("--inputs", default="./test/stage1_input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default=None, help="comma list of condition ids (default: all)")
    ap.add_argument("--per_cond", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--no_judgment", action="store_true", help="keep all stage-1 samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = Path(args.ckpt)
    s1 = Diffusion_stage1(model_path=str(ck / "model_stage1.pth"))
    s2 = Diffusion_stage2(model_path=str(ck / "model_stage2.pth"))
    inp = Path(args.inputs)
    conds = sorted({int(p.name.split("_")[0]) for p in inp.glob("*_living_room.png")})
    if args.conds:
        conds = [int(c) for c in args.conds.split(",")]
    out = Path(args.out)
    for d in ("stage1", "stage2", "stage1_rejected"):
        (out / d).mkdir(parents=True, exist_ok=True)
    meta = {}
    t0 = time.time()
    for c in conds:
        torch.manual_seed(args.seed + c)
        np.random.seed(args.seed + c)
        paths = {t: str(inp / f"{c}_{t}.png") for t in TYPES}
        accepted, n_drawn = [], 0
        for r in range(args.max_rounds):
            src, target = P.read_img_stage1(paths["balcony"], paths["bedroom"], paths["kitchen"],
                                            paths["living_room"], paths["toilet"], args.batch)
            # target order: [living, bedroom, toilet, kitchen, balcony]
            imgs = P.show_pred(args.batch, s1.net, "cuda", source=src)
            for i in range(args.batch):
                x = np.uint8(P.postprocess_output(imgs[i].numpy().transpose(1, 2, 0)))
                x[x == 255] = 0
                sid = f"c{c:02d}_s{n_drawn:03d}"
                n_drawn += 1
                ok = args.no_judgment or room_counts(x.copy()) == list(target)
                if ok and len(accepted) < args.per_cond:
                    Image.fromarray(x).save(out / "stage1" / f"{sid}.png")
                    accepted.append(sid)
                elif not ok:
                    Image.fromarray(x).save(out / "stage1_rejected" / f"{sid}.png")
            print(f"cond {c} round {r}: accepted {len(accepted)}/{n_drawn} ({time.time() - t0:.0f}s)", flush=True)
            if len(accepted) >= args.per_cond:
                break
        # stage 2 on accepted stage-1 layouts
        for b0 in range(0, len(accepted), args.batch):
            ids = accepted[b0:b0 + args.batch]
            src = P.read_img_stage2([str(out / "stage1" / f"{s}.png") for s in ids], len(ids))
            gen = P.show_pred(len(ids), s2.net, "cuda", source=src)
            for s, g in zip(ids, gen):
                y = np.uint8(P.postprocess_output(g.numpy().transpose(1, 2, 0)))
                Image.fromarray(P.remove_noise(y.copy())).save(out / "stage2" / f"{s}.png")
        meta[c] = {"inputs": paths, "target_counts_living_bed_toilet_kitchen_balcony": list(map(int, target)),
                   "drawn": n_drawn, "accepted": accepted}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
