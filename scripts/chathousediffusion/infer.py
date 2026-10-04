"""Run ChatHouseDiffusion on the Tell2Design test split (pre-parsed graphs, no LLM call).

Mirrors Trainer.val() from external/methods/chathousediffusion (test.py) but writes raw
label maps (uint8, RPLAN labels 0..17) instead of only RGB previews, and skips the IoU
bookkeeping. Must be run with cwd = the submodule root (t5_feature.pkl is loaded relatively).

Usage (fpe-chathousediffusion env):
  python infer.py --data DATA_DIR --ckpt CKPT_DIR --out OUT_RAW_DIR [--limit N] [--seed 1029]
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet  # noqa: E402
from denoising_diffusion_pytorch.utils import seed_torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with image_test/ mask_test/ text_test/")
    ap.add_argument("--ckpt", required=True, help="dir with params.pkl and model-98.pt")
    ap.add_argument("--milestone", type=int, default=98)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1029)  # upstream test.py: seed_torch() default
    args = ap.parse_args()

    with open(os.path.join(args.ckpt, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    params["trainer_dict"]["train_batch_size"] = args.batch
    model = Unet(**params["unet_dict"])
    diffusion = GaussianDiffusion(model, **params["diffusion_dict"])
    data = Path(args.data)
    trainer = Trainer(diffusion, str(data / "image"), str(data / "mask"), str(data / "text"),
                      **params["trainer_dict"], results_folder=args.ckpt,
                      train_num_workers=0, mode="val")
    trainer.load(args.milestone)
    trainer.ema.copy_params_from_model_to_ema()
    trainer.ema.ema_model.eval()
    seed_torch(args.seed)

    out = Path(args.out)
    (out / "label").mkdir(parents=True, exist_ok=True)
    (out / "text").mkdir(parents=True, exist_ok=True)
    n_done, t0 = 0, time.time()
    with torch.inference_mode():
        for img, feature, text, gdict, idx in trainer.val_dl:
            b = img.shape[0]
            feature = feature.to("cuda")
            gdict = {k: v.to("cuda") for k, v in gdict.items()}
            images = trainer.ema.ema_model.sample(
                batch_size=b, feature=feature, text=None if trainer.use_graphormer else text,
                graphormer_dict=gdict if trainer.use_graphormer else None,
                cond_scale=trainer.cond_scale)
            # same as Trainer.val (onehot=False): pixels outside the boundary -> External (13)
            images = torch.where(feature > 0.5, 13 / 17, images)
            labels = torch.round(images.clamp(0, 1) * 17).to(torch.uint8).cpu().numpy()[:, 0]
            for j in range(b):
                Image.fromarray(labels[j]).save(out / "label" / f"{idx[j]}.png")
                (out / "text" / f"{idx[j]}.json").write_text(text[j])
            n_done += b
            print(f"{n_done}/{len(trainer.val_ds)} {time.time() - t0:.1f}s", flush=True)
            if args.limit and n_done >= args.limit:
                break
    print(f"done {n_done} samples in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
