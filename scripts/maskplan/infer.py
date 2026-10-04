"""Run MaskPLAN inference (Inference/MaskPLAN_Inference_iterate_*.py) without touching the submodule.

The upstream script resolves everything relative to its REPO_ROOT (data, weights, boundary
images, output dir). We import it unmodified, then point its REPO_ROOT at a "shadow root" made
of symlinks:  Processed_data, VQ_Pretrained -> submodule;  parsed_img -> extracted boundary
images;  MaskPLAN_Trained -> checkpoints/maskplan;  Inference -> <out>/native.
The per-site loop is the same as upstream __main__ but (a) seeds NumPy/TF per site so runs are
resumable/reproducible, and (b) stores the partial input masks (M_*) and final predicted
attributes (In_*) for every site in <out>/meta/<site>.npz.

Usage (fpe-maskplan env):
  python infer.py --out OUT_RAW --script cross_Deep --model Base [--par_T ... ] [--test_cases N]
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUB = ROOT / "external/methods/maskplan"


def build_shadow(shadow, out_native, weights, parsed_img):
    shadow.mkdir(parents=True, exist_ok=True)
    links = {
        "Processed_data": SUB / "Processed_data",
        "VQ_Pretrained": SUB / "VQ_Pretrained",
        "parsed_img": parsed_img,
        "MaskPLAN_Trained": weights,
        "Inference": out_native,
    }
    for name, target in links.items():
        p = shadow / name
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(Path(target).resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="raw output dir (outputs/maskplan/<variant>/raw)")
    ap.add_argument("--script", default="cross_Deep", choices=["cross_Deep", "cross_Single", "vec_Deep", "vec_Single"])
    ap.add_argument("--weights", default=str(ROOT / "checkpoints/maskplan/MaskPLAN_Trained"))
    ap.add_argument("--parsed_img", default=str(ROOT / "data/method_inputs/maskplan/parsed_img"))
    ap.add_argument("--seed", type=int, default=0, help="per-site seed = seed + site_id")
    ap.add_argument("--start", type=int, default=0)
    known, rest = ap.parse_known_args()

    out = Path(known.out).resolve()
    (out / "native").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    shadow = out / "_shadow_root"
    build_shadow(shadow, out / "native", known.weights, known.parsed_img)

    sys.path.insert(0, str(SUB))
    path = SUB / f"Inference/MaskPLAN_Inference_iterate_{known.script}.py"
    spec = importlib.util.spec_from_file_location("maskplan_infer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(path)] + rest
    spec.loader.exec_module(mod)
    mod.REPO_ROOT = shadow  # data arrays were loaded at import from the real repo (identical)

    import numpy as np
    import tensorflow as tf

    args = mod.parse_args()
    model = mod.main(args)
    ids = mod.Testset_ids[: args.test_cases]
    post_dir = next((out / "native").glob("*/iteration")) / "post"
    t0 = time.time()
    for k, site in enumerate(ids):
        if k < known.start:
            continue
        site = int(site)
        if (post_dir / f"{site}.png").exists() and (out / "meta" / f"{site}.npz").exists():
            continue
        np.random.seed(known.seed + site)
        tf.random.set_seed(known.seed + site)
        model.reset(site)
        model.partial_input(site)
        given = {f"M_{a}": getattr(model, f"M_{a}").copy() for a in "TLASR"}
        model.inference_interation(site)
        pred = {f"In_{a}": getattr(model, f"In_{a}").copy() for a in "TLASR"}
        np.savez_compressed(out / "meta" / f"{site}.npz", site=site, **given, **pred)
        print(f"[{k + 1}/{len(ids)}] site {site} {time.time() - t0:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
