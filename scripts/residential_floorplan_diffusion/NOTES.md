# Residential Floorplan Diffusion — notes

Status: **BLOCKED on weights.** The env, driver, converter and run script are in place and were
smoke-tested end to end with randomly initialised weights, but no real outputs exist.

## Weights
- `model_stage1.pth` (Drive `1ONAu_i2q0FUGJClArBC4mBMwFAchmo1H`) and `model_stage2.pth`
  (Drive `1cVL4dLMM7j0n3nSKkapjdTWgHD9H45rA`) are **access-restricted**. gdown fails, and the Drive
  view page returns HTTP 401 / "request access" (checked 2026-10-04). Upstream GitHub issue #1 reports
  the same problem, with no answer from the authors.
- To unblock: request access from the authors (we did not, because that needs the user's Google
  account), put both files in `checkpoints/residential_floorplan_diffusion/`, then run
  `bash scripts/residential_floorplan_diffusion/run.sh`.

## Inputs
- The only public test inputs are the 11 condition sets bundled in the repo
  (`test/stage1_input/<k>_{living_room,bedroom,kitchen,toilet,balcony}.png`, k=0..10). Each one is a
  set of per-room-type rectangle masks (256 px, RPLAN-toolbox colours). The training/test dataset
  itself is not released (`datasets/` holds only a README).
- The local Graph2Plan RPLAN copy (`/datawaha/cggroup/datasets/RPLAN/Network/data`) could be
  rasterised into this input format (room bounding boxes per type) to get more conditions. We did
  not do this: there are no weights, and the exact preprocessing (box vs. polygon, scale, centring) is
  not documented.
- Plan: 48 accepted samples per condition × 11 = 528 samples (`run.sh [PER_COND]`). Variant name:
  `bundled_roommasks`.

## Env (`fpe-resdiff`)
- Python 3.10, torch 2.1.2+cu118 (upstream pins torch 1.12 and Python 3.8). numpy 1.26.4,
  opencv-python-headless 4.8.1.78, scikit-image 0.22, natsort.
- `PYTHONNOUSERSITE=1` is set as an env var. Otherwise `~/.local/lib/python3.10` shadows the env's
  packages.

## Deviations / workarounds (no submodule changes, no patch)
- At import time, upstream `nets/unet.py` sets `os.environ['CUDA_VISIBLE_DEVICES']` to a garbage
  string, which hides every GPU. `infer.py` calls `torch.cuda.init()` before importing upstream
  modules, so the caller's `CUDA_VISIBLE_DEVICES` takes effect.
- Upstream `predict.py` is a demo: it mixes `1_living_room` with `0_*` inputs, and stage 2 breaks
  when fewer than `predict_num` stage-1 samples pass. `infer.py` reuses its functions but:
  - uses matched condition sets;
  - keeps sampling stage-1 batches (64) until 48 pass the upstream room-count check
    (`Room_Judgment`), with at most 4 batches;
  - runs stage 2 on every accepted image.
- The upstream stage-2 PSNR filter is dropped. It compares every output with one fixed reference
  image, using data_range=255 on [0,1] images, so it never removes anything.
- Converter: each pixel is assigned to the nearest palette colour (white background, 5 room colours,
  salmon wall). Each room mask gets a 2×2 opening, which removes 1-px wall-antialiasing slivers that
  look like the kitchen colour. Then one polygon is made per 4-connected component. Boundary = outer
  contour of the non-background pixels. Units are px on the 64×64 output grid.
- The converter was checked only on the repo's stage-2 reference image `test/stage2_psnr.png`,
  where the result looks correct. Re-check it on real outputs.
- No GT is available for the bundled conditions, so there is no `gt_samples/`.
