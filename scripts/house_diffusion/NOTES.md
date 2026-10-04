# HouseDiffusion (CVPR 2023): run notes

Scripts:
- `setup_env.sh` creates the env `fpe-house_diffusion`.
- `download.sh` gets the weights from gdrive.
- `infer.py` is the sampling driver. `infer.py cndist` computes the corner-count distribution.
- `convert.py` writes the common format.
- `run.sh` runs everything end to end on the public inputs.
- `run_g2p.sh` does the same for the Graph2Plan-derived test variant.

The HG++ JSON helpers and the Graph2Plan converter are shared with House-GAN++. They live in
`scripts/houseganpp/hg_json.py` and `scripts/houseganpp/g2p_to_hgjson.py`.

## Environment
- Python 3.10.
- torch 2.4.1+cu121. The repo pins the nightly `torch==2.0.0.dev20221212` plus tensorflow 2.11.
- tensorflow and pytorch_fid are dropped. Only the FID part of `image_sample.py` uses them, and we
  don't run it.
- Shapely is kept at 1.8.5.post1 because the code uses `shapely.geos.lgeos`. drawSvg 1.9.0 (old
  API), cairosvg, cairo and mpi4py come from conda-forge. Also installed: blobfile, imageio,
  opencv-python-headless and numpy<2.
- The `house_diffusion` package is used via `PYTHONPATH`, not `pip -e`, so the submodule stays clean.
  No patch was needed: the submodule is unmodified.
- `PYTHONNOUSERSITE=1` is set because `~/.local` site-packages would otherwise shadow the env's own.

## Weights
`checkpoints/house_diffusion/model250000.pt` is the README's "temporary model", Drive id
16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR. Per `scripts/script.sh` it was trained with `--target_set 8`:
8-room plans are held out, and the paper's eval set is the 8-room plans.

## Why our own driver instead of `scripts/image_sample.py`
The original script has several problems for this setup:
- It reads `../datasets/rplan` and `processed_rplan/*.npz`.
- It needs `rplan_train_8_cndist.npz`, which comes from the RPLAN train split.
- It loops 5× and computes FID.
- It only writes SVG/PNG.

`infer.py` reproduces the eval path line by line:
- **Graph and GT:** `RPlanhgDataset.reader/build_graph` → 64 px masks → INTER_AREA to 256 →
  contours.
- **"syn" conditions:** room/door nodes with corner counts sampled from the cndist, and the
  living-room fallback for unconnected nodes in the door mask.
- **Sampling:** `p_sample_loop` (1000 cosine steps, discrete bit decoding in the last 32 steps),
  taking the last step.
- **Output:** coordinates are mapped with `(x/2+0.5)*256` as in `save_samples`.

Differences from the original:
1. Several samples per graph: the corner counts are re-sampled for each sample.
2. A fixed seed.
3. The output is `raw/<inputid>_<k>.npz` holding the polygons, HD types, graph and GT contours.
4. The room type is read per corner. The original's `room_types[j-1]` indexing only matters in
   degenerate cases.

Houses with more than 100 GT corners are skipped, as the original dataset class does.

## Inputs and variants
RPLAN is gated behind a Google Form. No public copy of the HG++-format JSON exists: HouseDiffusion
only links the form, and suggests MagicPlan as an alternative, but these weights are RPLAN-only.

1. **`rplan_bubble_public`**: the 7 RPLAN graphs that ship with the repos (6 in House-GAN++
   `data/json`, plus the data reader's `sample_output/0.json`), at 100 samples per graph, giving
   **700 samples**.
   - These plans have 5–7 rooms, so they fall in the training size range.
   - They may well be training plans, so this is not a held-out test.
   - The cndist is pooled from these 7 plans (`cndist_pooled.json`), because the original uses the
     train split.
   - Sampling took 633 s.
2. **`rplan-g2p_bubble_test1000_syndoors`**: bubble diagrams built from the Graph2Plan-preprocessed
   RPLAN **test** split, `/datawaha/cggroup/datasets/RPLAN/Network/data/data_test.mat`.
   - It uses the shared 1000-id subset `data/method_inputs/common_rplan_test/ids_test_1000.txt`, the
     same ids as iPLAN, DiffPlanner and WallPlan, with 1 sample each. **997 samples**: 3 inputs
     were skipped for having more than 100 GT corners.
   - The cndist comes from 3000 random Graph2Plan *train* plans converted the same way
     (`cndist_g2p_train3000.json`).
   - Sampling took 512 s, batch size 100, about 3 GB of GPU memory.
   - **Approximation:** Graph2Plan data has no interior doors, so `g2p_to_hgjson.py` *synthesizes*
     them. Every non-living room gets one 2 px × ≤12 px door on its shared wall: with the living
     room if they share a wall, otherwise with the best neighbour (dining > entrance > longest
     shared wall).
   - The front door comes from the Graph2Plan boundary. Room polygons and types are exact, with
     RPLAN→HG++ labels mapped as in `read_dd.py`. Wall adjacency means a shared boundary of at
     least 3 px.
   - Rooms have no wall gaps here, unlike the real HG++ JSON. So the input distribution differs
     somewhat from what the model was trained on, the graphs include 8-room plans (held out), and
     the doors are not RPLAN's.

To run on real HG++ JSON (e.g. the 8-room RPLAN eval set), use:
`INPUT_LIST=<list> CNDIST=<cndist from train> VARIANT=rplan_bubble_test bash scripts/house_diffusion/run.sh`.

## Conversion
- Each input node yields one polygon, in the centred 256 px frame. Room nodes go to `rooms`. HD type
  11 goes to `front_door` and type 12 to `interior_door` in `doors`.
- Consecutive duplicate corners are merged. Polygons with fewer than 3 distinct corners are dropped
  and listed in `condition.missing_nodes`. No validity repair is done, so polygons can
  self-intersect or overlap, as in the paper.
- `graph.edges` and the condition follow the House-GAN++ conventions (see houseganpp/NOTES.md).
  `condition.frame_shift_px` gives the offset from the centred frame back to the original RPLAN /
  Graph2Plan pixel frame.
- `gt_samples/` holds the input layouts traced from the JSON (the same centred frame).
- Renders were checked against a re-implementation of the native `save_samples` colouring: the
  orientation and the rooms and doors match.

## CPU and threads
`run.sh` sets:
- `OMP/MKL/OPENBLAS_NUM_THREADS=1`
- `TORCH_NUM_THREADS=4` (`th.set_num_threads` in `infer.py`)
- `CONVERT_WORKERS=8` for `convert.py` (process pool, at most 16)

Sampling itself is batched on the GPU: 70–100 graphs × 1000 steps per batch, using about 3 GB.
