# MaskPLAN — notes

Variants (RPLAN test split; first 500 ids of `Processed_data/Test_set.npy`; 1 sample per input):

| variant | condition (paper Table 1) | samples | inference time |
|---|---|---|---|
| `rplan_partial25` | "Our II": boundary + front door + random 25% of each of T, L, A, S, R (upstream default) | 500 | 3445 s |
| `rplan_boundary` | "Our I": boundary + front door only | 500 | 3692 s |

Both variants ran at the same time on GPU 3 (about 1 h wall clock in total). A site takes ~5.5 s
alone and ~7 s when two jobs share the GPU. Inference is autoregressive: about 50 Keras `predict`
calls per site.

We ran 500 sites instead of upstream's default 1000 to stay within the ~2 GPU-h budget. Run more with
`bash scripts/maskplan/run.sh 1000 rplan_partial25`; it resumes per site.

## Weights / inputs
- `All_Base_Deep_cross.7z` (Drive id `1mzZdK313lYybr4GMN6_pF6lW3tRFwLjM`, 10.4 GB) →
  `checkpoints/maskplan/MaskPLAN_Trained/All_Base_Deep_cross/All.*`. This is the hybrid vec-img
  "Deep" model that `Inference/MaskPLAN_Inference_iterate_cross_Deep.py --model Base` loads; the
  README calls it the paper's main hybrid model. The other three Drive archives (ablations) were not
  downloaded; their ids are in `download.sh`.
- VQ-VAE (`VQ_Pretrained/mix_5564`) and the preprocessed RPLAN attributes (`Processed_data/`) are in
  the repo.
- Boundary images: the bundled `parsed_img/img_room_sqe/0.7z` is extracted to
  `data/method_inputs/maskplan/parsed_img/`. Only the 8079 test ids are needed. Extracting 80k tiny
  files on this NFS is very slow; `7z x -i@list` with the test ids is faster.

## No submodule changes
`infer.py` imports the upstream inference script unmodified and points its module-level
`REPO_ROOT` at a symlink "shadow root" in `outputs/maskplan/<variant>/raw/_shadow_root`. That root
links:
- `Processed_data`, `VQ_Pretrained` → submodule
- `parsed_img` → data dir
- `MaskPLAN_Trained` → checkpoints
- `Inference` → `raw/native`

The per-site loop is the upstream `__main__` loop, plus two additions:
- It seeds NumPy and TF per site (seed = site id), so runs are reproducible and resumable. Upstream
  is unseeded.
- It saves the partial-input masks `M_*` and the predicted attributes `In_*` to `raw/meta/<site>.npz`.

Upstream outputs are in `raw/native/Base_Deep_cross/iteration/{raw,post,partial_input}`.

## Env (`fpe-maskplan`)
- Python 3.9, conda-forge `cudatoolkit 11.2` + `cudnn 8.1`, pip `tensorflow-gpu==2.6.0`,
  `keras==2.6.0`, numpy 1.19.5, protobuf 3.20.3, h5py 3.1.0, opencv-python-headless 4.5.5.64,
  shapely 2.0.2. `LD_LIBRARY_PATH` is set via an activate.d hook.
- TF 2.6 has no sm_80 kernels, but it ships `compute_80` PTX, so it runs on the A100 via driver JIT.
  The "Failed to launch ptxas" warning is harmless. GPU verified; `TF_FORCE_GPU_ALLOW_GROWTH=true` so
  two jobs fit on one GPU.

## Conversion (`convert.py`)
- Generated samples come from the upstream post-processed image (`post/<site>.png`, 128×128). Upstream
  post-processing turns each room into a rectangle, aligns it to the boundary and neighbours, and
  fills leftover gaps.
- Rooms are painted with the exact T_list colours, so each type mask is matched exactly. One polygon
  per 4-connected component, traced on an 8× upsampled mask and simplified with eps 0.75 px.
- MaskPLAN types: 1 Living, 2 Bathroom, 3 Storage/closet (RPLAN Storage + Wall-in), 4 Bedroom (all
  bedroom kinds), 5 Kitchen, 6 Dining, 7 Balcony. RPLAN Entrance is dropped in MaskPLAN's
  preprocessing.
- Upstream quirk: leftover gaps not touching the living room are filled with T_list[7] = balcony,
  so some spurious "balcony" pieces are part of the method's output.
- Boundary = upstream `load_boundary_points()` on the boundary PNG. Units are px in a 128-px frame
  (RPLAN 256 px / 2).
- `condition` lists the given types and the number of given attributes, from `raw/meta`.
- GT (`gt_samples/`): the exact RPLAN room polygons for the same site ids (site id = RPLAN file id),
  read from the local Graph2Plan-format copy `/datawaha/cggroup/datasets/RPLAN/Network/data/data_*.mat`
  (all 500 found) and scaled ×0.5 to the 128-px frame. They use full RPLAN labels (Entrance etc.).
  - Some records have empty room polygons; those rooms are skipped.
  - Fallback when a site is missing from that copy: MaskPLAN's own 20-level quantised GT boxes,
    drawn as upstream `Generate_GroundTruth.py --format vec` does.
- The converters use an explicit 8-process pool, with OMP/BLAS/OpenCV threads set to 1.

## Spot check / known issues
- Renders look like the paper's figures: rectangular rooms that tile the boundary, with the living
  room as the connected leftover. The partial25 layouts resemble GT more than boundary-only ones.
- Mean rooms per plan: partial25 6.2, boundary-only 5.1, GT 6.8. Boundary-only tends to predict fewer
  rooms, and overlapping rectangles are merged or dropped by the upstream post-process.
- Upstream did not record the random state for the Table-1 runs, so the exact masks used there
  cannot be reproduced.
