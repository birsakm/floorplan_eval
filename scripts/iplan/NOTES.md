# iPLAN (CVPR 2022) — run notes

Pipeline: `setup_env.sh` -> `download.sh` -> `run.sh [ids] [variant]` (inference in `fpe-iplan`,
conversion/rendering in `fpe`). GPU via `CUDA_VISIBLE_DEVICES` (default 2).

## Environment (`fpe-iplan`)
- Python 3.8, torch 1.8.1+cu111 / torchvision 0.9.1+cu111 (original: Python 3.6, torch 1.7.0, CUDA 10.1,
  which has no sm_80 kernels for the A100), numpy 1.21.6 (<1.24, code uses `np.int`), scipy 1.7.3,
  opencv-python-headless 4.5.5, shapely 1.8.5, scikit-image 0.19.3, torchnet 0.0.4.

## Weights
- Google Drive folder from the README -> `checkpoints/iplan/iPLAN/{room_type,room_location,room_partition}`
  (roomtype_cvae_150, living_*_300, location_*_100, resnet18-5c106cde, G_net_210, renderer.pkl).

## Inputs
- iPLAN needs RPLAN converted to its own 128 px `.mat` format; only 2 such files ship with the repo and the
  RPLAN images themselves are behind a Google Form. We instead use the public **Graph2Plan-preprocessed RPLAN**
  (`Data.zip` GitHub release -> `Network/data/data_test.mat`, 12110 test plans = exactly iPLAN's `data/test.txt`).
  Conversion: iPLAN `Boundary` (row, col, dir, isNew) at 128 px = `floor(G2P[:, [y, x]] / 2)`.
  Checked against the two bundled files (`0.mat`, `1.mat`, which are in Graph2Plan's validation split):
  same vertices/order/directions, coordinates agree within 1 px at 128 px (iPLAN derives them from a 2x2
  max-pooled mask; it also widens short front doors by 1 px).
- Test ids: `data/method_inputs/common_rplan_test/ids_test_1000.txt` = fixed random 1000-plan subset (seed 0) of
  DiffPlanner's 12002-plan test set (Graph2Plan test split minus plans DiffPlanner filters out). The same 1000
  boundaries are used for WallPlan and are contained in the DiffPlanner full-test run.

## Headless driver (`infer.py`) — no submodule changes
- Chains `synth/test_roomtype.py` -> `test_roomlocation.py` -> `test_roompartition.py` per sample (the originals
  process `mat_list[100:200]` of a folder, mutating the .mat files in place; partition results were never saved).
- Bugs worked around by shims in the driver (no patch file needed):
  - `test_roompartition.py` calls `models.FloorPlanRNNTest`, which does not exist; the test class is
    `FloorPlanRNN` in `room_partition/models/floorplan_rnn_test.py`.
  - `room_partition/models/loss_layer.py` imports `room_partition.models.utils.box_utils`, which does not exist
    (file is `room_partition/utils/box_utils.py`) -> aliased in `sys.modules`.
  - `floorplan_rnn.py` does `import utils` (top-level) -> `room_partition/` is put on `sys.path`.
  - `RendererNet` loads `./weights/renderer.pkl` relative to cwd -> driver chdirs into `raw/_work/`.
- Hyper-parameters as in the scripts: CVAE threshold 0.7, <=50 location attempts, partition fine-tuning
  lr 10, coverage 1, inside 0.5, mutex 0, max 200 iterations. One sample per boundary, torch/numpy seed 0.
- Failures: if room locating fails 50 times the plan is skipped (logged in `raw/_log.json`, also in run_info).

## Output conversion (`convert.py`)
- Raw: `raw/<name>.npz` with the 128x128 label layout, an instance map painted in the same order as iPLAN's
  `get_image` (living room = the remainder region, index 0), boxes, types, centres, boundaries.
- Each room instance is vectorized per connected component along pixel edges and scaled x2 into the 256 px
  RPLAN/Graph2Plan frame (x right, y down). Boundary = Graph2Plan input boundary; front door = rectangle on the
  first boundary segment. iPLAN predicts no interior doors/windows.
- Small unassigned interior slivers (label 16 left after fine-tuning) are not rooms and are dropped.
- Room labels: full RPLAN 13-class vocabulary (MasterRoom/SecondRoom/ChildRoom/GuestRoom -> bedroom,
  Wall-in -> closet, ...).
- `gt_samples/`: Graph2Plan GT for the same ids (room polygons `rBoundary`, RPLAN types, adjacency `rEdge`).

## Runtime
- ~2.1 s/sample on an idle A100 (most time in the 200-iteration box fine-tuning and Python loops);
  ~2.5-4 s/sample with GPU/CPU sharing. 12002 samples would take ~7-12 h, hence the 1000-plan subset.
- Actual run (`rplan_boundary_test1000`): 3576 s for 1000 inputs (GPU/CPU shared with other jobs);
  966 samples, 34 inputs failed ("room location failed after 50 attempts", listed in run_info.json).
- Sanity check (no GT leakage: gt_* fields are never read by the inference code; iPLAN train/val/test lists
  are disjoint): sorted room-type multiset equals GT for 179/966 iPLAN, 163/992 WallPlan, 152/1000 DiffPlanner.
