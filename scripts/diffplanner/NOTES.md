# DiffPlanner (TVCG 2025) — run notes

Pipeline: `setup_env.sh` -> `download.sh` -> `run.sh [ids] [variant] [batch]` (sampling + post-processing in
`fpe-diffplanner`, conversion/rendering in `fpe`). GPU via `CUDA_VISIBLE_DEVICES` (default 2).

## Environment (`fpe-diffplanner`)
- Python 3.9 (conda-forge, with mpi4py + openmpi — `dist_util.py` imports mpi4py), torch 2.0.1+cu118
  (original 2.0.0), numpy 1.21.5, scipy 1.10.1, shapely 2.0.6, opencv-python-headless 4.8.1, blobfile 2.1.1.
- Harmless warnings: c10d IPv6 socket warnings at `setup_dist()`.

## Weights
- `trained_model.zip` (GitHub release) -> `checkpoints/diffplanner/trained_model/<stage>/scripts/trained_model/`.
  Boundary-only setting uses `node_diff/b_model300000.pt`, `adjacency_diff/bncsl_model300000.pt`,
  `partitioning_diff/bncsla_model300000.pt` (README's "boundary with entrance, no additional conditions").

## Inputs
- DiffPlanner's own preprocessed json (`dataset.zip` release, = Graph2Plan `Data.zip` run through
  `dataset/data_preparation.py`): 12002 test plans. The Graph2Plan `Data.zip` (`Network/data/data_test.mat`,
  12110 plans; identical md5 to /datawaha/cggroup/datasets/RPLAN/Network/data/data_test.mat) is used for GT.
  108 Graph2Plan test plans are filtered out by DiffPlanner (boundary >40 corners, rooms <150 px^2, ...).
- `download_g2p_data.sh` also writes the shared id lists `data/method_inputs/common_rplan_test/ids_test_all.txt`
  (the 12002) and `ids_test_1000.txt` (fixed random subset, seed 0, used for iPLAN and WallPlan).

## How it is run (no submodule changes)
- `sample.py` scripts use cwd-relative paths (`../../dataset/dataset_json/data_test.json`,
  `../../output/output_json`). `prepare_work.py` builds `raw/_work/` mirroring that layout (with the test json
  filtered to the requested ids) and each stage runs with its cwd there; scripts are called from the submodule.
- Stages: NodeDiff (rooms: type/size/location) -> AdjacencyDiff -> PartitioningDiff (boxes), 1000-step DDPM,
  batch 1024, one sample per boundary (no seed control in the original code — not bit-reproducible).
- `postprocess.py` calls the official `output/post_processing.main()` (alignment to boundary, room polygons,
  doors & windows via `decorate.py`) and writes `raw/b_post.json` (the original `__main__` overwrites its input).
  Raw unprocessed samples: `raw/b_raw.json`.

## Output conversion (`convert.py`)
- Room polygons = `r_boundary_aligned` (256 px RPLAN frame, same as the Graph2Plan boundary). 6 room classes:
  LivingRoom, Bedroom, Kitchen, Bathroom, Balcony, Storage.
- Rooms whose aligned polygon is empty (post-processing subtracts higher-priority rooms; such rooms are fully
  covered and also invisible in the official box-painting visualization) are dropped:
  1635 of 81223 rooms, in 1552 of 12002 plans (`raw/convert_stats.json`).
- Doors/windows: decorate.py segments `[room, x, y, dx, dy, dir]` -> thin rectangles (half width 1.5 px, matching
  the official 512 px drawing); front door = rectangle on the first boundary segment (entrance).
- `graph.edges` = `adjacencies_aligned` (box adjacency after alignment).
- `gt_samples/`: Graph2Plan GT for the same ids (RPLAN 13-class labels, polygons `rBoundary`).

## Runtime
- Full test set (12002 plans): 1298 s for the three diffusion stages + post-processing on one A100 (GPU shared
  with other jobs).
