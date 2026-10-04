# WallPlan (SIGGRAPH 2022) — run notes

Pipeline: `setup_env.sh` -> `download.sh` -> `run.sh g2p [ids] [variant]` and/or `run.sh bundled`
(inference in `fpe-wallplan`, conversion/rendering in `fpe`). GPU via `CUDA_VISIBLE_DEVICES` (default 2).

## Environment (`fpe-wallplan`)
- Python 3.8, torch 1.8.1+cu111 / torchvision 0.9.1+cu111 (same version as the paper; the cu111 wheel has sm_80),
  numpy 1.19.5, scipy 1.6.3, opencv-python-headless 4.5.5, shapely 1.8.5, torchnet 0.0.4 (only for training).
- On first model construction torchvision downloads ImageNet resnet34 weights to ~/.cache/torch (`DinkNet34_no`
  uses `resnet34(pretrained=True)`); they are overwritten by the WallPlan checkpoints.

## Weights
- Drive link in `README.pdf`. Text extraction splits the file ID over a line break:
  `https://drive.google.com/file/d/1Ae9fisgl-r3AJUq_16VSygw0Jeh-` + `XooS/view` -> ID
  `1Ae9fisgl-r3AJUq_16VSygw0Jeh-XooS` (33 chars, verified by download). It is a 460 MB RAR
  (`Boundary_constraint.rar`) with `WindowLiving.pth, WindowOther.pth, LabelNet.pth, GraphNet.pth`
  -> `checkpoints/wallplan/Boundary_constraint/`. Hybrid-constraint models are not released.

## Inputs / variants
1. `rplan_boundary_test1000` (main, comparable with iPLAN/DiffPlanner): the 1000 Graph2Plan RPLAN test
   boundaries of `data/method_inputs/common_rplan_test/ids_test_1000.txt`, converted to WallPlan's input:
   boundary polygon (corners only, (row, col)), front door (centre of the first boundary segment, ori 0 = on a
   horizontal wall) -> 120 px masks with WallPlan's own drawing primitives (`(p - 8) // 2`, 2 px boundary lines,
   5 px boundary variant, filled inside minus boundary, 5x7 door box) and start node = lexicographically smallest
   corner. Validation: rebuilding the masks of the 500 bundled pkls from their wall graphs with the same code
   gives identical boundary/5px/door masks for 500/500 and identical inside masks for 498/500.
   Small deviation: WallPlan's training wall graph sits on wall centre lines, the Graph2Plan boundary on the inner
   face of the exterior wall (~1-2 px at 256 px, <=1 px at 120 px).
2. `bundled_boundary_test500`: the 500 `test/input/*.pkl` shipped with the repo (as in `Boundary_Test.py`).
   Their file names are NOT RPLAN ids (e.g. `45304.pkl` has a different boundary than RPLAN plan 45304), and the
   pkls contain no room labels, so there is no GT for this variant (boundary only).

## Headless driver (`infer.py`) — no submodule changes
- Re-implements `WallPlan_Main.generate_from_val()` (window assembly -> LabelNet/GraphNet coupling -> room
  circles -> door/window arrangement -> render). `Boundary_Test.py` is not imported because it sets
  `CUDA_VISIBLE_DEVICES="0"` at import time.
- Plans whose generated wall graph fails WallPlan's own checks (overlapping / non-axis-aligned junctions) are
  skipped, exactly like the original script (it silently writes no PNG): 8/1000 (g2p), 1/500 (bundled).
- Networks are deterministic; window/door arrangement uses `random`/`np.random` (seeded 0).
- Raw: `raw/<name>.pkl` (wall graph, room circles, front door, interior doors, windows, input masks) and
  `raw/png/<name>.png` (the official 512 px rendering).

## Output conversion (`convert.py`)
- Room polygons = room circles over wall-graph junctions (wall centre lines), 120 px -> 256 px RPLAN frame
  (`2p + 8`), (row, col) -> (x, y). 6 classes: LivingRoom, Bedroom, Kitchen, Bathroom, Balcony, Storage.
- Doors (front + interior, incl. balcony doors spanning 2/3 of the wall) and windows as rectangles with the sizes
  used by `floorplan_render()`.
- g2p variant: boundary = Graph2Plan input boundary; `gt_samples/` = Graph2Plan GT for the same ids.
  bundled variant: boundary = outer ring of the input wall graph (pkl's 256 px frame).

## Runtime
- ~0.7 s/sample on an idle GPU (mostly CPU-side graph search); 1000 g2p plans took 2034 s and 500 bundled
  plans 332 s while sharing the GPU/CPU with other jobs.
