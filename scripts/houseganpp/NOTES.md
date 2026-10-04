# House-GAN++ (CVPR 2021) — run notes

Scripts: `setup_env.sh` (conda env `fpe-houseganpp`), `download.sh` (weights + public inputs),
`infer.py` (inference driver), `convert.py` (to the common format), `hg_json.py` (HG++ JSON
helpers, also used by HouseDiffusion), `run.sh` (does all steps end to end).

## Environment
- Python 3.10. torch 2.4.1+cu121 and torchvision 0.19.1 from the PyTorch cu121 wheels. The repo pins
  torch 1.8.1, which has no sm_80 (A100) kernels, so we upgraded it. No code changes were needed.
- numpy<2 (1.26.4), opencv-python-headless, matplotlib, networkx, Pillow, svgwrite, webcolors.
- `pygraphviz` comes from conda-forge. `misc/utils.py` does `from pygraphviz import *` at import time.
- `PYTHONNOUSERSITE=1` is set in the scripts because `~/.local/lib/python3.10` holds packages
  (cv2, numpy, ...) that would otherwise shadow the env's own.
- The submodule is unmodified and needs no patch. `infer.py` imports the repo modules with cwd set to
  the submodule.

## Weights
`checkpoints/houseganpp/pretrained.pth` is a copy of the repo's `checkpoints/pretrained.pth`.

## Inputs (main limitation)
RPLAN is only available through a Google Form, and nothing public provides the HG++-preprocessed
test split. The README links point only to RPLAN and the data reader. I searched the group storage
too. `/datawaha/cggroup/datasets/RPLAN` holds only Graph2Plan's preprocessed RPLAN (`data.mat`),
which has room polygons and the front door but **no interior doors**, so it cannot be converted
faithfully into HG++ inputs. So the run uses the **7 RPLAN graphs that are publicly bundled**:
- `external/methods/houseganpp/data/json/{7513,18477,19307,36233,45012,45161}.json` (the repo's demo set)
- `external/data_tools/housegan_data_reader/sample_output/0.json`

These graphs have 5–7 rooms. The paper evaluates per graph-size group (target set held out of
training), and we don't know which split these files belong to. They may well be training samples,
so treat results as qualitative, not as a held-out test.

To run on the real test set later, put the HG++ JSON paths in a list file (one per line, trailing
newline) and run `INPUT_LIST=... VARIANT=rplan_bubble_test bash scripts/houseganpp/run.sh`.

## Inference
`infer.py` follows `test.py` exactly. It runs an initial pass from all-unknown masks, then 10
refinement rounds that fix room types incrementally in sorted type order. It differs only in that:
(a) it can draw several samples per graph (`--n_per_graph`, new noise each time), (b) it uses a
fixed seed, and (c) it saves the raw 64×64 masks plus the graph as `raw/<inputid>_<k>.npz`, along
with the native HG++ rendering `raw/<id>.png`.

Variants:
1. `rplan_bubble_public`: 7 public graphs × 100 samples = **700 samples**. About 90 s on 1 A100,
   using under 3 GB.
2. `rplan-g2p_bubble_test1000_syndoors` (`run_g2p.sh`): **1000 samples**, 1 per graph. The bubble
   diagrams are built by `g2p_to_hgjson.py` from the Graph2Plan-preprocessed RPLAN **test** split
   (`/datawaha/cggroup/datasets/RPLAN/Network/data/data_test.mat`), restricted to the shared ids in
   `data/method_inputs/common_rplan_test/ids_test_1000.txt`, the same ids as iPLAN, DiffPlanner and
   WallPlan. About 6.5 min.
   - **Approximation:** Graph2Plan has no interior doors, so they are *synthesized*. Each
     non-living room gets one door on its wall with the living room if they share one, otherwise
     with the best neighbour (dining > entrance > longest shared wall).
   - The front door comes from the boundary. Room polygons and types are exact, with labels mapped
     as in `read_dd.py`. Adjacency means a shared boundary of at least 3 px. There are no wall gaps.
   - So this is close to RPLAN's bubble diagrams, but not the original HG++ test inputs.

## Conversion
- Each mask goes through: logits > 0, then upsample to 256 px (INTER_AREA, ≥ 0.5). Rooms are then
  painted in node order, with later nodes overwriting earlier ones, the same way HG++'s
  `draw_masks` renders. Each room's visible region is traced per connected component
  (`cv2.findContours` + `approxPolyDP`, eps 1 px). Units are px in the 256 px frame, and the plan is
  centred as HG++ does.
- Door nodes become `doors` (largest component): type 17 maps to `interior_door`, 15 to `front_door`.
- `graph.edges` uses room indices (first component per node). Kind is `door` when an interior door
  node touches both rooms, otherwise `adjacent` (shared wall). The full input node graph, which
  includes door nodes and is what the model sees, is stored in `condition.node_edges`/`nodes`.
  `condition.front_door_rooms` and `condition.missing_nodes` are stored as well.
- Type map: living_room→living_room, kitchen, bedroom, bathroom, balcony, entrance,
  "dining room"→dining_room, "study room"→study, storage, unknown→other.
- `gt_samples/<inputid>.json` holds the input layouts traced exactly from the JSON edge loops, in the
  same centred frame (source `rplan`).

`condition.frame_shift_px` gives the offset from the centred frame back to the original RPLAN /
Graph2Plan pixel frame.

## Known issues
- In 185 of 700 samples at least one room is completely covered by later rooms after painting
  (listed in `condition.missing_nodes`). This is how HG++'s own rendering behaves.
- The generator sometimes produces oversized front-door masks, e.g. for 45161.
