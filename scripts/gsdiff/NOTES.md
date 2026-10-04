# GSDiff (AAAI 2025): run notes

Scripts: `setup_env.sh` (env `fpe-gsdiff`), `download.sh` (5 Drive checkpoints + public RPLAN-derived json),
`prepare_inputs.py` (bubble diagrams, boundary images and GT from that json), `infer.py` (all three settings),
`convert.py` (structural graph -> common format), `run.sh` (end to end; `bash scripts/gsdiff/run.sh [uncond|topo|boun]`).

## Environment
- Python 3.10, torch 2.0.1+cu118, torchvision 0.15.2, numpy 1.26.0, opencv-python-headless 4.9.0.80, shapely 2.0.6,
  networkx 3.1, scipy 1.10.1. These follow the repo's `requirements.txt`. cu118 runs on the A100, so no upgrade was needed.
- The submodule is unmodified (no patch). The repo's `test_*.py` scripts hard-code `/home/user00/...` paths, read
  pre-processed RPLAN `.npy` files and compute FID against GT renders. `infer.py` instead re-implements their sampling
  loops verbatim (cosine schedule, 1000 DDPM steps, x0 thresholding at 0.5/0.75, stage-2 edge transformer,
  `get_cycle_basis_and_semantic_3_semansimplified`, `merge_points=False`, `align_points=False`, resolution 512) and
  imports the models and utils from the submodule. The only difference is where the inputs come from (below).

## Checkpoints (`checkpoints/gsdiff/`, extracted from the README's Drive files)
| setting | node diffusion | edge model | encoder |
|---|---|---|---|
| uncond | `outputs/structure-1/model1000000.pt` | `outputs/structure-2/model_stage2_best_061000.pt` | none |
| topo | `topo-params/structure-80-106-2/model1000000.pt` | `topo-params/structure-56-35-interval1000/model_stage2_best_076000.pt` | `topo-params/structure-57-16/model_stage0_best_006000.pt` (graph AE) |
| boun | `outputs/structure-81-106-3/model1000000.pt` | `outputs/structure-56-36-interval1000/model_stage2_best_065000.pt` | `structure-78-12/model_stage0_best_006700.pt` (boundary CNN) |

These are the paths used in `test_main.py`, `test_topo.py`, `test_boun.py` and `prerunningCNN.py`. The "sloping walls" and LIFULL models were not used.

## Inputs and deviations
GSDiff's own test inputs come from raw RPLAN PNGs (Google-Form gated) through `datasets/rplan-process*.py`, using their own
3000-plan test split, which is not published. Instead:
- **uncond (`rplan_uncond`)**: needs no inputs. `test_main.py` uses the test loader only for tensor shapes and an all-ones
  attention matrix. We draw 3000 samples (the test-set size the paper uses per run), seed 0.
- **topo (`rplan_bubble_test1000`) and boun (`rplan_boundary_test1000`)**: inputs come from DiffPlanner's public release
  (`dataset.zip`, i.e. the Graph2Plan RPLAN test split as json with room polygons tiling the interior, 6 room
  classes and adjacencies, in the RPLAN 256 px frame). We use the shared id list
  `data/method_inputs/common_rplan_test/ids_test_1000.txt`. All 1000 have 4-8 rooms, which is the range GSDiff topo supports.
  For these ids, the json matches the local Graph2Plan copy `/datawaha/cggroup/datasets/RPLAN/Network/data/data_test.mat`
  exactly (boundary and room count checked for 1000/1000).
  - Bubble diagram: nodes are the rooms, mapped to GSDiff's 7 classes (DiffPlanner 0 living->0, 1 bedroom->1, 2 kitchen->3,
    3 bathroom->4, 4 balcony->5, 5 storage->2). Edges are pairs of rooms sharing a wall segment longer than 1 px, recomputed from the polygons
    (GSDiff's rplan-process8 definition). This agrees with Graph2Plan's rEdge on 91% of pairs.
    *Approximation:* DiffPlanner merges Wall-in into living, while GSDiff merges it into storage. GSDiff's GT bubble diagrams come from
    faces of its wall graph, which can differ slightly from Graph2Plan's room segmentation.
  - Boundary: the Graph2Plan boundary polygon (without the 2 front-door points) drawn exactly like
    `rplang_edge_semantics_simplified_78_10_prerunCNN` (black lines of width 7/5/3/1 plus 7x7 corner squares on 256x256 white), then
    passed through the frozen boundary CNN to get the 16x16 feature map (as in `prerunningCNN.py`, computed on the fly).
    *Approximation:* GSDiff's boundary is the outer wall-centerline loop of its wall graph. Graph2Plan's is the interior outline,
    which differs by about half a wall thickness (2-3 px).
  - *Possible train/test overlap:* GSDiff was trained on its own split of RPLAN, so some of these Graph2Plan test plans may be
    in its training set. This cannot be checked without their split.
- GT (`gt_samples/`, topo/boun only): the DiffPlanner json rooms (6 classes, canonical names), boundary, front door,
  and the recomputed adjacency as `graph.edges`, in the RPLAN 256 px frame.

## Conversion (`convert.py`)
- Room polygons are the faces of the generated wall graph, with semantics voted by their corners. Both are computed by GSDiff's own
  `get_cycle_basis_and_semantic_3_semansimplified` at inference time and stored in `raw/vr4stat_<id>.npy`.
- Non-simple faces (pinched at a repeated vertex, which happens occasionally) are split with `shapely.make_valid` into polygonal parts
  that keep the face label. Parts under 1 px^2 are dropped.
- Classes: 0 LivingRoom/DiningRoom/Entrance -> living_room, 1 Master/Child/Study/Second/GuestRoom -> bedroom,
  2 Storage/Wall-in -> storage, 3 kitchen, 4 bathroom, 5 balcony. Class 6 (External) is excluded from face voting by GSDiff.
  Study rooms are therefore labelled bedroom, and dining/entrance are labelled living_room.
- Coordinates: GSDiff's 512 frame divided by 2 gives the RPLAN 256 px frame, the same frame as the GT and conditions.
- `boundary` = outline of the union of the generated rooms. No doors or windows (GSDiff doesn't generate them).
  The wall graph itself (corners and adjacency) is kept in `raw/graph_<id>.npz`.

## Runtime (1 A100, GPU 1, batch 1000)
- uncond: 3000 samples in 864 s (stage 1 diffusion 685 s).
- topo: 1000 samples in 811 s (batch 1000). boun: 1000 samples in 1283 s (batch 250; batch 1000 OOMs on a 40 GB A100).
- Converted room counts: uncond averages 6.65 rooms (9 of 3000 samples have fewer than 3 rooms). topo averages 6.73 rooms vs. 6.75 in GT.
- 0 cycle-extraction failures in all variants. Spot-checked renders against GT: boundary outputs follow the GT outline,
  and topo outputs follow the room program. Occasional failures: missing faces, or a living room labelled as a bedroom.

## Known issues
- One sample per input (seed 0). The paper averages 5 runs and evaluates on 757 (topo) / 378 (boun) plans of its own test split.
- The original test scripts' FID/KID computation is not reproduced here (it is done by the shared evaluation instead).
