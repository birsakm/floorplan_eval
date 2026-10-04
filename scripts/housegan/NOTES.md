# House-GAN (ECCV 2020): run notes

Scripts: `setup_env.sh` (conda env `fpe-housegan`), `download.sh` (weights + LIFULL data from Dropbox),
`infer.py` (batched inference driver), `convert.py` (raw masks -> common format), `run.sh` (end to end).

## Environment
- Python 3.9, torch 2.1.2+cu118 / torchvision 0.16.2, numpy 1.23.5, Pillow 9.5.0, opencv-python-headless 4.8.1.78,
  scikit-image 0.21.0, pycocotools 2.0.7, networkx 2.8.8, matplotlib 3.7.3.
- **Deviation:** the repo pins torch 1.5.0+cu101, which has no sm_80 (A100) support, so we upgraded torch.
  The checkpoint loads with `strict=True` and no code changes were needed.
- `utils.py` runs `from pygraphviz import *`, but it is only used to draw graphs. `infer.py` puts an
  empty stub module in `sys.modules`, so no graphviz is needed. A `mamba install pygraphviz` hung on the shared package-cache lock.
- The submodule is unmodified. No patch is needed.

## Weights / data
- The Dropbox folder holds `exp_demo_D_500000.pth`, the only released model. It was trained with group **D** (10-12 rooms) held out.
  It is stored as `checkpoints/housegan/exp_demo_D_500000.pth`.
- The data is in `data/method_inputs/housegan/`: `dataset_paper/{train,valid}_data.npy` (room types + boxes)
  and `housegan_clean_data.{npy,pkl}` (the full vector data with edges and doors).

## Test set
- This follows the repo's eval scripts (`variation_*.py`): `FloorplanGraphDataset(split='eval', target_set='D')` on
  `dataset_paper/train_data.npy`, i.e. graphs with 10-12 rooms (excluded from training for this model).
  It uses the first 5000 of them (the loader's cap): 1960/1799/1241 graphs with 10/11/12 rooms.
- One sample per graph (seed 0), batched 64 graphs at a time. The GAN is graph-batched like in training, so batching does not change the model.
- Only the group-D model is public, so the paper's per-group results for A, B, C and E cannot be reproduced.

## Output: `outputs/housegan/lifull_bubble_testD/`
- `raw/<graph>_v0.npz`: 32x32 generator masks (float16, [-1,1]), node types, input edge triples, GT boxes.
- `samples/<graph>.json`: masks are binarized at >0 and upsampled to 256 (INTER_AREA, threshold 127, as in the repo's
  `draw_masks`). Then `cv2.findContours` (external), one polygon per connected component, dropping components < 64 px^2.
  Coordinates are in a 256 px frame with x = mask column and y = mask row. This is the convention of the repo's
  `mask_to_bb`/`bb_to_im_fid` paper visualizations. Rooms are sorted by decreasing area, as `bb_to_im_fid` draws them.
  `graph.edges` = input bubble-diagram adjacencies (type `"adjacent"`), remapped to the first polygon of each node.
- `gt_samples/<graph>.json`: GT room **boxes** of the same graphs (House-GAN data only has boxes), with the loader's
  centering and the same axis convention.
- Room classes: 1 living_room, 2 kitchen, 3 bedroom, 4 bathroom, 5 missing->other, 6 closet, 7 balcony, 8 corridor,
  9 dining_room, 10 laundry_room->laundry. In LIFULL, "kitchen" (2) is usually the large LDK room, and living_room is rare.

## Runtime
- Inference for 5000 graphs takes about 90 s on 1 A100. Conversion and rendering take about 1 min.

## Known issues
- About 9% of input rooms (5176 of about 55k) produce an empty mask (no pixel > 0), so they are missing from the sample.
  This is the generator's own output, usually for tiny closets and bathrooms. Masks also overlap and can be fragmented
  (several components), which is typical for House-GAN.
- No building boundary, doors or walls are generated (`boundary`=null).
