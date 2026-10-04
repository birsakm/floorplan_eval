# floorplan_eval

Runs existing synthetic floorplan generation methods and evaluates the quality of what they produce against real ground-truth floorplans.

Every method's output and every ground-truth dataset is converted to one shared JSON format, so one evaluation pipeline can read all of them.

## Repository layout

```
external/methods/     # generation methods as git submodules (pinned commits); see external/README.md
external/data_tools/  # RPLAN data readers
external/datasets/    # dataset parser repos (CubiCasa5k, ResPlan)
fpeval/               # shared Python package: common format, renderer
fpeval/datasets/      # GT dataset -> common format converters
scripts/<method>/     # setup_env.sh, download.sh, run.sh, convert.py, NOTES.md per method
scripts/datasets/     # GT dataset download scripts
docs/                 # floorplan_format.md (format spec), datasets.md (GT sources)
```

These paths are gitignored and live on disk only:
- `data/`: GT datasets, raw and converted, plus method inputs
- `checkpoints/`: pretrained weights
- `outputs/`: generated samples

## Setup

```bash
git clone --recursive https://github.com/birsakm/floorplan_eval.git
# or, in an existing clone:
git submodule update --init --recursive --depth 1
```

The shared environment `fpe` (Python 3.11) runs conversion, rendering and evaluation:

```bash
mamba create -n fpe -c conda-forge python=3.11 numpy pillow shapely opencv matplotlib scipy tqdm gdown
pip install lxml networkx pandas pyarrow
```

Each method has its own environment, `fpe-<method>`. Run one method end to end with:

```bash
bash scripts/<method>/setup_env.sh   # create the conda env
bash scripts/<method>/download.sh    # weights + inputs
bash scripts/<method>/run.sh         # inference -> outputs/<method>/<variant>/
```

`scripts/<method>/NOTES.md` covers each method's changes from upstream, version upgrades (old pinned PyTorch builds don't support A100s), runtime and known issues.

## Common format

One JSON file per floorplan. It holds rooms as polygons with a canonical room type, plus optional boundary, doors, windows, walls, room graph and input condition. The spec is in [docs/floorplan_format.md](docs/floorplan_format.md).

Outputs are laid out as:

```
outputs/<method>/<variant>/
  raw/          # native method output
  samples/      # common-format JSON
  renders/      # PNG renderings
  gt_samples/   # GT for the same inputs (paired evaluation), when available
  run_info.json # commit, checkpoint, command, sample counts, runtime
```

Render any sample directory with `python -m fpeval.render <dir>`.

## Generation methods

| Method | Venue | Condition | Variants run | Status |
|---|---|---|---|---|
| House-GAN | ECCV 2020 | bubble diagram | LIFULL test group D (5000) | done |
| House-GAN++ | CVPR 2021 | bubble diagram + doors | RPLAN test subset (1000, made-up interior doors), 7 public graphs ×100 | done |
| HouseDiffusion | CVPR 2023 | bubble diagram | RPLAN test subset (997 of 1000, made-up interior doors), 7 public graphs ×100 | done |
| GSDiff | AAAI 2025 | none / bubble / boundary | uncond (3000), bubble (1000), boundary (1000) | done |
| iPLAN | CVPR 2022 | boundary | RPLAN test subset (1000) | done |
| DiffPlanner | TVCG 2025 | boundary | RPLAN full test (12002) | done |
| WallPlan | SIGGRAPH 2022 | boundary | RPLAN test subset (1000), bundled samples (500) | done |
| MaskPLAN | CVPR 2024 | boundary (+ partial attributes) | RPLAN test (500 each) | done |
| ChatHouseDiffusion | arXiv 2024 | text → graph | Tell2Design test (2308) | done |
| DS2D | arXiv 2024 | JSON constraints (Llama-3-8B LoRA) | RPLAN 5–8 rooms, ProcTHOR (10 variants) | in progress |
| Residential Floorplan Diffusion | Autom. Constr. 2024/25 | room masks | — | blocked: weights need Drive access approval |
| Floor-plan RLVR | ACL Findings 2026 | bubble + areas (Llama-3.3-70B) | — | not run yet (needs all 4 GPUs) |

"RPLAN test subset" is a fixed set of 1000 plans (seed 0) drawn from the Graph2Plan RPLAN test split, listed in `data/method_inputs/common_rplan_test/ids_test_1000.txt`. The boundary methods are all compared on it. The Graph2Plan data has no interior doors, so for House-GAN++ and HouseDiffusion they are made up (one per non-living room, on the wall shared with the living room or a neighbour). This means neither run reproduces the papers' own test set.

Methods without pretrained weights (Graph2Plan, Tell2Design, FloorplanGAN, MSD baselines, …) are included as submodules but not run. See [external/README.md](external/README.md).

## Ground-truth datasets

| Dataset | Plans | Units |
|---|---|---|
| CubiCasa5k | 6,103 floors | m |
| ResPlan | 17,000 | m |
| RPLAN (Graph2Plan version) | 80,436 | px (18/256 m) |
| House-GAN LIFULL | 145,811 | px |
| Swiss Dwellings v3 | 46,830 | m |
| Modified Swiss Dwellings (MSD) | 4,167 | m |
| Tell2Design | 80,788 | px |
| MagicPlan | 155,329 | m |
| ProcTHOR-10K (synthetic) | 12,000 | m |

[docs/datasets.md](docs/datasets.md) covers sources, licenses, splits, conversion notes, and datasets that need a form or registration. Convert one with:

```bash
conda run -n fpe python -m fpeval.datasets.<name> --workers 16
```

## Background

The PI's design notes cover the broader pipeline: program sampling → layout → furniture → decoration. They also give the rule set (size, shape, overlap, density) that the evaluation metrics will build on.
