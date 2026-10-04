# External floorplan generation methods

Each method is a git submodule pinned to a fixed commit. Clone with:

```bash
git submodule update --init --recursive --depth 1
```

The checkpoint and framework details below were taken from each repo's README (checked 2026-10-04).

## Methods (`methods/`)

| Dir | Method | Venue | Input → output | Data | Pretrained weights | Framework |
|---|---|---|---|---|---|---|
| `housegan` | House-GAN | ECCV 2020 | bubble diagram → room masks | LIFULL | Dropbox (see repo README) | PyTorch |
| `houseganpp` | House-GAN++ | CVPR 2021 | bubble diagram (+doors) → layout | RPLAN | in repo: `checkpoints/pretrained.pth` | PyTorch |
| `house_diffusion` | HouseDiffusion | CVPR 2023 | bubble diagram → vector polygons | RPLAN | Google Drive ("temporary model") | PyTorch |
| `graph2plan` | Graph2Plan | SIGGRAPH 2020 | boundary + graph → room boxes | RPLAN | none (data in GitHub release) | PyTorch ≥1.5, GUI needs Matlab |
| `iplan` | iPLAN | CVPR 2022 | boundary → types → locations → partition | RPLAN | Google Drive | PyTorch 1.7 |
| `wallplan` | WallPlan | SIGGRAPH 2022 | boundary → wall graph → plan | Kujiale (test samples only) | Drive link in `README.pdf` | PyTorch 1.8 |
| `tell2design` | Tell2Design (T5 baseline) | ACL 2023 | text → room boxes | Tell2Design | none (fine-tune `t5-base`) | PyTorch 1.10 |
| `gsdiff` | GSDiff | AAAI 2025 | uncond. / boundary / bubble → vector graph | RPLAN, LIFULL | Google Drive (5 models) | PyTorch 2.0 |
| `maskplan` | MaskPLAN | CVPR 2024 | partial attributes + graph → layout | RPLAN | Google Drive + `VQ_Pretrained/` | **TensorFlow 2.6** (~1.1 GB repo) |
| `ds2d` | DStruct2Design | arXiv 2024 | JSON constraints → JSON plan (Llama-3-8B LoRA) | RPLAN, ProcTHOR | Google Drive (LoRA) | PyTorch 2.3 |
| `floorplan_rlvr` | Floor-plan RLVR | ACL Findings 2026 | bubble + areas → JSON polygons (Llama-3.3-70B) | RPLAN | HF `ludolara/fp5-{sft,rlvr}-Llama3.3-70B` | PyTorch, vLLM |
| `chathousediffusion` | ChatHouseDiffusion | arXiv 2024 | text → graph → raster | RPLAN / Tell2Design | Tsinghua cloud | PyTorch |
| `diffplanner` | DiffPlanner | TVCG 2025 | boundary (+graph/partial) → vector plan | RPLAN | GitHub release `trained_model.zip` | PyTorch 2.0 |
| `residential_floorplan_diffusion` | Residential Floorplan Diffusion | Autom. Constr. 2024/25 | multi-condition → raster (2-stage) | ? | Google Drive (stage 1 + 2) | PyTorch ≥1.10 |
| `floorplangan` | FloorplanGAN | Autom. Constr. 2022 | room set → vector layout | RPLAN subset | none | PyTorch |
| `what_a_comfortable_world` | What a Comfortable World | EG 2026 short | GPT-2 autoregressive + ergonomic loss | RPLAN | none | PyTorch |
| `floorplan_llm` | FloorplanLLM (Shim et al., RLVR preprint) | preprint 2026 | room spec + connectivity → tokens | RPLAN | none | PyTorch (branch `new_branch`) |
| `msd` | MSD baselines (MHD, GNN+U-Net) | ECCV 2024 | structure + graph → multi-apartment layout | MSD | none | PyTorch/PyG (branch `wip-house-diffusion-msd`) |

## Data tools (`data_tools/`)

- `housegan_data_reader`: the RPLAN reader used by House-GAN++, HouseDiffusion and RLVR.
- `rplan_toolbox`: RPLAN loading and visualization.

You request RPLAN through a Google Form (wutomwu.github.io). Most of these methods need it.

## Methods without public code (not included)

GTGAN (CVPR 2023), the Liu et al. panoptic-refinement method (ECCV 2022), HouseLLM, Cons2Plan, FloorplanSBS, FMLM (CVPR 2026), FloorPlan-DeepSeek, GreenPlanner, GFLAN, GRE-Diff (ECCV 2026), SSPT, HouseMind, DPLAN.

TLC-Plan and FloorPlan-LLaMa have repos, but they contain only a README.

RPLAN's own "Data-driven Interior Plan Generation" code is a Drive zip only, not a GitHub repo.

Excluded as out of scope: Building-GAN (3D volumetric), FloorGenT (robotics line segments), BubbleFormer (outputs bubble diagrams only).

## Dataset code (`datasets/`)

- `cubicasa5k`: the official CubiCasa5k parser.
- `resplan`: the ResPlan repo.

See `docs/datasets.md` for all ground-truth sources, how to get them, and their converters.
