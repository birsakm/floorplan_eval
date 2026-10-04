# ChatHouseDiffusion — notes

Variant `t2d_test_textgraph`: the full Tell2Design test split, 2308 inputs, 1 sample per input.
Outputs are in `outputs/chathousediffusion/t2d_test_textgraph/`.

## Pipeline / LLM stage
- Upstream: text → (LLM, OpenAI-compatible API; the paper uses moonshot-v1-8k) → room-graph JSON →
  Graphormer + Imagen-style diffusion (64×64 label map) conditioned on the boundary mask.
- **No LLM was called.** We used the authors' released test data `kimi_test_data.rar`
  (https://cloud.tsinghua.edu.cn/f/2844208e0c344d18bd72/), in which the Tell2Design test texts were
  already parsed by moonshot-v1-8k. The kimi data contains `text_test/json.csv` (graph JSON per
  sample), `mask_test/` (boundary + front door) and `image_test/` (GT label maps). This is the same
  input the upstream `test.py` uses (`../chat_test_data/0614-kimi`).
- The raw Tell2Design natural-language descriptions are not in that archive. `condition.graph_json`
  stores the parsed graph.

## Weights
- `predict_model.rar` from https://cloud.tsinghua.edu.cn/f/a01a8205be55462685fd/?dl=1 →
  `checkpoints/chathousediffusion/{model-98.pt, params.pkl}` (milestone 98, the one `predict.py` and
  `test.py` load).
- params: `onehot=False` (grayscale label/17), DDIM with 50 sampling steps, `cond_scale=1`,
  Graphormer conditioning (T5 node features come from the bundled `t5_feature.pkl`, so no HF
  download is needed).

## Inference (`scripts/chathousediffusion/infer.py`)
- Same as `Trainer.val(load_model=98)` from upstream `test.py`: EMA model, `seed_torch()` default
  seed 1029, pixels outside the boundary forced to External (13). Batch size 64 (upstream: 32), so
  per-sample noise differs from an upstream run.
- It saves raw uint8 label maps (`raw/label/<id>.png`, RPLAN labels 0..17) instead of only the
  RGB previews + IoU that upstream writes.
- It must run with cwd = the submodule root (`t5_feature.pkl` is opened by relative path). The
  submodule is not modified.
- Runtime: 366 s of sampling for 2308 samples on 1×A100 (GPU 3, shared with MaskPLAN), plus about
  1 min of model load.

## Conversion (`convert.py`)
- Room labels 0..12 are vectorized per 4-connected component. Contours are traced on an 8× upsampled
  mask so polygons follow pixel edges, then simplified with approxPolyDP at eps 0.5 px. Doors:
  FrontDoor (15) and InteriorDoor (17) components. Walls (14, 16) are dropped.
- Boundary = interior of the condition mask (`mask_test != 255`).
- Types: RPLAN labels map to canonical types. Master/Second/Child/Guest → bedroom,
  StudyRoom → study, Wall-in → closet.
- `gt_samples/` + `gt_renders/`: the Tell2Design GT label maps (`image_test/`), converted the same way.
- Units are px on the 64×64 grid (each px ≈ 4 RPLAN px ≈ 0.28 m, but `scale_m_per_unit` is left null).

## Env (`fpe-chathousediffusion`)
- Python 3.10, torch 2.1.2+cu118, torchvision 0.16.2, **CPU** DGL 2.0.0. The cu118 DGL wheel needs
  a system `libcusparse.so.11`. DGL is used only for CPU graph preprocessing (shortest paths), and the
  Graphormer layers are plain torch modules.
- Other packages: transformers 4.40.2, ema_pytorch 0.4.3, einops 0.7.0, pandas 2.1.4, numpy 1.26.0,
  torchdata 0.7.1.
- `PYTHONNOUSERSITE=1` is set (otherwise `~/.local/lib/python3.10` leaks in). During setup, two
  concurrent pip runs once corrupted pillow's dist-info; re-running setup_env.sh fixes it.

## Spot check
- Generated vs. GT renders match closely in layout and room types (the parsed graphs are very
  detailed: types, links, location and size per room). Typical errors: study vs. bedroom confusion,
  and small shifts of room extents.
