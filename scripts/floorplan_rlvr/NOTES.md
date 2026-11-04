# Floor-plan RLVR (ACL Findings 2026) — notes

Submodule: `external/methods/floorplan_rlvr` @ `cf98578` (unmodified, so there is no `patches/floorplan_rlvr.patch`).
Paper: "Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards" (Lara et al.).

## Status (2026-10-07 16:50): BLOCKED on GPUs — full run not started

Done: env, RLVR weights (bf16, 141 GB), 998 prompts, bf16 and fp8 smoke tests (16 inputs each), converter, eval
config entries. The bf16 smoke test is clean: 16/16 valid, room count/types/ids match the prompt for 16/16, 12 strict-JSON + 4
repaired, 0 truncated, renders look RPLAN-like.

The full run could not start. A 70B bf16 model needs all 4 GPUs at about 38 GB each. From 16:25 on, another user's automated
submitter (`cvejica`, `morphos_vqa/.../mevis_selection_arms.py` via `run.sh --gpu N`) kept 2–4 GPUs busy
(19–39 GB each) and refilled any freed GPU within about 20 s. Our first full launch (16:25) OOMed at load because of this.
We did not touch their jobs or hold GPUs to block them. Resume once the 4 GPUs are actually free:

```bash
SKIP_SETUP=1 bash scripts/floorplan_rlvr/run.sh rlvr        # resumable; ~17 h at the bf16 smoke throughput (see below)
PYTHONPATH=. OMP_NUM_THREADS=1 conda run -n fpe python -m fpeval.evaluate variants floorplan_rlvr/rplan-g2p_bubble_areas_test1000_syndoors
PYTHONPATH=. OMP_NUM_THREADS=1 conda run -n fpe python -m fpeval.evaluate report
# SFT variant (only if the RLVR run is clean; +145 GB base model download):
bash scripts/floorplan_rlvr/download.sh sft && SKIP_SETUP=1 bash scripts/floorplan_rlvr/run.sh sft
```

Runtime estimate: the bf16 smoke test ran at 298 output tok/s (KV cache 29,840 tokens at `gpu_mem 0.94`) with 1,800 tokens
per sample. Best-of-10 × 998 prompts ≈ 18M tokens ≈ 17 h. `run.sh` now defaults to `GPU_MEM=0.95` and
`MAX_BATCHED=2048` to get more KV cache. That is **untested** (the launch OOMed on the other user's memory before
profiling). If it OOMs on an idle machine, retry with `GPU_MEM=0.94 MAX_BATCHED=8192`.

## Scripts

| script | env | what |
|---|---|---|
| `setup_env.sh` | – | `mamba create -n fpe-floorplan_rlvr python=3.11`; pip `vllm==0.8.5.post1` + `torch==2.6.0` (requirements_pred.txt: vllm, torch 2.6), then pins `transformers==4.51.3` (the version in the checkpoint's config), `huggingface_hub<1`, `datasets<4`. Unpinned installs pull transformers 5.x / hub 1.x, which vLLM 0.8.5 does not support. `PYTHONNOUSERSITE=1`. |
| `download.sh` | env | `snapshot_download` into `checkpoints/floorplan_rlvr/` with `HF_HOME`/`HF_HUB_CACHE` under `checkpoints/floorplan_rlvr/hf_home` (the model dirs hold symlinks into that cache, so nothing is stored twice). Refuses a download that would leave < 1000 GB free on `/datawaha/cggroup`. `rlvr` (default) or `sft`. |
| `prepare_inputs.py` | env | HG++ JSONs → RLVR prompts (see Inputs). |
| `generate.py` | env | vLLM best-of-n generation; writes `raw/<id>.json` with all n candidates. Resumable. |
| `convert.py` | fpe | raw → `samples/`, `gt_samples/`, `run_info.json`. |
| `run.sh` | – | all of the above + renders (`renders/`, `gt_renders/`). |

## Models (checked on HF 2026-10-07)

| repo | content | size | gated |
|---|---|---|---|
| `ludolara/fp5-rlvr-Llama3.3-70B` | **full merged** bf16 checkpoint (SFT + RLVR/GRPO), 30 safetensors shards | 141 GB | no |
| `ludolara/fp5-sft-Llama3.3-70B` | **LoRA adapter** (r=64, alpha=128, q/k/v/o/gate/up/down) on `meta-llama/Llama-3.3-70B-Instruct` | 3.3 GB | no |
| `meta-llama/Llama-3.3-70B-Instruct` | base for the SFT adapter; we fetch only `*.safetensors`/`*.json` (not `original/`, another 141 GB) | 141 GB | manual gate; **our token has access** |

"fp5" = the 5-room held-out model: trained on all 6-, 7- and 8-room RPLAN plans, tested on 5-room plans.

## Inputs

1000 ids of `data/method_inputs/common_rplan_test/ids_test_1000.txt` → the House-GAN++-style JSONs of
`data/method_inputs/houseganpp/rplan-g2p_test1000/` (Graph2Plan RPLAN test split, synthesized interior
doors; see `scripts/houseganpp/NOTES.md`) → `data/method_inputs/floorplan_rlvr/rplan-g2p_test1000/inputs.jsonl`.

`prepare_inputs.py` calls the repo's own `RPLANConverter._convert_entry` (`src/dataset_convert/rplan.py`):
room polygons are polygonized from the HG edge segments, scaled to metres (18/256 m/px), areas and
sizes rounded to 0.01; rectangles are given as `height`+`width`, other polygons as `area`; the
front door is a space; `room_count` excludes doors; `total_area` sums the rooms.

- **998 of 1000 prompts.** Ids 23486 and 73648 are rejected by the converter (a living-room
  polygon cannot be polygonized from its segments). They have no prompt and no GT here and are not counted
  in `n_requested`.
- **Bubble graph = door graph (adaptation).** The paper derives the bubble diagram "from interior-door
  connectivity". In the real RPLAN HG++ JSONs the room–room `ed_rm` pairs coincide with the door
  connections (checked on all 7 bundled RPLAN JSONs), so `RPLANGraph.from_housegan` returns the door graph.
  Our g2p JSONs list every shared wall as a room–room pair, which would give a denser wall-adjacency graph than the model
  was trained on (and the system prompt asks for one door per bubble edge). `prepare_inputs.py --graph door`
  (default) therefore drops room–room `ed_rm` pairs not bridged by a synthesized door before calling the
  converter. Polygons and areas are unaffected. Every resulting graph equals the synthesized door pairs (+ front door).
  So the input graph is: every non-living room connected to the living room if they share a wall, otherwise to its
  best neighbour (synthesized, not RPLAN's real doors).
- **Areas are wall-centre areas.** Graph2Plan polygons run along wall centre lines (no gaps), while the
  RPLAN data reader's polygons are room interiors with ~4 px (0.28 m) walls between them. Our requested room
  sizes and totals are therefore somewhat larger than what the model saw in training (about +0.28 m per room
  dimension). Not corrected, to stay with the shared input.
- **Train/test overlap.** fp5 was trained on 100% of RPLAN's 6/7/8-room plans (its own conversion of raw
  RPLAN). Our test ids come from Graph2Plan's test split of RPLAN, so the 909 inputs with 6–8 rooms
  (320/332/257) were very likely in the model's training data; only the 89 five-room
  (and 2 four-room) inputs are held out. Treat the numbers as an upper bound on held-out quality.

## Decoding (paper App. A.3 and `src/pred/floorplan_generator.py`)

- Prompt: the repo's `build_prompt` (Llama-3 chat template, the repo's `SYSTEM_PROMPT`, user message = prompt JSON),
  tokenized with `add_special_tokens=False` (one BOS), as in SFT training (`floorplan_dataset.py`). The repo's own
  vLLM driver passes the string, so vLLM would add a second BOS.
- **best-of-10**, temperature 0.7, top_p 0.9, max 4096 new tokens, stop at `<|eot_id|>`, `<|eom_id|>`, `<|end_of_text|>`.
  The paper uses best-of-10 for all main results. The repo's `pred.sh` passes `--use_sampling`, which hard-codes
  n=best_of=100 (the paper's n=100 ablation).
- **Selection:** the repo's `FloorplanGenerator._select_least` (called unmodified): valid JSON first, then
  "overlap area", then RPLANGraph compatibility. Note: its overlap term reads `output["rooms"]`, but outputs use
  `spaces`, so overlap is always 0 and the selection is effectively valid JSON → min compatibility. The paper's
  stated rule (min overlap, ties by compatibility) is also computed per input (`selected_paper` in raw; 
  `convert.py --select paper` converts that pick instead). `run_info.n_selected_code_ne_paper` counts disagreements.

## Precision / memory (4× A100-40GB, TP=4)

- **bf16 weights are used** (no quantization). With bf16, 141 GB of weights leave ~30k tokens of KV cache at
  `gpu_memory_utilization=0.94` (≈6 sequences of 5k). We raised it with `gpu_memory_utilization=0.95` and
  `max_num_batched_tokens=2048` (smaller prefill chunks → smaller activation reservation). Not yet verified (see Status).
- **fp8 (W8A16 Marlin) was tried and rejected.** Same 16 smoke inputs, same settings, all 160 candidates:

  | weights | KV cache | tok/s | strict-JSON candidates | candidates hitting 4096 tokens | median tokens |
  |---|---|---|---|---|---|
  | bf16 | 29,840 tok | 298 | 116/160 | 5/160 | 1740 |
  | fp8 (vLLM `quantization=fp8`) | 219,888 tok | 644 | 77/160 | 79/160 | 3779 |

  fp8 roughly doubles throughput but half its candidates run away to the length cap with degenerate text (doors
  labelled with room ids, repeated vertices). Best-of-10 hides most of this in the selected sample, but it changes the
  model, so fp8 is not used. `run.sh` defaults to `QUANT=none`. The smoke outputs are in
  `outputs/floorplan_rlvr/*_smoke_{bf16,fp8}/`.
- `max_model_len=5120` (prompts are 600–1000 tokens + 4096 new tokens).
- vLLM prints NCCL "out of memory" tracebacks at process exit (during teardown after `done`). They are harmless:
  all outputs are written before that.

## Output conversion

- The model writes `floor_polygon` in **metres** in the RPLAN image frame (pixel × 18/256, y down), like its
  training targets. `convert.py` maps back to the 256 px frame (px = m × 256/18) and writes `units: "px"`,
  `scale_m_per_unit: 18/256`. `eval_config` uses scale 18/256.
- `interior_door` and `front_door` spaces → `doors`; `study_room` → study, `unknown` → other.
- `gt_samples/` are built from the same prompt record (the converter's spaces, i.e. the Graph2Plan polygons
  rounded to 0.01 m) in the exact RPLAN frame. They match `outputs/houseganpp/rplan-g2p_bubble_test1000_syndoors/gt_samples`
  (same ids and polygons). Those are shifted by ≈0.5–1 px (HG++'s centring, `frame_shift_px`), which
  does not matter because the variant is not frame-aligned.
- Validity accounting (`run_info.json`), on the selected candidate of each input: `n_parsed` (strict `json.loads`),
  `n_parsed_incl_repair` (+ repo `json_repair`, which is what `extract_output_json` does), `n_valid` (≥1 room with
  a ≥3-vertex non-zero-area polygon), `n_valid_all_rooms_ok`, `n_truncated`, room count/type/id match with the prompt,
  plus candidate-level strict-JSON counts.
