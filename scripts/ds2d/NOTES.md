# DStruct2Design (DS2D) — notes

Submodule: `external/methods/ds2d` @ `d3f734bd465d6239100b50bb11b3903741a2be08`. It is not modified, so there is no `patches/ds2d.patch`.

## Status (2026-10-04)

**Blocked: no generations yet.** The base model `meta-llama/Meta-Llama-3-8B-Instruct` is gated on Hugging Face. The HF token at `~/.cache/huggingface/token` (account `BirsakM`) gets HTTP 403. `meta-llama/Meta-Llama-3-8B` also gets 403. We did not request access. Once the account has been granted access:

```bash
SKIP_SETUP=1 bash scripts/ds2d/run.sh            # all 10 variants
SKIP_SETUP=1 bash scripts/ds2d/run.sh rplan5R_bubble_roomarea_test   # just one
```

`DS2D_BASE_MODEL=<hub id or local path>` can point to a local copy of the same weights.

Everything else is ready:

- The env works.
- The LoRAs are downloaded.
- The test inputs and GT are built.
- `outputs/ds2d/<variant>/gt_samples` and `gt_renders` are filled.
- The generation and conversion pipeline passed a smoke test with `Qwen2.5-0.5B-Instruct` and no LoRA. That output was thrown away.

## Scripts

| script | env | what |
|---|---|---|
| `setup_env.sh` | – | `mamba create -n fpe-ds2d python=3.10`, then pip: torch 2.3.0+cu121, transformers 4.40.1, tokenizers 0.19.1, peft 0.10.0, accelerate 0.29.3, bitsandbytes 0.43.1, datasets 2.19.0 (the pins from DS2D's `requirements.txt`). Sets `PYTHONNOUSERSITE=1`, because packages in `~/.local` otherwise hide missing deps. |
| `download.sh` | fpe (gdown) | Gets 3 Drive archives into `checkpoints/ds2d/{rplan,procthor_bd,procthor_nonbd}/`. The archives are named `.zip` but are tar.gz. The BD and nonBD archives use the same folder names, so each is extracted into its own folder. Also gets HF `ludolara/DStruct2Design` `{train,validation,test}.json` into `data/method_inputs/ds2d/procthor/`. |
| `prepare_inputs.py` | fpe | Builds the test-input jsonl files (see below). |
| `generate.py` | fpe-ds2d | Builds the prompts (copied from DS2D) and runs batched greedy decoding. Writes `raw/<id>.json`. Can resume. |
| `convert.py` | fpe | Parses raw output, then writes `samples/`, `gt_samples/`, `gt_renders/` and `run_info.json`. |
| `run.sh` | – | Runs everything end to end, then renders with `python -m fpeval.render`. |

## LoRAs → variants (10)

All LoRAs have r=8, alpha=32, q_proj and v_proj, base Meta-Llama-3-8B-Instruct.

| variant | LoRA | condition |
|---|---|---|
| `rplan{5,6,7,8}R_bubble_roomarea_test` | `rplan/{N}R` | RPLAN, held-out room count: model NR never saw N-room plans and is tested on N-room plans. Input is the adjacency pairs plus per-room `{area, room_type, id}` (DS2D level `only_room_area`). 500 plans each. |
| `procthor_bubble_constraints_test_lora-{fullprompt,mask,presetmask}` | `procthor_bd/*` | ProcTHOR. Input is the adjacency pairs plus the full specification (level `full_prompt`). 1000 plans (the full test split). |
| `procthor_constraints_test_lora-{fullprompt,mask,presetmask}` | `procthor_nonbd/*` | ProcTHOR. Same full specification, no bubble diagram. 1000 plans. |

`generate.py --level` also supports DS2D's other test levels (`only_total_area`, `some_room_area`, `full_prompt`). Those levels are not in `run.sh`.

## Deviations / decisions (important)

1. **The released inference scripts do not run as-is.**
   - `run_generation_rplan.py` leaves `start_idx` undefined unless it runs under SLURM.
   - Its `partial_prompt==0` branch treats rooms as dicts, while its adjacency code treats them as lists.
   - `run_generation_procthor.py` passes `prompt_style={version}` (a set), so no prompt branch matches.

   So `generate.py` re-implements the prompts exactly as in `rplan_dataset.py`, `procthor_dataset.py` and `src/pred/pred.py`: the training format `f"{bos}<|start_header_id|>system..."` with `add_special_tokens=False`.
2. **RPLAN inputs.** DS2D's RPLAN conversion needs the raw RPLAN PNGs, which we don't have. It also uses an unseeded split and loops over only the first 1000 images. We rebuilt DS2D's RPLAN records from the Graph2Plan RPLAN test split (`/datawaha/cggroup/datasets/RPLAN/Network/data/data_test.mat`):
   - room polygons from `rBoundary`
   - types from `rType`
   - area, height and width as in DS2D, with the 18/256 m/px scale
   - edges from DS2D's bbox-proximity rule (th=9)

   Corrupted plans are skipped: 34 have an empty room polygon and a few more have non-finite or out-of-range coordinates. From each room-count pool (875, 3727, 4338 and 3088 plans for 5, 6, 7 and 8 rooms), we take the first 500 of a `np.random.seed(12345)` permutation. Room ids follow Graph2Plan's room order, not DS2D's regionprops order.
3. **RPLAN output format is unknown.** `rplan_dataset.py` has two target formats: plain `{"rooms":[{room_type, floor_polygon, id}]}`, and a `new` format with room_count, total_area and per-room area. The released inference code reads `total_area`, so we assume the `new` format and give areas in the prompt. The converter accepts either output format.
4. **ProcTHOR `prompt` field.** The HF dataset has no `prompt` column, and the code that built it was not released. We assume the full spec is `{room_count, total_area, room_types, rooms:[all room keys except floor_polygon]}`. The test split is HF `validation.json`, following DS2D: "test set was used for validation, just naming difference". We use all 1000 plans, ordered by a seed-12345 permutation.
5. **Decoding.**
   - Greedy (DS2D's "greedy" setting, num_samples=1).
   - Weights in bf16 with the LoRA merged in, instead of 8-bit as in DS2D's scripts.
   - Batched with left padding.
   - Generation stops at `<|eot_id|>` or `<|end_of_text|>`.
   - max_new_tokens is 2800 for RPLAN and 4000 for ProcTHOR, as in DS2D.
   - Batch size is 12. Our estimate is about 16 GB of weights plus up to about 7 GB of KV cache, under the 25 GB cap. GPU 0 only.
6. **Coordinates.**
   - ProcTHOR: `[x, z]` in meters (`units: "m"`, `scale_m_per_unit: 1`).
   - RPLAN: DS2D writes `"x"` = image row and `"z"` = image column, so we store `[z, x]` in px (`units: "px"`, `scale_m_per_unit: 18/256`).
   - Type mapping: MasterRoom, SecondRoom, ChildRoom and GuestRoom become bedroom; StudyRoom becomes study; Wall-in becomes closet.
7. **Validity accounting** (`run_info.json`):
   - `n_requested`
   - `n_generated`
   - `n_parsed`: strict `json.loads`, allowing trailing text
   - `n_parsed_incl_repair`: adds DS2D's `json_repair` fallback
   - `n_valid`: at least one room polygon with 3 or more distinct numeric vertices and non-zero area. Malformed rooms are dropped and counted in `n_valid_all_rooms_ok`.
   - `n_truncated_no_eos`
   - `n_room_count_matches_gt`
   - per-sample parse status

   Invalid outputs are kept in `raw/` and listed under `failures`. Only valid outputs are written to `samples/`. Outputs have no doors, windows or boundary; DS2D only generates room polygons.

## Runtime

There are no Llama runs yet. Our estimate is about 0.3 to 0.7 GPU-h per ProcTHOR variant (1000 plans) and less for RPLAN (500 plans) on an A100-40GB, about 4 to 6 GPU-h in total. Each run writes its actual runtime to `run_info.json`.

## Known issues

- The conda package cache is shared with other agents. One `mamba create` hung on a lock for about 25 minutes. Killing it and retrying worked, and `setup_env.sh` retries on its own.
- RPLAN GT polygons come from Graph2Plan's vectorization, not DS2D's own. This may differ slightly, by about 1 px, from what the RPLAN LoRAs were trained on.
