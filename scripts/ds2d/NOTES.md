# DStruct2Design (DS2D) — notes

Submodule: `external/methods/ds2d` @ `d3f734bd465d6239100b50bb11b3903741a2be08`. It is not modified, so there is no `patches/ds2d.patch`.

## Status (2026-10-04): all 10 variants done

The base-model access (gated `meta-llama/Meta-Llama-3-8B-Instruct`) was granted to the HF token. All 10 variants were then generated, converted and rendered:

```bash
GPUS="0 1 2 3" SKIP_SETUP=1 bash scripts/ds2d/run.sh            # all variants, one worker per GPU
GPUS="1" BATCH=12 SKIP_SETUP=1 bash scripts/ds2d/run.sh rplan6R_bubble_roomarea_test   # one variant
```

Generation resumes: ids that already have `raw/<id>.json` are skipped, and runtime accumulates in `run_info.json`.

### Results

Validity rules are under "Deviations" item 7. "count match" means the generated room count equals the GT room count.

| variant | requested | parsed (strict) | +repaired | valid | count match | truncated | GPU-min |
|---|---|---|---|---|---|---|---|
| procthor_bubble_constraints_test_lora-fullprompt | 1000 | 1000 | 0 | 1000 | 1000 | 0 | 28 |
| procthor_bubble_constraints_test_lora-mask | 1000 | 998 | 2 | 1000 | 999 | 0 | 37 |
| procthor_bubble_constraints_test_lora-presetmask | 1000 | 998 | 2 | 995 | 995 | 0 | 29 |
| procthor_constraints_test_lora-fullprompt | 1000 | 1000 | 0 | 1000 | 1000 | 0 | 24 |
| procthor_constraints_test_lora-mask | 1000 | 998 | 2 | 998 | 997 | 1 | 41 |
| procthor_constraints_test_lora-presetmask | 1000 | 1000 | 0 | **790** | 789 | 0 | 30 |
| rplan5R_bubble_roomarea_test | 500 | 495 | 5 | 500 | **4** | 4 | 64 |
| rplan6R_bubble_roomarea_test | 500 | 485 | 15 | 500 | 485 | 15 | 73 |
| rplan7R_bubble_roomarea_test | 500 | 488 | 12 | 500 | **295** | 9 | 50 |
| rplan8R_bubble_roomarea_test | 500 | 481 | 19 | 500 | 481 | 18 | 95 |

Observations:

- **The ProcTHOR outputs copy the requested room ids, types and counts almost perfectly.**
- **nonBD preset_mask emits room-spec JSON without `floor_polygon` for 21% of plans.** These are counted as invalid. The likely cause is our assumed full-spec prompt: it contains `height`, `width` and `is_regular`, which never appear in preset_mask's training prompts. The training prompts have only room_count, total_area, room_types and room areas.
- **The held-out-count RPLAN models behave very differently from each other:**
  - 5R outputs 8 rooms for 493/500 inputs.
  - 7R outputs 7 rooms 294 times and 8 rooms 201 times.
  - 6R and 8R hit the requested count about 97% of the time.
  - In all cases the model copies the listed room ids and types first, then appends extra rooms.
  - This is a model property, not a parse problem: the smoke tests on seen counts also follow the given rooms.
- Validated by eye: the RPLAN outputs look like RPLAN, axis-aligned with wall gaps between rooms. Our Graph2Plan-derived GT polygons are tight, with no gaps. The ProcTHOR outputs are plausible, with occasional overlaps or gaps on large plans.

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
   - Batch size is 12 on GPU 0 (peak about 24 GB) and 24 on GPUs 1–3 (up to about 38 GB), with automatic halving on OOM. See Known issues.
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

About 7.8 GPU-h in total, on 4x A100-40GB, from about 20:49 to 23:15 wall-clock. Per-variant minutes are in the table above. They include model load and time wasted on OOM retries; the rplan8R and rplan6R totals also include a run that was killed and restarted at batch 12.

The RPLAN variants are slower than ProcTHOR: about 900 tokens per output, a few runaway generations of 2800 tokens, and OOM retries at batch 24.

## Known issues

- The conda package cache is shared with other agents. One `mamba create` hung on a lock for about 25 minutes. Killing it and retrying worked, and `setup_env.sh` retries on its own.
- RPLAN GT polygons come from Graph2Plan's vectorization, not DS2D's own. This may differ slightly, by about 1 px, from what the RPLAN LoRAs were trained on.
- Batched HF generation (transformers 4.40, sdpa) computes full-vocab fp32 logits over the whole prompt at prefill. Runaway generations also grow the KV cache. Together these cause OOMs at batch 24 on 40 GB GPUs. `generate.py` halves the batch on OOM; greedy decoding gives the same results at any batch size. Use `BATCH=12` for RPLAN: the first RPLAN runs at 24 thrashed, which is why rplan6R and rplan8R were restarted at 12.
- The first version of the OOM fallback retried inside the `except` block, which kept the failed tensors alive. It OOMed all the way down to batch size 1. This is fixed.
- `run.sh` shards variants round-robin over `GPUS`, with ProcTHOR (heavier) first. Batch size is `BATCH0` on GPU 0 (shared, default 12, peak about 24 GB) and `BATCH` elsewhere (default 24). `OMP_NUM_THREADS=4` is set and `torch.set_num_threads(4)` is called per worker.
- Throughput: 0.4 to 0.7 min per 24 ProcTHOR plans, with GPU utilisation at 95% or more.

