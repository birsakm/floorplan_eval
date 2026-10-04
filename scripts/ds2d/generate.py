"""DS2D batched inference (run in the `fpe-ds2d` env).

Re-implements the prompt construction of external/methods/ds2d/{run_generation_rplan.py,
src/pred/pred.py, rplan_dataset.py, procthor_dataset.py} without modifying the submodule
(the released inference scripts do not run as-is, see scripts/ds2d/NOTES.md), but with batched
greedy decoding in bf16 instead of per-sample 8-bit decoding.

Writes one raw/<id>.json per test input:
  {id, level, prompt_text, prompt_spec, generated_text, n_new_tokens, hit_eos, gt}
"""
import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOM_LABEL = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom", 4: "DiningRoom",
              5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom", 8: "GuestRoom", 9: "Balcony",
              10: "Entrance", 11: "Storage", 12: "Wall-in", 13: "External", 14: "ExteriorWall",
              15: "FrontDoor", 16: "InteriorWall", 17: "InteriorDoor"}
LEVELS = {"full_prompt": 0, "only_total_area": 1, "only_room_area": 2, "some_room_area": 3}


# ---------------------------------------------------------------- RPLAN (run_generation_rplan.py)
def rplan_prompt(rec, pp):
    gt = rec["gt"]
    prompt = {}
    if pp == 0:  # training-style random masking of the spec (seeded by caller)
        prompt = deepcopy(gt)
        for room in prompt["rooms"]:
            del room["floor_polygon"]
            for k in list(room.keys()):
                if random.random() < 0.5:
                    del room[k]
        prompt["rooms"] = [r for r in prompt["rooms"] if r]
        if len(prompt["rooms"]) == 0:
            del prompt["rooms"]
        rands = np.random.random(len(prompt.keys()))
        rands[np.argmax(rands)] = 1.0
        for i, k in enumerate(list(prompt.keys())):
            if rands[i] < 0.5:
                del prompt[k]
    if pp in (1, 3):
        prompt["total_area"] = gt["total_area"]
    if pp in (2, 3):
        rooms = deepcopy(gt["rooms"])
        if pp == 3:
            rands = np.random.random(len(rooms))
            rands[np.argmax(rands)] = 1.0
            for i in sorted(np.where(rands < 0.5)[0], reverse=True):
                del rooms[i]
        for room in rooms:
            for k in list(room.keys()):
                if k not in ("area", "room_type", "id"):
                    del room[k]
        prompt["rooms"] = rooms
    rooms_raw = rec["rooms_raw"]
    instruction = ('you are to generate a floor plan in a JSON structure. you have to satisfy the adjacency '
                   'constraints given as pairs of neighboring rooms; two connecting rooms are presented as '
                   '(room_type1 room_id1, room_type2 room_id2). you also need to satisfy additional contraints '
                   'given by the user.')
    adj = f"total number of rooms: {len(rooms_raw)}; adjacency pairs: "
    for u, v, _ in rec["edges"]:
        adj += f'({ROOM_LABEL[rooms_raw[u][4]]} = "room|{u}", {ROOM_LABEL[rooms_raw[v][4]]} = "room|{v}"), '
    user = adj.strip(", ")
    if len(prompt.keys()) > 0:
        user += f". additional constraints: {str(prompt)}"
    text = (f"<|start_header_id|>system<|end_header_id|> {instruction}<|eot_id|><|start_header_id|>user"
            f"<|end_header_id|> {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|> ")
    return text, prompt


# ---------------------------------------------------------------- ProcTHOR (src/pred/pred.py + procthor_dataset.py)
def procthor_prompt(sample, pp, version):
    d = deepcopy(sample)
    prompt = {"room_count": d["room_count"]}
    if pp == 0:
        prompt = d["prompt"]
    if pp in (1, 3):
        prompt["total_area"] = d["prompt"]["total_area"]
    if pp in (2, 3):
        rooms = deepcopy(d["rooms"])
        if pp == 3:
            rands = np.random.random(len(rooms))
            rands[np.argmax(rands)] = 1.0
            for i in sorted(np.where(rands < 0.5)[0], reverse=True):
                del rooms[i]
        for room in rooms:
            for k in list(room.keys()):
                if k not in ("area", "room_type", "id"):
                    del room[k]
        prompt["rooms"] = rooms
    if version == "bd":
        instruction = ('you are to generate a floor plan in a JSON structure where each room is defined by polygon '
                       'vertices, make sure to not overlap the polygons. you have to satisfy the adjacency constraints '
                       'given as pairs of neighboring rooms; two connecting rooms, room1 and room2, are presented as '
                       '(room1_type/"room1_id", room2_type/"room2_id"). you have to also match the specifications '
                       'passed by the user in a JSON structure when they exist. when room area and total area '
                       'requirements exist, make sure the polygon areas add up to the required number.')
        user = ""
        adj = ""
        for u, v in d["edges"]:
            ru, rv = d["rooms"][u], d["rooms"][v]
            adj += f'({ru["room_type"]}/"{ru["id"]}", {rv["room_type"]}/"{rv["id"]}"), '
        adj = adj.strip(", ")
        if len(adj):
            user += f"adjacency constraints: {adj}. "
        user += f"specifications: {str(prompt)}"
        text = (f"<|start_header_id|>system<|end_header_id|> {instruction}<|eot_id|><|start_header_id|>user"
                f"<|end_header_id|>{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|> ")
    else:
        instruction = ('you are to generate a floor plan in a JSON structure where each room is defined by polygon '
                       'vertices, make sure to not overlap the polygons. you have to satisfy the requirements passed '
                       'by the user in a JSON structure. when room area and total area requirements exist, make sure '
                       'the polygon areas add up to the required number.')
        text = (f"<|start_header_id|>system<|end_header_id|> {instruction}<|eot_id|><|start_header_id|>user"
                f"<|end_header_id|> {str(prompt)}<|eot_id|><|start_header_id|>assistant<|end_header_id|> ")
    return text, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["rplan", "procthor"], required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--lora", default=None, help="PEFT LoRA dir (omit only for smoke tests)")
    ap.add_argument("--version", default="bd", choices=["bd", "non_bd"], help="procthor prompt style")
    ap.add_argument("--level", default="full_prompt", choices=list(LEVELS))
    ap.add_argument("--out_raw", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--base_model", default=os.environ.get("DS2D_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dry_run", action="store_true", help="only write prompts, no model")
    a = ap.parse_args()
    pp = LEVELS[a.level]
    max_new = a.max_new_tokens or (2800 if a.dataset == "rplan" else 4000)

    recs = [json.loads(l) for l in open(a.inputs) if l.strip()][: a.limit]
    out = Path(a.out_raw)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for k, rec in enumerate(recs):
        random.seed(a.seed + k)
        np.random.seed(a.seed + k)
        if a.dataset == "rplan":
            text, spec = rplan_prompt(rec, pp)
            gt = rec["gt"]
        else:
            text, spec = procthor_prompt(rec["sample"], pp, a.version)
            gt = {kk: vv for kk, vv in rec["sample"].items() if kk != "prompt"}
        items.append({"id": rec["id"], "level": a.level, "prompt_text": text, "prompt_spec": str(spec),
                      "prompt_spec_obj": spec, "gt": gt,
                      "edges": rec.get("edges", rec.get("sample", {}).get("edges"))})
    if a.dry_run:
        for it in items[:3]:
            print(it["prompt_text"][:1500], "\n---")
        return
    todo = [it for it in items if not (out / f"{it['id']}.json").exists()]
    print(f"{len(items)} inputs, {len(todo)} to generate, max_new_tokens={max_new}", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora)
        model = model.merge_and_unload()
    model.eval()
    eos_ids = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>"),
                           tok.convert_tokens_to_ids("<|end_of_text|>")} if i is not None and i >= 0]

    for it in todo:  # training tokenization: f"{bos}{prompt_str}" with add_special_tokens=False
        it["ids"] = tok(f"{tok.bos_token or ''}{it['prompt_text']}", add_special_tokens=False)["input_ids"]
    todo.sort(key=lambda it: len(it["ids"]))
    t0 = time.time()
    done = 0
    for b in range(0, len(todo), a.batch_size):
        batch = todo[b: b + a.batch_size]
        L = max(len(it["ids"]) for it in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, it in enumerate(batch):
            ids[i, L - len(it["ids"]):] = torch.tensor(it["ids"])
            att[i, L - len(it["ids"]):] = 1
        with torch.no_grad():
            gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=max_new,
                                 do_sample=False, num_beams=1, eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        gen = gen[:, L:].cpu()
        for i, it in enumerate(batch):
            g = gen[i].tolist()
            hit = any(t in eos_ids for t in g)
            if hit:
                g = g[: min(g.index(t) for t in eos_ids if t in g)]
            rec = {k: v for k, v in it.items() if k not in ("ids",)}
            rec.update({"generated_text": tok.decode(g, skip_special_tokens=True), "n_new_tokens": len(g),
                        "hit_eos": hit, "prompt_tokens": len(it["ids"])})
            with open(out / f"{it['id']}.json", "w") as f:
                json.dump(rec, f)
        done += len(batch)
        el = time.time() - t0
        print(f"[{done}/{len(todo)}] {el / 60:.1f} min, max_mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
              flush=True)


if __name__ == "__main__":
    main()
