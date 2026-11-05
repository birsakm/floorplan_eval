"""Holodeck floor plans for the 1,000 RPLAN inputs of Table 1(a), with an open-weight LLM served by vLLM.

Holodeck (CVPR 2024, external/methods/holodeck = allenai/Holodeck 362b8ed) builds a house from a text query with an
LLM (gpt-4o-2024-05-13 in the release). We run its floor-plan stages unchanged (prompts, parsing, validity check,
wall and door placement: ai2holodeck.generation.{rooms,walls,doors}) and swap only the LLM: every call
llm(prompt) -> text goes to an OpenAI-compatible vLLM server (scripts/holodeck/NOTES.md), as one user message, with
Holodeck's settings: max_tokens 2048 and nothing else (langchain 0.0.171 turns Holodeck's OpenAI(gpt-4o...) into
OpenAIChat, which sends only max_tokens, so gpt-4o ran with OpenAI's default temperature 1.0). Sampling therefore uses
each model's own defaults, which vLLM takes from generation_config.json (both: temperature 1.0, top_p 0.95; top_k 20
for Qwen, 64 for Gemma).
Thinking is off (enable_thinking=False): gpt-4o is not a reasoning model and Holodeck parses the raw reply.

Stages run: floor plan (1 call), wall height (1 call), doors (1 call). Windows, objects and the AI2-THOR stages
are not run (Table 1 scores the room layout; ProcTHOR and Infinigen run the same stages).

Input (bubble+areas, as ProcTHOR and Infinigen): the program of outputs/rplan/g2p_test1000_realdoors_programs is
written as Holodeck's text query: rooms with unique names and net areas, the door connections, the entrance room.
Holodeck's own prompt constraints stay (rectangular rooms, 3-8 m sides).
A query whose floor plan fails Holodeck's own validity check (or cannot be parsed) is retried with a new LLM seed, at
most --attempts times in total (Holodeck itself raises and stops); the attempts are recorded. Inputs without a valid
plan after that count as failures (N < 1,000).

Output (common format, metres): rooms = Holodeck's room rectangles, typed by the program room whose name the LLM
used; interior doors / front door = Holodeck's door segments (doorSegment) as 0.1 m deep rectangles; 'open' pairs =
an opening over the whole shared wall. raw/<id>.json keeps every prompt and reply.

    V=/datawaha/cggroup/birsakm/venvs/fpe-holodeck/bin/python   # on kw61498, next to the vLLM servers
    PYTHONPATH=. $V scripts/holodeck/generate.py --model qwen [--limit 20] [--workers 16]
"""
import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "methods" / "holodeck"))
# Holodeck base data (materials, doors; AI2 bucket, holodeck/2023_09_23.tar) at <this>/holodeck/2023_09_23
os.environ.setdefault("OBJATHOR_ASSETS_BASE_DIR", str(ROOT / "data" / "holodeck" / "objathor"))

from fpeval.format import canonical_room_type, make_sample, save_sample  # noqa: E402

PROGRAMS = ROOT / "outputs" / "rplan" / "g2p_test1000_realdoors_programs" / "samples"
MODELS = {"qwen": ("Qwen/Qwen3.8-27B", 8201, "qwen3.8-27b"),
          "gemma": ("google/gemma-4-31B-it", 8202, "gemma-4-31b")}
HOLODECK_BASE = ROOT / "data" / "holodeck" / "objathor" / "holodeck" / "2023_09_23"
NAMES = {"living_room": "living room", "bedroom": "bedroom", "bathroom": "bathroom", "kitchen": "kitchen",
         "balcony": "balcony", "study room": "study", "dining room": "dining room", "storage": "storage room",
         "unknown": "storage room", "entrance": "entrance hall"}
ALIASES = {"circulation": "corridor", "outdoor": "balcony", "office": "study", "living": "living_room",
           "dining": "dining_room"}
DOOR_T = 0.1
MAX_TOKENS = 2048  # Holodeck: OpenAI(max_tokens=2048); no temperature (model defaults, see docstring)


class VLLM:
    """Holodeck calls self.llm(prompt) -> str; this sends it to the vLLM server and logs prompt and reply."""

    def __init__(self, model, port, seed):
        self.model, self.url, self.seed, self.log = model, f"http://127.0.0.1:{port}/v1/chat/completions", seed, []

    def __call__(self, prompt, *a, **k):
        import requests
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
                "seed": self.seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        for k_try in range(5):
            try:
                r = requests.post(self.url, json=body, timeout=600)
                r.raise_for_status()
                break
            except Exception:  # noqa: BLE001 - server busy/restarting
                if k_try == 4:
                    raise
                time.sleep(10 * (k_try + 1))
        out = r.json()
        text = out["choices"][0]["message"]["content"] or ""
        self.log.append({"prompt": prompt, "reply": text, "usage": out.get("usage"),
                         "finish_reason": out["choices"][0].get("finish_reason")})
        return text


def room_names(program):
    """Unique Holodeck room names for the program rooms (numbered when a name repeats)."""
    base = [NAMES[r["type_rplan"]] for r in program["rooms"]]
    count, seen, out = Counter(base), Counter(), {}
    for r, b in zip(program["rooms"], base):
        seen[b] += 1
        out[r["id"]] = f"{b} {seen[b]}" if count[b] > 1 else b
    return out


def query(program):
    """Holodeck's text query for one program (bubble+areas)."""
    names = room_names(program)
    rooms = sorted(program["rooms"], key=lambda r: -r["target_net_area"])
    total = sum(r["target_net_area"] for r in rooms)
    parts = ", ".join(f"{names[r['id']]} ({r['target_net_area']:.1f} m2)" for r in rooms)
    doors = ", ".join(f"{names[e['room_a']]} - {names[e['room_b']]}" for e in program["edges"])
    return (f"an apartment of {total:.0f} m2 with {len(rooms)} rooms: {parts}. Doors connect: {doors}. "
            f"The entrance door from the exterior opens into the {names[program['entrance']]}")


_GEN = {}


def generators():
    """Holodeck's floor-plan, wall and door generators (one set per worker process)."""
    if not _GEN:
        import open_clip
        import torch
        torch.set_num_threads(2)
        from ai2holodeck.generation.doors import DoorGenerator
        from ai2holodeck.generation.rooms import FloorPlanGenerator
        from ai2holodeck.generation.walls import WallGenerator
        clip, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
        tok = open_clip.get_tokenizer("ViT-L-14")
        _GEN.update(rooms=FloorPlanGenerator(clip, prep, tok, None), walls=WallGenerator(None),
                    doors=DoorGenerator(clip, prep, tok, None))
    return _GEN


def holodeck(q, llm):
    """Holodeck.generate_rooms / generate_walls / generate_doors on an empty scene (generation/holodeck.py)."""
    g = generators()
    for k in ("rooms", "walls", "doors"):
        g[k].llm = llm
    scene = {"query": q}
    scene["rooms"] = g["rooms"].generate_rooms(scene, "N/A")
    scene["wall_height"], scene["walls"] = g["walls"].generate_walls(scene)
    raw, doors, pairs, open_pairs = g["doors"].generate_doors(scene, "N/A")
    scene.update(raw_doorway_plan=raw, doors=doors, room_pairs=pairs, open_room_pairs=open_pairs)
    scene["walls"], scene["open_walls"] = g["walls"].update_walls(scene["walls"], open_pairs)
    return scene


def ring(p):
    return [[round(float(a), 4), round(float(b), 4)] for a, b in p.exterior.coords[:-1]]


def to_sample(scene, program, sid, variant, cond):
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    names = {v: k for k, v in room_names(program).items()}
    prog = {r["id"]: r for r in program["rooms"]}
    polys, rooms, index = {}, [], {}
    for r in scene["rooms"]:
        name = r["roomType"]
        pid = names.get(name) or names.get(re.sub(r"-\d+$", "", name))
        poly = Polygon([(float(x), float(z)) for x, z in r["vertices"]])
        index[name] = len(rooms)
        polys[name] = poly
        t = prog[pid]["type"] if pid else name
        rooms.append({"type": canonical_room_type(t, ALIASES), "type_native": name, "polygon": ring(poly),
                      "program_room": pid})
    doors, edges = [], []
    for d in scene["doors"]:
        a, b = d["room0"], d["room1"]
        seg = LineString([tuple(map(float, p)) for p in d["doorSegment"]])
        rect = ring(seg.buffer(DOOR_T / 2, cap_style="flat"))
        if "exterior" in (a, b):
            doors.append({"type": "front_door", "polygon": rect, "width_m": round(seg.length, 3)})
        else:
            doors.append({"type": "interior_door", "polygon": rect, "width_m": round(seg.length, 3),
                          "holodeck_type": "doorframe" if not d.get("openable") else "doorway"})
            edges.append([index[a], index[b], "door"])
    for a, b in scene["open_room_pairs"]:
        if a in polys and b in polys:
            shared = polys[a].boundary.intersection(polys[b].boundary)
            if shared.length > 0:
                g = shared.buffer(DOOR_T / 2, cap_style="flat")
                doors.append({"type": "opening", "polygon": ring(g.envelope), "width_m": round(shared.length, 3)})
                edges.append([index[a], index[b], "open"])
    outline = unary_union(list(polys.values()))
    return make_sample("holodeck", variant, sid, rooms, units="m", scale_m_per_unit=1.0,
                       boundary=ring(outline) if outline.geom_type == "Polygon" else None,
                       doors=doors, windows=[], graph={"edges": edges},
                       condition=dict(cond, type="bubble+areas", holodeck_query=scene["query"]))


def solve(args):
    sid, cond, model, port, variant, seed, attempts, out = args
    q = query(cond["program"])
    tries = []
    for k in range(attempts):
        s = seed * 1000 + k
        random.seed(s)
        llm = VLLM(model, port, s)
        try:
            scene = holodeck(q, llm)
            sample = to_sample(scene, cond["program"], sid, variant, cond)
            tries.append({"seed": s, "ok": True, "calls": llm.log})
            json.dump({"query": q, "attempts": tries}, open(out / "raw" / f"{sid}.json", "w"), indent=1)
            save_sample(sample, out / "samples" / f"{sid}.json")
            return sid, k + 1, None
        except Exception as e:  # noqa: BLE001 - Holodeck raises ValueError on an invalid plan
            tries.append({"seed": s, "ok": False, "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc(limit=3), "calls": llm.log})
    json.dump({"query": q, "attempts": tries}, open(out / "raw" / f"{sid}.json", "w"), indent=1)
    return sid, attempts, tries[-1]["error"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None, help="first N inputs only (test run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    assert HOLODECK_BASE.exists(), HOLODECK_BASE
    model, port, short = MODELS[a.model]
    variant = a.name or f"rplan-g2p_text_test1000_realdoors_{short}"
    out = ROOT / "outputs" / "holodeck" / variant
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    files = sorted(PROGRAMS.glob("*.json"), key=lambda f: int(f.stem))[:a.limit]
    jobs = [(f.stem, json.load(open(f))["condition"], model, port, variant, a.seed * 100000 + int(f.stem) % 100000,
             a.attempts, out) for f in files]
    t0, done, failed, tries = time.time(), 0, {}, Counter()
    with Pool(a.workers) as pool:
        for sid, n_try, err in pool.imap_unordered(solve, jobs):
            done += 1
            tries[n_try] += 1
            if err:
                failed[sid] = err
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} done, {len(failed)} failed, {time.time() - t0:.0f}s", flush=True)
    usage = Counter()
    for f in (out / "raw").glob("*.json"):
        for t in json.load(open(f))["attempts"]:
            for c in t["calls"]:
                for k, v in (c.get("usage") or {}).items():
                    if isinstance(v, int):
                        usage[k] += v
    info = {"method": "Holodeck (allenai/Holodeck 362b8ed), floor plan + walls + doors stages",
            "llm": model, "server": "vLLM 0.31.0, bf16, 1x H100 (kw61498)", "sampling": "model defaults (generation_config.json)",
            "max_tokens": MAX_TOKENS, "thinking": False, "max_attempts": a.attempts,
            "input": "bubble+areas as text query (see generate.py query())",
            "n_inputs": len(jobs), "n_samples": len(jobs) - len(failed), "failed": failed,
            "attempts_histogram": dict(sorted(tries.items())), "tokens": dict(usage),
            "seconds": round(time.time() - t0)}
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "failed"}, indent=1))


if __name__ == "__main__":
    main()
