"""Floor-plan RLVR generation with vLLM (best-of-N), following src/pred/floorplan_generator.py.

Per prompt: the repo's build_prompt (system prompt + user JSON in the Llama-3 chat template),
n samples with temperature/top_p, then the repo's own FloorplanGenerator._select_least picks one
candidate (valid JSON first, then RPLANGraph compatibility; see NOTES.md about overlap). We also
record which candidate the paper's stated criterion (min. room-room overlap area, ties broken by
compatibility) would pick.

Writes raw/<id>.json: {"id", "prompt", "candidates": [{"text", "finish_reason", "n_tokens"}],
"selected_code", "selected_paper", "output" (extract_output_json of the code-selected candidate)}.
Resumable: ids with an existing raw/<id>.json are skipped.
Run in fpe-floorplan_rlvr with cwd anywhere (the repo is put on sys.path).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external/methods/floorplan_rlvr"
sys.path.insert(0, str(REPO))


def paper_select(cands, input_prompt):
    """Paper criterion: valid JSON, then min. overlap area between room (non-door) polygons, then
    min. compatibility."""
    from shapely.geometry import Polygon
    from src.pred.extract_output_json import extract_output_json
    from src.dataset_convert.rplan_graph import RPLANGraph

    def key(text):
        try:
            out = extract_output_json(text)
            if not out:
                return (1, float("inf"), float("inf"))
        except Exception:
            return (1, float("inf"), float("inf"))
        try:
            polys = []
            for s in out.get("spaces", []):
                if s.get("room_type") in ("interior_door", "front_door"):
                    continue
                p = Polygon([(float(q["x"]), float(q["y"])) for q in s["floor_polygon"]])
                polys.append(p if p.is_valid else p.buffer(0))
            ov = sum(polys[i].intersection(polys[j]).area
                     for i in range(len(polys)) for j in range(i + 1, len(polys)))
        except Exception:
            ov = float("inf")
        try:
            comp = RPLANGraph.from_floorplan_json(out).compatibility_score(
                RPLANGraph.from_labeled_adjacency(input_prompt.get("input_graph", {})))
        except Exception:
            comp = float("inf")
        return (0, ov, comp)
    keys = [key(c) for c in cands]
    return min(range(len(cands)), key=lambda i: keys[i])


class _Cand:  # what _select_least expects (has .text)
    def __init__(self, text):
        self.text = text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out_raw", required=True)
    ap.add_argument("--n_best", type=int, default=10, help="best-of-n (paper: 10)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=5120)
    ap.add_argument("--gpu_mem", type=float, default=0.93)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (W8A16 Marlin on A100)")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=100, help="prompts per generate() call (checkpointing)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_num_batched_tokens", type=int, default=None,
                    help="prefill chunk size; smaller -> smaller activation reservation -> more KV cache")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from src.utils import build_prompt
    from src.pred.extract_output_json import extract_output_json
    from src.pred.floorplan_generator import FloorplanGenerator

    recs = [json.loads(l) for l in open(a.inputs)]
    if a.limit:
        recs = recs[:a.limit]
    os.makedirs(a.out_raw, exist_ok=True)
    todo = [r for r in recs if not os.path.exists(os.path.join(a.out_raw, f"{r['id']}.json"))]
    print(f"{len(recs)} inputs, {len(todo)} to do", flush=True)
    if not todo:
        return

    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", quantization=a.quantization,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enable_lora=bool(a.lora), max_lora_rank=64 if a.lora else 16,
              enable_prefix_caching=True, max_num_seqs=a.max_num_seqs, seed=a.seed,
              enforce_eager=a.enforce_eager, max_num_batched_tokens=a.max_num_batched_tokens)
    t_load = time.time() - t0
    tok = llm.get_tokenizer()
    lora = LoRARequest("floorplan_adapter", 1, a.lora) if a.lora else None
    sp = SamplingParams(n=a.n_best, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                        stop_token_ids=[128001, 128008, 128009], seed=None)
    stats = {"load_sec": round(t_load, 1), "gen_sec": 0.0, "n_tokens": 0}
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]
        # Training tokenizes build_prompt(...) with add_special_tokens=False (single BOS); do the same.
        prompts = [TokensPrompt(prompt_token_ids=tok(build_prompt(r), add_special_tokens=False)["input_ids"])
                   for r in chunk]
        t1 = time.time()
        outs = llm.generate(prompts, sp, lora_request=lora)
        dt = time.time() - t1
        ntok = 0
        for r, o in zip(chunk, outs):
            input_prompt = json.loads(r["prompt"])["input"]
            cands = [{"text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)}
                     for c in o.outputs]
            ntok += sum(c["n_tokens"] for c in cands)
            texts = [c["text"] for c in cands]
            wrapped = [_Cand(t) for t in texts]
            best = FloorplanGenerator._select_least(None, wrapped, input_prompt)
            sel_code = next(i for i, w in enumerate(wrapped) if w is best)
            sel_paper = paper_select(texts, input_prompt)
            rec = {"id": r["id"], "prompt": r["prompt"], "n_prompt_tokens": len(o.prompt_token_ids),
                   "candidates": cands, "selected_code": sel_code, "selected_paper": sel_paper,
                   "output": extract_output_json(texts[sel_code])}
            with open(os.path.join(a.out_raw, f"{r['id']}.json"), "w") as f:
                json.dump(rec, f)
        stats["gen_sec"] += dt
        stats["n_tokens"] += ntok
        print(f"[{c0 + len(chunk)}/{len(todo)}] {dt:.0f}s, {ntok / dt:.0f} tok/s, "
              f"{ntok / len(chunk) / a.n_best:.0f} tok/sample", flush=True)
    stats["gen_sec"] = round(stats["gen_sec"], 1)
    with open(os.path.join(a.out_raw, f"_gen_stats_{int(time.time())}.json"), "w") as f:
        json.dump({**stats, "args": vars(a)}, f, indent=1)
    print("done", stats, flush=True)


if __name__ == "__main__":
    main()
