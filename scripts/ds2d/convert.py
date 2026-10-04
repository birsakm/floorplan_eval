"""Convert DS2D raw generations to the common format (run in the `fpe` env, repo root on PYTHONPATH).

  conda run -n fpe python scripts/ds2d/convert.py outputs/ds2d/<variant> --dataset {rplan,procthor}

Reads   <variant>/raw/*.json (written by scripts/ds2d/generate.py) and raw/_generation_info.json
Writes  <variant>/samples/<id>.json      parsed LLM outputs with >=1 valid room polygon
        <variant>/gt_samples/<id>.json   ground truth for the same test inputs
        <variant>/gt_renders/<id>.png
        <variant>/run_info.json          incl. n_requested / n_parsed / n_valid and per-sample failures
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpeval.format import canonical_room_type, make_sample, save_sample, validate  # noqa: E402
from fpeval.render import render  # noqa: E402

DS2D = ROOT / "external" / "methods" / "ds2d"
EXTRA_ALIASES = {"Wall-in": "closet", "StudyRoom": "study", "SecondRoom": "bedroom", "ChildRoom": "bedroom",
                 "GuestRoom": "bedroom", "MasterRoom": "bedroom"}
RPLAN_M_PER_PX = 18 / 256  # DS2D pixel2len


def _load_repair():
    spec = importlib.util.spec_from_file_location("ds2d_json_repair", DS2D / "src" / "utils" / "json_repair.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.repair_json


repair_json = _load_repair()


def parse_output(text):
    """Return (obj or None, status) with status in {json, json_prefix, repaired, unparseable}."""
    s = text
    k = s.find("Output:")
    if k != -1:
        s = s[k + len("Output:"):]
    j = s.find("{")
    if j == -1:
        return None, "unparseable"
    s = s[j:]
    try:
        return json.loads(s), "json"
    except json.JSONDecodeError:
        pass
    try:  # valid JSON object followed by trailing text
        obj, _ = json.JSONDecoder().raw_decode(s)
        return obj, "json_prefix"
    except json.JSONDecodeError:
        pass
    try:  # DS2D's own fallback (src/utils/json_repair.py), as used in src/pred/extract_output_json.py
        obj = repair_json(s, return_objects=True)
        if isinstance(obj, dict) and obj:
            return obj, "repaired"
    except Exception:
        pass
    return None, "unparseable"


def _num(v):
    if isinstance(v, bool):
        raise ValueError
    return float(v)


def shoelace(poly):
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2


def extract_rooms(obj, dataset):
    """-> (rooms in common format, n_rooms_in_output, n_dropped)."""
    rooms_in = obj.get("rooms") if isinstance(obj, dict) else None
    if not isinstance(rooms_in, list):
        return [], 0, 0
    rooms, dropped = [], 0
    for r in rooms_in:
        try:
            pts = []
            for p in r["floor_polygon"]:
                x = _num(p["x"])
                z = _num(p["z"] if "z" in p else p["y"])
                # RPLAN: DS2D "x" = image row, "z" = image column -> [col, row] (y down)
                pts.append([z, x] if dataset == "rplan" else [x, z])
            dedup = [q for i, q in enumerate(pts) if q != pts[i - 1]] if len(pts) > 1 else pts
            if len(dedup) < 3 or shoelace(dedup) <= 1e-9:
                raise ValueError("degenerate polygon")
            native = str(r.get("room_type", "unknown"))
            rooms.append({"type": canonical_room_type(native, EXTRA_ALIASES), "type_native": native,
                          "polygon": dedup, "id_native": r.get("id")})
        except (KeyError, TypeError, ValueError):
            dropped += 1
    return rooms, len(rooms_in), dropped


def to_sample(rooms, variant, sid, dataset, condition, graph=None):
    units = "px" if dataset == "rplan" else "m"
    s = make_sample("ds2d", variant, sid, [{k: v for k, v in r.items() if k != "id_native"} for r in rooms],
                    units=units, condition=condition, graph=graph)
    s["scale_m_per_unit"] = RPLAN_M_PER_PX if dataset == "rplan" else 1.0
    return s


def write_gt(vd, variant, dataset, sid, gt, edges):
    gt_rooms, _, _ = extract_rooms(gt, dataset)
    gsamp = to_sample(gt_rooms, variant, sid, dataset,
                      {"type": "ground_truth", "ref_id": f"ds2d_{dataset}_{sid}"},
                      graph={"edges": [[int(e[0]), int(e[1]), "adjacent"] for e in edges]} if edges else None)
    save_sample(gsamp, vd / "gt_samples" / f"{sid}.json")
    render(gsamp).save(vd / "gt_renders" / f"{sid}.png")
    return gt_rooms


def write_gt_only(vd, variant, dataset, inputs, n):
    for sub in ("gt_samples", "gt_renders"):
        (vd / sub).mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(inputs) if l.strip()][:n]
    for rec in recs:
        if dataset == "rplan":
            write_gt(vd, variant, dataset, rec["id"], rec["gt"], rec["edges"])
        else:
            write_gt(vd, variant, dataset, rec["id"], rec["sample"], rec["sample"]["edges"])
    print(f"{variant}: wrote {len(recs)} GT samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--dataset", choices=["rplan", "procthor"], required=True)
    ap.add_argument("--gt_from_inputs", default=None,
                    help="only write gt_samples/ + gt_renders/ from a test_inputs jsonl (no generations needed)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    vd = Path(a.variant_dir)
    variant = vd.name
    if a.gt_from_inputs:
        write_gt_only(vd, variant, a.dataset, a.gt_from_inputs, a.limit)
        return
    raw = sorted(p for p in (vd / "raw").glob("*.json") if not p.name.startswith("_"))
    ginfo_p = vd / "raw" / "_generation_info.json"
    ginfo = json.load(open(ginfo_p)) if ginfo_p.exists() else {}
    for sub in ("samples", "gt_samples", "gt_renders"):
        (vd / sub).mkdir(exist_ok=True)
        for f in (vd / sub).glob("*"):
            f.unlink()

    stats = {"json": 0, "json_prefix": 0, "repaired": 0, "unparseable": 0}
    n_valid = n_valid_all = n_trunc = n_count_match = n_type_match = 0
    failures, per_sample = [], {}
    for p in raw:
        r = json.load(open(p))
        sid = r["id"]
        edges = r.get("edges") or []
        cond_type = "bubble_diagram+constraints" if (a.dataset == "rplan" or ginfo.get("version") == "bd") \
            else "constraints"
        condition = {"type": cond_type, "ref_id": f"ds2d_{a.dataset}_{sid}", "level": r["level"],
                     "constraints": r.get("prompt_spec_obj"),
                     "adjacency": [[int(e[0]), int(e[1])] for e in edges] if cond_type.startswith("bubble") else None,
                     "lora": ginfo.get("lora")}
        # ground truth
        gt_rooms = write_gt(vd, variant, a.dataset, sid, r["gt"], edges)
        # generation
        if not r.get("hit_eos", True):
            n_trunc += 1
        obj, status = parse_output(r["generated_text"])
        stats[status] += 1
        info = {"parse": status, "hit_eos": r.get("hit_eos"), "n_new_tokens": r.get("n_new_tokens")}
        if obj is None:
            failures.append({"id": sid, "reason": "unparseable"}); per_sample[sid] = info
            continue
        rooms, n_out, dropped = extract_rooms(obj, a.dataset)
        info.update({"n_rooms_output": n_out, "n_rooms_dropped": dropped, "n_rooms_gt": len(gt_rooms)})
        if not rooms:
            failures.append({"id": sid, "reason": "no valid room polygon"}); per_sample[sid] = info
            continue
        samp = to_sample(rooms, variant, sid, a.dataset, condition)
        validate(samp)
        save_sample(samp, vd / "samples" / f"{sid}.json")
        n_valid += 1
        n_valid_all += dropped == 0
        n_count_match += len(rooms) == len(gt_rooms)
        n_type_match += sorted(x["type_native"] for x in rooms) == sorted(x["type_native"] for x in gt_rooms)
        info["valid"] = True
        per_sample[sid] = info

    n_req = ginfo.get("n_requested", len(raw))
    run_info = {
        "method": "ds2d", "variant": variant, "dataset": a.dataset,
        "submodule_commit": ginfo.get("commit"), "base_model": ginfo.get("base_model"),
        "checkpoint": ginfo.get("lora"), "prompt_version": ginfo.get("version"), "condition_level": ginfo.get("level"),
        "condition": ginfo.get("condition_description"),
        "decoding": ginfo.get("decoding"), "command": ginfo.get("command"),
        "test_inputs": ginfo.get("inputs"),
        "units": "px (RPLAN 256x256 grid, 18/256 m per px as in DS2D)" if a.dataset == "rplan" else "m",
        "n_requested": n_req, "n_generated": len(raw),
        "n_parsed": stats["json"] + stats["json_prefix"],
        "n_parsed_incl_repair": stats["json"] + stats["json_prefix"] + stats["repaired"],
        "n_unparseable": stats["unparseable"], "parse_status_counts": stats,
        "n_valid": n_valid, "n_valid_all_rooms_ok": n_valid_all,
        "n_truncated_no_eos": n_trunc,
        "n_room_count_matches_gt": n_count_match, "n_room_type_multiset_matches_gt": n_type_match,
        "validity_definition": "n_parsed: strict json.loads (optionally ignoring trailing text); n_valid: parsed "
                               "(incl. DS2D json_repair fallback) with >=1 room whose floor_polygon has >=3 distinct "
                               "numeric vertices and non-zero area (malformed rooms are dropped, see "
                               "n_valid_all_rooms_ok). Invalid outputs are not written to samples/.",
        "runtime_sec": ginfo.get("runtime_sec"), "gpu": ginfo.get("gpu"),
        "date_generated": ginfo.get("date"), "date_converted": datetime.now().isoformat(timespec="seconds"),
        "failures": failures, "per_sample": per_sample,
    }
    json.dump(run_info, open(vd / "run_info.json", "w"), indent=1)
    print(f"{variant}: requested {n_req}, generated {len(raw)}, parsed {run_info['n_parsed']} "
          f"(+{stats['repaired']} repaired), valid {n_valid} (all rooms ok {n_valid_all}), truncated {n_trunc}, "
          f"room-count match {n_count_match}")


if __name__ == "__main__":
    main()
