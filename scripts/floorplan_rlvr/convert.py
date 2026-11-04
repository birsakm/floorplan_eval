"""Convert Floor-plan RLVR raw outputs to the common format (run in the `fpe` env).

Reads   <variant>/raw/<id>.json (scripts/floorplan_rlvr/generate.py) and raw/_generation_info.json (run.sh)
Writes  <variant>/samples/<id>.json   generated plan (selected best-of-n candidate)
        <variant>/gt_samples/<id>.json GT plan for the same input (from the prompt's source spaces)
        <variant>/run_info.json        incl. n_requested / n_parsed / n_valid, per-sample status

Units: the model works in metres in the RPLAN image frame (pixel * 18/256, origin at the image
corner, y down), so polygons are mapped back to the 256 px RPLAN frame with px = m * 256/18,
the same frame as the Graph2Plan GT. Interior and front doors are kept as `doors`.
--select code|paper picks which best-of-n candidate to convert (default: code = the repo's
FloorplanGenerator._select_least, as stored in raw[...]["selected_code"]).
"""
import argparse
import datetime
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fpeval.format import make_sample, save_sample  # noqa: E402

REPO = ROOT / "external/methods/floorplan_rlvr"
PX_PER_M = 256 / 18
TYPE_MAP = {"living_room": "living_room", "kitchen": "kitchen", "bedroom": "bedroom",
            "bathroom": "bathroom", "balcony": "balcony", "entrance": "entrance",
            "dining_room": "dining_room", "study_room": "study", "storage": "storage",
            "unknown": "other"}
DOOR_TYPES = {"interior_door", "front_door"}


def _load_repair():
    spec = importlib.util.spec_from_file_location("rlvr_json_repair", REPO / "src/utils/json_repair.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.repair_json


repair_json = _load_repair()


def parse_output(text):
    """Mirror of src/pred/extract_output_json.py with a status: json | repaired | unparseable."""
    s = text
    k = s.find("assistant")
    if k != -1:
        s = s[k + len("assistant"):].lstrip()
    status = "json"
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        status = "repaired"
        try:
            obj = repair_json(s, return_objects=True)
            obj = obj if obj != "" else {}
        except Exception:
            obj = {}
    if isinstance(obj, dict) and "output" in obj:
        obj = obj["output"]
    elif isinstance(obj, dict) and "floor_plan" in obj:
        obj = obj["floor_plan"]
    elif isinstance(obj, list):
        obj = {"rooms": obj}
    if not isinstance(obj, dict) or not obj:
        return None, "unparseable"
    return obj, status


def _poly_px(fp):
    pts = []
    for q in fp:
        x, y = float(q["x"]), float(q["y"])
        if not (abs(x) < 1e6 and abs(y) < 1e6):
            raise ValueError("non-finite")
        pts.append([round(x * PX_PER_M, 3), round(y * PX_PER_M, 3)])
    dedup = []
    for p in pts:
        if not dedup or p != dedup[-1]:
            dedup.append(p)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup


def _area(p):
    return 0.5 * abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p))))


def extract(obj):
    """-> rooms, doors, n_room_entries, n_dropped."""
    spaces = obj.get("spaces", obj.get("rooms", []))
    rooms, doors, n_entries, dropped = [], [], 0, 0
    if not isinstance(spaces, list):
        return rooms, doors, 0, 0
    for s in spaces:
        if not isinstance(s, dict):
            dropped += 1
            continue
        rt = str(s.get("room_type", "unknown"))
        is_door = rt in DOOR_TYPES
        n_entries += not is_door
        try:
            poly = _poly_px(s["floor_polygon"])
            ok = len({tuple(p) for p in poly}) >= 3 and _area(poly) > 0
        except Exception:
            ok = False
        if not ok:
            dropped += not is_door
            continue
        if is_door:
            doors.append({"type": rt, "polygon": poly})
        else:
            rooms.append({"type": TYPE_MAP.get(rt, "other"), "type_native": rt, "polygon": poly,
                          "id_native": s.get("id"), "area_m2_reported": s.get("area")})
    return rooms, doors, n_entries, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--inputs", required=True, help="inputs.jsonl from prepare_inputs.py")
    ap.add_argument("--select", choices=["code", "paper"], default="code")
    a = ap.parse_args()
    vd = Path(a.variant_dir)
    variant = vd.name
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(a.inputs)}
    gp = vd / "raw" / "_generation_info.json"
    ginfo = json.load(open(gp)) if gp.exists() else {}
    n_req = ginfo.get("n_requested", len(recs))
    ids = list(recs)[:n_req]
    for d in ("samples", "gt_samples"):
        (vd / d).mkdir(parents=True, exist_ok=True)
        for f in (vd / d).glob("*.json"):
            f.unlink()

    stats = {"json": 0, "repaired": 0, "unparseable": 0}
    n_gen = n_valid = n_valid_all = n_trunc = n_cnt = n_types = n_ids = n_sel_diff = 0
    n_cands = n_cands_parsed = 0
    per, failures = {}, []
    for sid in ids:
        r = recs[sid]
        inp = json.loads(r["prompt"])["input"]
        gt_rooms, gt_doors, _, _ = extract({"spaces": r["spaces"]})
        cond = {"type": "bubble_diagram+areas", "ref_id": f"rplan_{sid}", "room_count": inp["room_count"],
                "total_area_m2": inp["total_area"], "input_graph": inp["input_graph"],
                "spaces_spec": inp["spaces"], "graph_mode": r.get("graph_mode")}
        save_sample(make_sample("rplan", variant, sid, [{k: x[k] for k in ("type", "type_native", "polygon")}
                                                        for x in gt_rooms],
                                units="px", scale_m_per_unit=18 / 256, doors=gt_doors,
                                condition={**cond, "gt_of": sid}), vd / "gt_samples" / f"{sid}.json")
        rp = vd / "raw" / f"{sid}.json"
        if not rp.exists():
            per[sid] = {"parse": "missing"}
            failures.append(sid)
            continue
        raw = json.load(open(rp))
        n_gen += 1
        for c in raw["candidates"]:
            n_cands += 1
            n_cands_parsed += parse_output(c["text"])[1] == "json"
        k = raw["selected_code"] if a.select == "code" else raw["selected_paper"]
        n_sel_diff += raw["selected_code"] != raw["selected_paper"]
        cand = raw["candidates"][k]
        obj, status = parse_output(cand["text"])
        stats[status] += 1
        trunc = cand["finish_reason"] != "stop"
        n_trunc += trunc
        info = {"parse": status, "candidate": k, "finish_reason": cand["finish_reason"], "n_tokens": cand["n_tokens"]}
        rooms, doors, n_entries, dropped = extract(obj) if obj else ([], [], 0, 0)
        # requested rooms = prompt spaces without the front door
        req = [s for s in inp["spaces"] if s["room_type"] != "front_door"]
        info.update({"n_rooms_output": n_entries, "n_rooms_dropped": dropped, "n_rooms_requested": len(req),
                     "n_doors_output": len(doors)})
        per[sid] = info
        if not rooms:
            failures.append(sid)
            continue
        n_valid += 1
        n_valid_all += dropped == 0
        n_cnt += len(rooms) == len(req)
        n_types += sorted(x["type_native"] for x in rooms) == sorted(s["room_type"] for s in req)
        n_ids += sorted(str(x["id_native"]) for x in rooms) == sorted(s["id"] for s in req)
        save_sample(make_sample("floorplan_rlvr", variant, sid, rooms, units="px", scale_m_per_unit=18 / 256,
                                doors=doors, condition=cond), vd / "samples" / f"{sid}.json")

    run_info = {
        "method": "floorplan_rlvr", "variant": variant, **{k: v for k, v in ginfo.items() if k != "n_requested"},
        "inputs": a.inputs, "units": "px (RPLAN 256x256 frame; model outputs metres, px = m * 256/18)",
        "selection": a.select, "n_requested": n_req, "n_generated": n_gen,
        "n_parsed": stats["json"], "n_parsed_incl_repair": stats["json"] + stats["repaired"],
        "n_unparseable": stats["unparseable"], "parse_status_counts": stats,
        "n_valid": n_valid, "n_valid_all_rooms_ok": n_valid_all, "n_truncated": n_trunc,
        "n_room_count_matches_prompt": n_cnt, "n_room_type_multiset_matches_prompt": n_types,
        "n_room_ids_match_prompt": n_ids,
        "n_candidates_total": n_cands, "n_candidates_strict_json": n_cands_parsed,
        "n_selected_code_ne_paper": n_sel_diff,
        "validity_definition": "per input, the selected best-of-n candidate: n_parsed = strict json.loads; "
                               "n_parsed_incl_repair adds the repo's json_repair fallback (as extract_output_json); "
                               "n_valid = parsed with >=1 non-door space whose floor_polygon has >=3 distinct numeric "
                               "vertices and non-zero area (malformed rooms dropped, see n_valid_all_rooms_ok). "
                               "Invalid outputs are not written to samples/ and count against yield.",
        "date_converted": datetime.datetime.now().isoformat(timespec="seconds"),
        "failures": failures, "per_sample": per,
    }
    json.dump(run_info, open(vd / "run_info.json", "w"), indent=1)
    print(f"{variant}: requested {n_req}, generated {n_gen}, parsed {stats['json']} (+{stats['repaired']} repaired), "
          f"valid {n_valid} (all rooms ok {n_valid_all}), truncated {n_trunc}, room-count match {n_cnt}, "
          f"type match {n_types}, id match {n_ids}, code!=paper selection {n_sel_diff}")


if __name__ == "__main__":
    main()
