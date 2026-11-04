"""Build Floor-plan RLVR prompts from the House-GAN++-style JSONs of the shared RPLAN test subset.

Input : data/method_inputs/houseganpp/rplan-g2p_test1000/<id>.json (scripts/houseganpp/g2p_to_hgjson.py;
        Graph2Plan RPLAN test split, synthesized interior doors), ids from
        data/method_inputs/common_rplan_test/ids_test_1000.txt
Output: data/method_inputs/floorplan_rlvr/rplan-g2p_test1000/inputs.jsonl, one record per id:
        {"id", "prompt" (the user-message JSON string exactly as RPLANConverter writes it),
         "room_count", "total_area", "input_graph", "spaces" (GT spaces, metres), "n_rooms_hg"}
        plus skipped.json.

The conversion calls the repo's own RPLANConverter._convert_entry (src/dataset_convert/rplan.py,
unmodified): room polygons from the HG edge segments, scaled by 18/256 m/px, area/width/height
rounded to 2 decimals, rectangles described by height+width and other polygons by area, front door
listed as a space, room_count = #non-door spaces - 1 (front door), input_graph from
RPLANGraph.from_housegan (non-door ed_rm pairs).

One adaptation (BUBBLE GRAPH): in the real RPLAN HG++ JSONs (data reader output), the room-room
ed_rm pairs coincide with the interior-door connections (checked on the 7 bundled RPLAN samples), and
the paper says the bubble diagram is "derived from interior-door connectivity". Our g2p JSONs
instead list every shared wall as a room-room ed_rm pair, so a straight conversion would give a
denser wall-adjacency graph than the model was trained on. With --graph door (default) room-room
ed_rm pairs that are not bridged by a (synthesized) interior door are reduced to [room] before
conversion; segment-to-room assignment (ed_rm[0]) and hence the polygons/areas are unchanged.
--graph wall keeps the HG JSON as is.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external/methods/floorplan_rlvr"
sys.path[:0] = [str(REPO / "src"), str(REPO)]  # rplan.py uses `dataset_convert.*`/`utils.*`, utils uses `src.utils.*`
from dataset_convert.rplan import RPLANConverter  # noqa: E402

DOOR, FRONT = 17, 15


def door_pairs(hg):
    """Room pairs bridged by an interior door node (from the door's ed_rm entries)."""
    t = hg["room_type"]
    att = {}
    for e in hg["ed_rm"]:
        if len(e) == 2 and t[e[0]] == DOOR and t[e[1]] not in (DOOR, FRONT):
            att.setdefault(e[0], set()).add(e[1])
    return {tuple(sorted(s)) for s in att.values() if len(s) == 2}


def to_door_graph(hg):
    t = hg["room_type"]
    pairs = door_pairs(hg)
    ed_rm = []
    for e in hg["ed_rm"]:
        if len(e) == 2 and t[e[0]] not in (DOOR, FRONT) and t[e[1]] not in (DOOR, FRONT) \
                and tuple(sorted(e)) not in pairs:
            e = [e[0]]
        ed_rm.append(e)
    return {**hg, "ed_rm": ed_rm}, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hg_dir", default=str(ROOT / "data/method_inputs/houseganpp/rplan-g2p_test1000"))
    ap.add_argument("--ids", default=str(ROOT / "data/method_inputs/common_rplan_test/ids_test_1000.txt"))
    ap.add_argument("--graph", choices=["door", "wall"], default="door")
    ap.add_argument("--out", default=str(ROOT / "data/method_inputs/floorplan_rlvr/rplan-g2p_test1000"))
    a = ap.parse_args()
    conv = RPLANConverter()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    recs, skipped, n_graph_mismatch = [], {}, 0
    for i in ids:
        hg = json.load(open(os.path.join(a.hg_dir, f"{i}.json")))
        pairs = None
        if a.graph == "door":
            hg, pairs = to_door_graph(hg)
        hg["rplan_id"] = i
        out = conv._convert_entry(hg)
        if out is None:
            skipped[i] = "RPLANConverter rejected (polygonize failed or disconnected graph)"
            continue
        if pairs is not None:  # sanity: resulting room-room edges == door pairs
            g = json.loads(out["input_graph"])
            n_edges = sum(1 for k, v in g.items() for n in v if k < n and "front_door" not in (k, n))
            if n_edges != len(pairs):
                n_graph_mismatch += 1
        recs.append({"id": i, "prompt": out["prompt"], "room_count": out["room_count"],
                     "total_area": out["total_area"], "input_graph": json.loads(out["input_graph"]),
                     "spaces": out["spaces"], "graph_mode": a.graph})
    with open(os.path.join(a.out, "inputs.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump({"skipped": skipped, "n_graph_mismatch": n_graph_mismatch, "graph_mode": a.graph},
              open(os.path.join(a.out, "skipped.json"), "w"), indent=1)
    print(f"wrote {len(recs)} prompts ({a.graph} graph), skipped {len(skipped)}, "
          f"graph/door-pair mismatches {n_graph_mismatch} -> {a.out}")


if __name__ == "__main__":
    main()
