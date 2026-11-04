"""Programs for the procedural baselines from the RPLAN bubble diagrams of Table 1 block (a).

  PYTHONPATH=. python scripts/rplan/make_programs.py

Input: the 1,000 Graph2Plan test inputs as given to House-GAN++ / HouseDiffusion
(outputs/houseganpp/rplan-g2p_bubble_test1000_syndoors: condition = bubble diagram, gt_samples = the real layouts).
Output, in the format of our samples so that scripts/procthor and scripts/infinigen read them with --source:
  outputs/rplan/g2p_test1000_programs/samples/<id>.json   condition = program (room types, door connections,
                                                           entrance, and target net areas = the real rooms' areas)
  outputs/rplan/g2p_test1000_programs/gt_samples/<id>.json the real layout in metres (FID floor of the baselines)

The bubble diagram has no areas, but ProcTHOR and Infinigen need them, so they get the real rooms' areas: their input
is 'bubble+areas' (Table 1 says so). Room types are mapped to the program vocabulary of scripts/ours/convert.py.
"""
import json
from collections import Counter
from pathlib import Path

from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "houseganpp" / "rplan-g2p_bubble_test1000_syndoors"
OUT = ROOT / "outputs" / "rplan" / "g2p_test1000_programs"
SCALE = 18 / 256  # m per px
NATIVE = {"living_room": "living", "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom",
          "balcony": "outdoor", "study room": "office", "storage": "storage", "dining room": "dining",
          "entrance": "circulation", "unknown": "storage"}
NEEDS_EXTERIOR = {"living", "bedroom", "kitchen", "dining", "outdoor"}
DOOR_TYPES = {"interior_door", "front door", "front_door"}


def metres(sample):
    s = json.loads(json.dumps(sample))

    def sc(poly):
        return [[x * SCALE, y * SCALE] for x, y in poly]
    for r in s["rooms"]:
        r["polygon"] = sc(r["polygon"])
    for d in s.get("doors") or []:
        d["polygon"] = sc(d["polygon"])
    for w in s.get("windows") or []:
        w["polygon"] = sc(w["polygon"])
    if s.get("boundary"):
        s["boundary"] = sc(s["boundary"])
    s.update(units="m", scale_m_per_unit=1.0)
    return s


def program(cond, gt):
    nodes = cond["nodes"]
    rooms_idx = [i for i, t in enumerate(nodes) if t not in DOOR_TYPES]
    area = {r["node"]: Polygon(r["polygon"]).buffer(0).area * SCALE ** 2 for r in gt["rooms"] if "node" in r}
    rooms = []
    for i in rooms_idx:
        t = NATIVE.get(nodes[i], "storage")
        rooms.append({"id": f"r{i}", "type": t, "type_rplan": nodes[i], "target_net_area": round(area.get(i, 0.0), 3),
                      "storey": 0, "needs_exterior": t in NEEDS_EXTERIOR})
    # room - door - room in the bubble diagram = a door connection
    by_door = {}
    for a, b in cond["node_edges"]:
        for x, y in ((a, b), (b, a)):
            if nodes[x] == "interior_door" and y in rooms_idx:
                by_door.setdefault(x, []).append(y)
    edges = []
    for d, rs in by_door.items():
        rs = sorted(set(rs))
        if len(rs) == 2:
            edges.append({"room_a": f"r{rs[0]}", "room_b": f"r{rs[1]}", "kind": "door"})
    front = cond.get("front_door_rooms") or []
    entrance = f"r{front[0]}" if front else None
    counts = Counter(r["type"] for r in rooms)
    net = sum(r["target_net_area"] for r in rooms if r["type"] != "outdoor")
    return {"type": "bubble+areas", "home": gt["id"], "storey": None, "n_storeys": 1, "ref_id": cond.get("ref_id"),
            "brief": {"building_type": "apartment", "net_area_m2": round(net, 3), "room_counts": dict(counts)},
            "program": {"rooms": rooms, "edges": edges, "entrance": entrance}}


def main():
    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    (OUT / "gt_samples").mkdir(parents=True, exist_ok=True)
    n, skipped = 0, []
    for p in sorted((SRC / "samples").glob("*.json")):
        s = json.load(open(p))
        gid = p.stem.rsplit("_", 1)[0]
        g = SRC / "gt_samples" / f"{gid}.json"
        if not g.exists():
            skipped.append(gid)
            continue
        gt = json.load(open(g))
        cond = s["condition"]
        if any(r.get("node") is None for r in gt["rooms"]):
            skipped.append(gid)
            continue
        gm = metres(gt)
        gm["id"] = gid
        json.dump(gm, open(OUT / "gt_samples" / f"{gid}.json", "w"))
        rec = dict(gm, source="rplan", variant="g2p_test1000_programs", condition=program(cond, gt))
        json.dump(rec, open(OUT / "samples" / f"{gid}.json", "w"))
        n += 1
    json.dump({"source": str(SRC.relative_to(ROOT)), "n_programs": n, "skipped": skipped, "scale_m_per_px": SCALE},
              open(OUT / "run_info.json", "w"), indent=1)
    print(f"{n} programs -> {OUT}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
