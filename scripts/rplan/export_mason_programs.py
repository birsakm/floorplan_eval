"""Export the 1,000 RPLAN programs of Table 1 block (a) in Mason2's program format, as input for our method (APM).

  python scripts/rplan/export_mason_programs.py [--out outputs/rplan/g2p_test1000_mason] [--push]

Source: outputs/rplan/g2p_test1000_realdoors_programs/samples (scripts/rplan/make_programs.py): the bubble diagrams that
House-GAN++ and HouseDiffusion receive (rooms + door connections + front door), plus the real rooms' net areas,
which ProcTHOR and Infinigen receive. The real layouts are NOT exported (they stay in floorplan_eval).

Per input, Mason2's program graph (docs/concepts/01-home-state.md of mason2):
  nodes    {id "<type>:<k>", type, area_m2, outdoor?, rplan_type}; plus one outdoor 'landing' node (area 2 m^2,
           not from RPLAN) because Mason anchors the entrance on a landing
  edges    {id "edge:<a>--<b>", a, b, kind "door"}; the bubble diagram's door connections + entrance room -- landing
  entrance "opening:edge:<entrance room>--landing:0"
Written: programs/<rplan id>.json, programs.jsonl (same records, one per line), manifest.json, README.md.
--push copies the folder to noface:/data/mason/michael/benchmark_inputs/<name>/ (group 'mason' can read it).
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "rplan" / "g2p_test1000_realdoors_programs" / "samples"
NAME = "rplan_g2p_test1000"
NOFACE = "michael@10.68.74.7"
REMOTE = "/data/mason/michael/benchmark_inputs"
# RPLAN bubble-diagram room names (House-GAN++ vocabulary) -> Mason2 room types
MASON_TYPE = {"living_room": "living_room", "bedroom": "bedroom", "bathroom": "bathroom", "kitchen": "kitchen",
              "balcony": "balcony", "study room": "office", "dining room": "dining_room", "storage": "storage",
              "entrance": "hallway", "unknown": "storage"}
OUTDOOR = {"balcony"}
LANDING_M2 = 2.0
INPUT_VERSION = 2  # 1: synthesised interior doors (withdrawn 2026-11-04); 2: RPLAN's real interior doors


def convert(sample):
    p = sample["condition"]["program"]
    k = Counter()
    ids, nodes = {}, []
    for r in p["rooms"]:
        t = MASON_TYPE[r["type_rplan"]]
        nid = f"{t}:{k[t]}"
        k[t] += 1
        ids[r["id"]] = nid
        n = {"id": nid, "type": t, "area_m2": r["target_net_area"]}
        if t in OUTDOOR:
            n["outdoor"] = True
        n["rplan_type"] = r["type_rplan"]
        nodes.append(n)
    nodes.append({"id": "landing:0", "type": "landing", "outdoor": True, "area_m2": LANDING_M2, "synthetic": True})
    edges, seen = [], set()
    for e in p["edges"]:
        a, b = ids[e["room_a"]], ids[e["room_b"]]
        if frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        edge = {"id": f"edge:{a}--{b}", "a": a, "b": b, "kind": "door"}
        if e.get("inferred"):
            edge["inferred"] = True
        edges.append(edge)
    ent = ids[p["entrance"]]
    edges.append({"id": f"edge:{ent}--landing:0", "a": ent, "b": "landing:0", "kind": "door"})
    rid = sample["id"]
    return {"input_id": f"rplan:{rid}", "rplan_id": rid, "benchmark": NAME, "input_version": INPUT_VERSION,
            "input": "bubble+areas",
            "program": {"nodes": nodes, "edges": edges, "entrance": f"opening:edge:{ent}--landing:0"}}


README = """# RPLAN benchmark inputs for APM (Table 1 block (a))

{n} programs, one per RPLAN apartment of the Graph2Plan test split (the same 1,000 inputs every method in
Table 1(a) gets). Written by floorplan_eval `scripts/rplan/export_mason_programs.py` on {today}.

**Input version 2** (`input_version: 2` in every record): the door connections are RPLAN's real interior doors.
Version 1 (synthesised doors, written earlier on 2026-11-04) is withdrawn: please run on this version only.

- `programs/<rplan id>.json` and `programs.jsonl` (the same records, one per line)
- `manifest.json`: counts, room types, sha256 of `programs.jsonl`

## Record

```json
{example}
```

- `program` is Mason2's program graph (`{{nodes, edges, entrance}}`, docs/concepts/01-home-state.md).
- Room types are Mason2 types. `rplan_type` keeps the RPLAN name (`study room` -> office, `entrance` -> hallway,
  `unknown` -> storage; all RPLAN bedroom kinds are `bedroom`).
- `area_m2` = the real room's net area in the RPLAN plan.
- `edges` = the door connections of the bubble diagram (the input of House-GAN++ / HouseDiffusion): RPLAN's real
  interior doors, plus the entrance door to `landing:0`. Where RPLAN draws no door to a room (mostly balconies,
  8% of the apartments), the room gets a door to the room it shares the longest wall with (`inferred: true`). The landing is not in RPLAN (2 m^2, `synthetic: true`); it is there only because
  Mason anchors the entrance on an outdoor landing.
- No footprint/boundary: bubble-diagram methods do not get one either.

## What we need: a run of APM with this program as input

1. **Program stage bypassed**: build the home from the given program (rooms, types, connections, entrance)
   instead of sampling one. Anything the Roomplan stage needs beyond the graph (e.g. circulation, wet-room plans)
   may be derived from it, but rooms must not be added, removed or retyped, and connections must not change:
   Table 1 scores room types and the door graph against this input (as for every other method).
2. **Areas**: two settings are possible; the paper uses one of them (to be decided):
   - `bubble+areas` (as ProcTHOR and Infinigen): use `area_m2` as the room's target area.
   - `bubble` (as House-GAN++ and HouseDiffusion): ignore `area_m2` and let the program stage's priors choose areas.
3. **One home per input, seed 0**, single storey, published as a generator version on noface like the other runs,
   with `rplan_g2p_test1000` (and the area setting) in the version note.
4. **Every generated home must carry its `input_id`** (e.g. `rplan:101`) so we can match it to its input and the
   real layout. Node ids should stay as given.
5. Failures count: an input without a home lowers N, as for the other methods.

The evaluation (Valid, Reach, Types, Graph, Area, dFID, dKID against RPLAN) then runs in floorplan_eval.
Contact: Michael.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "rplan" / "g2p_test1000_mason"))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "programs").mkdir(parents=True, exist_ok=True)
    recs = [convert(json.load(open(f))) for f in sorted(SRC.glob("*.json"), key=lambda f: int(f.stem))]
    lines = []
    for r in recs:
        s = json.dumps(r, separators=(",", ":"))
        (out / "programs" / f"{r['rplan_id']}.json").write_text(json.dumps(r, indent=1) + "\n")
        lines.append(s)
    body = "\n".join(lines) + "\n"
    (out / "programs.jsonl").write_text(body)
    types = Counter(n["type"] for r in recs for n in r["program"]["nodes"])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    json.dump({"benchmark": NAME, "n_programs": len(recs), "room_types": dict(types),
               "rooms_per_program": round(sum(len(r["program"]["nodes"]) - 1 for r in recs) / len(recs), 2),
               "programs_jsonl_sha256": hashlib.sha256(body.encode()).hexdigest(),
               "source": "floorplan_eval outputs/rplan/g2p_test1000_realdoors_programs (scripts/rplan/make_programs.py)",
               "floorplan_eval_commit": commit, "written": date.today().isoformat(), "input_version": INPUT_VERSION,
               "doors": "RPLAN's real interior doors; inferred door where a room has none"},
              open(out / "manifest.json", "w"), indent=1)
    (out / "README.md").write_text(README.format(n=len(recs), today=date.today().isoformat(),
                                                 example=json.dumps(recs[0], indent=1)))
    print(f"{len(recs)} programs -> {out}; room types {dict(types)}")
    if a.push:
        dst = f"{REMOTE}/{NAME}"
        subprocess.run(["ssh", "-o", "BatchMode=yes", NOFACE, f"mkdir -p {dst}"], check=True)
        subprocess.run(["rsync", "-a", "--delete", "--chmod=Dg+rwxs,Fg+rw", f"{out}/", f"{NOFACE}:{dst}/"], check=True)
        print(f"copied to {NOFACE}:{dst}")


if __name__ == "__main__":
    main()
