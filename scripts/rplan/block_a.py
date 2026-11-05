"""Table 1 block (a): every method on the 1,000 RPLAN inputs (Graph2Plan test split), with the same metrics.

  PYTHONPATH=. python scripts/rplan/block_a.py      # -> results/rplan/block_a.json, block_a.md

Per method (columns of Table 1):
  Valid   layouts breaking no basic rule (fpeval.evaluate variants: per_sample.csv any_rule_violation)
  Reach   layouts with a front door from which every room is reachable through doors and openings
  Types   room types and counts equal the input (bubble methods: the input's rooms; boundary input: n/a)
  Graph   door graph read from the geometry equals the input door graph (typed graph isomorphism = GED 0)
  Area    relative error of the total net room area vs the input's target areas (only for inputs with areas)
  dFID, dKID  FID / KID x1000 against RPLAN minus the score of the real layouts of the same inputs
              (fpeval.evaluate fid: eval/fid_rplan_graph2plan_test.json, 'floor')
The Real row is the real layouts of the 1,000 inputs (outputs/rplan/g2p_test1000_realdoors_programs/gt_samples, metres);
its Reach comes from the original RPLAN masks with their real doors (fpeval/datasets/rplan_kaggle.py): Graph2Plan's
RPLAN has no interior doors, and its room polygons are simplified boxes, so the door graph between them is not usable.
Inputs: bubble methods read the bubble diagram from their sample's condition; ProcTHOR / Infinigen read the program
of scripts/rplan/make_programs.py (same rooms and connections, plus the real rooms' areas).
"""
import json
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from fpeval import rules  # noqa: E402
from fpeval.eval_config import VARIANTS  # noqa: E402
from fpeval.format import canonical_room_type, load_sample  # noqa: E402
from fpeval.geometry import rooms_as_shapes, to_polygon  # noqa: E402
from fpeval.metrics import door_edges  # noqa: E402

PROGRAMS = ROOT / "outputs" / "rplan" / "g2p_test1000_realdoors_programs"
REAL_DOORS = ROOT / "data" / "rplan_kaggle" / "converted" / "samples"  # RPLAN with its real doors
ROWS = [  # (row, variant key or None for Real, input kind)
    ("Real", None, "-"),
    ("House-GAN++", "houseganpp/rplan-g2p_bubble_test1000_realdoors", "bubble"),
    ("HouseDiffusion", "house_diffusion/rplan-g2p_bubble_test1000_realdoors", "bubble"),
    ("WallPlan", "wallplan/rplan_boundary_test1000", "boundary"),
    ("ProcTHOR", "procthor/rplan-g2p_program_test1000_realdoors", "bubble+areas"),
    ("Infinigen", "infinigen/rplan-g2p_program_test1000_realdoors", "bubble+areas"),
    ("Holodeck (Qwen3.8-27B)", "holodeck/rplan-g2p_text_test1000_realdoors_qwen3.8-27b", "bubble+areas"),
    ("Holodeck (Gemma-4-31B)", "holodeck/rplan-g2p_text_test1000_realdoors_gemma-4-31b", "bubble+areas"),
]
# APM (ours): every converted Mason2 version generated from these programs (run_all.py --benchmark rplan), one row
# per input setting. Versions named "...-retry<k>" (best of k seeds) deviate from the one-sample protocol and are
# left out unless named with --apm.


def apm_input(version):
    return "bubble" if version.endswith("-bubble") else "bubble+areas"


def apm_versions():
    runs = []
    for d in (ROOT / "outputs" / "ours").glob("mason2_*"):
        try:
            if json.load(open(d / "run_info.json")).get("benchmark") == "rplan_g2p_test1000" and "-retry" not in d.name:
                runs.append(d)
        except (OSError, ValueError):
            pass
    latest = {}  # newest run per input setting
    for d in sorted(runs, key=lambda d: (d / "run_info.json").stat().st_mtime):
        v = d.name[len("mason2_"):]
        latest[apm_input(v)] = v
    return [latest[k] for k in ("bubble", "bubble+areas") if k in latest]


APT_ALIASES = {"circulation": "corridor", "outdoor": "balcony", "office": "study", "small_bathroom": "bathroom"}
DOOR_NODES = {"interior_door", "front door", "front_door"}


def typed_graph(types, edges):
    G = nx.Graph()
    G.add_nodes_from((i, {"t": t}) for i, t in enumerate(types))
    G.add_edges_from(edges)
    return G


def input_graph(s, kind):
    """(types, door edges) of the input, or None when the input has no rooms (boundary)."""
    if kind == "bubble":
        gt = load_sample(PROGRAMS / "gt_samples" / f"{s['id'].rsplit('_', 1)[0]}.json")
        node_type = {r["node"]: r["type"] for r in gt["rooms"]}
        c = s["condition"]
        rooms = [i for i, t in enumerate(c["nodes"]) if t not in DOOR_NODES]
        pos = {n: k for k, n in enumerate(rooms)}
        by_door = {}
        for a, b in c["node_edges"]:
            for x, y in ((a, b), (b, a)):
                if c["nodes"][x] == "interior_door" and y in pos:
                    by_door.setdefault(x, set()).add(pos[y])
        edges = [tuple(sorted(v)) for v in by_door.values() if len(v) == 2]
        return [node_type.get(n, "other") for n in rooms], edges
    if kind == "bubble+areas":
        p = s["condition"]["program"]
        ids = {r["id"]: k for k, r in enumerate(p["rooms"])}
        edges = [(ids[e["room_a"]], ids[e["room_b"]]) for e in p["edges"] if e["kind"] in ("door", "open")]
        return [canonical_room_type(r["type"], APT_ALIASES) for r in p["rooms"]], edges
    return None


def one(args):
    path, scale, kind = args
    s = load_sample(path)
    shapes = rooms_as_shapes(s)
    tol = rules.WALL_HALF_THICKNESS_M / scale
    D = nx.Graph()
    D.add_nodes_from(range(len(shapes)))
    D.add_edges_from(door_edges(s, shapes, tol))
    front = None
    if shapes:
        tree = shapely.STRtree([g for _, g, _ in shapes])
        for d in s.get("doors") or []:
            if d.get("type") != "front_door":
                continue
            probe = to_polygon(d["polygon"])[0].buffer(tol)
            hits = [(probe.intersection(shapes[k][1]).area, int(k)) for k in tree.query(probe, predicate="intersects")]
            if hits:
                front = max(hits)[1]
                break
    r = {"id": s["id"], "has_front_door": front is not None,
         "reach": bool(front is not None and len(nx.node_connected_component(D, front)) == len(shapes))}
    inp = input_graph(s, kind)
    if inp is not None:
        types_in, edges_in = inp
        types_out = [t for t, _, _ in shapes]
        r["types"] = sorted(types_in) == sorted(types_out)
        G_in, G_out = typed_graph(types_in, edges_in), typed_graph(types_out, list(D.edges))
        r["graph"] = r["types"] and nx.is_isomorphic(G_in, G_out, node_match=lambda a, b: a["t"] == b["t"])
    if kind == "bubble+areas":
        want = s["condition"]["brief"]["net_area_m2"]
        got = sum(g.area for t, g, _ in shapes if t != "balcony") * scale ** 2
        r["area_rel_err"] = abs(got - want) / want
    return r


def evaluate_row(name, key, kind, workers, shown=None):
    """kind = format of the input stored in the samples; shown = the input the method used, if different (APM given
    the bubble setting still stores the program; its Area is then not adherence to a target and is kept apart)."""
    if key is None:
        files, scale, vdir = sorted((PROGRAMS / "gt_samples").glob("*.json")), 1.0, None
    else:
        vdir = ROOT / "outputs" / key
        files, scale = sorted((vdir / "samples").glob("*.json")), VARIANTS[key]["scale"]
    with Pool(workers) as pool:
        df = pd.DataFrame(pool.map(one, [(f, scale, kind) for f in files], chunksize=8))
    out = {"method": name, "variant": key, "input": shown or kind, "N": len(df),
           "reach": float(df.reach.mean()), "front_door": float(df.has_front_door.mean())}
    if "types" in df:
        out["types"], out["graph"] = float(df.types.mean()), float(df.graph.mean())
    if "area_rel_err" in df:
        out["area_rel_err" if (shown or kind) != "bubble" else "area_rel_err_vs_real"] = float(df.area_rel_err.mean())
    if key is None:
        # The doors in these files were synthesised for the bubble inputs (Graph2Plan's RPLAN has no interior doors).
        # Reach of the real layouts comes from the original RPLAN masks with their real doors instead
        # (fpeval/datasets/rplan_kaggle.py, condition.mask_reach: front door -> free space and doors -> every room).
        out.pop("reach")
        out.pop("front_door")
        real = [REAL_DOORS / f"test_{Path(f).stem}.json" for f in files]
        if all(p.exists() for p in real):
            m = [json.load(open(p))["condition"]["mask_reach"] for p in real]
            out["reach"] = float(np.mean([x["all"] for x in m]))
            out["reach_source"] = "original RPLAN masks (rplan_kaggle mask_reach)"
    if key is None:  # rules of the real layouts of the 1,000 inputs (metres), same per-sample evaluation as methods
        from fpeval.evaluate import _eval_dir
        g, _, _ = _eval_dir(files, workers, scale=1.0, scale_known=True)
        out["valid"] = float(1 - g.any_rule_violation.mean())
        out["dfid"] = out["dkid"] = 0.0
    else:
        ps = pd.read_csv(vdir / "eval" / "per_sample.csv")
        out["valid"] = float(1 - ps.any_rule_violation.mean())
        f = json.load(open(vdir / "eval" / "fid_rplan_graph2plan_test.json"))
        out["dfid"], out["dkid"] = f["fid"] - f["fid_floor"], f["kid_x1000"] - f["kid_floor_x1000"]
        out["fid_floor_is"] = f["floor"]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--apm", nargs="*", default=None,
                    help="Mason2 versions of the APM rows (default: newest seed-0 rplan benchmark run per input setting)")
    a = ap.parse_args()
    rows = []
    a_rows = list(ROWS)
    for v in (apm_versions() if a.apm is None else a.apm):
        a_rows.append((f"APM (ours, {apm_input(v)})", f"ours/mason2_{v}", "bubble+areas", apm_input(v)))
    for name, key, kind, *shown in a_rows:
        if key is not None and not (ROOT / "outputs" / key / "eval" / "fid_rplan_graph2plan_test.json").exists():
            print(f"{name}: not evaluated yet, skipped")
            continue
        rows.append(evaluate_row(name, key, kind, a.workers, *shown))
        print(rows[-1], flush=True)
    out = ROOT / "results" / "rplan"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "block_a.json", "w"), indent=1)

    def pct(r, k):
        return f"{100 * r[k]:.1f}" if k in r else "n/a"
    L = ["# Table 1 block (a): RPLAN, Graph2Plan test inputs", "", "Generated by `scripts/rplan/block_a.py`.", "",
         "| method | input | N | Valid % | Reach % | Types % | Graph % | Area | ΔFID | ΔKID |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        area = f"{100 * r['area_rel_err']:.1f}%" if "area_rel_err" in r else "n/a"
        L.append(f"| {r['method']} | {r['input']} | {r['N']} | {pct(r, 'valid')} | {pct(r, 'reach')} | "
                 f"{pct(r, 'types')} | {pct(r, 'graph')} | {area} | {r['dfid']:.1f} | {r['dkid']:.1f} |")
    (out / "block_a.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
