"""Evaluate our generated apartments or houses with metrics from recent floor plan papers (`fpe` env).

  PYTHONPATH=. conda run -n fpe python scripts/ours/evaluate_ours.py --kind {apartment,house}

Houses are stored one sample per storey (<home>_s<k>.json). Brief, program and reachability metrics are
computed per home, with the stair rooms of consecutive storeys connected; realism and diversity per storey,
against real floors (CubiCasa5k, MagicPlan) with the same numbers of bedrooms and bathrooms.

Needs scripts/ours/convert.py (samples) and `python -m fpeval.evaluate refs` (reference per-sample CSVs).
Writes results/<method>/<variant name>.json (--variant <method>/<name>, default ours).

Metric groups (paper lineage in brackets):
  brief adherence     room counts and net area vs the brief          [DS2D prompt consistency, RLVR area error]
  program consistency door/open/wall edges of the program graph realised in the layout; typed graph edit
                      distance between program graph and the door graph read from geometry
                                                                     [House-GAN compatibility, DS2D self-consistency]
  reachability        all rooms reachable from the front door through doors; rooms only reachable through a
                      bedroom                                       [GRE-Diff reachability]
  daylight            habitable rooms with a window; window area / floor area (needs sill/head)
  realism             per-type area / aspect distributions vs composition-matched real apartments, and a
                      1-NN classifier two-sample test on plan features (0.5 = indistinguishable)
                                                                     [replaces FID/KID, which need renders]
  diversity           distinct typed door-graph topologies among 100 plans, vs matched real samples of 100
  FID / KID           Inception-V3 features of uniform layout renders (fpeval.render.render_layout), against the
                      matched real plans; with a real-vs-real floor at the same sample size
                                                                     [House-GAN/HouseDiffusion FID, GSDiff/GRE-Diff KID]
Rule checks and building norms come from fpeval.evaluate / fpeval.norms and are read from their outputs.
"""
import argparse
import json
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shapely

from fpeval import rules
from fpeval.eval_config import REFERENCES
from fpeval.format import load_sample
from fpeval.geometry import closing, fill_holes, n_corners, to_polygon
from shapely.ops import unary_union
from fpeval.metrics import door_edges, sample_metrics

ROOT = Path(__file__).resolve().parents[2]
HABITABLE = {"living_room", "bedroom", "kitchen", "dining_room", "study"}
N_MATCH = 20  # real plans drawn per generated plan with the same composition
KINDS = {
    "apartment": dict(
        types=["living_room", "kitchen", "bedroom", "bathroom", "balcony", "hallway"],
        merge={"corridor": "hallway", "entrance": "hallway"},
        real={"swiss_dwellings:all": "Swiss Dwellings", "resplan:all": "ResPlan"},
        match=["n_bedroom", "n_bathroom", "n_balcony"]),
    "house": dict(
        types=["living_room", "kitchen", "bedroom", "bathroom", "hallway", "storage", "study", "laundry", "outdoor"],
        merge={"corridor": "hallway", "entrance": "hallway", "balcony": "outdoor", "closet": "storage"},
        real={"cubicasa5k:all": "CubiCasa5k", "magicplan:train": "MagicPlan"},
        match=["n_bedroom", "n_bathroom"]),
}
CFG = KINDS["apartment"]  # set in main(); worker processes inherit it (fork)


def _t(t):
    return CFG["merge"].get(t, t)


# ---------- per-plan features (same code for ours and real data) -----------------------------------------

def plan_features(path):
    s = load_sample(path)
    out, recs, adj, shapes = sample_metrics(s, 1.0, True)
    n = len(shapes)
    tol = rules.WALL_HALF_THICKNESS_M
    D = nx.Graph()
    D.add_nodes_from(range(n))
    D.add_edges_from(door_edges(s, shapes, tol))
    types = [_t(t) for t, _, _ in shapes]

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
    f = {"id": s["id"], "n_rooms": n, "total_area_m2": out.get("total_area_m2"),
         "has_front_door": float(front is not None), "n_doors": D.number_of_edges(),
         "door_components": nx.number_connected_components(D) if n else np.nan}
    f["front_room"] = front if front is not None else -1
    for t in CFG["types"]:
        f[f"n_{t}"] = sum(x == t for x in types)
        f[f"area_{t}"] = sum(r["area_m2"] for r, x in zip(recs, types) if x == t)
    if front is not None:
        reach = nx.node_connected_component(D, front)
        f["frac_reachable"] = len(reach) / n
        f["all_reachable"] = float(len(reach) == n)
        # Habitable rooms that can only be reached by walking through a bedroom.
        nobed = D.subgraph([k for k in range(n) if types[k] != "bedroom" or k == front])
        ok = nx.node_connected_component(nobed, front)
        through = [k for k in range(n) if types[k] in HABITABLE and k not in ok
                   and not any(nb in ok for nb in D[k])]
        f["n_habitable_via_bedroom"] = len(through)
    else:
        f["frac_reachable"] = f["all_reachable"] = f["n_habitable_via_bedroom"] = np.nan
    halls = [k for k in range(n) if types[k] == "hallway"]
    others = [k for k in range(n) if types[k] != "hallway"]
    f["frac_rooms_off_hallway"] = (np.mean([any(types[nb] == "hallway" for nb in D[k]) for k in others])
                                   if halls and others else 0.0)
    deg = [D.degree(k) for k in range(n)]
    f["mean_door_degree"] = float(np.mean(deg)) if deg else np.nan
    f["max_door_degree"] = max(deg) if deg else np.nan
    # Windows: habitable rooms touched by a window polygon.
    wins = [to_polygon(w["polygon"])[0].buffer(tol) for w in s.get("windows") or []]
    hab = [k for k in range(n) if types[k] in HABITABLE]
    # ProcTHOR's layout stage places no windows: not measured (NaN) instead of 0.
    f["frac_habitable_with_window"] = (np.mean([any(w.intersects(shapes[k][1]) for w in wins) for k in hab])
                                       if hab and wins else (0.0 if hab and s["source"] != "procthor" else np.nan))
    f["has_windows"] = float(bool(wins))
    # Typed room-adjacency topology (shared walls, Weisfeiler-Lehman hash) for diversity. Shared walls,
    # not doors: the real datasets draw no door for open connections, so their door graphs are incomplete.
    for k in adj:
        adj.nodes[k]["t"] = D.nodes[k]["t"] = types[k]
    f["topology"] = nx.weisfeiler_lehman_graph_hash(adj, node_attr="t", iterations=3)
    # Shape of rooms and of the apartment outline.
    f["mean_room_aspect"] = float(np.mean([min(r["aspect"], 10) for r in recs])) if recs else np.nan
    f["frac_rect_rooms"] = float(np.mean([r["corners"] == 4 for r in recs])) if recs else np.nan
    f["mean_room_convexity"] = float(np.mean([r["convexity"] for r in recs])) if recs else np.nan
    if shapes:
        outline = fill_holes(closing(unary_union([g for _, g, _ in shapes]), tol))
        f["outline_rectangularity"] = outline.area / outline.minimum_rotated_rectangle.area
        f["outline_corners"] = n_corners(outline, simplify_tol=0.05) if outline.geom_type == "Polygon" else np.nan
        f["mean_adjacency_degree"] = float(np.mean([adj.degree(k) for k in adj]))
    rooms = [{"id": s["id"], "type": x, "area_m2": r["area_m2"], "aspect": r["aspect"], "width_m": r["width_m"]}
             for r, x in zip(recs, types)]
    return f, rooms, D, shapes, s


def plan_features_light(path):
    f, rooms, *_ = plan_features(path)
    return f, rooms


# ---------- ours only: brief, program graph, window areas -------------------------------------------------

def home_extra(paths):
    """Brief, program and reachability metrics for one home (one path per storey)."""
    from shapely.geometry import Polygon
    per = [plan_features(p) for p in paths]
    s0 = per[0][4]
    cond = s0["condition"]
    home = cond.get("home", s0["id"])
    brief, prog = cond["brief"], cond["program"]
    H, adj_pairs, nodes = nx.Graph(), set(), {}
    front = None
    for k, (f, _, D, shapes, s) in enumerate(per):
        adj = sample_metrics(s, 1.0, True)[2]
        for i, r in enumerate(s["rooms"]):
            nodes[(k, i)] = r
            H.add_node((k, i), t=_t(r["type"]), native=r["type_native"])
        H.add_edges_from(((k, i), (k, j)) for i, j in D.edges)
        adj_pairs |= {frozenset(((k, i), (k, j))) for i, j in adj.edges}
        if f["front_room"] >= 0 and front is None:
            front = (k, f["front_room"])
    stairs = [n for n in H if H.nodes[n]["native"] == "stairs"]
    H.add_edges_from((a, b) for a in stairs for b in stairs if b[0] == a[0] + 1)
    rooms = list(nodes.values())

    native = Counter(r["type_native"] for r in rooms)
    want = dict(brief.get("room_counts") or {})
    e = {"home": home, "n_storeys": len(per), "brief_net_area_m2": brief.get("net_area_m2")}
    e["room_counts_match"] = float(all(native.get(k, 0) == v for k, v in want.items()) and all(k in want for k in native))
    e["room_count_abs_err"] = sum(abs(native.get(k, 0) - want.get(k, 0)) for k in set(native) | set(want))
    area = Counter()
    for r in rooms:
        area[r["type_native"]] += Polygon(r["polygon"]).area
    net = sum(v for t, v in area.items() if t not in ("outdoor", "stairs"))
    e["net_area_m2"], e["stair_area_m2"] = net, area.get("stairs", 0.0)
    e["net_area_rel_err"] = abs(net - brief["net_area_m2"]) / brief["net_area_m2"]
    e["net_area_incl_stairs_rel_err"] = abs(net + e["stair_area_m2"] - brief["net_area_m2"]) / brief["net_area_m2"]

    # Program graph vs layout. Rooms map to program rooms by id (State keeps the link).
    idx = {r.get("program_room"): n for n, r in nodes.items()}
    pe = [x for x in prog["edges"] if x["room_a"] in idx and x["room_b"] in idx]
    conn = {frozenset((idx[x["room_a"]], idx[x["room_b"]])) for x in pe if x["kind"] in ("door", "open", "stair")}
    walls = [frozenset((idx[x["room_a"]], idx[x["room_b"]])) for x in pe if x["kind"] == "wall"]
    got = {frozenset(x) for x in H.edges}
    e["program_door_recall"] = len(got & conn) / len(conn) if conn else np.nan
    e["program_door_precision"] = len(got & conn) / len(got) if got else np.nan
    e["program_wall_adjacent"] = np.mean([w in adj_pairs for w in walls]) if walls else np.nan
    P = nx.Graph()
    P.add_nodes_from((n, {"t": H.nodes[n]["t"]}) for n in H)
    P.add_edges_from(tuple(x) for x in conn)
    ged = nx.graph_edit_distance(P, H, node_match=lambda a, b: a["t"] == b["t"], timeout=20)
    e["compatibility_ged"] = float(ged) if ged is not None else np.nan

    # Reachability over the whole home (GRE-Diff), through doors, openings and stairs.
    n = H.number_of_nodes()
    if front is not None:
        reach = nx.node_connected_component(H, front)
        e["all_reachable"], e["frac_reachable"] = float(len(reach) == n), len(reach) / n
        nobed = H.subgraph([x for x in H if H.nodes[x]["t"] != "bedroom" or x == front])
        ok = nx.node_connected_component(nobed, front)
        e["n_habitable_via_bedroom"] = sum(1 for x in H if H.nodes[x]["t"] in HABITABLE and x not in ok
                                           and not any(nb in ok for nb in H[x]))
    else:
        e["all_reachable"] = e["frac_reachable"] = e["n_habitable_via_bedroom"] = np.nan

    # Window area / floor area per habitable room (needs sill and head: from the State for ours, otherwise from
    # windows that carry width_m, sill_m, head_m and the index of their room, e.g. scripts/infinigen/convert.py).
    win = Counter()
    if per[0][4]["source"] == "ours" and cond.get("pipeline") != "mason2":  # mason_old: windows from its State
        st = json.load(open(ROOT / "data" / "ours" / "state" / home / "roomplan" / "openings.json"))
        room_of = {r["id"]: r.get("program_room") for r in st["rooms"]}
        for line in open(ROOT / "data" / "ours" / "state" / home / "replay.jsonl"):  # restored rooms' windows
            d = json.loads(line)
            if d.get("verdict") == "rejected" and d["command"]["op"] == "AddRoom":
                room_of[d["command"]["value"]["id"]] = d["command"]["value"].get("program_room")
            if d.get("verdict") == "rejected" and d["command"]["op"] == "AddOpening" and d["command"]["value"]["kind"] == "window":
                o = d["command"]["value"]
                win[room_of.get(o["room_a"])] += o["width"] * ((o.get("head_m") or 0) - (o.get("sill_m") or 0))
        for o in st["openings"]:
            if o["kind"] == "window":
                win[room_of.get(o["room_a"])] += o["width"] * ((o.get("head_m") or 0) - (o.get("sill_m") or 0))
    else:
        for *_, s in per:
            for w in s.get("windows") or []:
                if w.get("room") is not None and "head_m" in w:
                    win[s["rooms"][w["room"]].get("program_room")] += w["width_m"] * (w["head_m"] - w["sill_m"])
    # A method that places no windows at all (ProcTHOR's layout stage) is not measured, rather than scored 0.
    no_windows = per[0][4]["source"] != "ours" and not any(s.get("windows") for *_, s in per)
    ratios = [] if no_windows else [win.get(r["program_room"], 0.0) / Polygon(r["polygon"]).area
                                    for r in rooms if _t(r["type"]) in HABITABLE]
    e["min_window_floor_ratio"] = min(ratios) if ratios else np.nan
    e["frac_habitable_window_ge_1_8"] = np.mean([x >= 1 / 8 for x in ratios]) if ratios else np.nan
    e["frac_habitable_window_ge_1_10"] = np.mean([x >= 1 / 10 for x in ratios]) if ratios else np.nan
    return [p[0] for p in per], [x for p in per for x in p[1]], e


# ---------- distribution comparison -----------------------------------------------------------------------

def w1(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return np.nan
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


# Geometry only: door features are left out because the real datasets draw no door for open connections.
def c2st_features():
    return ["n_rooms", "total_area_m2", "mean_room_aspect", "frac_rect_rooms", "mean_room_convexity",
            "outline_rectangularity", "outline_corners", "mean_adjacency_degree",
            *[f"n_{t}" for t in CFG["types"]], *[f"share_{t}" for t in CFG["types"]]]


def c2st_1nn(A, B, seed=0):
    """Leave-one-out 1-NN accuracy telling A from B on standardised features (equal-size draw)."""
    rng = np.random.default_rng(seed)
    m = min(len(A), len(B))
    A, B = A[rng.choice(len(A), m, replace=False)], B[rng.choice(len(B), m, replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(m), np.ones(m)]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # Nearest other point via a k-d tree: O(n) memory. (A dense n x n x features difference array needs ~280 GB for
    # the 2 x 20k matched real plans of 1,000 generated homes.) Exact duplicates count as each other's neighbour.
    from scipy.spatial import cKDTree
    _, idx = cKDTree(X).query(X, k=2)
    nn = np.where(idx[:, 0] == np.arange(len(X)), idx[:, 1], idx[:, 0])
    return float((y[nn] == y).mean())


def feat_matrix(df):
    df = df.copy()
    tot = df[[f"area_{t}" for t in CFG["types"]]].sum(axis=1).replace(0, np.nan)
    for t in CFG["types"]:
        df[f"share_{t}"] = df[f"area_{t}"] / tot
    return df[c2st_features()].fillna(0).to_numpy(float)


def matched_reference(name, ours, workers):
    """Real plans with the same composition (CFG["match"] counts; balcony as yes/no) as each generated plan."""
    d, *_ = REFERENCES[name]
    from fpeval.evaluate import ref_dir
    cols = CFG["match"]
    ps = pd.read_csv(ref_dir(name) / "per_sample.csv", usecols=["id", *cols], dtype={"id": str})
    key = lambda df: list(zip(*[(df[c].clip(upper=1) if c == "n_balcony" else df[c]).astype(int) for c in cols]))
    ps["key"] = key(ps)
    rng = np.random.default_rng(0)
    pools = ps.groupby("key").id.apply(list).to_dict()
    pick, group, missing = [], [], 0
    mine = ours.rename(columns={"n_hallway": "_h"})
    if "n_balcony" in cols and "n_balcony" not in mine:
        mine["n_balcony"] = 0
    for g, k in enumerate(key(mine)):
        pool = pools.get(tuple(int(x) for x in k))
        if not pool:
            missing += 1
            continue
        got = list(rng.choice(pool, min(N_MATCH, len(pool)), replace=False))
        group += [g] * len(got)
        pick += got
    files = [ROOT / d / "samples" / f"{i}.json" for i in pick]
    with Pool(workers) as pool:
        res = pool.map(plan_features_light, files, chunksize=8)
    RF = pd.DataFrame([r for r, _ in res])
    RF["file"], RF["group"] = [str(f) for f in files], group
    return RF, pd.DataFrame([x for _, rs in res for x in rs]), missing


# ---------- FID / KID on rendered layouts ----------------------------------------------------------------

N_FID_ROUNDS = 5


def fid_block(ours_feats, RF, ref_feats, other=None):
    """FID/KID of ours vs the matched real plans, and the same for held-out matched real plans (the floor).

    Each round holds out one real plan per generated plan (same composition), so the held-out set has about as
    many plans as ours, and scores both against the rest. other: (RF, feats) of the other real dataset, whose
    held-out plans give a cross-dataset value on the same footing.
    """
    from fpeval.distribution import fid, kid
    out = {k: [] for k in ["fid", "kid_x1000", "fid_real_vs_real", "kid_real_vs_real_x1000",
                           "fid_other_real", "kid_other_real_x1000"]}
    for s in range(N_FID_ROUNDS):
        held = RF.groupby("group").sample(1, random_state=s).index.to_numpy()
        rest = np.setdiff1d(np.arange(len(RF)), held)
        ref = ref_feats[rest]
        out["fid"].append(fid(ours_feats, ref))
        out["kid_x1000"].append(1000 * kid(ours_feats, ref, seed=s)[0])
        out["fid_real_vs_real"].append(fid(ref_feats[held], ref))
        out["kid_real_vs_real_x1000"].append(1000 * kid(ref_feats[held], ref, seed=s)[0])
        if other is not None:
            oh = other[0].groupby("group").sample(1, random_state=s).index.to_numpy()
            out["fid_other_real"].append(fid(other[1][oh], ref))
            out["kid_other_real_x1000"].append(1000 * kid(other[1][oh], ref, seed=s)[0])
    res = {}
    for k, v in out.items():
        if v:
            res[k], res[k + "_std"] = float(np.mean(v)), float(np.std(v))
    res["n_ours"], res["n_ref"], res["n_held_out"] = len(ours_feats), len(rest), len(held)
    return res


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="apartment", choices=list(KINDS))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    CFG = KINDS[a.kind]
    TYPES = CFG["types"]
    variant = a.variant or f"ours/mason_old_{a.kind}_n100_seed0"
    vdir = ROOT / "outputs" / variant
    homes = {}
    for p in sorted((vdir / "samples").glob("*.json")):
        homes.setdefault(p.stem.split("_s")[0], []).append(p)
    with Pool(a.workers) as pool:
        res = pool.map(home_extra, list(homes.values()))
    F = pd.DataFrame([x for r in res for x in r[0]])
    R = pd.DataFrame([x for r in res for x in r[1]])
    E = pd.DataFrame([r[2] for r in res])
    (vdir / "eval").mkdir(exist_ok=True)
    F.to_csv(vdir / "eval" / "ours_plan_features.csv", index=False)
    E.to_csv(vdir / "eval" / "ours_home_metrics.csv", index=False)

    out = {"variant": variant, "kind": a.kind, "n_homes": len(E), "n_plans": len(F)}
    def block(df, cols):
        return {k: float(df[k].mean()) for k in cols}
    groups = {"all": E}
    if E.n_storeys.nunique() > 1:
        groups.update({f"{k} storeys": E[E.n_storeys == k] for k in sorted(E.n_storeys.unique())})
    out["homes"] = {}
    for g, df in groups.items():
        b = block(df, ["room_counts_match", "room_count_abs_err", "net_area_rel_err", "net_area_incl_stairs_rel_err",
                       "program_door_recall", "program_door_precision", "program_wall_adjacent", "compatibility_ged",
                       "all_reachable", "frac_reachable", "n_habitable_via_bedroom",
                       "frac_habitable_window_ge_1_8", "frac_habitable_window_ge_1_10"])
        b.update(n=len(df), net_area_rel_err_max=float(df.net_area_rel_err.max()),
                 frac_ged_zero=float((df.compatibility_ged == 0).mean()),
                 plans_all_habitable_ge_1_8=float((df.min_window_floor_ratio >= 1 / 8).mean()),
                 stair_area_m2=float(df.stair_area_m2.mean()))
        out["homes"][g] = b

    plan_cols = ["outline_rectangularity", "outline_corners", "frac_rect_rooms", "mean_room_aspect",
                 "mean_adjacency_degree", "frac_habitable_with_window", "n_rooms", "total_area_m2",
                 *[f"n_{t}" for t in TYPES]]
    out["plans"] = {"ours": block(F, plan_cols)}
    out["areas"], out["realism"], out["diversity"] = {}, {}, {}
    refs = {}
    feats = c2st_features()
    for name, label in CFG["real"].items():
        RF, RR, missing = matched_reference(name, F, a.workers)
        refs[label] = (RF, RR)
        p = block(RF, plan_cols)
        p["n_matched_plans"], p["unmatched_generated"] = len(RF), missing
        out["plans"][label] = p
        A, B = feat_matrix(F), feat_matrix(RF)
        out["realism"][label] = {"c2st_1nn_acc": float(np.mean([c2st_1nn(A, B, s) for s in range(10)]))}
        # Which features separate ours from real most: standardised mean difference (Cohen's d).
        sd = np.sqrt((A.var(0) + B.var(0)) / 2) + 1e-9
        dd = (A.mean(0) - B.mean(0)) / sd
        out["realism"][label]["largest_differences"] = {
            feats[i]: {"d": float(dd[i]), "ours": float(A[:, i].mean()), "real": float(B[:, i].mean())}
            for i in np.argsort(-np.abs(dd))[:8]}
        # Real-vs-real floor: two disjoint matched halves of the same dataset.
        h = len(RF) // 2
        perm = np.random.default_rng(1).permutation(len(RF))
        out["realism"][label]["c2st_real_vs_real"] = float(np.mean(
            [c2st_1nn(B[perm[:h]], B[perm[h:]], s) for s in range(10)]))
        # Diversity: distinct topologies among our plans vs equally many matched real plans.
        draws = [RF.topology.sample(len(F), random_state=s).nunique() / len(F) for s in range(20)]
        out["diversity"][label] = float(np.mean(draws))
        for t in TYPES:
            a_, b_ = R[R.type == t], RR[RR.type == t]
            if not len(a_):
                continue
            lo, hi = np.percentile(b_.area_m2, [5, 95]) if len(b_) else (np.nan, np.nan)
            out["areas"].setdefault(t, {"ours_median": float(a_.area_m2.median()), "ours_n": len(a_)})
            out["areas"][t][label] = {
                "median": float(b_.area_m2.median()) if len(b_) else None,
                "p5": float(lo), "p95": float(hi),
                "ours_within_p5_p95": float(a_.area_m2.between(lo, hi).mean()) if len(b_) else None,
                "w1_area_m2": w1(a_.area_m2, b_.area_m2),
                "w1_aspect": w1(a_.aspect.clip(upper=10), b_.aspect.clip(upper=10))}
    out["diversity"]["ours"] = float(F.topology.nunique() / len(F))
    (l1, (RF1, _)), (l2, (RF2, _)) = refs.items()
    out["realism"][f"{l2} vs {l1}"] = {"c2st_1nn_acc": float(np.mean(
        [c2st_1nn(feat_matrix(RF2), feat_matrix(RF1), s) for s in range(10)]))}

    # FID / KID. Inception features are cached per variant in its eval/ directory.
    from fpeval.distribution import layout_features
    cache = vdir / "eval" / "fid_features.npz"
    ours_feats = layout_features([p for ps in homes.values() for p in ps], cache, workers=a.workers)
    feats = {lab: layout_features(RF.file.tolist(), cache, workers=a.workers) for lab, (RF, _) in refs.items()}
    out["fid"] = {}
    for lab, (RF, _) in refs.items():
        o = next(x for x in refs if x != lab)
        out["fid"][lab] = fid_block(ours_feats, RF, feats[lab], other=(refs[o][0], feats[o]))
        out["fid"][lab]["other_real"] = o

    res_dir = ROOT / "results" / variant.split("/")[0]
    res_dir.mkdir(parents=True, exist_ok=True)
    stem = variant.split("/")[-1]
    json.dump(out, open(res_dir / f"{stem}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
