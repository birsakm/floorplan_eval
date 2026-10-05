"""Statistics of real floorplan datasets (inputs: the reference eval CSVs).

Writes results/datasets/:
  composition.csv          presence and mean count per room type
  area_percentiles.csv     p5 / p50 / p95 of room area and clear width per type (m)
  violations.csv           share of rooms per type breaking each per-room rule
  connections.csv          room connection statistics per type pair (wall / door)
  <dataset>.json           everything above for one dataset, plus the average graph

Connection statistics (one row per dataset, kind, type_a, type_b):
  p_room   P(a room of type_a has >= 1 neighbour of type_b). Rows sum to more than 1.
  p_plan   P(plan has an a-b connection | plan has both types); for a == b, >= 2 rooms.
  mean_edges  mean number of a-b connections per plan (the average graph's edge weight).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import rules
from .eval_config import REFERENCES, STAT_DATASETS
from .format import ROOM_TYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "datasets"
PCTS = (5, 50, 95)

# Datasets label hallway-like space differently (CubiCasa "Entry Lobby" -> entrance,
# MagicPlan "Hall" and Swiss Dwellings "CORRIDOR" -> corridor), so the statistics
# merge both into one "hallway" type.
TYPE_GROUPS = {"entrance": "hallway", "corridor": "hallway"}
STAT_TYPES = list(dict.fromkeys(TYPE_GROUPS.get(t, t) for t in ROOM_TYPES))


def _load(name):
    from .evaluate import ref_dir
    d = ref_dir(name)
    rooms = pd.read_csv(d / "per_room.csv", dtype={"id": str})
    rooms["type_canonical"] = rooms.type
    rooms["type"] = rooms.type.replace(TYPE_GROUPS)
    return (pd.read_csv(d / "per_sample.csv", dtype={"id": str}), rooms,
            pd.read_csv(d / "per_edge.csv", dtype={"id": str}))


def composition(samples, rooms):
    n = len(samples)
    counts = rooms.groupby(["id", "type"]).size().unstack(fill_value=0).reindex(
        index=samples.id, columns=STAT_TYPES, fill_value=0)
    return pd.DataFrame({"presence": (counts > 0).sum() / n, "mean_count": counts.sum() / n})


def area_percentiles(rooms):
    rows = []
    for t, g in rooms.groupby("type"):
        rec = {"type": t, "n_rooms": len(g)}
        for col in ("area_m2", "width_m"):
            vals = g[col].dropna()
            for p in PCTS:
                rec[f"{col}_p{p}"] = float(np.percentile(vals, p)) if len(vals) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("type")


def violations(rooms, scale_known):
    rows = []
    # Thresholds are per canonical type (entrance and corridor differ), grouped for display.
    min_area = rooms.type_canonical.map(rules.MIN_AREA_M2)
    min_width = rooms.type_canonical.map(rules.MIN_WIDTH_M)
    max_aspect = rooms.type_canonical.map(lambda t: rules.MAX_ASPECT[t] if rules.MAX_ASPECT[t] else np.inf)
    flags = pd.DataFrame({"type": rooms.type, "area": rooms.area_m2 < min_area,
                          "width": rooms.width_m < min_width, "aspect": rooms.aspect > max_aspect,
                          "has_aspect": np.isfinite(max_aspect)})
    for t, g in flags.groupby("type"):
        rows.append({
            "type": t, "n_rooms": len(g),
            "below_min_area": float(g.area.mean()) if scale_known else np.nan,
            "below_min_width": float(g.width.mean()) if scale_known else np.nan,
            "above_max_aspect": float(g.aspect[g.has_aspect].mean()) if g.has_aspect.any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("type")


def connections(samples, rooms, edges, kind):
    e = edges[edges.kind == kind].drop_duplicates(["id", "i", "j"])
    types = rooms.set_index(["id", "idx"]).type
    e = e.assign(ta=types.reindex(list(zip(e.id, e.i))).values, tb=types.reindex(list(zip(e.id, e.j))).values)
    n_plans = len(samples)

    # Per-room: which neighbour types does each room have?
    nb = pd.concat([e[["id", "i", "tb"]].set_axis(["id", "idx", "nb"], axis=1),
                    e[["id", "j", "ta"]].set_axis(["id", "idx", "nb"], axis=1)]).drop_duplicates()
    nb = nb.merge(rooms[["id", "idx", "type"]], on=["id", "idx"])
    room_counts = rooms.type.value_counts()
    with_nb = nb.groupby(["type", "nb"]).size()
    p_room = with_nb / room_counts.reindex(with_nb.index.get_level_values(0)).values

    # Per-plan: is there an a-b connection, given both types are present?
    counts = rooms.groupby(["id", "type"]).size().unstack(fill_value=0)
    pair = e.assign(a=np.minimum(e.ta, e.tb), b=np.maximum(e.ta, e.tb))
    plan_pairs = pair.drop_duplicates(["id", "a", "b"]).groupby(["a", "b"]).size()
    mean_edges = pair.groupby(["a", "b"]).size() / n_plans

    rows = []
    present = [t for t in STAT_TYPES if t in counts]
    for ia, a in enumerate(present):
        for b in present[ia:]:
            both = ((counts[a] >= 2) if a == b else ((counts[a] > 0) & (counts[b] > 0))).sum()
            key = tuple(sorted((a, b)))  # pair keys are stored alphabetically (min, max)
            k = plan_pairs.get(key, 0)
            for x, y in ((a, b), (b, a)) if a != b else ((a, a),):
                rows.append({"kind": kind, "type_a": x, "type_b": y,
                             "p_room": float(p_room.get((x, y), 0.0)),
                             "p_plan": float(k / both) if both else np.nan,
                             "n_plans_both": int(both),
                             "mean_edges": float(mean_edges.get(key, 0.0))})
    return pd.DataFrame(rows)


def average_graph(comp, conn, min_count=0.1, min_edges=0.1):
    nodes = [{"type": t, "mean_count": float(c)} for t, c in comp.mean_count.items() if c >= min_count]
    keep = {n["type"] for n in nodes}
    edges = [{"a": r.type_a, "b": r.type_b, "mean_edges": r.mean_edges}
             for r in conn.itertuples() if r.type_a <= r.type_b and r.type_a in keep and r.type_b in keep
             and r.mean_edges >= min_edges]
    return {"nodes": nodes, "edges": edges}


def write_stats():
    OUT.mkdir(parents=True, exist_ok=True)
    all_comp, all_area, all_viol, all_conn = [], [], [], []
    for name, meta in STAT_DATASETS.items():
        samples, rooms, edges = _load(name)
        scale_known = REFERENCES[name][3]
        comp = composition(samples, rooms)
        area = area_percentiles(rooms)
        if not scale_known:
            area.loc[:, [c for c in area if c != "n_rooms"]] = np.nan
        viol = violations(rooms, scale_known)
        has_doors = samples.get("n_interior_doors", pd.Series([0])).mean() >= 0.5
        kinds = ["wall", "door"] if has_doors else ["wall"]
        conn = pd.concat([connections(samples, rooms, edges, k) for k in kinds], ignore_index=True)
        info = {"dataset": name, **meta, "n_samples": len(samples), "n_rooms": len(rooms),
                "scale_known": scale_known, "connection_kinds": kinds,
                "mean_rooms": float(samples.n_rooms.mean()),
                "mean_total_area_m2": float(samples.total_area_m2.mean()) if scale_known else None}
        graphs = {k: average_graph(comp, conn[conn.kind == k]) for k in kinds}
        slug = meta["label"].lower().replace(" ", "_").replace("-", "")
        json.dump({"info": info, "composition": comp.reset_index(names="type").to_dict("records"),
                   "area_percentiles": area.reset_index().to_dict("records"),
                   "violations": viol.reset_index().to_dict("records"),
                   "connections": conn.to_dict("records"), "average_graph": graphs},
                  open(OUT / f"{slug}.json", "w"), indent=1, default=float)
        for df, acc in ((comp, all_comp), (area, all_area), (viol, all_viol)):
            acc.append(df.reset_index(names="type").assign(dataset=meta["label"]))
        all_conn.append(conn.assign(dataset=meta["label"]))
        print(f"[stats] {meta['label']}: {len(samples)} samples, {len(rooms)} rooms, kinds={kinds}")
    for acc, fname in ((all_comp, "composition"), (all_area, "area_percentiles"),
                       (all_viol, "violations"), (all_conn, "connections")):
        df = pd.concat(acc, ignore_index=True)
        cols = ["dataset"] + [c for c in df.columns if c != "dataset"]
        df[cols].to_csv(OUT / f"{fname}.csv", index=False)
    print(f"wrote {OUT}")
