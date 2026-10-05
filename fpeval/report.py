"""Collect eval summaries and dataset statistics into results/summary.{csv,md}."""
import json
from pathlib import Path

import pandas as pd

from . import rules
from .eval_config import REFERENCES, STAT_DATASETS, VARIANTS
from .format import ROOM_TYPES

ROOT = Path(__file__).resolve().parents[1]

# (column, header, format)
RULE_COLS = [
    ("n_samples", "N", "{:.0f}"), ("yield", "yield", "{:.2f}"), ("n_rooms", "rooms", "{:.1f}"),
    ("overlap_ratio", "overlap", "{:.3f}"), ("hole_ratio", "holes", "{:.3f}"),
    ("boundary_coverage", "bnd cov", "{:.3f}"), ("outside_boundary_ratio", "outside", "{:.3f}"),
    ("adj_components", "pieces", "{:.2f}"), ("frac_nonrectilinear_rooms", "non-rect", "{:.3f}"),
    ("frac_area_violation", "< min area", "{:.3f}"), ("frac_width_violation", "< min width", "{:.3f}"),
    ("frac_aspect_violation", "> max aspect", "{:.3f}"), ("any_rule_violation", "plans breaking a rule", "{:.2f}"),
]
ADHERENCE_COLS = [
    ("room_count_match", "room count right", "{:.2f}"), ("type_multiset_match", "room types right", "{:.2f}"),
    ("adjacency_f1", "adjacency F1", "{:.2f}"), ("type_iou", "type IoU", "{:.2f}"),
]
MAIN_TYPES = ["living_room", "kitchen", "bedroom", "bathroom", "balcony", "hallway", "storage"]  # stats types


def _fmt(v, f):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "–"
    return f.format(v)


def _table(df, cols, label_col):
    head = "| " + " | ".join([label_col] + [h for _, h, _ in cols]) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = ["| " + " | ".join([str(idx)] + [_fmt(r.get(c), f) for c, _, f in cols]) + " |"
            for idx, r in df.iterrows()]
    return "\n".join([head, sep, *rows])


def rules_table():
    lines = ["| room type | min area (m²) | min clear width (m) | max aspect ratio |", "|---|---|---|---|"]
    for t in ROOM_TYPES:
        mx = rules.MAX_ASPECT[t]
        lines.append(f"| {t} | {rules.MIN_AREA_M2[t]:g} | {rules.MIN_WIDTH_M[t]:g} | {'–' if mx is None else f'{mx:g}'} |")
    return "\n".join(lines)


def _dataset_sections():
    d = ROOT / "results" / "datasets"
    if not (d / "composition.csv").exists():
        return []
    comp = pd.read_csv(d / "composition.csv")
    area = pd.read_csv(d / "area_percentiles.csv")
    conn = pd.read_csv(d / "connections.csv")
    labels = [m["label"] for m in STAT_DATASETS.values()]
    md = ["## Dataset statistics (real data)", "",
          "Computed by `python -m fpeval.evaluate stats`. Full tables, including connection matrices "
          "that can be sampled from, are in `results/datasets/`. Entrance and corridor are merged into "
          "`hallway` here, because datasets label hallway-like space differently. CubiCasa labels about "
          "20% of its rooms as Undefined (`other`).", "",
          "### Mean rooms of each type per plan", ""]
    piv = comp.pivot(index="dataset", columns="type", values="mean_count").reindex(labels)[MAIN_TYPES + ["other"]]
    md += [_table(piv, [(t, t, "{:.2f}") for t in piv.columns], "dataset"), ""]
    md += ["### Room area in m² (p5 / median / p95)", ""]
    a = area.assign(v=area.apply(lambda r: None if pd.isna(r.area_m2_p50) else
                                 f"{r.area_m2_p5:.1f} / {r.area_m2_p50:.1f} / {r.area_m2_p95:.1f}", axis=1))
    piv = a.pivot(index="dataset", columns="type", values="v").reindex(labels)
    piv = piv[[t for t in MAIN_TYPES if t in piv]]
    md += [_table(piv, [(t, t, "{}") for t in piv.columns], "dataset"), ""]
    md += ["### Most frequent connections: P(a room of type A has a neighbour of type B)", ""]
    for kind, title in (("door", "Through a door"), ("wall", "Sharing a wall")):
        sub = conn[(conn.kind == kind) & (conn.type_a != "other") & (conn.type_b != "other")]
        if sub.empty:
            continue
        top = sub.groupby(["type_a", "type_b"]).p_room.mean().sort_values(ascending=False).head(10).index
        sub = sub[sub.set_index(["type_a", "type_b"]).index.isin(top)]
        piv = sub.assign(pair=sub.type_a + " → " + sub.type_b).pivot(index="dataset", columns="pair", values="p_room")
        piv = piv.reindex([l for l in labels if l in piv.index])
        md += [f"**{title}**", "", _table(piv, [(c, c, "{:.2f}") for c in piv.columns], "dataset"), ""]
    return md


def write_report():
    from .evaluate import ref_dir
    var = {k: json.load(open(p)) for k in VARIANTS
           if (p := ROOT / "outputs" / k / "eval" / "summary.json").exists()}
    refs = {n: json.load(open(p)) for n in REFERENCES if (p := ref_dir(n) / "summary.json").exists()}
    vdf = pd.DataFrame.from_dict(var, orient="index")
    if "raster" in vdf:
        # Rectilinearity of approxPolyDP-vectorized rasters measures the converter, not the model.
        vdf.loc[vdf.raster.fillna(False).astype(bool), "frac_nonrectilinear_rooms"] = None
    rdf = pd.DataFrame.from_dict(refs, orient="index")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    vdf.to_csv(out / "summary_variants.csv")
    rdf.to_csv(out / "summary_references.csv")

    md = ["# Evaluation results", "",
          "Generated by `python -m fpeval.evaluate report`. Metric definitions: docs/metrics.md.", "",
          "## Rules", "",
          "A plan breaks a rule when any of these hold: summed room area exceeds the footprint by more than 2% "
          "(overlap); enclosed holes wider than a wall cover more than 2% of the footprint; the room adjacency "
          "graph has more than one piece; or any room is below the minimum area or minimum clear width, or above "
          "the maximum aspect ratio, for its type. Clear width is the diameter of the largest circle that fits "
          "inside the room; aspect ratio is long / short side of the smallest rotated rectangle around it.", "",
          rules_table(), "",
          "## Real datasets: rule baselines", "", _table(rdf, RULE_COLS, "dataset"), ""]
    md += _dataset_sections()
    for ref in vdf["ref"].unique():
        sub = vdf[vdf.ref == ref].copy()
        sub.index = [f"{i} ({c})" for i, c in zip(sub.index, sub.condition)]
        md += [f"## Generated plans with inputs from {ref}", "",
               "### Rules", "", _table(sub, RULE_COLS, "variant (condition)"), "",
               "### Following the input (against the GT plan for the same input)", "",
               _table(sub, ADHERENCE_COLS, "variant (condition)"), ""]
    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out}/summary.md, summary_variants.csv, summary_references.csv")
