"""Build the shareable results page (HTML) from results/ and sample renders.

    PYTHONPATH=. python scripts/report/build_results_page.py <out.html>
"""
import base64
import io
import json
import sys
from pathlib import Path

import pandas as pd

from fpeval import rules
from fpeval.eval_config import STAT_DATASETS
from fpeval.format import ROOM_TYPES, load_sample
from fpeval.render import PALETTE, render

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("results_page_template.html")

VARIANT_NAMES = {
    "housegan/lifull_bubble_testD": "House-GAN · LIFULL, held-out 10–12 rooms",
    "houseganpp/rplan-g2p_bubble_test1000_syndoors": "House-GAN++ · bubble",
    "houseganpp/rplan_bubble_public": "House-GAN++ · bubble, 7 bundled graphs",
    "house_diffusion/rplan-g2p_bubble_test1000_syndoors": "HouseDiffusion · bubble",
    "house_diffusion/rplan_bubble_public": "HouseDiffusion · bubble, 7 bundled graphs",
    "gsdiff/rplan_uncond": "GSDiff · unconditional",
    "gsdiff/rplan_bubble_test1000": "GSDiff · bubble",
    "gsdiff/rplan_boundary_test1000": "GSDiff · boundary",
    "iplan/rplan_boundary_test1000": "iPLAN · boundary",
    "diffplanner/rplan_boundary_test": "DiffPlanner · boundary (all 12k)",
    "wallplan/rplan_boundary_test1000": "WallPlan · boundary",
    "wallplan/bundled_boundary_test500": "WallPlan · boundary, bundled samples",
    "maskplan/rplan_boundary": "MaskPLAN · boundary",
    "maskplan/rplan_partial25": "MaskPLAN · boundary + 25% hints",
    "chathousediffusion/t2d_test_textgraph": "ChatHouseDiffusion · text + boundary",
    **{f"ds2d/rplan{n}R_bubble_roomarea_test": f"DS2D · {n}-room model" for n in (5, 6, 7, 8)},
    **{f"ds2d/procthor_{c}_test_lora-{l}": f"DS2D · {'bubble + ' if c.startswith('bubble') else ''}constraints, {l}"
       for c in ("bubble_constraints", "constraints") for l in ("fullprompt", "mask", "presetmask")},
}
REF_LABELS = {"rplan_graph2plan:test": "RPLAN", "procthor10k:test": "ProcTHOR-10K",
              "tell2design:test": "Tell2Design", "housegan_lifull:all": "LIFULL"}

# (key, header, decimals, lower_is_better or None)
RULE_COLS = [("n_samples", "Plans", 0, None), ("n_rooms", "Rooms", 1, None),
             ("overlap_ratio", "Overlap", 3, True), ("hole_ratio", "Holes", 3, True),
             ("boundary_coverage", "Boundary covered", 3, False), ("adj_components", "Pieces", 2, True),
             ("frac_area_violation", "Rooms below min. area", 3, True),
             ("frac_width_violation", "Rooms below min. width", 3, True),
             ("frac_aspect_violation", "Rooms above max. aspect", 3, True),
             ("any_rule_violation", "Plans breaking a rule", 2, True)]
ADHERENCE_COLS = [("yield", "Yield", 2, False), ("room_count_match", "Room count right", 2, False),
                  ("type_multiset_match", "Room types right", 2, False),
                  ("adjacency_f1", "Adjacency F1", 2, False), ("type_iou", "Type IoU", 2, False)]
GROUPS = [
    ("rplan_graph2plan:test", "RPLAN inputs", "Inputs from the RPLAN test split. Baseline: 12,063 real RPLAN plans."),
    ("procthor10k:test", "ProcTHOR inputs", "Inputs from ProcTHOR-10K (synthetic houses). Baseline: its test split."),
    ("tell2design:test", "Tell2Design inputs", "Text descriptions from the Tell2Design test split. Baseline: those 2,308 plans."),
    ("housegan_lifull:all", "LIFULL inputs", "Bubble diagrams from LIFULL. Scale unknown, so area and width rules are skipped."),
]
# Room types shown in the dataset comparison (stats types: entrance + corridor = hallway).
STAT_SHOW = ["living_room", "kitchen", "dining_room", "bedroom", "bathroom", "hallway", "balcony",
             "storage", "closet", "study", "laundry", "other"]
SIZE_TYPES = ["living_room", "kitchen", "bedroom", "bathroom", "balcony", "hallway", "storage"]

GALLERY_COLS = [
    ("Real (GT)", None), ("iPLAN", "iplan/rplan_boundary_test1000"),
    ("DiffPlanner", "diffplanner/rplan_boundary_test"), ("WallPlan", "wallplan/rplan_boundary_test1000"),
    ("GSDiff (boundary)", "gsdiff/rplan_boundary_test1000"), ("MaskPLAN", "maskplan/rplan_boundary"),
    ("House-GAN++", "houseganpp/rplan-g2p_bubble_test1000_syndoors"),
    ("HouseDiffusion", "house_diffusion/rplan-g2p_bubble_test1000_syndoors"),
    ("GSDiff (bubble)", "gsdiff/rplan_bubble_test1000"),
]
GALLERY_IDS = ["31098", "79807", "10960"]


def img_uri(sample, size=200):
    buf = io.BytesIO()
    render(sample, size=size, margin=8).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def find_sample(variant, sid, sub="samples"):
    d = ROOT / "outputs" / variant / sub
    for name in (f"{sid}.json", f"{sid}_0.json"):
        if (d / name).exists():
            return load_sample(d / name)
    return None


def _num(v, dec):
    return None if v is None or pd.isna(v) else round(float(v), dec)


def table_rows(df, cols, names):
    return [{"name": names.get(k, k), "cond": r.get("condition", ""),
             "cells": [_num(r.get(c), d) for c, _, d, _ in cols]} for k, r in df.iterrows()]


def dataset_stats():
    out = []
    for name, meta in STAT_DATASETS.items():
        slug = meta["label"].lower().replace(" ", "_").replace("-", "")
        js = json.load(open(ROOT / "results" / "datasets" / f"{slug}.json"))
        comp = {r["type"]: r for r in js["composition"]}
        area = {r["type"]: r for r in js["area_percentiles"]}
        viol = {r["type"]: r for r in js["violations"]}
        conn = {}
        for r in js["connections"]:
            if r["type_a"] in STAT_SHOW and r["type_b"] in STAT_SHOW:
                conn.setdefault(r["kind"], {})[f"{r['type_a']}|{r['type_b']}"] = [
                    _num(r["p_room"], 3), _num(r["mean_edges"], 3)]
        out.append({
            "key": name, "label": meta["label"], "kind": meta["kind"], "unit": meta["unit"],
            "n": js["info"]["n_samples"], "nRooms": js["info"]["n_rooms"],
            "meanRooms": _num(js["info"]["mean_rooms"], 1),
            "meanArea": _num(js["info"]["mean_total_area_m2"], 0),
            "kinds": js["info"]["connection_kinds"],
            "count": {t: _num(comp.get(t, {}).get("mean_count", 0), 2) for t in STAT_SHOW},
            "roomsOfType": {t: int(area[t]["n_rooms"]) if t in area else 0 for t in STAT_SHOW},
            "area": {t: [_num(area[t].get(f"area_m2_p{p}"), 1) for p in (5, 50, 95)] for t in SIZE_TYPES if t in area},
            "width": {t: [_num(area[t].get(f"width_m_p{p}"), 2) for p in (5, 50, 95)] for t in SIZE_TYPES if t in area},
            "viol": {t: [_num(viol[t].get(k), 3) for k in ("below_min_area", "below_min_width", "above_max_aspect")]
                     for t in STAT_SHOW if t in viol},
            "conn": conn,
            "graph": js["average_graph"],
        })
    return out


def main(out):
    var = pd.read_csv(ROOT / "results/summary_variants.csv", index_col=0)
    refs = pd.read_csv(ROOT / "results/summary_references.csv", index_col=0)

    groups = []
    for ref, short, blurb in GROUPS:
        sub = var[var.ref == ref]
        base = table_rows(refs.loc[[ref]], RULE_COLS, REF_LABELS)[0]
        groups.append({"short": short, "blurb": blurb, "baseline": base,
                       "rules": table_rows(sub, RULE_COLS, VARIANT_NAMES),
                       "adherence": table_rows(sub, ADHERENCE_COLS, VARIANT_NAMES)})

    skip = {"houseganpp/rplan_bubble_public", "house_diffusion/rplan_bubble_public", "wallplan/bundled_boundary_test500"}
    shared = [k for k in var.index if var.loc[k, "ref"] == "rplan_graph2plan:test" and k not in skip]
    chart_rules = sorted(({"name": VARIANT_NAMES[k], "value": float(var.loc[k, "any_rule_violation"])}
                          for k in shared), key=lambda d: d["value"])

    gallery = []
    for sid in GALLERY_IDS:
        cells = []
        for label, variant in GALLERY_COLS:
            s = (find_sample("iplan/rplan_boundary_test1000", sid, "gt_samples") if variant is None
                 else find_sample(variant, sid))
            cells.append({"label": label, "img": img_uri(s) if s else None})
        gallery.append({"id": sid, "cells": cells})

    palette = {t: "#%02x%02x%02x" % PALETTE[t] for t in ROOM_TYPES}
    palette["hallway"] = palette["corridor"]
    stats = dataset_stats()
    data = {
        "ruleCols": [h for _, h, _, _ in RULE_COLS], "ruleBetter": [b for *_, b in RULE_COLS],
        "ruleDec": [d for _, _, d, _ in RULE_COLS],
        "adhCols": [h for _, h, _, _ in ADHERENCE_COLS], "adhBetter": [b for *_, b in ADHERENCE_COLS],
        "adhDec": [d for _, _, d, _ in ADHERENCE_COLS],
        "groups": groups, "chartRules": chart_rules,
        "rplanBaseline": float(refs.loc["rplan_graph2plan:test", "any_rule_violation"]),
        "gallery": gallery, "palette": palette,
        "thresholds": [{"type": t, "area": rules.MIN_AREA_M2[t], "width": rules.MIN_WIDTH_M[t],
                        "aspect": rules.MAX_ASPECT[t]} for t in ROOM_TYPES],
        "wallGap": 2 * rules.WALL_HALF_THICKNESS_M, "minShared": rules.MIN_SHARED_WALL_M,
        "stats": stats, "statTypes": STAT_SHOW, "sizeTypes": SIZE_TYPES,
        "nVariants": len(var), "nPlans": int(var.n_samples.sum()),
        "nMethods": len({k.split("/")[0] for k in var.index}),
        "nRealPlans": sum(s["n"] for s in stats if s["kind"] == "real"),
        "nDatasets": sum(1 for s in stats if s["kind"] == "real"),
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    Path(out).write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
