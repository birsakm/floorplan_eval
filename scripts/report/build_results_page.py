"""Build the shareable results page (HTML) from results/*.csv and sample renders.

    PYTHONPATH=. python scripts/report/build_results_page.py <out.html>
"""
import base64
import io
import json
import sys
from pathlib import Path

import pandas as pd

from fpeval.format import ROOM_TYPES, load_sample
from fpeval.render import PALETTE, render

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("results_page_template.html")

METHOD_NAMES = {
    "housegan": "House-GAN", "houseganpp": "House-GAN++", "house_diffusion": "HouseDiffusion",
    "gsdiff": "GSDiff", "iplan": "iPLAN", "diffplanner": "DiffPlanner", "wallplan": "WallPlan",
    "maskplan": "MaskPLAN", "chathousediffusion": "ChatHouseDiffusion", "ds2d": "DS2D",
}
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
REF_NAMES = {
    "rplan_graph2plan:test": "RPLAN (test)", "tell2design:test": "Tell2Design (test)",
    "housegan_lifull:all": "LIFULL (House-GAN)", "procthor10k:test": "ProcTHOR-10K (test, synthetic)",
    "cubicasa5k:test": "CubiCasa5k (test)", "resplan:test": "ResPlan (test)",
    "swiss_dwellings:all": "Swiss Dwellings", "msd:train": "MSD (building floors)",
    "magicplan:train": "MagicPlan",
}

# Columns shown in the tables: (key, header, decimals, lower_is_better or None).
RULE_COLS = [("n_samples", "Plans", 0, None), ("n_rooms", "Rooms", 1, None),
             ("overlap_ratio", "Overlap", 3, True), ("hole_ratio", "Holes", 3, True),
             ("boundary_coverage", "Boundary covered", 3, False), ("adj_components", "Pieces", 2, True),
             ("frac_nonrectilinear_rooms", "Non-rect. rooms", 3, True),
             ("frac_area_violation", "Too small", 3, True), ("frac_width_violation", "Too narrow", 3, True),
             ("frac_aspect_violation", "Too elongated", 3, True), ("any_rule_violation", "Plans breaking a rule", 2, True)]
PAIRED_COLS = [("yield", "Yield", 2, False), ("room_count_match", "Room count right", 2, False),
               ("type_multiset_match", "Room types right", 2, False), ("adjacency_f1", "Adjacency F1", 2, False),
               ("type_miou", "Type mIoU", 2, False)]
DIST_COLS = [("fid", "FID vs dataset", 1, True), ("kid_x1000", "KID ×10³", 1, True),
             ("fid_vs_gt", "FID vs own GT", 1, True), ("fid_gt_vs_ref", "Real-data floor", 1, None),
             ("dist_room_count_tv", "Room-count TV", 2, True), ("dist_room_type_jsd", "Room-type JSD", 3, True),
             ("dist_room_area_m2_w1", "Room area W1 (m²)", 1, True)]

GROUPS = [
    ("rplan_graph2plan:test", "RPLAN", "Compared with the RPLAN test split (12,063 real plans)."),
    ("procthor10k:test", "ProcTHOR", "Compared with the ProcTHOR-10K test split (synthetic houses, in meters)."),
    ("tell2design:test", "Tell2Design", "Compared with the Tell2Design test split (2,308 plans described in text)."),
    ("housegan_lifull:all", "LIFULL", "Compared with 10,000 LIFULL plans. Scale unknown, so rules in meters are skipped."),
]

GALLERY_COLS = [
    ("GT", None), ("iPLAN", "iplan/rplan_boundary_test1000"),
    ("DiffPlanner", "diffplanner/rplan_boundary_test"), ("WallPlan", "wallplan/rplan_boundary_test1000"),
    ("GSDiff (boundary)", "gsdiff/rplan_boundary_test1000"), ("MaskPLAN", "maskplan/rplan_boundary"),
    ("House-GAN++", "houseganpp/rplan-g2p_bubble_test1000_syndoors"),
    ("HouseDiffusion", "house_diffusion/rplan-g2p_bubble_test1000_syndoors"),
    ("GSDiff (bubble)", "gsdiff/rplan_bubble_test1000"),
]
GALLERY_IDS = ["31098", "79807", "10960"]
PAIR_ROWS = [  # (label, variant, sample id or None for first)
    ("House-GAN · LIFULL", "housegan/lifull_bubble_testD", "00010"),
    ("ChatHouseDiffusion · Tell2Design", "chathousediffusion/t2d_test_textgraph", "10344"),
    ("DS2D · RPLAN 6-room", "ds2d/rplan6R_bubble_roomarea_test", "00146"),
    ("DS2D · ProcTHOR", "ds2d/procthor_bubble_constraints_test_lora-fullprompt", "0010"),
]


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


def table_rows(df, cols):
    rows = []
    for key, r in df.iterrows():
        cells = []
        for c, _, dec, _ in cols:
            v = r.get(c)
            cells.append(None if v is None or pd.isna(v) else round(float(v), dec))
        rows.append({"key": key, "name": VARIANT_NAMES.get(key, REF_NAMES.get(key, key)),
                     "cond": r.get("condition", ""), "cells": cells})
    return rows


def main(out):
    var = pd.read_csv(ROOT / "results/summary_variants.csv", index_col=0)
    refs = pd.read_csv(ROOT / "results/summary_references.csv", index_col=0)
    var.loc[var.raster.fillna(False).astype(bool), "frac_nonrectilinear_rooms"] = None
    var.loc[var.n_gt_inputs < 50, ["fid_vs_gt", "kid_vs_gt_x1000", "fid_gt_vs_ref"]] = None

    groups = []
    for ref, short, blurb in GROUPS:
        sub = var[var.ref == ref]
        groups.append({"ref": ref, "short": short, "blurb": blurb,
                       "rules": table_rows(sub, RULE_COLS), "paired": table_rows(sub, PAIRED_COLS),
                       "dist": table_rows(sub, DIST_COLS),
                       "baseline": table_rows(refs.loc[[ref]], RULE_COLS)[0]})

    shared = [k for k in var.index if var.loc[k, "ref"] == "rplan_graph2plan:test"
              and k not in ("houseganpp/rplan_bubble_public", "house_diffusion/rplan_bubble_public",
                            "wallplan/bundled_boundary_test500")]
    chart_rules = sorted(({"name": VARIANT_NAMES[k], "value": float(var.loc[k, "any_rule_violation"])}
                          for k in shared), key=lambda d: d["value"])
    chart_fid = sorted(({"name": VARIANT_NAMES[k], "value": float(var.loc[k, "fid_vs_gt"])}
                        for k in shared if pd.notna(var.loc[k, "fid_vs_gt"])), key=lambda d: d["value"])

    gallery = []
    for sid in GALLERY_IDS:
        cells = []
        for label, variant in GALLERY_COLS:
            s = (find_sample("iplan/rplan_boundary_test1000", sid, "gt_samples") if variant is None
                 else find_sample(variant, sid))
            cells.append({"label": label, "img": img_uri(s) if s else None})
        gallery.append({"id": sid, "cells": cells})
    pairs = []
    for label, variant, sid in PAIR_ROWS:
        pairs.append({"label": label, "gt": img_uri(find_sample(variant, sid, "gt_samples")),
                      "gen": img_uri(find_sample(variant, sid))})

    data = {
        "ruleCols": [h for _, h, _, _ in RULE_COLS], "ruleBetter": [b for *_, b in RULE_COLS],
        "ruleDec": [d for _, _, d, _ in RULE_COLS],
        "pairedCols": [h for _, h, _, _ in PAIRED_COLS], "pairedBetter": [b for *_, b in PAIRED_COLS],
        "pairedDec": [d for _, _, d, _ in PAIRED_COLS],
        "distCols": [h for _, h, _, _ in DIST_COLS], "distBetter": [b for *_, b in DIST_COLS],
        "distDec": [d for _, _, d, _ in DIST_COLS],
        "groups": groups, "refs": table_rows(refs, RULE_COLS),
        "chartRules": chart_rules, "chartFid": chart_fid,
        "rplanBaseline": float(refs.loc["rplan_graph2plan:test", "any_rule_violation"]),
        "gallery": gallery, "pairs": pairs,
        "palette": {t: "#%02x%02x%02x" % PALETTE[t] for t in ROOM_TYPES},
        "nVariants": len(var), "nPlans": int(var.n_samples.sum()), "nRefs": len(refs),
        "nMethods": len({k.split("/")[0] for k in var.index}),
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data))
    Path(out).write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
