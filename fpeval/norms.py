"""Check floor plans against the building norms of one jurisdiction (floorplan-rules project).

    python -m fpeval.norms AT-9 rplan_graph2plan:test wallplan/rplan_boundary_test1000 ...
    python -m fpeval.norms AT-9 --all          # every reference set and variant with a metric scale

Targets are reference names (fpeval.eval_config.REFERENCES) or variant keys (VARIANTS). Writes
<dir>/eval/norms_<JUR>.csv (one row per plan) and norms_<JUR>.json (summary), and
results/norms/<JUR>.md across all checked targets.

A common-format sample is turned into a plan of the floorplan-rules checker (tools/check_plan.py):
rooms with area, width and depth (sides of the minimum rotated rectangle) and doors with clear width
(long side of the door polygon) and the rooms they join. Ceiling heights and window opening areas are
not in the common format, so rules on them are not measured.
"""
import argparse
import json
import os
import sys
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import rules
from .eval_config import REFERENCES, VARIANTS
from .format import load_sample
from .geometry import min_rect_sides, rooms_as_shapes, to_polygon
from .metrics import door_edges

ROOT = Path(__file__).resolve().parents[1]
FPR_ROOT = Path(os.environ.get("FLOORPLAN_RULES", ROOT.parent / "floorplan-rules"))
BINDING = ("law", "model_code")
HABITABLE = {"living_room", "bedroom", "study", "dining_room"}  # rooms counted as "Zimmer"
NOT_DWELLING_AREA = {"balcony", "outdoor", "garage"}

_CHECKER = {}


def _checker(jurisdiction):
    """(check function, Country, rule pack by id, from_room_type map), loaded once per process."""
    if jurisdiction not in _CHECKER:
        sys.path.insert(0, str(FPR_ROOT / "tools"))
        import yaml
        from check_plan import check
        from common import Country
        from rulepacks import load_pack
        country = Country(jurisdiction.split("-")[0].split(":")[0].lower())
        pack = load_pack(country)["rules"]
        els = yaml.safe_load(open(FPR_ROOT / "vocabulary" / "elements.yaml"))
        _CHECKER[jurisdiction] = (check, country, pack, {r["id"]: r for r in pack},
                                  els["fpeval"]["from_room_type"])
    return _CHECKER[jurisdiction]


def sample_to_plan(sample, scale, jurisdiction, from_type, accessibility="none",
                   building_type="multi_family_building"):
    """Common-format sample -> plan dict of the floorplan-rules checker (lengths in m)."""
    shapes = rooms_as_shapes(sample)
    rooms = []
    for k, (rtype, geom, _) in enumerate(shapes):
        short, long_ = min_rect_sides(geom)
        rooms.append({"id": f"r{k}", "type": from_type.get(rtype, "room"), "fpeval_type": rtype,
                      "area_m2": round(geom.area * scale ** 2, 3),
                      "width_m": round(short * scale, 3), "depth_m": round(long_ * scale, 3), "doors": []})
    tol = rules.WALL_HALF_THICKNESS_M / scale
    for door in sample.get("doors") or []:
        geom, _ = to_polygon(door["polygon"])
        if geom.is_empty:
            continue
        width = min_rect_sides(geom)[1] * scale
        if door.get("type") == "front_door":
            hit = [k for k, (_, g, _) in enumerate(shapes) if g.distance(geom) <= tol]
            if hit:
                rooms[hit[0]]["doors"].append({"to": "stairwell", "clear_width_m": round(width, 3)})
            continue
        edge = door_edges({"doors": [door]}, shapes, tol)
        if edge:
            i, j = edge[0]
            rooms[i]["doors"].append({"to": f"r{j}", "clear_width_m": round(width, 3)})
    living = [r for r in rooms if r["fpeval_type"] not in NOT_DWELLING_AREA]
    return {
        "name": f"{sample['source']}/{sample['variant']}/{sample['id']}", "jurisdiction": jurisdiction,
        "building": {"type": building_type}, "subsidised": False, "accessibility": accessibility,
        "regime": "new_build",
        "dwelling": {"type": "dwelling", "area_m2": round(sum(r["area_m2"] for r in living), 3),
                     "rooms_count": sum(r["fpeval_type"] in HABITABLE for r in rooms), "rooms": rooms},
    }


def check_sample(path, scale, jurisdiction, accessibility):
    check, country, _, by_id, from_type = _checker(jurisdiction)
    sample = load_sample(path)
    plan = sample_to_plan(sample, scale, jurisdiction, from_type, accessibility)
    res = check(plan, list(by_id.values()), country, jurisdiction)
    measured = [f for f in res["findings"] if f["result"] in ("met", "not met")]
    binding = [f for f in measured if by_id[f["rule"]]["bindingness"] in BINDING]
    failed = [f for f in binding if f["result"] == "not met"]
    # Door widths are kept apart: several generators draw doors as fixed-size symbols, so for them
    # the door rule measures a drawing convention (WallPlan 0.70 m, DiffPlanner 0.84 m).
    failed_rooms = [f for f in failed if by_id[f["rule"]].get("parameter") != "door_clear_width"]
    return {"id": sample["id"], "n_binding_checked": len(binding), "n_binding_not_met": len(failed),
            "any_binding_not_met": float(bool(failed)),
            "any_binding_not_met_excl_doors": float(bool(failed_rooms)),
            "n_other_not_met": sum(f["result"] == "not met" for f in measured) - len(failed),
            "failed_rules": ";".join(sorted({f["rule"] for f in failed}))}


def _target(name):
    """(label, sample files, scale, out dir) for a reference name or variant key."""
    from .evaluate import ref_dir, ref_files
    if name in REFERENCES:
        _, _, scale, known, _ = REFERENCES[name]
        return name, ref_files(name), scale, known, ref_dir(name)
    cfg = VARIANTS[name]
    files = sorted((ROOT / "outputs" / name / "samples").glob("*.json"))
    return name, files, cfg["scale"], cfg["scale_known"], ROOT / "outputs" / name / "eval"


def run(jurisdiction, names, workers=16, accessibility="none", max_samples=3000):
    _, _, _, by_id, _ = _checker(jurisdiction)
    rows = []
    for name in names:
        label, files, scale, known, out = _target(name)
        if not known:
            print(f"[norms] skip {label}: no metric scale")
            continue
        if len(files) > max_samples:
            files = sorted(np.random.default_rng(0).choice(files, max_samples, replace=False))
        with Pool(workers) as pool:
            res = pool.map(partial(check_sample, scale=scale, jurisdiction=jurisdiction,
                                   accessibility=accessibility), files, chunksize=8)
        df = pd.DataFrame(res)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"norms_{jurisdiction}.csv", index=False)
        counts = Counter(r for s in df.failed_rules if s for r in s.split(";"))
        top = [{"rule": r, "share_of_plans": n / len(df), "statement": by_id[r]["statement"],
                "jurisdiction": by_id[r]["jurisdiction"], "edition": by_id[r].get("edition"),
                "source": "; ".join(f"{s['source_id']} {s['locator']}" for s in by_id[r]["sources"][:1])}
               for r, n in counts.most_common(10)]
        summary = {"target": label, "jurisdiction": jurisdiction, "accessibility": accessibility,
                   "n_plans": len(df), "share_breaking_binding_rule": float(df.any_binding_not_met.mean()),
                   "share_breaking_excl_doors": float(df.any_binding_not_met_excl_doors.mean()),
                   "mean_binding_checked": float(df.n_binding_checked.mean()),
                   "mean_binding_not_met": float(df.n_binding_not_met.mean()), "top_failed": top}
        json.dump(summary, open(out / f"norms_{jurisdiction}.json", "w"), indent=1, ensure_ascii=False)
        rows.append(summary)
        print(f"[norms] {label}: {len(df)} plans, {summary['share_breaking_binding_rule']:.2f} break a binding rule")
    write_report(jurisdiction, rows)


def write_report(jurisdiction, rows):
    out = ROOT / "results" / "norms"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Building norms: {jurisdiction}", "",
          "Generated by `python -m fpeval.norms`. Rules from the floorplan-rules project "
          "(law and model code only count as binding). Ceiling heights and window areas are not in "
          "the plan format and are not measured.", "",
          "Door widths are shown apart: several generators draw doors as fixed-size symbols (WallPlan "
          "0.70 m, DiffPlanner 0.84 m), and real-data door polygons give the nominal, not the clear width.", "",
          "| target | plans | binding rules measured / plan | plans breaking a binding rule | without door widths | most frequent |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        top = ", ".join(f"{t['rule']} ({t['share_of_plans']:.0%})" for t in r["top_failed"][:3]) or "–"
        md.append(f"| {r['target']} | {r['n_plans']} | {r['mean_binding_checked']:.1f} | "
                  f"{r['share_breaking_binding_rule']:.2f} | {r['share_breaking_excl_doors']:.2f} | {top} |")
    (out / f"{jurisdiction}.md").write_text("\n".join(md) + "\n")
    json.dump(rows, open(out / f"{jurisdiction}.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {out / (jurisdiction + '.md')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jurisdiction", help="e.g. AT-9 (Wien), AT-6 (Steiermark)")
    ap.add_argument("targets", nargs="*", help="reference names or variant keys")
    ap.add_argument("--all", action="store_true", help="all references and variants with a metric scale")
    ap.add_argument("--accessibility", default="none", choices=["none", "basic", "R"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=3000)
    a = ap.parse_args()
    names = a.targets or []
    if a.all:
        names = [n for n in REFERENCES if REFERENCES[n][3]] + [k for k, v in VARIANTS.items() if v["scale_known"]
                                                              and (ROOT / "outputs" / k / "samples").is_dir()]
    run(a.jurisdiction, names, a.workers, a.accessibility, a.max_samples)


if __name__ == "__main__":
    main()
