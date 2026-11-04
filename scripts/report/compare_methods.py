"""Side-by-side table of all methods: rules, input adherence, FID/KID with their floors.

  python scripts/report/compare_methods.py [--version <mason2 version>]   # -> results/comparison.{md,csv}

--version picks the Mason2 generator version of the same-programs table (default: the most recently evaluated one).

Reads outputs/<variant>/eval/summary.json (python -m fpeval.evaluate variants),
outputs/<variant>/eval/fid_<ref>.json (python -m fpeval.evaluate fid) and
results/ours/<variant>.json (scripts/ours/evaluate_ours.py).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from fpeval.eval_config import STAT_DATASETS, VARIANTS, mason2_variants

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "housegan": "House-GAN", "houseganpp": "House-GAN++", "house_diffusion": "HouseDiffusion", "gsdiff": "GSDiff",
    "iplan": "iPLAN", "diffplanner": "DiffPlanner", "wallplan": "WallPlan", "maskplan": "MaskPLAN",
    "chathousediffusion": "ChatHouseDiffusion", "ds2d": "DS2D", "floorplan_rlvr": "Floor-plan RLVR",
    "infinigen": "Infinigen Indoors", "procthor": "ProcTHOR", "scenesmith": "SceneSmith", "ours": "Ours",
}
REF_LABELS = {**{k: v["label"] for k, v in STAT_DATASETS.items()}, "tell2design:test": "Tell2Design"}
# Our old pipeline (mason_old, 100 + 100 homes) and the baselines run on its programs are superseded by Mason2.
OBSOLETE = ("ours/mason_old_", "procthor/apartment_program_n100", "procthor/house_program_n100",
            "infinigen/apartment_program", "infinigen/house_program")


def rows(only=None):
    out = []
    for key, cfg in VARIANTS.items():
        vdir = ROOT / "outputs" / key
        summ = vdir / "eval" / "summary.json"
        if not summ.exists() or key.startswith(OBSOLETE) or (only is not None and "mason2" in key and key not in only):
            continue
        s = json.load(open(summ))
        base = dict(method=METHODS[key.split("/")[0]], variant=key.split("/", 1)[1], condition=cfg["condition"],
                    breaks_rule=s.get("any_rule_violation"), types_as_input=s.get("type_multiset_match"))
        res = ROOT / "results" / key.split("/")[0] / f"{key.split('/')[1]}.json"
        if key.startswith(("ours/", "infinigen/", "procthor/")) and res.exists():
            o = json.load(open(res))
            base["types_as_input"] = o["homes"]["all"]["room_counts_match"]
            for lab, f in o["fid"].items():
                out.append(dict(base, ref=lab, ref_kind="composition-matched", n=f["n_ours"], n_ref=f["n_ref"],
                                floor="held-out matched real", fid=f["fid"], fid_floor=f["fid_real_vs_real"],
                                kid_x1000=f["kid_x1000"], kid_floor_x1000=f["kid_real_vs_real_x1000"]))
        for p in sorted((vdir / "eval").glob("fid_*.json")):
            f = json.load(open(p))
            out.append(dict(base, ref=REF_LABELS[f["ref"]], ref_kind=f"{f['ref_split'] or 'all'} split",
                            n=f["n"], n_ref=f["n_ref"], floor=f["floor"], fid=f["fid"], fid_floor=f["fid_floor"],
                            kid_x1000=f["kid_x1000"], kid_floor_x1000=f["kid_floor_x1000"]))
    df = pd.DataFrame(out)
    df["fid_above_floor"] = df.fid - df.fid_floor
    df["kid_above_floor_x1000"] = df.kid_x1000 - df.kid_floor_x1000
    return df


def latest_mason2():
    """The Mason2 version whose evaluation finished last."""
    res = sorted((ROOT / "results" / "ours").glob("mason2_*.json"), key=lambda p: p.stat().st_mtime)
    return res[-1].stem[len("mason2_"):] if res else None


def program_table(version):
    """Ours and the baselines on the same Mason2 programs: adherence, rules, realism, diversity."""
    if version is None:
        return []
    n = json.load(open(ROOT / "results" / "ours" / f"mason2_{version}.json"))["n_plans"]
    L = [f"## On the same programs (Mason2 `{version}`, {n:,} apartments)", "",
         "ProcTHOR and Infinigen get our program (room types, target net areas, room-room connections) as input. "
         "area = mean relative net-area error; doors = program connections realised (recall); GED 0 = layouts whose "
         "connection graph equals the program; reach = all rooms reachable from the front door; win ≥ 1/8 = layouts "
         "where every habitable room has window area ≥ 1/8 of its floor area (– = method places no windows); "
         "C2ST = leave-one-out 1-NN accuracy telling layouts from composition-matched Swiss Dwellings layouts (50% = "
         "indistinguishable); div = diversity relative scale (Swiss Dwellings itself 0.75); ΔKID = KID − KID of "
         "held-out matched real layouts (×1000).", "",
         "| method | N | rule | types | area | doors | GED 0 | reach | win ≥ 1/8 | C2ST | div | ΔKID Swiss | ΔKID ResPlan |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in mason2_variants(version).values():
        res = ROOT / "results" / key.split("/")[0] / f"{key.split('/')[1]}.json"
        summ = ROOT / "outputs" / key / "eval" / "summary.json"
        if not res.exists():
            continue
        o, h = json.load(open(res)), json.load(open(res))["homes"]["all"]
        rule = json.load(open(summ)).get("any_rule_violation") if summ.exists() else None
        dk = {k: f["kid_x1000"] - f["kid_real_vs_real_x1000"] for k, f in o["fid"].items()}
        name = METHODS[key.split("/")[0]]
        L.append(f"| {'**Ours**' if name == 'Ours' else name} | {o['n_plans']} | {fmt(rule, True)} | "
                 f"{fmt(h['room_counts_match'], True)} | {100 * h['net_area_rel_err']:.1f}% | "
                 f"{fmt(h['program_door_recall'], True)} | {fmt(h['frac_ged_zero'], True)} | "
                 f"{fmt(h['all_reachable'], True)} | {fmt(h['plans_all_habitable_ge_1_8'] if h['frac_habitable_window_ge_1_8'] == h['frac_habitable_window_ge_1_8'] else None, True)} | "
                 f"{fmt(o['realism']['Swiss Dwellings']['c2st_1nn_acc'], True)} | {o['diversity']['ours']:.2f} | "
                 f"{fmt(dk.get('Swiss Dwellings'))} | {fmt(dk.get('ResPlan'))} |")
    return L + [""]


def fmt(x, pct=False):
    if x is None or pd.isna(x):
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="Mason2 version of the same-programs table (default: latest)")
    version = ap.parse_args().version or latest_mason2()
    df = rows(set(mason2_variants(version).values()) if version else None)
    out = ROOT / "results"
    df.to_csv(out / "comparison.csv", index=False)
    L = ["# All methods side by side", "",
         "Generated by `scripts/report/compare_methods.py`. Rules and input adherence: `python -m fpeval.evaluate "
         "variants` (docs/metrics.md §1–2). FID/KID: every plan rendered by `fpeval.render.render_layout` (256 px, "
         "type colours, 4 px walls, no doors or windows; docs/metrics.md §5), Inception-V3 features, KID ×1000.", "",
         "How to read FID/KID: each row is scored against real plans of its own dataset, and the **floor** is the "
         "score of real plans at the same sample size against the same reference: for methods with paired inputs, "
         "the GT plans of exactly those inputs (what reproducing the GT would score); for our plans, held-out "
         "composition-matched real plans; otherwise random real plans. FID depends strongly on N (biased upwards at "
         "small N), so compare **FID − floor** and KID across rows with different N; KID is unbiased. Rows with "
         "different references measure different things and are not a ranking. None of these numbers are comparable "
         "with the numbers printed in the papers, which use their own renderers and sample counts.", "",
         "Columns: N generated plans scored (random subset of at most 1,000); rule = plans breaking a basic rule "
         "(overlap, holes, pieces, minimum area/width, maximum aspect); types = plans whose room types and counts "
         "equal the input (paired methods: the GT plan; ours: the brief).", ""]
    L += program_table(version)
    order = ["RPLAN", "LIFULL", "Tell2Design", "ProcTHOR-10K", "Swiss Dwellings", "ResPlan", "CubiCasa5k", "MagicPlan"]
    for ref in [r for r in order if r in set(df.ref)]:
        d = df[df.ref == ref]
        L += [f"## Against {ref}" + (" (composition-matched to our plans)" if (d.ref_kind == "composition-matched").all()
                                      else ""), "",
              f"Reference: {d.n_ref.iloc[0]:,} real plans ({d.ref_kind.iloc[0]})." if d.n_ref.nunique() == 1
              else f"Reference: {', '.join(sorted({f'{x:,}' for x in d.n_ref}))} real plans.", "",
              "| method | variant | condition | N | rule | types | FID | floor | FID − floor | KID | floor | KID − floor | floor is |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {'**Ours**' if r.method == 'Ours' else r.method} | {r.variant} | {r.condition} | {r.n} | "
                     f"{fmt(r.breaks_rule, True)} | {fmt(r.types_as_input, True)} | {fmt(r.fid)} | {fmt(r.fid_floor)} | "
                     f"**{fmt(r.fid_above_floor)}** | {fmt(r.kid_x1000)} | {fmt(r.kid_floor_x1000)} | "
                     f"**{fmt(r.kid_above_floor_x1000)}** | {r.floor} |")
        L.append("")
    L += ["Our apartments against RPLAN are scored against a reference they were not composed to match (0.3% of RPLAN "
          "plans have a separate hallway; all of ours do), the same way as the RPLAN-trained baselines; their "
          "composition-matched scores are in the Swiss Dwellings and ResPlan sections. Floor-plan RLVR has not been "
          "run yet.", ""]
    (out / "comparison.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
