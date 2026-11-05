"""Write the measured numbers of Table 1 blocks (a), (b), (c) as LaTeX macros: paper/sec/results_auto.tex.

  python scripts/ours2/paper_numbers.py [--version <mason2 version>] [--house-version <mason2 version>]
  (defaults: the most recently evaluated apartment / multi-storey house version; block (a) from results/rplan/block_a.json)

paper/sec/eval_tables.tex uses the macros (\\RBoursValid, \\RBprocthorDFID, ...), so re-running this after a new
evaluation updates the table; the best value of each column (among the methods, not the Real row) is bold.
Do not edit results_auto.tex by hand.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from fpeval.eval_config import VARIANTS, mason2_variants  # noqa: E402

# metric -> (higher is better, format)
METRICS = {"Valid": (True, "{:.1f}"), "Reach": (True, "{:.1f}"), "Types": (True, "{:.1f}"), "Graph": (True, "{:.1f}"),
           "Area": (False, "{:.1f}\\%"), "DFID": (False, "{:.1f}"), "DKID": (False, "{:.1f}")}
HOLODECK_LLMS = {"holoqwen": "qwen3.8-27b", "hologemma": "gemma-4-31b"}
REF_LABEL = {"swiss_dwellings:all": "Swiss Dwellings", "cubicasa5k:all": "CubiCasa5k"}


def kind_of(version):
    try:
        info = json.load(open(ROOT / "outputs" / "ours" / f"mason2_{version}" / "run_info.json"))
    except OSError:
        return None
    return None if info.get("benchmark") else info.get("kind", "apartment")


def latest(kind="apartment"):
    res = sorted((ROOT / "results" / "ours").glob("mason2_*.json"), key=lambda p: p.stat().st_mtime)
    res = [r.stem[len("mason2_"):] for r in res if kind_of(r.stem[len("mason2_"):]) == kind]
    return res[-1] if res else None


def measure(key, ref_label):
    """Valid and the image metrics per plan (one plan per storey for houses); Reach, Types, Graph, Area per home."""
    o = json.load(open(ROOT / "results" / f"{key}.json"))
    h, f = o["homes"]["all"], o["fid"][ref_label]
    rule = json.load(open(ROOT / "outputs" / key / "eval" / "summary.json"))["any_rule_violation"]
    return {"N": o["n_homes"], "Valid": 100 * (1 - rule), "Reach": 100 * h["all_reachable"],
            "Types": 100 * h["room_counts_match"], "Graph": 100 * h["frac_ged_zero"], "Area": 100 * h["net_area_rel_err"],
            "DFID": f["fid"] - f["fid_real_vs_real"], "DKID": f["kid_x1000"] - f["kid_real_vs_real_x1000"],
            "NRef": f["n_ref"]}


# Table 1 block (a): results/rplan/block_a.json (scripts/rplan/block_a.py); macros \RA<row><metric>
A_ROWS = {"Real": "real", "House-GAN++": "hgpp", "HouseDiffusion": "hd", "WallPlan": "wallplan",
          "ProcTHOR": "procthor", "Infinigen": "infinigen",
          "Holodeck (Qwen3.8-27B)": "holoqwen", "Holodeck (Gemma-4-31B)": "hologemma",
          "APM (ours, bubble)": "apmbubble", "APM (ours, bubble+areas)": "apmareas"}
A_METRICS = {"Valid": ("valid", True, 100, "{:.1f}"), "Reach": ("reach", True, 100, "{:.1f}"),
             "Types": ("types", True, 100, "{:.1f}"), "Graph": ("graph", True, 100, "{:.1f}"),
             "Area": ("area_rel_err", False, 100, "{:.1f}\\%"), "DFID": ("dfid", False, 1, "{:.1f}"),
             "DKID": ("dkid", False, 1, "{:.1f}")}


def block_a():
    f = ROOT / "results" / "rplan" / "block_a.json"
    rows = {r["method"]: r for r in json.load(open(f))} if f.exists() else {}
    L = ["% Table 1 block (a), from results/rplan/block_a.json"]
    for name, tag in A_ROWS.items():
        r = rows.get(name)
        n = str(r["N"]) if r else "\\tbd"
        L.append(f"\\def\\RA{tag}N{{{n}}}")
    for metric, (k, higher, mult, fmt) in A_METRICS.items():
        shown = {n: fmt.format(mult * r[k]) for n, r in rows.items() if k in r and n != "Real"}
        num = {n: float(v.replace("\\%", "")) for n, v in shown.items()}
        best = (max if higher else min)(num.values()) if len(num) > 1 else None
        for name, tag in A_ROWS.items():
            r = rows.get(name)
            if name == "Real" and metric in ("DFID", "DKID"):
                v = "0"
            elif r is None:
                v = "\\tbd"
            elif k not in r:
                v = "\\na"
            elif name == "Real":
                v = fmt.format(mult * r[k])
            else:
                v = shown[name] if num[name] != best else "\\textbf{" + shown[name] + "}"
            L.append(f"\\def\\RA{tag}{metric}{{{v}}}")
    return L


def block(tag, version, kind):
    """Macros \\R<tag>... of one block (b: apartments, c: houses) from a Mason2 version and its two baselines."""
    keys = mason2_variants(version, kind)
    # Holodeck on the same briefs as text (scripts/holodeck/generate.py --block b|c): one row per LLM
    keys.update({tag: f"holodeck/{kind}_mason2-{version}_{llm}" for tag, llm in HOLODECK_LLMS.items()})
    ref = VARIANTS[keys["ours"]]["ref"]
    label = REF_LABEL[ref]
    vals = {m: measure(k, label) for m, k in keys.items() if (ROOT / "results" / f"{k}.json").exists()}
    missing = sorted(set(keys) - set(vals))
    ref_row = next(r for r in csv.DictReader(open(ROOT / "results" / "summary_references.csv")) if r["reference"] == ref)
    R = f"\\R{tag}"
    L = [f"% Table 1 block ({tag.lower()}): Mason2 generator version {version}; reference {ref}",
         f"\\def{R}version{{{version}}}",
         f"\\def{R}N{{{vals['ours']['N']:,}}}".replace(",", "{,}"),
         f"\\def{R}NRef{{{vals['ours']['NRef']:,}}}".replace(",", "{,}"),
         f"\\def{R}NRealAll{{{int(ref_row['n_samples']):,}}}".replace(",", "{,}"),
         f"\\def{R}realValid{{{100 * (1 - float(ref_row['any_rule_violation'])):.1f}}}"]
    for metric, (higher, fmt) in METRICS.items():
        # Holodeck gets only the brief (text), never the program, so its door graph has nothing to match: Graph n/a
        na = {m for m in vals if metric == "Graph" and m in HOLODECK_LLMS}
        shown = {m: fmt.format(v[metric]) for m, v in vals.items() if m not in na}
        # best by the printed value, so ties after rounding are all bold
        num = {m: float(s.replace("\\%", "")) for m, s in shown.items()}
        best = (max if higher else min)(num.values()) if num else None
        for m in keys:
            if m in na:
                L.append(f"\\def{R}{m}{metric}{{\\na}}")
            elif m not in vals:
                L.append(f"\\def{R}{m}{metric}{{\\tbd}}")
            else:
                s = shown[m]
                if num[m] == best and len(num) > 1:
                    s = "\\textbf{" + s + "}"
                L.append(f"\\def{R}{m}{metric}{{{s}}}")
    for m in keys:
        n = str(vals[m]["N"]) if m in vals else "\\tbd"
        L.append(f"\\def{R}{m}N{{{n}}}")
    return L, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None, help="apartment version (block b)")
    ap.add_argument("--house-version", default=None, help="multi-storey house version (block c)")
    ap.add_argument("--out", default=str(ROOT / "paper" / "sec" / "results_auto.tex"))
    a = ap.parse_args()
    version, house = a.version or latest("apartment"), a.house_version or latest("house")
    L = ["% Generated by scripts/ours2/paper_numbers.py -- do not edit by hand.", f"% Written {date.today().isoformat()}."]
    Lb, missing = block("B", version, "apartment")
    L += Lb
    if house:
        Lc, miss_c = block("C", house, "house")
        L += Lc
        missing += [f"house {m}" for m in miss_c]
    L += block_a()
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out} for {version}" + (f", houses {house}" if house else "")
          + (f"; missing: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
