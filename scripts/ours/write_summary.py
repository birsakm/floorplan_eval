"""Write results/ours/<variant>.md from the outputs of evaluate_ours.py, fpeval.evaluate and fpeval.norms.

  python scripts/ours/write_summary.py --kind {apartment,house}
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NORMS = [("AT-9", "Vienna"), ("AT-3", "Lower Austria"), ("AT-4", "Upper Austria")]


def pct(x, nd=0):
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="apartment", choices=["apartment", "house"])
    a = ap.parse_args()
    variant = f"ours/mason_old_{a.kind}_n100_seed0"
    stem = variant.split("/")[-1]
    o = json.load(open(ROOT / "results" / "ours" / f"{stem}.json"))
    rules = json.load(open(ROOT / "outputs" / variant / "eval" / "summary.json"))
    norms = {j: json.load(open(ROOT / "outputs" / variant / "eval" / f"norms_{j}.json")) for j, _ in NORMS}
    H, P, A, R, D = o["homes"], o["plans"], o["areas"], o["realism"], o["diversity"]
    refs = [k for k in P if k != "ours"]
    unit = "plans" if a.kind == "apartment" else "storeys"
    groups = [g for g in H if g != "all"] or ["all"]
    cols = ["all"] + ([g for g in groups if g != "all"] if len(H) > 1 else [])

    L = [f"# Our {a.kind}s: evaluation (mason_old, run 20261024-full-n100-seed0)", ""]
    if a.kind == "apartment":
        L += ["100 apartments from Mason's old program (state export in `data/ours`, room-plan stage), converted by "
              "`scripts/ours/convert.py`. Real baselines: Swiss Dwellings and ResPlan apartments with the same numbers "
              "of bedrooms and bathrooms and a balcony or not, 20 per generated plan.", ""]
    else:
        L += ["100 houses from Mason's old program (62 one-storey, 38 two-storey), converted by `scripts/ours/convert.py` "
              "into one sample per storey (138). In the two-storey houses the State replay had rejected 50 rooms and "
              "97 openings because a newer check counts stair rooms into the net area; the rejected commands carry the "
              "original geometry and are restored, so all 100 houses have the generator's full room list. "
              "Real baselines: CubiCasa5k and MagicPlan floors with the same numbers of bedrooms and bathrooms, "
              "20 per generated storey. Neither dataset says whether a floor belongs to a house or an apartment.", ""]
    L += ["## Validity, input and building norms", "",
          "| metric (lineage) | " + " | ".join(f"{c.replace('1 storeys', '1 storey')} (n={H[c]['n']})" for c in cols) + " |",
          "|---|" + "---|" * len(cols)]

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(H[c]) for c in cols) + " |")
    L.append(f"| {unit} breaking a basic rule: overlap, holes, pieces, min area/width, max aspect "
             f"(HouseDiffusion-era validity, GRE-Diff) | {pct(rules['any_rule_violation'])} |" + " |" * (len(cols) - 1))
    row("room counts exactly as in the brief (DS2D prompt consistency)", lambda h: pct(h["room_counts_match"]))
    row("net area error vs brief, mean / max (DS2D, RLVR area error)",
        lambda h: f"{pct(h['net_area_rel_err'], 2)} / {pct(h['net_area_rel_err_max'], 2)}")
    if a.kind == "house":
        row("same, counting stair rooms as net area", lambda h: pct(h["net_area_incl_stairs_rel_err"], 1))
    row("program connections built, recall / precision (DS2D self-consistency)",
        lambda h: f"{h['program_door_recall']:.2f} / {h['program_door_precision']:.2f}")
    row("program 'wall' neighbours sharing a wall", lambda h: f"{h['program_wall_adjacent']:.3f}")
    row("compatibility: typed graph edit distance program ↔ built connections (House-GAN)",
        lambda h: f"{h['compatibility_ged']:.2f}")
    row("all rooms reachable from the front door" + (" (stairs join storeys)" if a.kind == "house" else "")
        + " (GRE-Diff)", lambda h: pct(h["all_reachable"]))
    row("habitable rooms only reachable through a bedroom, per home", lambda h: f"{h['n_habitable_via_bedroom']:.2f}")
    row("habitable rooms with window area ≥ 1/8 of floor area", lambda h: pct(h["frac_habitable_window_ge_1_8"]))
    L.append(f"| {unit} breaking a binding Austrian rule: " + ", ".join(n for _, n in NORMS) + " | "
             + " / ".join(pct(norms[j]["share_breaking_binding_rule"]) for j, _ in NORMS) + " |" + " |" * (len(cols) - 1))
    fails = [t for j, _ in NORMS for t in norms[j]["top_failed"]]
    if fails:
        L += ["", "Building-norm failures: " + "; ".join(
            f"{t['rule']} ({pct(t['share_of_plans'])} of {unit}): {t['statement']}" for t in fails)]
    L += ["", "Program connections and the brief come from the generator's own program stage, so these rows measure "
          "internal consistency. Reachability is not compared with real data: the real datasets draw no door for "
          "open connections.", "",
          f"## Realism against matched real {'apartments' if a.kind == 'apartment' else 'floors'}", "",
          "| | ours | " + " | ".join(refs) + " |", "|---|---|" + "---|" * len(refs)]
    for k, lab in [("total_area_m2", f"net area per {'plan' if a.kind == 'apartment' else 'storey'} (m²)"),
                   ("n_rooms", "rooms"), ("outline_corners", "corners of the outline"),
                   ("outline_rectangularity", "outline area / bounding rectangle"),
                   ("frac_rect_rooms", "rooms that are plain rectangles"), ("mean_room_aspect", "mean room aspect ratio"),
                   ("mean_adjacency_degree", "neighbours per room"),
                   ("frac_habitable_with_window", "habitable rooms with a window")]:
        L.append(f"| {lab} | {P['ours'][k]:.2f} | " + " | ".join(f"{P[r][k]:.2f}" for r in refs) + " |")
    L += ["", "Room area in m², median (share of our rooms inside the real p5–p95 range):", "",
          "| room | ours (n) | " + " | ".join(refs) + " |", "|---|---|" + "---|" * len(refs)]
    for t, x in A.items():
        cells = []
        for r in refs:
            y = x.get(r)
            cells.append("–" if not y or y["median"] is None else f"{y['median']:.1f} ({pct(y['ours_within_p5_p95'])})")
        L.append(f"| {t} | {x['ours_median']:.1f} ({x['ours_n']}) | " + " | ".join(cells) + " |")
    cross = next(k for k in R if " vs " in k)
    L += ["", "Classifier two-sample test (leave-one-out 1-NN on geometric plan features; 50% = cannot tell apart): "
          + ", ".join(f"ours vs {r} {pct(R[r]['c2st_1nn_acc'])} (real vs real {pct(R[r]['c2st_real_vs_real'])})" for r in refs)
          + f"; {cross} {pct(R[cross]['c2st_1nn_acc'])}. The test is very sensitive: two real datasets are about as "
          "separable from each other as ours is from either.", ""]
    for r in refs:
        L.append(f"Largest differences to {r} (by Cohen's d): " + "; ".join(
            f"{k} {v['ours']:.2f} vs {v['real']:.2f}" for k, v in list(R[r]["largest_differences"].items())[:6]) + ".")
    L += ["", "## Diversity", "",
          f"Distinct typed room-adjacency topologies per {unit[:-1]}: ours {pct(D['ours'])}, "
          + ", ".join(f"matched {r} {pct(D[r])}" for r in refs) + f" (mean of 20 draws of {o['n_plans']}).", ""]
    Fd = o["fid"]
    L += ["## FID / KID", "",
          f"Inception-V3 features of uniform 256 px layout renders (`fpeval.render.render_layout`: type colours, 4 px "
          f"walls, no doors or windows), against the matched real {unit}. Mean ± std over 5 rounds; each round holds "
          f"out one matched real {unit[:-1]} per generated one, and the held-out set is scored the same way as ours. "
          "That row is the floor: what a generator that samples the real data would get at this sample size. "
          "KID ×1000; lower is better for both.", "",
          "| reference | " + " | ".join(f"{r} (n={Fd[r]['n_ref']})" for r in refs) + " |",
          "|---|" + "---|" * len(refs)]

    def frow(label, key):
        L.append(f"| {label} | " + " | ".join(
            f"{Fd[r][key]:.1f} ± {Fd[r][key + '_std']:.1f}" for r in refs) + " |")
    frow(f"FID, ours (n={Fd[refs[0]]['n_ours']})", "fid")
    frow("FID, held-out real of the same dataset (floor)", "fid_real_vs_real")
    frow("FID, held-out real of the other dataset", "fid_other_real")
    frow("KID ×1000, ours", "kid_x1000")
    frow("KID ×1000, held-out real of the same dataset (floor)", "kid_real_vs_real_x1000")
    frow("KID ×1000, held-out real of the other dataset", "kid_other_real_x1000")
    L += ["", "FID at about 100 samples has a large positive bias (the floor is well above 0) and absolute values are "
          "not comparable with numbers in papers, which use other renderers and 1,000–12,000 samples. KID is unbiased.",
          "", "## Not measured", "",
          "- Human or VLM realism studies (House-GAN-style pairwise realism)."]
    out = ROOT / "results" / "ours" / f"{stem}.md"
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
