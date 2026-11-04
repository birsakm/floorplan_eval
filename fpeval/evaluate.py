"""Evaluate GT reference sets and generated variants.

    python -m fpeval.evaluate refs  [names...]   # GT reference sets (run first)
    python -m fpeval.evaluate variants [method/variant ...]
    python -m fpeval.evaluate stats              # dataset statistics -> results/datasets/
    python -m fpeval.evaluate fid [method/variant ...] [--ref NAME]   # FID/KID -> <variant>/eval/fid_<ref>.json
    python -m fpeval.evaluate report             # results/summary.{csv,md}

Writes per-sample / per-room (/ per-edge for references) CSVs and summary.json into <dir>/eval/.
"""
import argparse
import json
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import rules
from .eval_config import FID_N, FID_REFS, REFERENCES, VARIANTS
from .format import load_sample
from .metrics import door_edges, paired_metrics, sample_metrics

ROOT = Path(__file__).resolve().parents[1]


def _gt_id(sample_id, gt_ids):
    if sample_id in gt_ids:
        return sample_id
    base = re.sub(r"_\d+$", "", sample_id)
    return base if base in gt_ids else None


def _eval_one(path, scale, scale_known, gt_dir=None, gt_ids=None, frame_aligned=False, edges=False):
    sample = load_sample(path)
    res = sample_metrics(sample, scale, scale_known)
    sid = sample["id"]
    row = {"id": sid, **res[0]}
    rooms = [{"id": sid, "idx": k, **r} for k, r in enumerate(res[1])]
    edge_rows = []
    if edges:
        edge_rows = [{"id": sid, "i": int(i), "j": int(j), "kind": "wall"} for i, j in res[2].edges]
        tol = rules.WALL_HALF_THICKNESS_M / scale
        edge_rows += [{"id": sid, "i": i, "j": j, "kind": "door"} for i, j in door_edges(sample, res[3], tol)]
        row["n_interior_doors"] = sum(1 for e in edge_rows if e["kind"] == "door")
    if gt_dir is not None:
        gid = _gt_id(sid, gt_ids)
        row["gt_id"] = gid
        if gid is not None:
            gt = sample_metrics(load_sample(Path(gt_dir) / f"{gid}.json"), scale, scale_known)
            row.update(paired_metrics(res, gt, frame_aligned))
    return row, rooms, edge_rows


def _eval_dir(files, workers, **kw):
    with Pool(workers) as pool:
        results = pool.map(partial(_eval_one, **kw), files, chunksize=16)
    rows = pd.DataFrame([r for r, _, _ in results])
    rooms = pd.DataFrame([x for _, rs, _ in results for x in rs])
    edges = pd.DataFrame([x for _, _, es in results for x in es], columns=["id", "i", "j", "kind"])
    return rows, rooms, edges


def _summarize(df, prefix=""):
    num = df.select_dtypes(include=[np.number])
    return {prefix + k: (None if pd.isna(v) else float(v)) for k, v in num.mean().items()}


def ref_dir(name):
    d, split, *_ = REFERENCES[name]
    tag = split if isinstance(split, str) else ("all" if split is None else "+".join(split))
    return ROOT / d / "eval" / tag


def ref_files(name):
    d, split, *_, max_n = REFERENCES[name]
    d = ROOT / d
    if split is None:
        ids = sorted(p.stem for p in (d / "samples").glob("*.json"))
    else:
        splits = json.load(open(d / "splits.json"))["splits"]
        ids = sorted(i for s in ([split] if isinstance(split, str) else split) for i in splits[s])
    if len(ids) > max_n:
        ids = sorted(np.random.default_rng(0).choice(ids, max_n, replace=False))
    return [d / "samples" / f"{i}.json" for i in ids]


def eval_reference(name, workers):
    _, _, scale, scale_known, _ = REFERENCES[name]
    out = ref_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    rows, rooms, edges = _eval_dir(ref_files(name), workers, scale=scale, scale_known=scale_known, edges=True)
    rows.to_csv(out / "per_sample.csv", index=False)
    rooms.to_csv(out / "per_room.csv", index=False)
    edges.to_csv(out / "per_edge.csv", index=False)
    summary = {"reference": name, "n_samples": len(rows), **_summarize(rows)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"[ref] {name}: {len(rows)} samples -> {out}")


def eval_variant(key, workers):
    cfg = VARIANTS[key]
    vdir = ROOT / "outputs" / key
    out = vdir / "eval"
    out.mkdir(exist_ok=True)
    files = sorted((vdir / "samples").glob("*.json"))
    gt_dir = vdir / "gt_samples"
    gt_ids = {p.stem for p in gt_dir.glob("*.json")} if gt_dir.is_dir() else set()
    kw = dict(scale=cfg["scale"], scale_known=cfg["scale_known"])
    rows, rooms, _ = _eval_dir(files, workers, gt_dir=gt_dir if gt_ids else None, gt_ids=gt_ids,
                               frame_aligned=cfg["frame_aligned"], **kw)
    rows.to_csv(out / "per_sample.csv", index=False)
    rooms.to_csv(out / "per_room.csv", index=False)

    summary = {"variant": key, **cfg, "n_samples": len(rows)}
    if gt_ids:
        # Yield: generated samples / expected samples (inputs x samples per input).
        per_input = rows.groupby("gt_id").size().max() if rows.gt_id.notna().any() else 1
        summary["n_gt_inputs"] = len(gt_ids)
        summary["yield"] = len(rows) / (len(gt_ids) * per_input)
    summary.update(_summarize(rows))
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print(f"[var] {key}: {len(rows)} samples, {summary['any_rule_violation']:.2f} break a rule")


def _split_files(name, split, max_n, seed=0):
    d = ROOT / REFERENCES[name][0]
    if split is None:
        ids = sorted(p.stem for p in (d / "samples").glob("*.json"))
    else:
        ids = sorted(json.load(open(d / "splits.json"))["splits"][split])
    if len(ids) > max_n:
        ids = sorted(np.random.default_rng(seed).choice(ids, max_n, replace=False))
    return [d / "samples" / f"{i}.json" for i in ids]


def eval_fid(key, workers, ref=None):
    """FID/KID of a variant (up to FID_N random plans) against real plans, with a floor at the same size.

    Floor: the GT plans of the same inputs when the variant has them (what reproducing the GT would score),
    otherwise the same number of random plans from the evaluation set. The reference comes from FID_REFS.
    """
    from .distribution import fid, kid, layout_features
    cfg = VARIANTS[key]
    ref = ref or cfg["ref"]
    vdir = ROOT / "outputs" / key
    files = sorted((vdir / "samples").glob("*.json"))
    if len(files) > FID_N:
        files = sorted(np.random.default_rng(0).choice(files, FID_N, replace=False))
    gt_dir = vdir / "gt_samples"
    gt_ids = {p.stem for p in gt_dir.glob("*.json")} if gt_dir.is_dir() and ref == cfg["ref"] else set()
    gts = sorted({g for g in (_gt_id(f.stem, gt_ids) for f in files) if g is not None})
    if len(gts) >= 0.9 * len(files):
        floor, floor_kind, floor_scale = [gt_dir / f"{g}.json" for g in gts], "GT of the same inputs", cfg["scale"]
    else:
        floor = list(np.random.default_rng(1).permutation(ref_files(ref)))[:len(files)]
        floor_kind, floor_scale = "random real plans", REFERENCES[ref][2]
    split, max_n = FID_REFS[ref]
    rfiles = _split_files(ref, split, max_n)
    rset = set(map(str, rfiles))
    floor = [f for f in floor if str(f) not in rset]
    rcache = ref_dir(ref) / f"fid_features_{split or 'all'}.npz"
    R = layout_features(rfiles, rcache, REFERENCES[ref][2], workers)
    G = layout_features(files, vdir / "eval" / "fid_features.npz", cfg["scale"], workers)
    Fl = layout_features(floor, (vdir / "eval" if floor_kind.startswith("GT") else ref_dir(ref)) / "fid_features_floor.npz",
                         floor_scale, workers)
    out = {"variant": key, "ref": ref, "ref_split": split, "n": len(G), "n_ref": len(R), "n_floor": len(Fl),
           "floor": floor_kind, "fid": fid(G, R), "fid_floor": fid(Fl, R)}
    out["kid_x1000"], out["kid_x1000_std"] = (1000 * v for v in kid(G, R))
    out["kid_floor_x1000"], out["kid_floor_x1000_std"] = (1000 * v for v in kid(Fl, R))
    json.dump(out, open(vdir / "eval" / f"fid_{ref.replace(':', '_')}.json", "w"), indent=1)
    print(f"[fid] {key} vs {ref}: FID {out['fid']:.1f} (floor {out['fid_floor']:.1f}), "
          f"KID {out['kid_x1000']:.1f} (floor {out['kid_floor_x1000']:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refs", "variants", "stats", "fid", "report"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ref", default=None, help="fid: reference set instead of the variant's own")
    args = ap.parse_args()
    if args.what == "refs":
        for n in args.names or REFERENCES:
            eval_reference(n, args.workers)
    elif args.what == "variants":
        for k in args.names or VARIANTS:
            eval_variant(k, args.workers)
    elif args.what == "fid":
        for k in args.names or [k for k in VARIANTS if not k.startswith("ours/")]:
            if any((ROOT / "outputs" / k / "samples").glob("*.json")):
                eval_fid(k, args.workers, args.ref)
    elif args.what == "stats":
        from .stats import write_stats
        write_stats()
    else:
        from .report import write_report
        write_report()


if __name__ == "__main__":
    main()
