"""Evaluate GT reference sets and generated variants.

    python -m fpeval.evaluate refs  [names...]   # GT reference sets (run first)
    python -m fpeval.evaluate variants [method/variant ...]
    python -m fpeval.evaluate stats              # dataset statistics -> results/datasets/
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
from .eval_config import REFERENCES, VARIANTS
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refs", "variants", "stats", "report"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.what == "refs":
        for n in args.names or REFERENCES:
            eval_reference(n, args.workers)
    elif args.what == "variants":
        for k in args.names or VARIANTS:
            eval_variant(k, args.workers)
    elif args.what == "stats":
        from .stats import write_stats
        write_stats()
    else:
        from .report import write_report
        write_report()


if __name__ == "__main__":
    main()
