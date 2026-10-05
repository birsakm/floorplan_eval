"""Evaluate generated variants and GT reference sets.

    python -m fpeval.evaluate refs  [names...]   # GT reference sets (run first)
    python -m fpeval.evaluate variants [method/variant ...]
    python -m fpeval.evaluate report             # results/summary.{csv,md}

Writes per-sample/per-room CSVs and summary.json into <dir>/eval/.
"""
import argparse
import json
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .eval_config import REFERENCES, VARIANTS
from .format import load_sample
from .metrics import paired_metrics, sample_metrics

ROOT = Path(__file__).resolve().parents[1]


def _gt_id(sample_id, gt_ids):
    if sample_id in gt_ids:
        return sample_id
    base = re.sub(r"_\d+$", "", sample_id)
    return base if base in gt_ids else None


def _eval_one(path, scale, scale_known, gt_dir=None, gt_ids=None, frame_aligned=False):
    sample = load_sample(path)
    res = sample_metrics(sample, scale, scale_known)
    row = {"id": sample["id"], **res[0]}
    rooms = [{"id": sample["id"], **r} for r in res[1]]
    if gt_dir is not None:
        gid = _gt_id(sample["id"], gt_ids)
        row["gt_id"] = gid
        if gid is not None:
            gt = sample_metrics(load_sample(Path(gt_dir) / f"{gid}.json"), scale, scale_known)
            row.update(paired_metrics(res, gt, frame_aligned))
    return row, rooms


def _eval_dir(files, workers, **kw):
    with Pool(workers) as pool:
        results = pool.map(partial(_eval_one, **kw), files, chunksize=16)
    rows = pd.DataFrame([r for r, _ in results])
    rooms = pd.DataFrame([x for _, rs in results for x in rs])
    return rows, rooms


def _fid_features(files, cache, device):
    from .distribution import inception_features
    from .render import render
    if cache.exists():
        return np.load(cache)
    feats = inception_features((render(load_sample(f), size=256, margin=8, rooms_only=True) for f in files),
                               device=device)
    np.save(cache, feats)
    return feats


def _summarize(df, prefix=""):
    num = df.select_dtypes(include=[np.number])
    return {prefix + k: (None if pd.isna(v) else float(v)) for k, v in num.mean().items()}


def ref_files(name):
    d, split, *_ , max_n = REFERENCES[name]
    d = ROOT / d
    if split is None:
        ids = sorted(p.stem for p in (d / "samples").glob("*.json"))
    else:
        ids = json.load(open(d / "splits.json"))["splits"][split]
    rng = np.random.default_rng(0)
    if len(ids) > max_n:
        ids = sorted(rng.choice(ids, max_n, replace=False))
    return [d / "samples" / f"{i}.json" for i in ids]


def eval_reference(name, workers, device):
    d, split, scale, scale_known, _ = REFERENCES[name]
    out = ROOT / d / "eval" / (split or "all")
    out.mkdir(parents=True, exist_ok=True)
    files = ref_files(name)
    rows, rooms = _eval_dir(files, workers, scale=scale, scale_known=scale_known)
    rows.to_csv(out / "per_sample.csv", index=False)
    rooms.to_csv(out / "per_room.csv", index=False)
    _fid_features(files, out / "inception.npy", device)
    summary = {"reference": name, "n_samples": len(rows), **_summarize(rows)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"[ref] {name}: {len(rows)} samples -> {out}")


def _load_ref(name):
    d, split, *_ = REFERENCES[name]
    out = ROOT / d / "eval" / (split or "all")
    return (pd.read_csv(out / "per_sample.csv"), pd.read_csv(out / "per_room.csv"),
            np.load(out / "inception.npy"))


def eval_variant(key, workers, device):
    from .distribution import fid, kid, stat_distances
    cfg = VARIANTS[key]
    vdir = ROOT / "outputs" / key
    out = vdir / "eval"
    out.mkdir(exist_ok=True)
    files = sorted((vdir / "samples").glob("*.json"))
    gt_dir = vdir / "gt_samples"
    gt_ids = {p.stem for p in gt_dir.glob("*.json")} if gt_dir.is_dir() else set()
    kw = dict(scale=cfg["scale"], scale_known=cfg["scale_known"])
    rows, rooms = _eval_dir(files, workers, gt_dir=gt_dir if gt_ids else None, gt_ids=gt_ids,
                            frame_aligned=cfg["frame_aligned"], **kw)
    rows.to_csv(out / "per_sample.csv", index=False)
    rooms.to_csv(out / "per_room.csv", index=False)

    summary = {"variant": key, **{k: v for k, v in cfg.items()}, "n_samples": len(rows)}
    if gt_ids:
        # Yield: generated samples / expected samples (inputs x samples per input).
        per_input = rows.groupby("gt_id").size().max() if rows.gt_id.notna().any() else 1
        summary["n_gt_inputs"] = len(gt_ids)
        summary["yield"] = len(rows) / (len(gt_ids) * per_input)
        gt_rows, _ = _eval_dir(sorted(gt_dir.glob("*.json")), workers, **kw)
        gt_rows.to_csv(out / "gt_per_sample.csv", index=False)
        summary.update(_summarize(gt_rows, "gt_"))
    summary.update(_summarize(rows))

    ref_rows, ref_rooms, ref_feats = _load_ref(cfg["ref"])
    summary.update(stat_distances(rows, rooms, ref_rows, ref_rooms,
                                  cfg["scale_known"] and REFERENCES[cfg["ref"]][3]))
    feats = _fid_features(files, out / "inception.npy", device)
    summary["fid"] = fid(feats, ref_feats)
    summary["kid_x1000"], summary["kid_std_x1000"] = (1000 * v for v in kid(feats, ref_feats))
    if gt_ids:
        # Same inputs and sample count as the generated set, so sample-size bias cancels.
        gt_feats = _fid_features(sorted(gt_dir.glob("*.json")), out / "gt_inception.npy", device)
        summary["fid_vs_gt"] = fid(feats, gt_feats)
        summary["kid_vs_gt_x1000"] = 1000 * kid(feats, gt_feats)[0]
        summary["fid_gt_vs_ref"] = fid(gt_feats, ref_feats)  # floor: real plans vs reference
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print(f"[var] {key}: {len(rows)} samples, FID {summary['fid']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refs", "variants", "report"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.what == "refs":
        for n in args.names or REFERENCES:
            eval_reference(n, args.workers, args.device)
    elif args.what == "variants":
        for k in args.names or VARIANTS:
            eval_variant(k, args.workers, args.device)
    else:
        from .report import write_report
        write_report()


if __name__ == "__main__":
    main()
