"""Write <variant>/run_info.json (commit, checkpoint, command, conditioning, #samples, date, runtime)."""
import argparse, datetime, json, os, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("variant_dir")
ap.add_argument("--method", required=True)
ap.add_argument("--submodule", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--command", required=True)
ap.add_argument("--conditioning", required=True)
ap.add_argument("--ids", default=None)
ap.add_argument("--infer_seconds", type=float, default=None)
ap.add_argument("--env", default=None)
ap.add_argument("--notes", default="")
ap.add_argument("--extra", default=None, help="json string merged into the record")
a = ap.parse_args()
root = Path(a.variant_dir)
try:
    commit = subprocess.check_output(["git", "-C", a.submodule, "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = None
n_in = sum(1 for l in open(a.ids) if l.strip()) if a.ids else None
info = {
    "method": a.method, "variant": root.name, "commit": commit, "checkpoint": a.checkpoint,
    "command": a.command, "conditioning": a.conditioning, "input_ids": a.ids, "num_inputs": n_in,
    "num_samples": len(list((root / "samples").glob("*.json"))),
    "num_gt_samples": len(list((root / "gt_samples").glob("*.json"))) if (root / "gt_samples").exists() else 0,
    "date": datetime.date.today().isoformat(), "runtime_inference_s": a.infer_seconds,
    "conda_env": a.env, "gpu": "1x A100-40GB (CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "unset") + ")", "notes": a.notes,
}
for logname in ("raw/_log.json",):
    p = root / logname
    if p.exists():
        lg = json.load(open(p))
        info["failed_inputs"] = lg.get("failed", {})
        if info["runtime_inference_s"] is None:
            info["runtime_inference_s"] = lg.get("seconds")
if a.extra:
    info.update(json.loads(a.extra))
json.dump(info, open(root / "run_info.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "failed_inputs"}, indent=1))
