#!/usr/bin/env bash
# End-to-end Residential Floorplan Diffusion on the 11 bundled condition sets (test/stage1_input/<k>_*.png),
# PER_COND accepted samples each (default 48 -> 528 samples). Requires the (currently inaccessible)
# weights in checkpoints/residential_floorplan_diffusion/.
# Usage: bash scripts/residential_floorplan_diffusion/run.sh [PER_COND]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PER="${1:-48}"
V=bundled_roommasks
OUT="$ROOT/outputs/residential_floorplan_diffusion/$V"
CK="$ROOT/checkpoints/residential_floorplan_diffusion"
for f in model_stage1.pth model_stage2.pth; do [ -s "$CK/$f" ] || { echo "missing $CK/$f (run download.sh)"; exit 1; }; done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/raw"
T0=$(date +%s)
(cd "$ROOT/external/methods/residential_floorplan_diffusion" && \
  conda run --no-capture-output -n fpe-resdiff python "$ROOT/scripts/residential_floorplan_diffusion/infer.py" \
    --ckpt "$CK" --out "$OUT/raw" --per_cond "$PER" --batch 64 --max_rounds 4 --seed 0) 2>&1 | tee "$OUT/raw/infer.log"
T1=$(date +%s)
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # converters use an explicit 8-worker pool
PYTHONPATH="$ROOT" conda run -n fpe python scripts/residential_floorplan_diffusion/convert.py "$OUT"
conda run -n fpe python -m fpeval.render "$OUT"
PYTHONPATH="$ROOT" conda run -n fpe python - "$OUT" "$((T1-T0))" "$(git -C external/methods/residential_floorplan_diffusion rev-parse HEAD 2>/dev/null)" <<'PY'
import json, sys, datetime, os
out, rt, commit = sys.argv[1:]
meta = json.load(open(f"{out}/raw/meta.json"))
json.dump({"method": "residential_floorplan_diffusion", "variant": os.path.basename(out), "commit": commit,
           "checkpoint": "checkpoints/residential_floorplan_diffusion/model_stage{1,2}.pth",
           "command": "scripts/residential_floorplan_diffusion/infer.py --per_cond N --batch 64 --max_rounds 4 --seed 0",
           "conditioning": "per-room-type rectangle masks (living, bedroom, kitchen, toilet, balcony), 11 bundled test sets",
           "filter": "upstream Room_Judgment (room counts per type must match the condition)",
           "num_samples": len(os.listdir(f"{out}/samples")),
           "drawn_stage1": sum(m["drawn"] for m in meta.values()),
           "date": datetime.date.today().isoformat(), "runtime_sec_inference": int(rt)},
          open(f"{out}/run_info.json", "w"), indent=1)
PY
