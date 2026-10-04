#!/usr/bin/env bash
# End-to-end ChatHouseDiffusion on the full Tell2Design test split (2308 samples, 1 sample each).
# LLM stage skipped: the authors' pre-parsed graphs (moonshot-v1-8k) from kimi_test_data.rar are used.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V=t2d_test_textgraph
OUT="$ROOT/outputs/chathousediffusion/$V"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" PYTHONNOUSERSITE=1
mkdir -p "$OUT/raw"
T0=$(date +%s)
(cd "$ROOT/external/methods/chathousediffusion" && conda run --no-capture-output -n fpe-chathousediffusion \
   python "$ROOT/scripts/chathousediffusion/infer.py" --data "$ROOT/data/method_inputs/chathousediffusion" \
   --ckpt "$ROOT/checkpoints/chathousediffusion" --out "$OUT/raw" --batch 64 --seed 1029) 2>&1 \
   | grep -v "sampling loop" | tee "$OUT/raw/infer.log"
T1=$(date +%s)
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # converters use an explicit 8-worker pool
PYTHONPATH="$ROOT" conda run -n fpe python scripts/chathousediffusion/convert.py "$OUT" --data data/method_inputs/chathousediffusion --gt
conda run -n fpe python -m fpeval.render "$OUT"
PYTHONPATH="$ROOT" conda run -n fpe python - "$OUT" "$((T1-T0))" "$(git -C external/methods/chathousediffusion rev-parse HEAD 2>/dev/null)" <<'PY'
import json, sys, datetime, os
out, rt, commit = sys.argv[1:]
json.dump({"method": "chathousediffusion", "variant": os.path.basename(out), "commit": commit,
           "checkpoint": "checkpoints/chathousediffusion/model-98.pt (+params.pkl), Tsinghua cloud a01a8205be55462685fd",
           "command": "scripts/chathousediffusion/infer.py --batch 64 --seed 1029 (== upstream test.py / Trainer.val sampling, EMA model, cond_scale 1, DDIM 50 steps)",
           "conditioning": "boundary mask (with front door) + room graph parsed from Tell2Design text by the authors with moonshot-v1-8k (pre-parsed test data; no LLM called here)",
           "dataset": "Tell2Design test split (RPLAN-based), 64x64", "num_samples": len(os.listdir(f"{out}/samples")),
           "samples_per_input": 1, "date": datetime.date.today().isoformat(), "runtime_sec_inference": int(rt),
           "gt": "gt_samples/: Tell2Design GT label maps (image_test) vectorized the same way"},
          open(f"{out}/run_info.json", "w"), indent=1)
PY
