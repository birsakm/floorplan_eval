#!/usr/bin/env bash
# End-to-end MaskPLAN: inference on the first N RPLAN test sites (default 500; upstream default 1000) for the
# paper's Table-1 partial-input settings, then conversion + rendering.
#   variants: rplan_partial25  = "Our II" (boundary + independently sampled 25% of T, L, A, S, R)
#             rplan_boundary   = "Our I"  (boundary only)
# Usage: bash scripts/maskplan/run.sh [N] [variant ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
N="${1:-500}"; shift || true  # 500 used for the reported run (≈1 GPU-h for both variants)
VARIANTS=("${@:-rplan_partial25 rplan_boundary}")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true
COMMIT="$(git -C external/methods/maskplan rev-parse HEAD 2>/dev/null || echo unknown)"
for V in ${VARIANTS[@]}; do
  case "$V" in
    rplan_partial25) ARGS="--par_T 0.25 --par_L 0.25 --par_A 0.25 --par_S 0.25 --par_R 0.25" ;;
    rplan_boundary)  ARGS="--par_T 0 --par_L 0 --par_A 0 --par_S 0 --par_R 0" ;;
    rplan_TLA)       ARGS="--par_T 1 --par_L 1 --par_A 1 --par_S 0 --par_R 0" ;;
    *) echo "unknown variant $V"; exit 1 ;;
  esac
  OUT="outputs/maskplan/$V"
  mkdir -p "$OUT/raw"
  T0=$(date +%s)
  conda run --no-capture-output -n fpe-maskplan python scripts/maskplan/infer.py --out "$OUT/raw" \
    --script cross_Deep --model Base --test_cases "$N" --seed 0 $ARGS 2>&1 | grep -v -E "^$|tensorflow/|ptxas|ptx compilation|PATH to customize|logged once" | tee "$OUT/raw/infer.log"
  T1=$(date +%s)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # converters use an explicit 8-worker pool
  PYTHONPATH="$ROOT" conda run -n fpe python scripts/maskplan/convert.py "$OUT" --gt
  conda run -n fpe python -m fpeval.render "$OUT"
  NS=$(ls "$OUT/samples" | wc -l)
  PYTHONPATH="$ROOT" conda run -n fpe python - "$OUT" "$V" "$COMMIT" "$N" "$ARGS" "$((T1-T0))" "$NS" <<'PY'
import json, sys, datetime
out, v, commit, n, args, rt, ns = sys.argv[1:]
info = {"method": "maskplan", "variant": v, "commit": commit,
        "checkpoint": "checkpoints/maskplan/MaskPLAN_Trained/All_Base_Deep_cross/All (Drive 1mzZdK313lYybr4GMN6_pF6lW3tRFwLjM) + repo VQ_Pretrained/mix_5564",
        "command": f"python scripts/maskplan/infer.py --script cross_Deep --model Base --test_cases {n} --seed 0 {args}",
        "upstream_script": "Inference/MaskPLAN_Inference_iterate_cross_Deep.py (imported unmodified)",
        "conditioning": {"rplan_partial25": "boundary + front door + random 25% of each attribute (T, L, A, S, R) = paper 'Our II'",
                          "rplan_boundary": "boundary + front door only = paper 'Our I'",
                          "rplan_TLA": "boundary + complete T, L, A = paper 'Our III'"}[v],
        "dataset": "RPLAN test split, first N ids of Processed_data/Test_set.npy (8079 total)",
        "num_samples": int(ns), "num_inputs": int(n), "samples_per_input": 1, "seed": "per-site seed = site_id (np + tf)",
        "date": datetime.date.today().isoformat(), "runtime_sec_inference": int(rt),
        "gpu": "1x A100-40GB (CUDA_VISIBLE_DEVICES=3), TF 2.6.0 via PTX JIT for sm_80",
        "gt": "gt_samples/: room bounding boxes from MaskPLAN's preprocessed RPLAN vec data (quantised)"}
json.dump(info, open(f"{out}/run_info.json", "w"), indent=1)
PY
done
