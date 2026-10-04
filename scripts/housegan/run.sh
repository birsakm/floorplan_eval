#!/usr/bin/env bash
# End-to-end House-GAN: inference on the held-out group D (10-12 rooms, first 5000 graphs of
# dataset_paper/train_data.npy, as in the repo's eval scripts) -> conversion -> render -> run_info.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=$ROOT
V=lifull_bubble_testD
OUT=outputs/housegan/$V
rm -rf $OUT; mkdir -p $OUT/raw
CMD="python scripts/housegan/infer.py --data_dir data/method_inputs/housegan/dataset_paper --checkpoint checkpoints/housegan/exp_demo_D_500000.pth --out_dir $OUT/raw --target_set D --max_graphs 5000 --num_variations 1 --batch_size 64 --seed 0"
START=$(date +%s.%N)
conda run --no-capture-output -n fpe-housegan $CMD 2>&1 | tee $OUT/infer.log
SECS=$(echo "$(date +%s.%N) - $START" | bc)
conda run -n fpe python scripts/housegan/convert.py $OUT --variant $V --gt
conda run -n fpe python -m fpeval.render $OUT
conda run -n fpe python scripts/common/render_gt.py $OUT
conda run -n fpe python scripts/common/write_run_info.py $OUT --method housegan --submodule external/methods/housegan \
  --checkpoint checkpoints/housegan/exp_demo_D_500000.pth --command "$CMD" \
  --conditioning "bubble diagram (room types + box-adjacency edges) of LIFULL held-out group D (10-12 rooms), first 5000 graphs of dataset_paper/train_data.npy" \
  --infer_seconds $SECS --env fpe-housegan \
  --extra '{"gpu": "1x A100 (CUDA_VISIBLE_DEVICES=1)", "coord_frame": "256 px (House-GAN normalized coords x 256)", "num_variations": 1}' \
  --notes "see scripts/housegan/NOTES.md"
