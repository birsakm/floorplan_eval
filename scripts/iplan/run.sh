#!/usr/bin/env bash
# End-to-end iPLAN run: inference (fpe-iplan) -> conversion + render (fpe).
# Usage: scripts/iplan/run.sh [ids_file] [variant]
set -euo pipefail
cd "$(dirname "$0")/../.."
IDS=${1:-data/method_inputs/common_rplan_test/ids_test_1000.txt}
VARIANT=${2:-rplan_boundary_test1000}
OUT=outputs/iplan/$VARIANT
G2P=data/method_inputs/diffplanner/graph2plan_data/Network/data/data_test.mat
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
mkdir -p $OUT/raw
T0=$(date +%s)
conda run --no-capture-output -n fpe-iplan python scripts/iplan/infer.py --g2p_mat $G2P --ids $IDS \
  --ckpt checkpoints/iplan/iPLAN --out $OUT/raw --seed 0 2>&1 | tee $OUT/infer.log
T1=$(date +%s)
rm -rf "${OUT:?}/samples" "${OUT:?}/gt_samples" "${OUT:?}/renders" "${OUT:?}/gt_renders"
PYTHONPATH=. conda run -n fpe python scripts/iplan/convert.py $OUT --g2p_mat $G2P
PYTHONPATH=. conda run -n fpe python -m fpeval.render $OUT
# GT renders: render gt_samples into gt_renders/
PYTHONPATH=. conda run -n fpe python scripts/common/render_gt.py $OUT
PYTHONPATH=. conda run -n fpe python scripts/common/write_run_info.py $OUT --method iplan \
  --submodule external/methods/iplan --checkpoint checkpoints/iplan/iPLAN \
  --command "scripts/iplan/run.sh $IDS $VARIANT" --conditioning "RPLAN boundary + front door (Graph2Plan test split)" \
  --ids $IDS --infer_seconds $((T1-T0)) --env fpe-iplan \
  --notes "torch 1.8.1+cu111 (orig 1.7.0/cu101); headless driver scripts/iplan/infer.py; 128px raster vectorized, x2 to 256px RPLAN frame; one sample per boundary (seed 0)"
