#!/usr/bin/env bash
# End-to-end WallPlan run (boundary-only models): inference (fpe-wallplan) -> convert + render (fpe).
# Usage: scripts/wallplan/run.sh g2p [ids_file] [variant]     (Graph2Plan RPLAN test boundaries)
#        scripts/wallplan/run.sh bundled                        (the 500 pkls shipped with the repo)
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=${1:-g2p}
G2P=data/method_inputs/diffplanner/graph2plan_data/Network/data/data_test.mat
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
if [ "$SRC" = g2p ]; then
  IDS=${2:-data/method_inputs/common_rplan_test/ids_test_1000.txt}
  VARIANT=${3:-rplan_boundary_test1000}
  INARGS="--source g2p --g2p_mat $G2P --ids $IDS"; CONVARGS="--g2p_mat $G2P"
  COND="RPLAN boundary + front door (Graph2Plan test split), converted to WallPlan's 120px input masks"
  IDARG="--ids $IDS"
else
  VARIANT=bundled_boundary_test500
  INARGS="--source bundled"; CONVARGS=""; IDARG=""
  COND="boundary + front door from the 500 test pkls bundled in the WallPlan repo (test/input)"
fi
OUT=outputs/wallplan/$VARIANT
mkdir -p $OUT/raw
T0=$(date +%s)
conda run --no-capture-output -n fpe-wallplan python scripts/wallplan/infer.py $INARGS \
  --ckpt checkpoints/wallplan/Boundary_constraint --out $OUT/raw --seed 0 2>&1 | grep -v "it/s" | tee $OUT/infer.log
T1=$(date +%s)
rm -rf "${OUT:?}/samples" "${OUT:?}/gt_samples" "${OUT:?}/renders" "${OUT:?}/gt_renders"
PYTHONPATH=. conda run -n fpe python scripts/wallplan/convert.py $OUT $CONVARGS
PYTHONPATH=. conda run -n fpe python -m fpeval.render $OUT
[ -d $OUT/gt_samples ] && PYTHONPATH=. conda run -n fpe python scripts/common/render_gt.py $OUT
PYTHONPATH=. conda run -n fpe python scripts/common/write_run_info.py $OUT --method wallplan \
  --submodule external/methods/wallplan --checkpoint checkpoints/wallplan/Boundary_constraint \
  --command "scripts/wallplan/run.sh $*" --conditioning "$COND" $IDARG --infer_seconds $((T1-T0)) --env fpe-wallplan \
  --notes "torch 1.8.1+cu111; headless driver scripts/wallplan/infer.py (Boundary_Test.generate_from_val logic); deterministic nets, random window/door arrangement seeded (0); raw/png = official 512px render"
