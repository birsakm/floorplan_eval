#!/usr/bin/env bash
# End-to-end GSDiff: inputs -> inference (3 variants) -> conversion -> render -> run_info.
# Usage: bash scripts/gsdiff/run.sh [uncond|topo|boun ...]   (default: all three)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH=$ROOT
IN=data/method_inputs/gsdiff
IDS=data/method_inputs/common_rplan_test/ids_test_1000.txt
MODES=${@:-uncond topo boun}
if [ ! -d $IN/rplan_test1000 ]; then
  conda run -n fpe python scripts/gsdiff/prepare_inputs.py --json $IN/dataset/dataset_json/data_test.json \
    --ids $IDS --out $IN/rplan_test1000 --gt_dir $IN/gt_test1000
fi
for MODE in $MODES; do
  case $MODE in
    uncond) V=rplan_uncond; COND=""; EXTRA="--num_samples 3000"; BS=1000; CONDDESC="none (unconditional)";;
    topo)   V=rplan_bubble_test1000; COND="--condition_type bubble_diagram"; EXTRA="--inputs $IN/rplan_test1000 --ids $IDS"; BS=1000;
            CONDDESC="bubble diagram (7-class rooms + wall adjacency) from DiffPlanner/Graph2Plan public RPLAN test json, common ids_test_1000";;
    boun)   V=rplan_boundary_test1000; COND="--condition_type boundary"; EXTRA="--inputs $IN/rplan_test1000 --ids $IDS"; BS=250;  # 1000 OOMs on 40 GB
            CONDDESC="boundary image (Graph2Plan boundary polygon, GSDiff drawing) from DiffPlanner/Graph2Plan public RPLAN test json, common ids_test_1000";;
  esac
  OUT=outputs/gsdiff/$V
  rm -rf $OUT; mkdir -p $OUT/raw
  CMD="python scripts/gsdiff/infer.py --mode $MODE --ckpt_root checkpoints/gsdiff --out_dir $OUT/raw $EXTRA --batch_size $BS --seed 0"
  conda run --no-capture-output -n fpe-gsdiff $CMD 2>&1 | tee $OUT/infer.log
  [ -f $OUT/raw/_log.json ] || { echo "inference failed for $MODE"; exit 1; }
  conda run -n fpe python scripts/gsdiff/convert.py $OUT --variant $V $COND
  conda run -n fpe python -m fpeval.render $OUT
  if [ "$MODE" != uncond ]; then
    mkdir -p $OUT/gt_samples
    for i in $(cat $IDS); do cp $IN/gt_test1000/$i.json $OUT/gt_samples/; done
    conda run -n fpe python scripts/common/render_gt.py $OUT
    IDARG="--ids $IDS"
  else IDARG=""; fi
  conda run -n fpe python scripts/common/write_run_info.py $OUT --method gsdiff --submodule external/methods/gsdiff \
    --checkpoint "checkpoints/gsdiff ($MODE: see raw/_log.json)" --command "$CMD" --conditioning "$CONDDESC" $IDARG \
    --env fpe-gsdiff --extra '{"gpu": "1x A100 (CUDA_VISIBLE_DEVICES=1)", "coord_frame": "RPLAN 256 px (GSDiff 512 frame / 2)"}' \
    --notes "see scripts/gsdiff/NOTES.md"
done
