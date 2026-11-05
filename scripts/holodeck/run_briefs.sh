#!/bin/bash
# Holodeck on the Table 1 block (b) and (c) briefs, one model: run_briefs.sh qwen|gemma (on kw61498, servers up)
M=$1
cd /datawaha/cggroup/birsakm/floorplan_eval
V=/datawaha/cggroup/birsakm/venvs/fpe-holodeck/bin/python
for spec in "b next-living-bay-ad5110bb7b58" "c next-multistory-nav-4648dcf51a12"; do
  set -- $spec
  PYTHONPATH=. nice -n 10 $V scripts/holodeck/generate.py --model "$M" --block "$1" --version "$2" --workers 16 \
    > "logs/holodeck/${M}_block$1.log" 2>&1
done
echo DONE >> "logs/holodeck/${M}_blockc.log"
