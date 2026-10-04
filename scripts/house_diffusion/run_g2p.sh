#!/usr/bin/env bash
# HouseDiffusion on bubble diagrams derived from the Graph2Plan-preprocessed RPLAN test split
# (shared 1000-id subset), with SYNTHESISED interior doors -- see houseganpp/g2p_to_hgjson.py, NOTES.md.
# Corner-count distribution from 3000 random Graph2Plan *train* plans converted the same way.
set -euo pipefail
export PYTHONNOUSERSITE=1
# limit CPU threads (shared machine): torch uses TORCH_NUM_THREADS (infer.py), conversion uses a
# process pool of CONVERT_WORKERS single-threaded workers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4} CONVERT_WORKERS=${CONVERT_WORKERS:-8}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
IDS="${IDS:-$ROOT/data/method_inputs/common_rplan_test/ids_test_1000.txt}"
D="$ROOT/data/method_inputs/house_diffusion"
[ -f "$D/rplan-g2p_test1000/list.txt" ] || conda run -n fpe python scripts/houseganpp/g2p_to_hgjson.py --ids "$IDS" --out "$D/rplan-g2p_test1000"
[ -f "$D/rplan-g2p_train3000/list.txt" ] || conda run -n fpe python scripts/houseganpp/g2p_to_hgjson.py --split train --n_random 3000 --out "$D/rplan-g2p_train3000"
[ -f "$D/cndist_g2p_train3000.json" ] || PYTHONPATH="$ROOT/external/methods/house_diffusion" \
  conda run -n fpe-house_diffusion python scripts/house_diffusion/infer.py cndist "$D/rplan-g2p_train3000/list.txt" "$D/cndist_g2p_train3000.json"
INPUT_LIST="$D/rplan-g2p_test1000/list.txt" CNDIST="$D/cndist_g2p_train3000.json" \
VARIANT=rplan-g2p_bubble_test1000_syndoors N_PER_GRAPH=1 BATCH=100 \
RUN_NOTES="Inputs built from Graph2Plan RPLAN test split (ids_test_1000) by scripts/houseganpp/g2p_to_hgjson.py: room polygons/types exact, wall adjacency from shared boundaries, front door from boundary, INTERIOR DOORS SYNTHESISED. cndist from 3000 G2P train plans. 3 inputs skipped (>100 GT corners, as in the original eval-set filter), see raw/_infer_info.json." \
  bash scripts/house_diffusion/run.sh
