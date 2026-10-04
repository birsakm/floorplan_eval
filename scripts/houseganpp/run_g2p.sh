#!/usr/bin/env bash
# House-GAN++ on bubble diagrams derived from the Graph2Plan-preprocessed RPLAN test split
# (shared 1000-id subset data/method_inputs/common_rplan_test/ids_test_1000.txt), with SYNTHESISED
# interior doors (Graph2Plan data has none) -- see g2p_to_hgjson.py and NOTES.md.
set -euo pipefail
export PYTHONNOUSERSITE=1
# limit CPU threads (shared machine): torch uses TORCH_NUM_THREADS (infer.py), conversion uses a
# process pool of CONVERT_WORKERS single-threaded workers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4} CONVERT_WORKERS=${CONVERT_WORKERS:-8}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
IDS="${IDS:-$ROOT/data/method_inputs/common_rplan_test/ids_test_1000.txt}"
IN="$ROOT/data/method_inputs/houseganpp/rplan-g2p_test1000"
[ -f "$IN/list.txt" ] || conda run -n fpe python scripts/houseganpp/g2p_to_hgjson.py --ids "$IDS" --out "$IN"
INPUT_LIST="$IN/list.txt" VARIANT=rplan-g2p_bubble_test1000_syndoors N_PER_GRAPH=1 \
RUN_NOTES="Inputs built from Graph2Plan RPLAN test split (ids_test_1000) by scripts/houseganpp/g2p_to_hgjson.py: room polygons/types exact, wall adjacency from shared boundaries, front door from boundary, INTERIOR DOORS SYNTHESISED (1 per non-living room, to living room if adjacent). Not the original HG++ JSON." \
  bash scripts/houseganpp/run.sh
