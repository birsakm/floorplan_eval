#!/usr/bin/env bash
# End-to-end House-GAN++ run: env -> weights/inputs -> inference -> conversion -> renders -> run_info.
# Overridable: INPUT_LIST (file with HG++ JSON paths), VARIANT, N_PER_GRAPH, SEED, GPU.
set -euo pipefail
export PYTHONNOUSERSITE=1  # ignore ~/.local site-packages (they shadow env packages)
# limit CPU threads (shared machine): torch uses TORCH_NUM_THREADS (infer.py), conversion uses a
# process pool of CONVERT_WORKERS single-threaded workers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-4} CONVERT_WORKERS=${CONVERT_WORKERS:-8}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GPU="${GPU:-0}"
VARIANT="${VARIANT:-rplan_bubble_public}"
N_PER_GRAPH="${N_PER_GRAPH:-100}"
SEED="${SEED:-0}"
INPUT_LIST="${INPUT_LIST:-$ROOT/data/method_inputs/houseganpp/rplan_public/list.txt}"
OUT="$ROOT/outputs/houseganpp/$VARIANT"
HG="$ROOT/external/methods/houseganpp"
CKPT="$ROOT/checkpoints/houseganpp/pretrained.pth"

conda env list | awk '{print $1}' | grep -qx fpe-houseganpp || bash scripts/houseganpp/setup_env.sh
[ -f "$CKPT" ] && [ -f "$INPUT_LIST" ] || bash scripts/houseganpp/download.sh

rm -rf "$OUT"; mkdir -p "$OUT/raw"
CMD="python scripts/houseganpp/infer.py --checkpoint $CKPT --list $INPUT_LIST --out $OUT/raw --n_per_graph $N_PER_GRAPH --seed $SEED --save_png"
echo "$CMD"
T0=$(date +%s)
(cd "$HG" && CUDA_VISIBLE_DEVICES=$GPU HG_DIR="$HG" conda run --no-capture-output -n fpe-houseganpp \
   python "$ROOT/scripts/houseganpp/infer.py" --checkpoint "$CKPT" --list "$INPUT_LIST" --out "$OUT/raw" \
   --n_per_graph "$N_PER_GRAPH" --seed "$SEED" --save_png)
T1=$(date +%s)

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
conda run -n fpe python scripts/houseganpp/convert.py "$OUT"
conda run -n fpe python -m fpeval.render "$OUT"
conda run -n fpe python scripts/common/render_gt.py "$OUT"
conda run -n fpe python scripts/common/write_run_info.py "$OUT" --method houseganpp --submodule "$HG" \
  --checkpoint "checkpoints/houseganpp/pretrained.pth (= repo checkpoints/pretrained.pth)" \
  --command "$CMD" --ids "$INPUT_LIST" --infer_seconds $((T1-T0)) --env fpe-houseganpp \
  --conditioning "bubble diagram (HG++ JSON: rooms + door nodes, wall/door adjacency) from RPLAN; ${N_PER_GRAPH} samples per input graph, seed ${SEED}" \
  --notes "${RUN_NOTES:-Only the 7 publicly bundled RPLAN graphs are available (RPLAN is form-gated); see scripts/houseganpp/NOTES.md}" \
  --extra "{\"gpu\": \"1x A100-40GB (CUDA_VISIBLE_DEVICES=$GPU)\", \"n_per_graph\": $N_PER_GRAPH, \"seed\": $SEED, \"torch\": \"2.4.1+cu121 (repo pins 1.8.1)\"}"
