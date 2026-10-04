#!/usr/bin/env bash
# End-to-end HouseDiffusion run: env -> weights/inputs -> cndist -> sampling -> conversion -> renders -> run_info.
# Overridable: INPUT_LIST (file with HG++ JSON paths), CNDIST (corner-count dist json; default: pooled
# from the inputs), VARIANT, N_PER_GRAPH, BATCH, SEED, GPU.
set -euo pipefail
export PYTHONNOUSERSITE=1  # ignore ~/.local site-packages (they shadow env packages)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GPU="${GPU:-0}"
VARIANT="${VARIANT:-rplan_bubble_public}"
N_PER_GRAPH="${N_PER_GRAPH:-100}"
BATCH="${BATCH:-70}"
SEED="${SEED:-0}"
INDIR="$ROOT/data/method_inputs/house_diffusion/rplan_public"
INPUT_LIST="${INPUT_LIST:-$INDIR/list.txt}"
CKPT="$ROOT/checkpoints/house_diffusion/model250000.pt"
OUT="$ROOT/outputs/house_diffusion/$VARIANT"
HD="$ROOT/external/methods/house_diffusion"

conda env list | awk '{print $1}' | grep -qx fpe-house_diffusion || bash scripts/house_diffusion/setup_env.sh
[ -f "$CKPT" ] || bash scripts/house_diffusion/download.sh
[ -f "$INPUT_LIST" ] || bash scripts/houseganpp/download.sh   # copies the public RPLAN graphs for both methods

# Corner-count distribution per room type. The original uses the RPLAN *train* split
# (processed_rplan/rplan_train_8_cndist.npz); without RPLAN we pool it from the input graphs.
CNDIST="${CNDIST:-$(dirname "$INPUT_LIST")/cndist_pooled.json}"
[ -f "$CNDIST" ] || PYTHONPATH="$HD" conda run -n fpe-house_diffusion python scripts/house_diffusion/infer.py cndist "$INPUT_LIST" "$CNDIST"

rm -rf "$OUT"; mkdir -p "$OUT/raw"
CMD="python scripts/house_diffusion/infer.py --dataset rplan --set_name eval --target_set 8 --model_path $CKPT --list $INPUT_LIST --cndist $CNDIST --out $OUT/raw --n_per_graph $N_PER_GRAPH --batch_size $BATCH --seed $SEED"
echo "$CMD"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$HD" conda run --no-capture-output -n fpe-house_diffusion $CMD 2>&1 | grep -v "it/s\]"
T1=$(date +%s)

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
conda run -n fpe python scripts/house_diffusion/convert.py "$OUT"
conda run -n fpe python -m fpeval.render "$OUT"
conda run -n fpe python scripts/common/render_gt.py "$OUT"
conda run -n fpe python scripts/common/write_run_info.py "$OUT" --method house_diffusion --submodule "$HD" \
  --checkpoint "checkpoints/house_diffusion/model250000.pt (README 'temporary model', gdrive 16zKmtxwY5lF6JE-CJGkRf3-OFoD1TrdR)" \
  --command "$CMD" --ids "$INPUT_LIST" --infer_seconds $((T1-T0)) --env fpe-house_diffusion \
  --conditioning "bubble diagram (rooms + door nodes, HG++ JSON adjacency) from RPLAN, 'syn' conditions: corner counts sampled from $(basename "$CNDIST"); 1000 DDPM steps; ${N_PER_GRAPH} samples per graph, seed ${SEED}" \
  --notes "${RUN_NOTES:-Only the 7 publicly bundled RPLAN graphs are available (RPLAN is form-gated); they have 5-7 rooms (sizes seen in training, the checkpoint holds out 8-room plans) and may be training plans. See scripts/house_diffusion/NOTES.md}" \
  --extra "{\"gpu\": \"1x A100-40GB (CUDA_VISIBLE_DEVICES=$GPU)\", \"n_per_graph\": $N_PER_GRAPH, \"seed\": $SEED, \"torch\": \"2.4.1+cu121 (repo pins 2.0.0.dev20221212)\"}"
