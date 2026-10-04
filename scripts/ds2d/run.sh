#!/usr/bin/env bash
# End-to-end DS2D: env -> downloads -> test inputs -> generation (sharded over $GPUS) -> conversion -> renders.
# Usage: bash scripts/ds2d/run.sh [variant ...]   (default: all 10 released LoRAs)
# Env vars: GPUS (default "0"), BATCH0 (batch on GPU 0, default 12), BATCH (other GPUs, default 24),
#           DS2D_BASE_MODEL (default meta-llama/Meta-Llama-3-8B-Instruct; gated on HF),
#           N_RPLAN (500), N_PROCTHOR (1000), SKIP_SETUP=1
set -euo pipefail
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the envs
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
BASE="${DS2D_BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
GPUS="${GPUS:-0}"   # e.g. GPUS="0 1 2 3": one worker per GPU, variants assigned round-robin
BATCH0="${BATCH0:-12}"; BATCH="${BATCH:-24}"   # GPU 0 is shared (cap ~25 GB); other GPUs up to ~35 GB
N_RPLAN="${N_RPLAN:-500}"; N_PROCTHOR="${N_PROCTHOR:-1000}"
if [ "${SKIP_SETUP:-0}" != 1 ]; then
  bash scripts/ds2d/setup_env.sh
  bash scripts/ds2d/download.sh
  conda run -n fpe python scripts/ds2d/prepare_inputs.py --rplan_n "$N_RPLAN" --procthor_n "$N_PROCTHOR"
fi
COMMIT=$(git -C external/methods/ds2d rev-parse HEAD)

# variant : dataset : lora dir : prompt version : condition level : inputs
VARIANTS=(
  "rplan5R_bubble_roomarea_test:rplan:rplan/5R:-:only_room_area:rplan/test_inputs_5R.jsonl"
  "rplan6R_bubble_roomarea_test:rplan:rplan/6R:-:only_room_area:rplan/test_inputs_6R.jsonl"
  "rplan7R_bubble_roomarea_test:rplan:rplan/7R:-:only_room_area:rplan/test_inputs_7R.jsonl"
  "rplan8R_bubble_roomarea_test:rplan:rplan/8R:-:only_room_area:rplan/test_inputs_8R.jsonl"
  "procthor_bubble_constraints_test_lora-fullprompt:procthor:procthor_bd/full_prompt:bd:full_prompt:procthor/test_inputs.jsonl"
  "procthor_bubble_constraints_test_lora-mask:procthor:procthor_bd/mask:bd:full_prompt:procthor/test_inputs.jsonl"
  "procthor_bubble_constraints_test_lora-presetmask:procthor:procthor_bd/preset_mask:bd:full_prompt:procthor/test_inputs.jsonl"
  "procthor_constraints_test_lora-fullprompt:procthor:procthor_nonbd/full_prompt:non_bd:full_prompt:procthor/test_inputs.jsonl"
  "procthor_constraints_test_lora-mask:procthor:procthor_nonbd/mask:non_bd:full_prompt:procthor/test_inputs.jsonl"
  "procthor_constraints_test_lora-presetmask:procthor:procthor_nonbd/preset_mask:non_bd:full_prompt:procthor/test_inputs.jsonl"
)
run_variant() {  # spec gpu batch
  local spec="$1" GPU="$2" BATCH="$3"
  IFS=: read -r V DS LORA VER LEVEL INP <<<"$spec"
  OUT="outputs/ds2d/$V"; mkdir -p "$OUT/raw"
  N=$([ "$DS" = rplan ] && echo "$N_RPLAN" || echo "$N_PROCTHOR")
  VERARG=$([ "$VER" = - ] && echo "" || echo "--version $VER")
  CMD="conda run -n fpe-ds2d python scripts/ds2d/generate.py --dataset $DS --inputs data/method_inputs/ds2d/$INP --lora checkpoints/ds2d/$LORA $VERARG --level $LEVEL --limit $N --batch_size $BATCH --base_model $BASE --out_raw $OUT/raw"
  echo "== $V"
  # GT layouts for the same test inputs (independent of generation)
  PYTHONPATH="$ROOT" conda run -n fpe python scripts/ds2d/convert.py "$OUT" --dataset "$DS" --gt_from_inputs "data/method_inputs/ds2d/$INP" --limit "$N"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU $CMD >> "$OUT/raw/_generation.log" 2>&1
  T1=$(date +%s)
  if [ "$DS" = rplan ]; then
    COND="RPLAN room-count hold-out: LoRA ${LORA##*/} never saw ${LORA##*/}-room plans; input = bubble diagram (DS2D adjacency pairs from box proximity) + per-room {area (m^2), room_type, id} (DS2D level '$LEVEL')"
  elif [ "$VER" = bd ]; then
    COND="ProcTHOR bubble diagram (adjacency pairs) + full specification {room_count,total_area,room_types,rooms[{area,height,width,is_regular,room_type,id}]} (DS2D level '$LEVEL')"
  else
    COND="ProcTHOR constraints only (no bubble diagram): full specification {room_count,total_area,room_types,rooms[{area,height,width,is_regular,room_type,id}]} (DS2D level '$LEVEL')"
  fi
  PREV=$(python3 -c "import json,sys;print(json.load(open('$OUT/raw/_generation_info.json')).get('runtime_sec',0))" 2>/dev/null || echo 0)
  python3 - "$OUT/raw/_generation_info.json" <<PY
import json,sys,datetime
json.dump({"commit":"$COMMIT","base_model":"$BASE","lora":"checkpoints/ds2d/$LORA","version":"$VER","level":"$LEVEL",
 "inputs":"data/method_inputs/ds2d/$INP","n_requested":$N,"condition_description":"""$COND""",
 "decoding":"greedy, bf16 (no 8-bit), batched left-padded, max_new_tokens=$([ "$DS" = rplan ] && echo 2800 || echo 4000)",
 "command":"""$CMD""","runtime_sec":$PREV+$((T1-T0)),"gpu":"$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)",
 "date":datetime.datetime.now().isoformat(timespec="seconds")},open(sys.argv[1],"w"),indent=1)
PY
  PYTHONPATH="$ROOT" conda run -n fpe python scripts/ds2d/convert.py "$OUT" --dataset "$DS"
  conda run -n fpe python -m fpeval.render "$OUT"
}

want=("$@"); SEL=()
for spec in "${VARIANTS[@]}"; do   # procthor (heavier) first so round-robin balances the load
  V="${spec%%:*}"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $V " ]]; then continue; fi
  [[ "$spec" == *:procthor:* ]] && SEL=("$spec" "${SEL[@]}") || SEL+=("$spec")
done
read -r -a GPU_ARR <<<"$GPUS"; NG=${#GPU_ARR[@]}
for k in "${!GPU_ARR[@]}"; do
  G=${GPU_ARR[$k]}; B=$([ "$G" = 0 ] && echo "$BATCH0" || echo "$BATCH")
  (
    for ((j=k; j<${#SEL[@]}; j+=NG)); do
      echo "[gpu $G] start ${SEL[$j]%%:*} (batch $B) $(date +%T)"
      run_variant "${SEL[$j]}" "$G" "$B" || echo "[gpu $G] FAILED ${SEL[$j]%%:*}"
      echo "[gpu $G] done ${SEL[$j]%%:*} $(date +%T)"
    done
  ) &
done
wait
echo "all DS2D variants finished"
