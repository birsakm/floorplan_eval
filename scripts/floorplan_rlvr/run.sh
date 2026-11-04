#!/usr/bin/env bash
# End-to-end Floor-plan RLVR: env -> weights -> prompts -> vLLM best-of-n generation (TP=4) -> conversion -> renders.
# Usage: bash scripts/floorplan_rlvr/run.sh [rlvr|sft ...]   (default: rlvr)
# Env vars: QUANT (none | fp8; default none = bf16; fp8 degrades this model, see NOTES.md), N_BEST (10), LIMIT (all 1000),
#           GPU_MEM (0.95), MAX_BATCHED (2048), MAX_MODEL_LEN (5120), MAX_TOKENS (4096), MAX_NUM_SEQS (256), CHUNK (100),
#           SUFFIX (appended to the variant name, e.g. _smoke), SKIP_SETUP=1, CUDA_VISIBLE_DEVICES (0,1,2,3)
set -euo pipefail
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the envs
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CK="$ROOT/checkpoints/floorplan_rlvr"
export HF_HOME="$CK/hf_home" HF_HUB_CACHE="$CK/hf_home/hub" HF_HUB_OFFLINE=1 \
       VLLM_CACHE_ROOT="$CK/vllm_cache" TRITON_CACHE_DIR="$CK/triton_cache" TORCHINDUCTOR_CACHE_DIR="$CK/inductor_cache"
QUANT="${QUANT:-none}"; N_BEST="${N_BEST:-10}"; GPU_MEM="${GPU_MEM:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-5120}"; MAX_TOKENS="${MAX_TOKENS:-4096}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
CHUNK="${CHUNK:-100}"; SUFFIX="${SUFFIX:-}"; MAX_BATCHED="${MAX_BATCHED:-2048}"
INPUTS="data/method_inputs/floorplan_rlvr/rplan-g2p_test1000/inputs.jsonl"
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(rlvr)
if [ "${SKIP_SETUP:-0}" != 1 ]; then
  bash scripts/floorplan_rlvr/setup_env.sh
  bash scripts/floorplan_rlvr/download.sh "${want[@]}"
fi
[ -s "$INPUTS" ] || conda run -n fpe-floorplan_rlvr python scripts/floorplan_rlvr/prepare_inputs.py
NIN=$(wc -l < "$INPUTS"); N=${LIMIT:-$NIN}
COMMIT=$(git -C external/methods/floorplan_rlvr rev-parse HEAD)
QARG=$([ "$QUANT" = none ] && echo "" || echo "--quantization $QUANT")
LIMARG=$([ -n "${LIMIT:-}" ] && echo "--limit $LIMIT" || echo "")

for w in "${want[@]}"; do
  case "$w" in
    rlvr) V=rplan-g2p_bubble_areas_test1000_syndoors; MODEL="$CK/fp5-rlvr-Llama3.3-70B"; LORA=""
          CKDESC="HF ludolara/fp5-rlvr-Llama3.3-70B (full merged SFT+RLVR checkpoint, 5-room held-out model)" ;;
    sft)  V=rplan-g2p_bubble_areas_test1000_syndoors_sft; MODEL="$CK/Llama-3.3-70B-Instruct"; LORA="--lora $CK/fp5-sft-Llama3.3-70B"
          CKDESC="HF meta-llama/Llama-3.3-70B-Instruct + LoRA ludolara/fp5-sft-Llama3.3-70B (r=64, SFT only, 5-room held-out model)" ;;
    *) echo "unknown model $w"; exit 1 ;;
  esac
  V="$V$SUFFIX"; OUT="outputs/floorplan_rlvr/$V"; mkdir -p "$OUT/raw"
  CMD="python scripts/floorplan_rlvr/generate.py --inputs $INPUTS --model $MODEL $LORA --out_raw $OUT/raw --n_best $N_BEST --temperature 0.7 --top_p 0.9 --max_tokens $MAX_TOKENS --max_model_len $MAX_MODEL_LEN --gpu_mem $GPU_MEM $QARG --tp 4 --max_num_seqs $MAX_NUM_SEQS --max_num_batched_tokens $MAX_BATCHED --chunk $CHUNK $LIMARG"
  echo "== $V: $CMD"
  T0=$(date +%s)
  conda run --no-capture-output -n fpe-floorplan_rlvr $CMD 2>&1 | tee -a "$OUT/raw/_generation.log"
  T1=$(date +%s)
  PREV=$(python3 -c "import json;print(json.load(open('$OUT/raw/_generation_info.json')).get('runtime_sec',0))" 2>/dev/null || echo 0)
  VERS=$(conda run -n fpe-floorplan_rlvr python -c "import torch,vllm,transformers;print(f'torch {torch.__version__}, vllm {vllm.__version__}, transformers {transformers.__version__}')" 2>/dev/null | tail -1)
  python3 - "$OUT/raw/_generation_info.json" <<PY
import json,sys,datetime
json.dump({"submodule_commit":"$COMMIT","checkpoint":"""$CKDESC""","model_path":"$MODEL","lora":"$LORA",
 "precision": "bf16 activations, " + ("bf16 weights" if "$QUANT"=="none" else "vLLM $QUANT weight quantization (W8A16 Marlin on A100; deviation from bf16)"),
 "n_requested":$N,
 "condition_description":"RPLAN (Graph2Plan test subset, 1000 ids) -> RLVR JSON prompt via repo RPLANConverter: room_count, total_area (m^2), spaces [{id, room_type, area | height+width}] incl. front door, input_graph = interior-door connectivity (doors synthesized, see scripts/houseganpp/NOTES.md)",
 "decoding":"best-of-$N_BEST sampling, temperature 0.7, top_p 0.9, max_tokens $MAX_TOKENS, stop at <|eot_id|>/<|eom_id|>/<|end_of_text|>; candidate chosen by repo FloorplanGenerator._select_least (paper criterion pick also recorded); single BOS (add_special_tokens=False, as in training)",
 "vllm":"tensor_parallel_size=4, max_model_len=$MAX_MODEL_LEN, gpu_memory_utilization=$GPU_MEM, max_num_batched_tokens=$MAX_BATCHED, max_num_seqs=$MAX_NUM_SEQS, prefix caching on",
 "env":"conda fpe-floorplan_rlvr: $VERS",
 "command":"""CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $CMD""","runtime_sec":$PREV+$((T1-T0)),
 "gpu":"4x $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)",
 "date_generated":datetime.datetime.now().isoformat(timespec="seconds")},open(sys.argv[1],"w"),indent=1)
PY
  PYTHONPATH="$ROOT" conda run -n fpe python scripts/floorplan_rlvr/convert.py "$OUT" --inputs "$INPUTS"
  PYTHONPATH="$ROOT" conda run -n fpe python -m fpeval.render "$OUT"
  # GT renders for side-by-side checks (fpeval.render only renders samples/)
  PYTHONPATH="$ROOT" conda run -n fpe python -c "
from pathlib import Path
from fpeval.render import render
from fpeval.format import load_sample
v = Path('$OUT'); (v / 'gt_renders').mkdir(exist_ok=True)
for p in sorted((v / 'gt_samples').glob('*.json')):
    render(load_sample(p)).save(v / 'gt_renders' / f'{p.stem}.png')
"
done
