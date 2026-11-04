#!/usr/bin/env bash
# Download Floor-plan RLVR weights into checkpoints/floorplan_rlvr/ (HF cache kept there too).
#   rlvr: ludolara/fp5-rlvr-Llama3.3-70B   full merged bf16 checkpoint (~141 GB, not gated)
#   sft : ludolara/fp5-sft-Llama3.3-70B    LoRA adapter r=64 (~3.3 GB, not gated)
#         + meta-llama/Llama-3.3-70B-Instruct base, safetensors only (~141 GB, gated: needs HF access)
# Usage: bash scripts/floorplan_rlvr/download.sh [rlvr] [sft]   (default: rlvr)
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CK="$ROOT/checkpoints/floorplan_rlvr"
export HF_HOME="$CK/hf_home" HF_HUB_CACHE="$CK/hf_home/hub" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"
ENV="${DL_ENV:-fpe-floorplan_rlvr}"
mkdir -p "$CK"
KEEP_FREE_GB="${KEEP_FREE_GB:-1000}"  # refuse to start a download that would leave < ~1 TB free
check_disk() {  # needed_GB
  local free_gb; free_gb=$(df -BG --output=avail /datawaha/cggroup | tail -1 | tr -dc 0-9)
  [ $((free_gb - $1)) -ge "$KEEP_FREE_GB" ] || {
    echo "not enough disk: ${free_gb} GB free, need $1 GB and must keep ~1 TB free"; exit 1; }
}
dl() {  # repo local_dir needed_GB [allow patterns...]
  local repo="$1" dir="$2" need="$3"; shift 3
  if [ -e "$CK/$dir/.complete" ]; then echo "$repo already downloaded"; return; fi
  check_disk "$need"
  conda run --no-capture-output -n "$ENV" python - "$repo" "$CK/$dir" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, d, *pats = sys.argv[1:]
snapshot_download(repo, local_dir=d, allow_patterns=pats or None, max_workers=8)
PY
  touch "$CK/$dir/.complete"
}
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(rlvr)
for w in "${want[@]}"; do
  case "$w" in
    rlvr) dl ludolara/fp5-rlvr-Llama3.3-70B fp5-rlvr-Llama3.3-70B 145 ;;
    sft)
      dl ludolara/fp5-sft-Llama3.3-70B fp5-sft-Llama3.3-70B 4
      # base: skip original/ (Meta consolidated .pth, another 141 GB)
      dl meta-llama/Llama-3.3-70B-Instruct Llama-3.3-70B-Instruct 145 "*.json" "*.safetensors" "LICENSE" "USE_POLICY.md" ;;
    *) echo "unknown: $w"; exit 1 ;;
  esac
done
du -sh "$CK"/* 2>/dev/null; df -h /datawaha/cggroup
