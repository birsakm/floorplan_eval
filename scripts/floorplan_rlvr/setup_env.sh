#!/usr/bin/env bash
# Create the fpe-floorplan_rlvr conda env for Floor-plan RLVR inference with vLLM.
# requirements_pred.txt asks for vllm + torch 2.6 (unpinned vllm); vllm 0.8.5.post1 is the last
# release built against torch 2.6.0 and still has the V0 engine that run_generation.py selects
# (VLLM_USE_V1=0) for best_of sampling.
set -euo pipefail
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the envs
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV=fpe-floorplan_rlvr
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  # retry on conda package-cache lock conflicts (other agents may create envs concurrently)
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" python=3.11 pip && break
    echo "mamba create failed (attempt $i), retrying in 60s"; sleep 60
  done
fi
conda run -n "$ENV" pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0"
# pin transformers/hub/datasets: unpinned installs pull transformers 5.x / hub 1.x, which vllm 0.8.5 does not support
conda run -n "$ENV" pip install --no-cache-dir "transformers==4.51.3" "huggingface_hub[hf_transfer]<1.0" "datasets<4" "fsspec<=2025.3.0" shapely networkx python-dotenv scikit-learn "numpy<2.3"
# The submodule is used unmodified; apply a local patch only if one exists (idempotent).
PATCH="$ROOT/patches/floorplan_rlvr.patch"
if [ -s "$PATCH" ]; then
  cd "$ROOT/external/methods/floorplan_rlvr"
  if git apply --reverse --check "$PATCH" 2>/dev/null; then echo "patch already applied";
  else git apply "$PATCH"; fi
fi
conda run -n "$ENV" python -c "import torch,vllm,transformers;print(torch.__version__,torch.cuda.is_available(),vllm.__version__,transformers.__version__)"
