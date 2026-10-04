#!/usr/bin/env bash
# Create the fpe-ds2d conda env for DStruct2Design inference.
# Versions follow external/methods/ds2d/requirements.txt (torch 2.3.0, transformers 4.40.1, peft 0.10.0).
set -euo pipefail
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the envs
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV=fpe-ds2d
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  # retry on conda package-cache lock conflicts (other agents may create envs concurrently)
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" python=3.10 pip && break
    echo "mamba create failed (attempt $i), retrying in 60s"; sleep 60
  done
fi
conda run -n "$ENV" pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
conda run -n "$ENV" pip install --no-cache-dir \
  transformers==4.40.1 tokenizers==0.19.1 peft==0.10.0 accelerate==0.29.3 bitsandbytes==0.43.1 \
  datasets==2.19.0 "huggingface-hub==0.22.2" safetensors==0.4.3 sentencepiece==0.2.0 \
  numpy==1.26.4 shapely==2.0.4 tqdm "pyarrow<16" "fsspec==2024.2.0"
# apply local patch (idempotent) if present
PATCH="$ROOT/patches/ds2d.patch"
if [ -s "$PATCH" ]; then
  cd "$ROOT/external/methods/ds2d"
  if git apply --reverse --check "$PATCH" 2>/dev/null; then echo "patch already applied";
  else git apply "$PATCH"; fi
fi
conda run -n "$ENV" python -c "import torch,transformers,peft;print(torch.__version__,torch.cuda.is_available(),transformers.__version__,peft.__version__)"
