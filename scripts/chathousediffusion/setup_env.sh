#!/usr/bin/env bash
# Create the fpe-chathousediffusion env (Linux; upstream tested on Windows / Python 3.10).
set -euo pipefail
export PYTHONNOUSERSITE=1  # ~/.local/lib/python3.10 would otherwise shadow env packages
ENV=fpe-chathousediffusion
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" -c conda-forge python=3.10 pip && break
    echo "mamba create failed (cache lock?), retrying in 60s"; sleep 60
  done
fi
conda env config vars set -n "$ENV" PYTHONNOUSERSITE=1
conda run -n "$ENV" pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
# CPU build of DGL: the cu118 wheel needs system libcusparse.so.11; DGL is only used for CPU graph preprocessing here
conda run -n "$ENV" pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/repo.html
# torchdata 0.7.x for dgl 2.0
conda run -n "$ENV" pip install "numpy==1.26.0" "ema_pytorch==0.4.3" "transformers==4.40.2" sentencepiece \
  "pandas==2.1.4" "einops==0.7.0" tqdm pillow requests fuzzywuzzy "torchdata==0.7.1" pydantic pyyaml \
  "opencv-python-headless==4.8.1.78"
conda run -n "$ENV" python -c "import torch, dgl; print(torch.__version__, dgl.__version__, torch.cuda.is_available())"
