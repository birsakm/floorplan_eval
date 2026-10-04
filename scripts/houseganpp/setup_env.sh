#!/usr/bin/env bash
# Create the fpe-houseganpp conda env. Original repo pins torch 1.8.1 (no sm_80 / A100
# support); we use torch 2.4.1 + CUDA 12.1 wheels instead (API-compatible for this code).
set -euo pipefail
export PYTHONNOUSERSITE=1  # ignore ~/.local site-packages (they shadow env packages)
ENV=fpe-houseganpp
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  # retry in case another process holds the package-cache lock
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" -c conda-forge python=3.10 pip pygraphviz && break || { echo "retry $i"; sleep 60; }
  done
fi
conda list -n "$ENV" pygraphviz | grep -q pygraphviz || mamba install -y -n "$ENV" -c conda-forge pygraphviz
conda run -n "$ENV" pip install --no-cache-dir \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n "$ENV" pip install --no-cache-dir \
  "numpy<2" scikit-image matplotlib networkx opencv-python-headless Pillow svgwrite webcolors scipy tqdm
conda run -n "$ENV" python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
