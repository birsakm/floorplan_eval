#!/usr/bin/env bash
# Create the fpe-housegan conda env. Original repo pins torch 1.5.0+cu101 (no sm_80 / A100 support),
# so we use torch 2.1.2+cu118 and Pillow<10 (code uses Image.ANTIALIAS), numpy<1.24.
set -euo pipefail
ENV=fpe-housegan
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do mamba create -y -n "$ENV" python=3.9 pip && break || { echo "retry mamba ($i)"; sleep 60; }; done
fi
conda run -n "$ENV" pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
conda run -n "$ENV" pip install --no-cache-dir "numpy==1.23.5" "pillow==9.5.0" "opencv-python-headless==4.8.1.78" \
  "scikit-image==0.21.0" "scipy==1.10.1" "networkx==2.8.8" "matplotlib==3.7.3" webcolors==1.13 svgwrite==1.4.3 \
  pycocotools==2.0.7 imageio==2.31.6 tqdm
conda run -n "$ENV" python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
