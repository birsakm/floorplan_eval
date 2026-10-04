#!/usr/bin/env bash
# Create the fpe-house_diffusion conda env. Original requirements pin a torch 2.0 nightly
# (2.0.0.dev20221212) + tensorflow; we use torch 2.4.1 + CUDA 12.1 wheels (A100/sm_80),
# drop tensorflow / pytorch_fid (only needed for FID inside image_sample.py, which we don't use),
# keep Shapely 1.8.x (code uses shapely.geos.lgeos) and drawSvg 1.9 (old API).
set -euo pipefail
export PYTHONNOUSERSITE=1  # ignore ~/.local site-packages (they shadow env packages)
ENV=fpe-house_diffusion
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" -c conda-forge python=3.10 pip mpi4py cairo && break || { echo "retry $i"; sleep 60; }
  done
fi
conda run -n "$ENV" pip install --no-cache-dir \
  torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n "$ENV" pip install --no-cache-dir \
  "numpy<2" blobfile "shapely==1.8.5.post1" "drawSvg==1.9.0" cairosvg imageio matplotlib networkx \
  opencv-python-headless Pillow tqdm webcolors scipy
# house_diffusion package is used via PYTHONPATH (no pip -e, keeps the submodule clean)
conda run -n "$ENV" python -c "import torch, shapely; print(torch.__version__, torch.cuda.is_available(), shapely.__version__)"
