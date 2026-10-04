#!/usr/bin/env bash
# Create the fpe-diffplanner conda env. Original: Python 3.9.18 / torch 2.0.0 / numpy 1.21.5.
# torch 2.0.1+cu118 supports A100 (sm_80). mpi4py from conda-forge (needed by dist_util import).
set -euo pipefail
ENV=fpe-diffplanner
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do mamba create -y -n "$ENV" -c conda-forge python=3.9 pip mpi4py openmpi && break || { echo "retry mamba ($i)"; sleep 60; }; done
fi
conda run -n "$ENV" pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
conda run -n "$ENV" pip install --no-cache-dir numpy==1.21.5 scipy==1.10.1 shapely==2.0.6 opencv-python-headless==4.8.1.78 \
  blobfile==2.1.1 tqdm pillow
conda run -n "$ENV" python -c "import torch, mpi4py;print(torch.__version__, torch.cuda.is_available())"
