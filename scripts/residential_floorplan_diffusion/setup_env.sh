#!/usr/bin/env bash
# Create fpe-resdiff (upstream pins torch 1.12 / py3.8; upgraded to torch 2.1.2 + cu118 for A100).
set -euo pipefail
export PYTHONNOUSERSITE=1  # ~/.local/lib/python3.10 would otherwise shadow env packages
ENV=fpe-resdiff
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do
    mamba create -y -n "$ENV" -c conda-forge python=3.10 pip && break
    echo "mamba create failed (cache lock?), retrying in 60s"; sleep 60
  done
fi
conda env config vars set -n "$ENV" PYTHONNOUSERSITE=1
conda run -n "$ENV" pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
conda run -n "$ENV" pip install "numpy==1.26.4" "opencv-python-headless==4.8.1.78" "matplotlib==3.7.1" \
  "scipy==1.11.4" "scikit-image==0.22.0" natsort "pillow==10.3.0" "tqdm==4.65.0" "pandas==2.1.4" thop
conda run -n "$ENV" python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
