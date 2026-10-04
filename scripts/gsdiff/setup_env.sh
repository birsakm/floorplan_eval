#!/usr/bin/env bash
# Create the fpe-gsdiff conda env following requirements.txt (torch 2.0.1 + cu118 supports sm_80/A100).
set -euo pipefail
ENV=fpe-gsdiff
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  for i in 1 2 3 4 5; do mamba create -y -n "$ENV" python=3.10 pip && break || { echo "retry mamba ($i)"; sleep 60; }; done
fi
conda run -n "$ENV" pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda run -n "$ENV" pip install --no-cache-dir numpy==1.26.0 opencv-python-headless==4.9.0.80 pillow==10.0.1 \
  networkx==3.1 scipy==1.10.1 shapely==2.0.6 scikit-image==0.24.0 scikit-learn==1.4.1.post1 matplotlib==3.9.2 \
  tqdm==4.65.0 pytorch-fid==0.3.0 sympy==1.12 tensorboardx==2.6.2.2
conda run -n "$ENV" python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
